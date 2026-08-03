from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.interimschema import (
    TMP_EVAL_ROOT,
    stage_file,
    stage_manifest_file,
    base_record,
)

INPUT_PATH = stage_file("generated_candidates")
OUT_PATH = stage_file("execution_results")
MANIFEST_PATH = stage_manifest_file("execution_results")

PYTEST_TIMEOUT_SEC = 20
MAX_TRACEBACK_CHARS = 1000
SCRIPT_VERSION = "run_execution_v5"


# =========================================================
# 基础 IO
# =========================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# 通用字符串 / 路径工具
# =========================================================

def clip_text(text: Optional[str], max_chars: int = MAX_TRACEBACK_CHARS) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    return text[:max_chars] if text else None


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def sanitize_fs_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", name)
    safe = safe.strip("._")
    return safe[:180] or "candidate"


def normalize_imports(imports: Any) -> str:
    if isinstance(imports, list):
        cleaned = [str(x).strip() for x in imports if str(x).strip()]
        return "\n".join(cleaned)
    if isinstance(imports, str):
        return imports.strip()
    return ""


def extract_prompt_import_lines(prompt: str) -> str:
    """
    从 prompt 顶部提取 import / from ... import ... 语句。
    只提取连续出现在前部的导入行，遇到 def 或其他正文就停止。
    """
    if not prompt or not isinstance(prompt, str):
        return ""

    lines = prompt.splitlines()
    imports: List[str] = []

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if not imports:
                continue
            else:
                continue

        if stripped.startswith("def "):
            break

        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
            continue

        break

    return "\n".join(imports)


def deduplicate_code_blocks(blocks: List[str]) -> str:
    """
    简单按完整块去重，避免重复 import / 重复片段。
    """
    seen = set()
    kept = []
    for block in blocks:
        block = (block or "").strip()
        if not block:
            continue
        if block in seen:
            continue
        seen.add(block)
        kept.append(block)
    return "\n\n".join(kept).strip()


# =========================================================
# Solution 组装
# =========================================================

def canonical_solution_is_full_function(solution_code: str, entry_point: Optional[str]) -> bool:
    """
    用 AST 判断 canonical_solution 是否已经包含完整的顶层函数定义。
    """
    if not solution_code or not solution_code.strip():
        return False

    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return False

    func_names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_names.append(node.name)

    if not func_names:
        return False

    if entry_point and isinstance(entry_point, str) and entry_point.strip():
        return entry_point.strip() in func_names

    return True


def _scan_signature_from_lines(
    lines: List[str],
    start_pattern: re.Pattern[str],
) -> Optional[str]:
    """
    从 prompt 的多行文本中扫描函数签名。
    兼容单行和多行签名，直到遇到签名结束的冒号。
    """
    for i, line in enumerate(lines):
        if not start_pattern.match(line):
            continue

        buf = [line.rstrip()]
        paren_balance = line.count("(") - line.count(")")
        j = i

        while True:
            current = buf[-1].strip()
            if current.endswith(":") and paren_balance <= 0:
                return "\n".join(x.rstrip() for x in buf).strip()

            j += 1
            if j >= len(lines):
                break

            next_line = lines[j]
            buf.append(next_line.rstrip())
            paren_balance += next_line.count("(") - next_line.count(")")

    return None


def extract_signature_from_prompt(prompt: str, entry_point: Optional[str]) -> Optional[str]:
    """
    从 prompt 中提取函数签名。
    优先匹配 entry_point；否则匹配第一个 def 签名。
    兼容：
      - 返回类型注解: def foo(x: int) -> int:
      - 多行签名
    """
    if not prompt or not isinstance(prompt, str):
        return None

    lines = prompt.splitlines()

    if entry_point and isinstance(entry_point, str) and entry_point.strip():
        ep = re.escape(entry_point.strip())
        strict_pattern = re.compile(rf"^\s*def\s+{ep}\s*\(")
        sig = _scan_signature_from_lines(lines, strict_pattern)
        if sig:
            return sig

    loose_pattern = re.compile(r"^\s*def\s+[A-Za-z_]\w*\s*\(")
    return _scan_signature_from_lines(lines, loose_pattern)


def indent_function_body(body: str) -> str:
    """
    先 dedent，再统一缩进到函数体层级。
    """
    dedented = textwrap.dedent(body).strip("\n")
    if not dedented.strip():
        return "    pass"

    lines = dedented.splitlines()
    indented: List[str] = []
    for line in lines:
        if line.strip():
            indented.append("    " + line)
        else:
            indented.append("")
    return "\n".join(indented)


def build_solution_file_code(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    返回:
      (solution_file_code, solution_assembly_mode)

    solution_assembly_mode:
      - full_function
      - wrapped_from_prompt_signature
      - missing_solution
      - signature_not_found
    """
    row_imports_code = normalize_imports(row.get("imports", []))
    prompt = row.get("prompt") or ""
    prompt_imports_code = extract_prompt_import_lines(prompt)

    solution_code = (row.get("canonical_solution") or "").rstrip()
    entry_point = row.get("entry_point")

    if not solution_code.strip():
        return "", "missing_solution"

    if canonical_solution_is_full_function(solution_code, entry_point):
        final_solution = solution_code
        assembly_mode = "full_function"
    else:
        signature = extract_signature_from_prompt(prompt, entry_point)
        if not signature:
            return "", "signature_not_found"
        final_solution = f"{signature}\n{indent_function_body(solution_code)}"
        assembly_mode = "wrapped_from_prompt_signature"

    merged = deduplicate_code_blocks([
        row_imports_code,
        prompt_imports_code,
        final_solution,
    ])

    return ((merged + "\n") if merged else "", assembly_mode)


# =========================================================
# Test 文件生成
# =========================================================

def build_test_file_code(row: Dict[str, Any]) -> str:
    """
    默认把被测函数从 solution_under_test 导入进来。
    """
    candidate_test_code = (row.get("candidate_test_code") or "").strip()

    if not candidate_test_code:
        return (
            "from solution_under_test import *\n\n"
            "raise ValueError('Empty candidate_test_code. Failing explicitly.')\n"
        )

    prelude_blocks = []
    if (
        "import solution_under_test" not in candidate_test_code
        and "from solution_under_test" not in candidate_test_code
    ):
        prelude_blocks.append("from solution_under_test import *")

    merged = "\n\n".join(
        block.strip()
        for block in [*prelude_blocks, candidate_test_code]
        if isinstance(block, str) and block.strip()
    )
    return merged + "\n"


# =========================================================
# 测试代码分析
# =========================================================

def analyze_test_code(candidate_test_code: str) -> Tuple[str, int]:
    """
    返回:
      runner_mode: pytest / script
      top_level_asserts_count
    """
    try:
        tree = ast.parse(candidate_test_code)
    except SyntaxError:
        return "script", 0

    has_pytest_style = False
    top_level_asserts = 0

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            has_pytest_style = True
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            has_pytest_style = True
        elif isinstance(node, ast.Assert):
            top_level_asserts += 1

    runner_mode = "pytest" if has_pytest_style else "script"
    return runner_mode, top_level_asserts


# =========================================================
# 执行环境与日志解析
# =========================================================

def build_subprocess_env(work_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    old_pythonpath = env.get("PYTHONPATH", "")
    prefix = f"{work_dir}{os.pathsep}{REPO_ROOT}"
    env["PYTHONPATH"] = f"{prefix}{os.pathsep}{old_pythonpath}" if old_pythonpath else prefix
    return env


def parse_pytest_counts(stdout: str, stderr: str) -> Tuple[Optional[int], int, int, int]:
    text = f"{stdout}\n{stderr}"

    patterns = {
        "passed": r"(\d+)\s+passed",
        "failed": r"(\d+)\s+failed",
        "errors": r"(\d+)\s+error(?:s)?",
        "skipped": r"(\d+)\s+skipped",
        "xfailed": r"(\d+)\s+xfailed",
        "xpassed": r"(\d+)\s+xpassed",
    }

    counts = {k: 0 for k in patterns}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            counts[key] = int(match.group(1))

    collected_match = re.search(r"collected\s+(\d+)\s+items?", text)
    collected = int(collected_match.group(1)) if collected_match else None

    if collected is None:
        summary_total = sum(counts.values())
        if summary_total > 0:
            collected = summary_total

    return collected, counts["passed"], counts["failed"], counts["errors"]


def extract_exception_type(text: str) -> Optional[str]:
    if not text:
        return None

    matches = re.findall(
        r"^([A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt))(?::|\b)",
        text,
        flags=re.MULTILINE,
    )
    if matches:
        return matches[-1]

    known = [
        "AssertionError",
        "SyntaxError",
        "TypeError",
        "ValueError",
        "IndexError",
        "KeyError",
        "RuntimeError",
        "ImportError",
        "ModuleNotFoundError",
        "NameError",
        "AttributeError",
        "ZeroDivisionError",
        "TimeoutExpired",
    ]
    for name in known:
        if name in text:
            return name

    return None


def run_with_pytest(test_path: Path, work_dir: Path) -> Tuple[subprocess.CompletedProcess[str], float]:
    env = build_subprocess_env(work_dir)
    start = time.time()
    proc = subprocess.run(
        [
            sys.executable,
            "-m", "pytest",
            "-q",
            "--tb=short",
            "-p", "no:cacheprovider",
            "-o", "cache_dir=/tmp",
            test_path.name,
        ],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=PYTEST_TIMEOUT_SEC,
        env=env,
    )
    return proc, round(time.time() - start, 4)


def run_as_script(test_path: Path, work_dir: Path) -> Tuple[subprocess.CompletedProcess[str], float]:
    env = build_subprocess_env(work_dir)
    start = time.time()
    proc = subprocess.run(
        [sys.executable, test_path.name],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=PYTEST_TIMEOUT_SEC,
        env=env,
    )
    return proc, round(time.time() - start, 4)


# =========================================================
# 主流程
# =========================================================

def main() -> None:
    rows = load_jsonl(INPUT_PATH)
    TMP_EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    out: List[Dict[str, Any]] = []

    for row in rows:
        candidate_id = row["candidate_id"]
        work_dir = TMP_EVAL_ROOT / sanitize_fs_name(candidate_id)

        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)

        solution_path = work_dir / "solution_under_test.py"
        test_path = work_dir / "test_candidate.py"
        stdout_path = work_dir / "stdout.txt"
        stderr_path = work_dir / "stderr.txt"

        rec: Dict[str, Any] = {
            **base_record(SCRIPT_VERSION),
            "candidate_id": candidate_id,
            "task_id": row["task_id"],
            "source": row["source"],
            "difficulty_bucket": row.get("difficulty_bucket", "unknown"),
            "test_source_type": row.get("test_source_type", "unknown"),
            "candidate_test_origin_detail": row.get("candidate_test_origin_detail"),
            "smoke_strength": row.get("smoke_strength"),
            "solution_assembly_mode": None,
            "runner_mode": None,
            "top_level_asserts_count": None,
            "syntax_pass": True,
            "execution_pass": False,
            "execution_status": None,
            "return_code": None,
            "exception_type": None,
            "traceback_snippet": None,
            "runtime_sec": None,
            "num_tests_collected": None,
            "num_tests_passed": None,
            "num_tests_failed": None,
            "num_tests_errors": None,
            "work_dir": str(work_dir),
            "solution_path": str(solution_path),
            "test_path": str(test_path),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
        }

        stdout_text = ""
        stderr_text = ""

        # 1) 组装 solution 文件
        solution_file_code, solution_assembly_mode = build_solution_file_code(row)
        rec["solution_assembly_mode"] = solution_assembly_mode

        if solution_assembly_mode == "missing_solution":
            rec.update({
                "syntax_pass": False,
                "execution_pass": False,
                "execution_status": "infra_error",
                "exception_type": "MissingSolution",
                "traceback_snippet": "canonical_solution is empty or missing.",
            })
            stderr_text = "canonical_solution is empty or missing."
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            out.append(rec)
            continue

        if solution_assembly_mode == "signature_not_found":
            rec.update({
                "syntax_pass": False,
                "execution_pass": False,
                "execution_status": "infra_error",
                "exception_type": "SolutionAssemblyError",
                "traceback_snippet": "Failed to extract function signature from prompt for body-only canonical_solution.",
            })
            stderr_text = "Failed to extract function signature from prompt for body-only canonical_solution."
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            out.append(rec)
            continue

        solution_path.write_text(solution_file_code, encoding="utf-8")

        # 2) 生成 test 文件
        test_file_code = build_test_file_code(row)
        test_path.write_text(test_file_code, encoding="utf-8")

        # 3) 语法检查
        try:
            ast.parse(solution_file_code)
            ast.parse(test_file_code)
        except SyntaxError as e:
            rec.update({
                "syntax_pass": False,
                "execution_status": "syntax_error",
                "exception_type": "SyntaxError",
                "traceback_snippet": clip_text(str(e)),
            })
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text(str(e), encoding="utf-8")
            out.append(rec)
            continue

        # 4) 分析测试代码，决定运行方式
        candidate_test_code = row.get("candidate_test_code", "") or ""
        runner_mode, top_level_asserts = analyze_test_code(candidate_test_code)
        rec["runner_mode"] = runner_mode
        rec["top_level_asserts_count"] = top_level_asserts

        # 5) 执行
        try:
            if runner_mode == "pytest":
                proc, runtime = run_with_pytest(test_path, work_dir)
                stdout_text = safe_text(proc.stdout)
                stderr_text = safe_text(proc.stderr)

                collected, passed, failed, errors = parse_pytest_counts(stdout_text, stderr_text)

                rec.update({
                    "runtime_sec": runtime,
                    "return_code": proc.returncode,
                    "num_tests_collected": collected,
                    "num_tests_passed": passed,
                    "num_tests_failed": failed,
                    "num_tests_errors": errors,
                    "execution_pass": (proc.returncode == 0),
                    "execution_status": "passed" if proc.returncode == 0 else "failed",
                })

            else:
                proc, runtime = run_as_script(test_path, work_dir)
                stdout_text = safe_text(proc.stdout)
                stderr_text = safe_text(proc.stderr)

                rec.update({
                    "runtime_sec": runtime,
                    "return_code": proc.returncode,
                    "num_tests_collected": top_level_asserts,
                    "num_tests_passed": top_level_asserts if proc.returncode == 0 else None,
                    "num_tests_failed": 1 if (proc.returncode != 0 and top_level_asserts > 0) else None,
                    "num_tests_errors": 0,
                    "execution_pass": (proc.returncode == 0),
                    "execution_status": "passed" if proc.returncode == 0 else "failed",
                })

            if not rec["execution_pass"]:
                error_body = stderr_text or stdout_text
                rec["exception_type"] = extract_exception_type(error_body) or (
                    "PytestFailure" if runner_mode == "pytest" else "ScriptExecutionFailure"
                )
                rec["traceback_snippet"] = clip_text(error_body)

        except subprocess.TimeoutExpired as e:
            stdout_text = safe_text(getattr(e, "stdout", ""))
            stderr_text = safe_text(getattr(e, "stderr", "")) or str(e)
            rec.update({
                "execution_status": "timeout",
                "exception_type": "TimeoutExpired",
                "traceback_snippet": clip_text(stderr_text or stdout_text or str(e)),
                "runtime_sec": float(PYTEST_TIMEOUT_SEC),
            })

        except Exception as e:
            stderr_text = str(e)
            rec.update({
                "execution_status": "infra_error",
                "exception_type": type(e).__name__,
                "traceback_snippet": clip_text(stderr_text),
            })

        finally:
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            out.append(rec)

    write_jsonl(out, OUT_PATH)

    status_counter = Counter(row["execution_status"] for row in out)
    manifest = {
        "run_id": out[0]["run_id"] if out else None,
        "script_version": SCRIPT_VERSION,
        "input_path": str(INPUT_PATH),
        "output_path": str(OUT_PATH),
        "tmp_eval_root": str(TMP_EVAL_ROOT),
        "num_rows": len(out),
        "status_counts": dict(status_counter),
        "timeout_sec": PYTEST_TIMEOUT_SEC,
    }
    write_json(manifest, MANIFEST_PATH)

    print(f"wrote {len(out)} rows to {OUT_PATH}")
    print(f"status_counts={dict(status_counter)}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()