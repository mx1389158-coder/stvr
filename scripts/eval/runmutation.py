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
    stage_file,
    stage_manifest_file,
    base_record,
)

EXEC_PATH = stage_file("execution_results")
INPUT_PATH = stage_file("generated_candidates")
OUT_PATH = stage_file("mutation_results")
MANIFEST_PATH = stage_manifest_file("mutation_results")

SCRIPT_VERSION = "run_mutation_v10"
MUTATION_TIMEOUT_SEC = int(os.environ.get("UTPLM_MUTATION_TIMEOUT_SEC", "180"))
MAX_ERROR_MSG_CHARS = 2000


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


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def clip_text(text: Optional[str], max_chars: int = MAX_ERROR_MSG_CHARS) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    return text[:max_chars] if text else None


# =========================================================
# runner_mode / wrapper 诊断
# =========================================================

def infer_runner_mode(candidate_test_code: str) -> str:
    try:
        tree = ast.parse(candidate_test_code)
    except SyntaxError:
        return "script"

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            return "pytest"
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            return "pytest"

    return "script"


def indent_block(code: str, prefix: str = "    ") -> str:
    dedented = textwrap.dedent(code).strip("\n")
    if not dedented.strip():
        return f"{prefix}raise ValueError('Empty candidate_test_code for mutation testing.')"
    return "\n".join(f"{prefix}{line}" if line.strip() else "" for line in dedented.splitlines())


SAFE_PRELUDE = """
import multiprocessing as _mp

_orig_set_start_method = _mp.set_start_method

def _safe_set_start_method(method=None, force=False):
    try:
        return _orig_set_start_method(method, force=force)
    except RuntimeError as e:
        if "context has already been set" in str(e):
            return None
        raise

_mp.set_start_method = _safe_set_start_method
""".strip()


# =========================================================
# HumanEval / doctest 相关
# =========================================================

def looks_like_humaneval_check_harness(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check" and len(node.args.args) >= 1:
            return True
    return False


def is_safe_single_line_expr(expr: str) -> bool:
    if not expr or not isinstance(expr, str):
        return False
    if "\n" in expr:
        return False
    expr = expr.strip()
    if not expr:
        return False
    if expr.startswith('"""') or expr.startswith("'''"):
        return False
    try:
        ast.parse(expr, mode="eval")
        return True
    except Exception:
        return False


def extract_doctest_cases_from_prompt(prompt: str, entry_point: Optional[str]) -> List[Tuple[str, str]]:
    """
    只提取最保守、最安全的 doctest:
        >>> func(...)
        expected_expr

    严格要求:
    - 调用行是单行
    - expected 是单行
    - 两者都能被 ast.parse(..., mode="eval") 解析
    """
    if not prompt or not isinstance(prompt, str):
        return []

    lines = prompt.splitlines()
    cases: List[Tuple[str, str]] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if not line.startswith(">>>"):
            i += 1
            continue

        call_expr = line[3:].strip()

        if entry_point and not call_expr.startswith(f"{entry_point}("):
            i += 1
            continue

        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1

        if j >= len(lines):
            break

        expected_expr = lines[j].strip()

        if (
            expected_expr
            and not expected_expr.startswith(">>>")
            and is_safe_single_line_expr(call_expr)
            and is_safe_single_line_expr(expected_expr)
        ):
            cases.append((call_expr, expected_expr))
            i = j + 1
            continue

        i += 1

    return cases


def extract_top_level_segments(code: str) -> Tuple[List[str], List[str]]:
    """
    返回:
      setup_segments: 顶层非 assert 语句源码
      assert_segments: 顶层 assert 语句源码
    """
    tree = ast.parse(code)
    setup_segments: List[str] = []
    assert_segments: List[str] = []

    for node in tree.body:
        seg = ast.get_source_segment(code, node)
        if not seg:
            continue
        seg = seg.strip("\n")
        if isinstance(node, ast.Assert):
            assert_segments.append(seg)
        else:
            setup_segments.append(seg)

    return setup_segments, assert_segments


def build_humaneval_check_wrapper(
    candidate_test_code: str,
    entry_point: Optional[str],
) -> str:
    """
    HumanEval 常见 harness:
        def check(candidate): ...
    包装成:
        def test_humaneval_check():
            check(entry_point)

    注意：这里不再叠加 doctest，避免把 prompt 内容拼坏。
    """
    code = (candidate_test_code or "").strip()
    ep = (entry_point or "").strip() or "solution"

    parts = [
        SAFE_PRELUDE,
        "",
        "from solution_under_test import *",
        "",
        code,
        "",
        "def test_humaneval_check():",
        f"    check({ep})",
        "",
    ]

    return "\n".join(parts).rstrip() + "\n"


def build_humaneval_strong_wrapper(
    candidate_test_code: str,
    prompt: str,
    entry_point: Optional[str],
) -> str:
    """
    HumanEval 非 harness 风格 script 样本:
    1. 尝试把顶层 assert 拆成多个 test_case
    2. 再追加严格过滤后的 doctest case
    """
    code = (candidate_test_code or "").strip()
    doctest_cases = extract_doctest_cases_from_prompt(prompt, entry_point)

    if not code:
        parts = [
            SAFE_PRELUDE,
            "",
            "from solution_under_test import *",
            "",
            "def test_case_001():",
            "    raise ValueError('Empty candidate_test_code for mutation testing.')",
            "",
        ]
        for idx, (call_expr, expected_expr) in enumerate(doctest_cases, start=1):
            parts.append(f"def test_prompt_case_{idx:03d}():")
            parts.append(f"    assert {call_expr} == {expected_expr}")
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    try:
        setup_segments, assert_segments = extract_top_level_segments(code)
    except Exception:
        parts = [
            SAFE_PRELUDE,
            "",
            "from solution_under_test import *",
            "",
            "def test_mutmut_smoke():",
            indent_block(code),
            "",
        ]
        for idx, (call_expr, expected_expr) in enumerate(doctest_cases, start=1):
            parts.append(f"def test_prompt_case_{idx:03d}():")
            parts.append(f"    assert {call_expr} == {expected_expr}")
            parts.append("")
        return "\n".join(parts).rstrip() + "\n"

    parts = [
        SAFE_PRELUDE,
        "",
        "from solution_under_test import *",
        "",
    ]

    if assert_segments:
        for seg in setup_segments:
            parts.append(seg)
            parts.append("")

        for i, assert_seg in enumerate(assert_segments, start=1):
            parts.append(f"def test_case_{i:03d}():")
            parts.append(indent_block(assert_seg))
            parts.append("")
    else:
        parts.append("def test_mutmut_smoke():")
        parts.append(indent_block(code))
        parts.append("")

    for idx, (call_expr, expected_expr) in enumerate(doctest_cases, start=1):
        parts.append(f"def test_prompt_case_{idx:03d}():")
        parts.append(f"    assert {call_expr} == {expected_expr}")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def build_mutmut_test_file(
    candidate_test_code: str,
    runner_mode: str,
    source: str,
    prompt: str | None = None,
    entry_point: str | None = None,
) -> str:
    code = (candidate_test_code or "").strip()

    # 1) HumanEval harness 优先
    if source == "humaneval" and looks_like_humaneval_check_harness(code):
        return build_humaneval_check_wrapper(
            candidate_test_code=code,
            entry_point=entry_point,
        )

    # 2) pytest 风格原样保留
    if runner_mode == "pytest":
        if not code:
            return f"{SAFE_PRELUDE}\n\nfrom solution_under_test import *\n"
        if "import solution_under_test" in code or "from solution_under_test" in code:
            return f"{SAFE_PRELUDE}\n\n{code}\n"
        return f"{SAFE_PRELUDE}\n\nfrom solution_under_test import *\n\n{code}\n"

    # 3) HumanEval 非 harness 走增强版
    if source == "humaneval":
        return build_humaneval_strong_wrapper(
            candidate_test_code=code,
            prompt=prompt or "",
            entry_point=entry_point,
        )

    # 4) MBPP 保持现状
    return (
        f"{SAFE_PRELUDE}\n\n"
        "from solution_under_test import *\n\n"
        "def test_mutmut_smoke():\n"
        f"{indent_block(code)}\n"
    )


def write_mutmut_setup_cfg(work_dir: Path) -> Path:
    cfg_path = work_dir / "setup.cfg"
    cfg_path.write_text(
        "[mutmut]\n"
        "paths_to_mutate=solution_under_test.py\n",
        encoding="utf-8",
    )
    return cfg_path


def write_pytest_ini(work_dir: Path) -> Path:
    ini_path = work_dir / "pytest.ini"
    ini_path.write_text(
        "[pytest]\n"
        "python_files = test_candidate_mutmut.py\n"
        "addopts = -q --tb=short -p no:cacheprovider\n",
        encoding="utf-8",
    )
    return ini_path


# =========================================================
# 环境与缓存清理
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


def clean_mutation_artifacts(work_dir: Path) -> None:
    for p in [
        work_dir / ".mutmut-cache",
        work_dir / "mutants",
        work_dir / ".pytest_cache",
        work_dir / ".coverage",
        work_dir / "coverage.json",
        work_dir / "htmlcov",
    ]:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)


# =========================================================
# mutmut 结果解析
# =========================================================

def parse_mutmut_results_text(
    combined_text: str,
    results_text: str,
) -> Dict[str, Any]:
    """
    combined_text:
        用于检查全局错误关键词 / summary
    results_text:
        只用于提取 mutant 明细行，避免 stdout + results 重复计数
    """
    combined_text = combined_text or ""
    results_text = results_text or ""

    summary_total: Optional[int] = None
    summary_killed: Optional[int] = None
    summary_survived: int = 0
    summary_timeout: int = 0
    summary_suspicious: int = 0

    survived_mutant_ids: List[str] = []
    not_checked_mutant_ids: List[str] = []
    timeout_mutant_ids: List[str] = []
    suspicious_mutant_ids: List[str] = []

    summary_line = None
    for line in combined_text.splitlines():
        s = line.strip()
        if re.search(r"\d+\s*/\s*\d+.*🎉", s):
            summary_line = s

    if summary_line:
        total_match = re.search(r"(\d+)\s*/\s*(\d+)", summary_line)
        killed_match = re.search(r"🎉\s*(\d+)", summary_line)
        survived_match = re.search(r"(?:🙁|🫥)\s*(\d+)", summary_line)
        timeout_match = re.search(r"⏰\s*(\d+)", summary_line)
        suspicious_match = re.search(r"🤔\s*(\d+)", summary_line)

        if total_match:
            summary_total = int(total_match.group(2))
        if killed_match:
            summary_killed = int(killed_match.group(1))
        if survived_match:
            summary_survived = int(survived_match.group(1))
        if timeout_match:
            summary_timeout = int(timeout_match.group(1))
        if suspicious_match:
            summary_suspicious = int(suspicious_match.group(1))

    survived_set = set()
    not_checked_set = set()
    timeout_set = set()
    suspicious_set = set()

    for line in results_text.splitlines():
        s = line.strip()
        if not s:
            continue

        if ": not checked" in s:
            mutant_id = s.split(":", 1)[0].strip()
            not_checked_set.add(mutant_id)
            continue

        if ": survived" in s:
            mutant_id = s.split(":", 1)[0].strip()
            survived_set.add(mutant_id)
            continue

        if ": timeout" in s or ": timed out" in s:
            mutant_id = s.split(":", 1)[0].strip()
            timeout_set.add(mutant_id)
            continue

        if ": suspicious" in s:
            mutant_id = s.split(":", 1)[0].strip()
            suspicious_set.add(mutant_id)

    survived_mutant_ids = sorted(survived_set)
    not_checked_mutant_ids = sorted(not_checked_set)
    timeout_mutant_ids = sorted(timeout_set)
    suspicious_mutant_ids = sorted(suspicious_set)

    invalid_run = False
    invalid_reason = None

    if "failed to collect stats" in combined_text:
        invalid_run = True
        invalid_reason = "failed_to_collect_stats"

    if "context has already been set" in combined_text:
        invalid_run = True
        invalid_reason = "multiprocessing_context_conflict"

    if "could not find any test case for any mutant" in combined_text:
        invalid_run = True
        invalid_reason = "no_tests_cover_mutants"

    if not_checked_mutant_ids:
        invalid_run = True
        invalid_reason = invalid_reason or "not_checked_mutants_present"

    line_survived = len(survived_mutant_ids)
    line_timeout = len(timeout_mutant_ids)
    line_suspicious = len(suspicious_mutant_ids)

    mutants_survived = line_survived if line_survived > 0 else summary_survived
    mutants_timeout = line_timeout if line_timeout > 0 else summary_timeout
    mutants_suspicious = line_suspicious if line_suspicious > 0 else summary_suspicious

    mutants_total = summary_total
    mutants_killed = summary_killed

    if mutants_total is None:
        observed = mutants_survived + mutants_timeout + mutants_suspicious + len(not_checked_mutant_ids)
        if observed > 0:
            mutants_total = observed

    if mutants_total is not None:
        derived_killed = mutants_total - mutants_survived - mutants_timeout - mutants_suspicious
        if derived_killed < 0:
            derived_killed = None
    else:
        derived_killed = None

    if derived_killed is not None:
        mutants_killed = derived_killed

    mutation_score: Optional[float] = None
    if not invalid_run and mutants_total is not None and mutants_killed is not None:
        mutation_score = round(mutants_killed / mutants_total, 6) if mutants_total > 0 else 0.0

    return {
        "mutants_total": mutants_total,
        "mutants_killed": mutants_killed,
        "mutants_survived": mutants_survived,
        "mutants_timeout": mutants_timeout,
        "mutants_suspicious": mutants_suspicious,
        "mutation_score": mutation_score,
        "survived_mutant_ids": survived_mutant_ids,
        "killed_mutant_ids": [],
        "not_checked_mutant_ids": not_checked_mutant_ids,
        "timeout_mutant_ids": timeout_mutant_ids,
        "suspicious_mutant_ids": suspicious_mutant_ids,
        "invalid_run": invalid_run,
        "invalid_reason": invalid_reason,
    }


# =========================================================
# mutmut 执行
# =========================================================

def run_cmd(
    cmd: List[str],
    work_dir: Path,
    timeout: int,
) -> Tuple[int, str, str, float]:
    env = build_subprocess_env(work_dir)
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    dt = round(time.time() - start, 4)
    return proc.returncode, safe_text(proc.stdout), safe_text(proc.stderr), dt


def run_mutmut_for_candidate(work_dir: Path) -> Dict[str, Any]:
    pytest_cmd = [sys.executable, "-m", "pytest", "-q", "test_candidate_mutmut.py"]
    pytest_rc, pytest_out, pytest_err, pytest_dt = run_cmd(pytest_cmd, work_dir, timeout=60)

    if pytest_rc != 0:
        raise RuntimeError(
            f"mutation wrapper pytest failed with returncode={pytest_rc}\n"
            f"STDOUT:\n{pytest_out}\nSTDERR:\n{pytest_err}"
        )

    run_cmd_list = [sys.executable, "-m", "mutmut", "run"]
    run_rc, run_out, run_err, run_dt = run_cmd(
        run_cmd_list,
        work_dir,
        timeout=MUTATION_TIMEOUT_SEC,
    )

    results_cmd = [sys.executable, "-m", "mutmut", "results"]
    res_rc, res_out, res_err, _ = run_cmd(
        results_cmd,
        work_dir,
        timeout=MUTATION_TIMEOUT_SEC,
    )

    combined_stdout = "\n".join(x for x in [pytest_out, run_out, res_out] if x).strip()
    combined_stderr = "\n".join(x for x in [pytest_err, run_err, res_err] if x).strip()
    results_text = "\n".join(x for x in [res_out, res_err] if x).strip()

    runtime_sec = round(pytest_dt + run_dt, 4)

    if run_rc not in (0, 2) and not (combined_stdout or combined_stderr or results_text):
        raise RuntimeError(
            f"mutmut run failed with returncode={run_rc}\n"
            f"STDOUT:\n{combined_stdout}\nSTDERR:\n{combined_stderr}"
        )

    return {
        "runtime_sec": runtime_sec,
        "combined_stdout": combined_stdout,
        "combined_stderr": combined_stderr,
        "results_text": results_text,
        "pytest_cmd": pytest_cmd,
        "mutmut_run_cmd": run_cmd_list,
        "mutmut_results_cmd": results_cmd,
        "pytest_rc": pytest_rc,
        "mutmut_run_rc": run_rc,
        "mutmut_results_rc": res_rc,
    }


# =========================================================
# 主流程
# =========================================================

def main() -> None:
    exec_rows = load_jsonl(EXEC_PATH)
    candidate_rows = {row["candidate_id"]: row for row in load_jsonl(INPUT_PATH)}

    out: List[Dict[str, Any]] = []
    status_counter = Counter()
    parse_counter = Counter()

    for row in exec_rows:
        work_dir = Path(row["work_dir"])
        mutmut_test_path = work_dir / "test_candidate_mutmut.py"
        setup_cfg_path = work_dir / "setup.cfg"
        pytest_ini_path = work_dir / "pytest.ini"
        mutation_stdout_path = work_dir / "mutation_stdout.txt"
        mutation_stderr_path = work_dir / "mutation_stderr.txt"
        mutation_results_path = work_dir / "mutation_results.txt"

        rec: Dict[str, Any] = {
            **base_record(SCRIPT_VERSION),
            "candidate_id": row["candidate_id"],
            "task_id": row["task_id"],
            "source": row["source"],
            "difficulty_bucket": row.get("difficulty_bucket", "unknown"),
            "test_source_type": row.get("test_source_type", "unknown"),
            "runner_mode": row.get("runner_mode"),
            "mutation_status": None,
            "mutation_runtime_sec": None,
            "mutants_total": None,
            "mutants_killed": None,
            "mutants_survived": None,
            "mutants_timeout": 0,
            "mutants_suspicious": 0,
            "mutation_score": None,
            "mutation_timeout_count": 0,
            "mutation_error_count": 0,
            "killed_mutant_ids": [],
            "survived_mutant_ids": [],
            "not_checked_mutant_ids": [],
            "timeout_mutant_ids": [],
            "suspicious_mutant_ids": [],
            "mutation_error_type": None,
            "mutation_error_msg": None,
            "mutation_raw_path": str(mutation_results_path),
            "mutation_stdout_path": str(mutation_stdout_path),
            "mutation_stderr_path": str(mutation_stderr_path),
            "mutmut_test_path": str(mutmut_test_path),
            "mutmut_setup_cfg_path": str(setup_cfg_path),
            "mutmut_pytest_ini_path": str(pytest_ini_path),
            "mutation_stdout": None,
            "mutation_stderr": None,
            "mutation_cmd": None,
            "mutation_exit_code": None,
            "mutation_results_excerpt": None,
        }

        if not row.get("execution_pass", False):
            rec["mutation_status"] = "skipped_execution_failed"
            out.append(rec)
            status_counter[rec["mutation_status"]] += 1
            continue

        candidate_row = candidate_rows.get(row["candidate_id"])
        if candidate_row is None:
            rec.update({
                "mutation_status": "tool_error",
                "mutation_error_count": 1,
                "mutation_error_type": "MissingCandidateRecord",
                "mutation_error_msg": "candidate_id not found in generated_candidates jsonl.",
            })
            out.append(rec)
            status_counter[rec["mutation_status"]] += 1
            parse_counter["mutation_parse_fail"] += 1
            continue

        candidate_test_code = candidate_row.get("candidate_test_code", "") or ""
        runner_mode = row.get("runner_mode") or infer_runner_mode(candidate_test_code)
        rec["runner_mode"] = runner_mode

        clean_mutation_artifacts(work_dir)
        write_mutmut_setup_cfg(work_dir)
        write_pytest_ini(work_dir)

        mutmut_test_path.write_text(
            build_mutmut_test_file(
                candidate_test_code=candidate_test_code,
                runner_mode=runner_mode,
                source=candidate_row.get("source", ""),
                prompt=candidate_row.get("prompt", ""),
                entry_point=candidate_row.get("entry_point"),
            ),
            encoding="utf-8",
        )

        try:
            run_info = run_mutmut_for_candidate(work_dir)
            runtime_sec = run_info["runtime_sec"]
            run_stdout = run_info["combined_stdout"]
            run_stderr = run_info["combined_stderr"]
            results_text = run_info["results_text"]

            mutation_stdout_path.write_text(run_stdout, encoding="utf-8")
            mutation_stderr_path.write_text(run_stderr, encoding="utf-8")
            mutation_results_path.write_text(results_text, encoding="utf-8")

            rec["mutation_stdout"] = clip_text(run_stdout)
            rec["mutation_stderr"] = clip_text(run_stderr)
            rec["mutation_cmd"] = " || ".join([" ".join(x) for x in [run_info["pytest_cmd"], run_info["mutmut_run_cmd"], run_info["mutmut_results_cmd"]]])
            rec["mutation_exit_code"] = run_info["mutmut_run_rc"]
            rec["mutation_results_excerpt"] = clip_text(results_text)

            combined_text = f"{run_stdout}\n{run_stderr}\n{results_text}"
            parsed = parse_mutmut_results_text(
                combined_text=combined_text,
                results_text=results_text,
            )

            total = parsed["mutants_total"]
            killed = parsed["mutants_killed"]
            survived = parsed["mutants_survived"]
            timeout = parsed["mutants_timeout"]
            suspicious = parsed["mutants_suspicious"]

            valid_counts = (
                total is not None
                and killed is not None
                and survived is not None
                and timeout >= 0
                and suspicious >= 0
                and killed >= 0
                and survived >= 0
                and killed + survived + timeout + suspicious == total
            )

            if parsed["invalid_run"] or not valid_counts:
                rec.update({
                    "mutation_status": "tool_error",
                    "mutation_error_count": 1,
                    "mutation_error_type": "InvalidMutationRun",
                    "mutation_error_msg": clip_text(
                        parsed["invalid_reason"] or "mutation_counts_inconsistent"
                    ),
                    "mutation_runtime_sec": runtime_sec,
                    "mutants_total": total,
                    "mutants_killed": killed,
                    "mutants_survived": survived,
                    "mutants_timeout": timeout,
                    "mutants_suspicious": suspicious,
                    "mutation_score": None,
                    "killed_mutant_ids": parsed["killed_mutant_ids"],
                    "survived_mutant_ids": parsed["survived_mutant_ids"],
                    "not_checked_mutant_ids": parsed["not_checked_mutant_ids"],
                    "timeout_mutant_ids": parsed["timeout_mutant_ids"],
                    "suspicious_mutant_ids": parsed["suspicious_mutant_ids"],
                })
                parse_counter["mutation_parse_fail"] += 1
            else:
                rec.update({
                    "mutation_status": "ok",
                    "mutation_runtime_sec": runtime_sec,
                    "mutants_total": total,
                    "mutants_killed": killed,
                    "mutants_survived": survived,
                    "mutants_timeout": timeout,
                    "mutants_suspicious": suspicious,
                    "mutation_score": parsed["mutation_score"],
                    "killed_mutant_ids": parsed["killed_mutant_ids"],
                    "survived_mutant_ids": parsed["survived_mutant_ids"],
                    "not_checked_mutant_ids": parsed["not_checked_mutant_ids"],
                    "timeout_mutant_ids": parsed["timeout_mutant_ids"],
                    "suspicious_mutant_ids": parsed["suspicious_mutant_ids"],
                })
                parse_counter["mutation_parse_ok"] += 1

        except subprocess.TimeoutExpired as e:
            stdout_text = safe_text(getattr(e, "stdout", ""))
            stderr_text = safe_text(getattr(e, "stderr", "")) or str(e)
            mutation_stdout_path.write_text(stdout_text, encoding="utf-8")
            mutation_stderr_path.write_text(stderr_text, encoding="utf-8")
            rec["mutation_stdout"] = clip_text(stdout_text)
            rec["mutation_stderr"] = clip_text(stderr_text)
            rec["mutation_results_excerpt"] = None

            rec.update({
                "mutation_status": "timeout",
                "mutation_runtime_sec": float(MUTATION_TIMEOUT_SEC),
                "mutation_timeout_count": 1,
                "mutation_error_type": "TimeoutExpired",
                "mutation_error_msg": clip_text(stderr_text or stdout_text),
            })
            parse_counter["mutation_parse_fail"] += 1

        except Exception as e:
            mutation_stdout_path.write_text("", encoding="utf-8")
            mutation_stderr_path.write_text(str(e), encoding="utf-8")
            rec["mutation_stdout"] = None
            rec["mutation_stderr"] = clip_text(str(e))
            rec["mutation_results_excerpt"] = None

            rec.update({
                "mutation_status": "tool_error",
                "mutation_error_count": 1,
                "mutation_error_type": type(e).__name__,
                "mutation_error_msg": clip_text(str(e)),
            })
            parse_counter["mutation_parse_fail"] += 1

        out.append(rec)
        status_counter[rec["mutation_status"]] += 1

    write_jsonl(out, OUT_PATH)

    manifest = {
        "run_id": out[0].get("run_id") if out else None,
        "script_version": SCRIPT_VERSION,
        "input_execution_path": str(EXEC_PATH),
        "input_candidates_path": str(INPUT_PATH),
        "output_path": str(OUT_PATH),
        "num_rows": len(out),
        "status_counts": dict(status_counter),
        "parse_counts": dict(parse_counter),
        "timeout_sec": MUTATION_TIMEOUT_SEC,
        "mutation_tool": "mutmut",
        "test_discovery_mode": "pytest_ini_restricted_to_test_candidate_mutmut_with_humaneval_harness_wrapper",
    }
    write_json(manifest, MANIFEST_PATH)

    print(f"wrote {len(out)} rows to {OUT_PATH}")
    print(f"status_counts={dict(status_counter)}")
    print(f"parse_counts={dict(parse_counter)}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()