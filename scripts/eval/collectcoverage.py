from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(os.environ["UTPLM_PROJECT_ROOT"]).resolve()
RUN_ID = os.environ.get("UTPLM_RUN_ID", "").strip()
if not RUN_ID:
    raise RuntimeError("UTPLM_RUN_ID is required for collect_coverage_v2.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCRIPT_VERSION = "collect_coverage_v2_final"
COVERAGE_TIMEOUT_SEC = 20
MAX_ERROR_MSG_CHARS = 1000

EXEC_PATH = PROJECT_ROOT / "data" / "interim" / "executionresults" / RUN_ID / "executionresults.jsonl"
OUT_PATH = PROJECT_ROOT / "data" / "interim" / "coverageresults" / RUN_ID / "coverageresults.jsonl"
MANIFEST_PATH = PROJECT_ROOT / "data" / "interim" / "coverageresults" / RUN_ID / "manifest.json"


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


def base_record() -> Dict[str, Any]:
    return {"run_id": RUN_ID, "script_version": SCRIPT_VERSION}


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
    if not text:
        return None
    return text[:max_chars]


def build_subprocess_env(work_dir: Path) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    old_pythonpath = env.get("PYTHONPATH", "")
    prefix = f"{work_dir}{os.pathsep}{REPO_ROOT}"
    env["PYTHONPATH"] = f"{prefix}{os.pathsep}{old_pythonpath}" if old_pythonpath else prefix
    return env


def build_coverage_run_cmd(test_path: Path, runner_mode: Optional[str]) -> List[str]:
    if runner_mode == "pytest":
        return [
            sys.executable, "-m", "coverage", "run", "--branch",
            "-m", "pytest", "-q", "--tb=short",
            "-p", "no:cacheprovider",
            "-o", "cache_dir=/tmp",
            test_path.name,
        ]
    return [sys.executable, "-m", "coverage", "run", "--branch", test_path.name]


def run_coverage_for_candidate(work_dir: Path, test_path: Path, runner_mode: Optional[str]) -> Tuple[Dict[str, Any], float, str, str]:
    env = build_subprocess_env(work_dir)
    coverage_json_path = work_dir / "coverage.json"

    subprocess.run([sys.executable, "-m", "coverage", "erase"], cwd=str(work_dir), capture_output=True, text=True, env=env, timeout=COVERAGE_TIMEOUT_SEC)
    start = time.time()
    run_proc = subprocess.run(build_coverage_run_cmd(test_path, runner_mode), cwd=str(work_dir), capture_output=True, text=True, timeout=COVERAGE_TIMEOUT_SEC, env=env)
    json_proc = subprocess.run([sys.executable, "-m", "coverage", "json", "-o", "coverage.json"], cwd=str(work_dir), capture_output=True, text=True, timeout=COVERAGE_TIMEOUT_SEC, env=env)
    runtime_sec = round(time.time() - start, 4)

    stdout_text = "\n".join(x for x in [safe_text(run_proc.stdout), safe_text(json_proc.stdout)] if x).strip()
    stderr_text = "\n".join(x for x in [safe_text(run_proc.stderr), safe_text(json_proc.stderr)] if x).strip()

    if run_proc.returncode != 0:
        raise RuntimeError(f"coverage run failed with returncode={run_proc.returncode}: {clip_text(stderr_text or stdout_text) or 'no stderr/stdout'}")
    if json_proc.returncode != 0:
        raise RuntimeError(f"coverage json failed with returncode={json_proc.returncode}: {clip_text(safe_text(json_proc.stderr) or safe_text(json_proc.stdout)) or 'no stderr/stdout'}")
    if not coverage_json_path.exists():
        raise RuntimeError("coverage.json was not generated.")

    with open(coverage_json_path, "r", encoding="utf-8") as f:
        coverage_json_data = json.load(f)
    return coverage_json_data, runtime_sec, stdout_text, stderr_text


def normalize_line_number_collection(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    return sorted({x for x in values if isinstance(x, int)})


def normalize_branch_collection(values: Any) -> List[List[int]]:
    if not isinstance(values, list):
        return []
    normalized = []
    seen = set()
    for item in values:
        if isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, int) for x in item):
            pair = (int(item[0]), int(item[1]))
            if pair not in seen:
                seen.add(pair)
                normalized.append([pair[0], pair[1]])
    normalized.sort(key=lambda x: (x[0], x[1]))
    return normalized


def _safe_percent_to_ratio(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return round(float(value) / 100.0, 6)
    except Exception:
        return None


def extract_percent_from_summary(summary: Dict[str, Any], key: str) -> Optional[float]:
    if not isinstance(summary, dict):
        return None
    block = summary.get(key)
    if isinstance(block, dict):
        ratio = _safe_percent_to_ratio(block.get("percent_covered"))
        if ratio is not None:
            return ratio
    if key == "lines":
        ratio = _safe_percent_to_ratio(summary.get("percent_covered"))
        if ratio is not None:
            return ratio
        num_statements = summary.get("num_statements")
        covered_lines = summary.get("covered_lines")
        try:
            if num_statements:
                return round(float(covered_lines) / float(num_statements), 6)
        except Exception:
            pass
    if key == "branches":
        ratio = _safe_percent_to_ratio(summary.get("percent_branches_covered"))
        if ratio is not None:
            return ratio
        num_branches = summary.get("num_branches")
        covered_branches = summary.get("covered_branches")
        try:
            if num_branches and num_branches > 0:
                return round(float(covered_branches) / float(num_branches), 6)
        except Exception:
            pass
    return None


def parse_solution_coverage(coverage_json_data: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    files_block = coverage_json_data.get("files", {})
    if not isinstance(files_block, dict):
        raise ValueError("coverage.json missing 'files' block.")
    target_key = next((k for k in files_block if k.endswith("solution_under_test.py")), None)
    if not target_key:
        target_key = next((k for k in files_block if "solution_under_test" in k), None)
    if not target_key:
        raise ValueError("Could not locate coverage record for solution_under_test.py")

    file_block = files_block[target_key]
    summary = file_block.get("summary", {})
    covered_lines = normalize_line_number_collection(file_block.get("executed_lines", []))
    missing_lines = normalize_line_number_collection(file_block.get("missing_lines", []))
    covered_branches = normalize_branch_collection(file_block.get("executed_branches", file_block.get("covered_branches", [])))
    missing_branches = normalize_branch_collection(file_block.get("missing_branches", []))

    line_cov = extract_percent_from_summary(summary, "lines")
    branch_cov = extract_percent_from_summary(summary, "branches")
    if line_cov is None:
        total_lines = len(covered_lines) + len(missing_lines)
        if total_lines > 0:
            line_cov = round(len(covered_lines) / total_lines, 6)
    if branch_cov is None:
        total_branches = len(covered_branches) + len(missing_branches)
        if total_branches > 0:
            branch_cov = round(len(covered_branches) / total_branches, 6)
    return {
        "coverage_file_key": target_key,
        "line_coverage": line_cov,
        "branch_coverage": branch_cov,
        "covered_lines": covered_lines,
        "missing_lines": missing_lines,
        "covered_branches": covered_branches,
        "missing_branches": missing_branches,
        "coverage_json_path": str(work_dir / "coverage.json"),
    }


class TestHeuristicVisitor(ast.NodeVisitor):
    def __init__(self, source_text: str) -> None:
        self.source_text = source_text.lower()
        self.test_function_names: List[str] = []
        self.test_functions_with_checks = 0
        self.current_function_has_check = False
        self.boundary_signal_count = 0
        self.exception_signal_count = 0
        self._fn_stack = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        is_test = node.name.startswith("test")
        if is_test:
            self.test_function_names.append(node.name)
            prev = self.current_function_has_check
            self.current_function_has_check = False
            self._fn_stack += 1
            self.generic_visit(node)
            self._fn_stack -= 1
            if self.current_function_has_check:
                self.test_functions_with_checks += 1
            self.current_function_has_check = prev
        else:
            self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _mark_check(self) -> None:
        if self._fn_stack > 0:
            self.current_function_has_check = True

    def visit_Assert(self, node: ast.Assert) -> Any:
        self._mark_check()
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> Any:
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                name = ast.unparse(ctx.func) if hasattr(ast, "unparse") else ""
                name_l = name.lower()
                if "pytest.raises" in name_l or "assertraises" in name_l:
                    self.exception_signal_count += 1
                    self._mark_check()
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> Any:
        if node.handlers:
            self.exception_signal_count += 1
            self._mark_check()
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        name_l = name.lower()
        if "assertraises" in name_l:
            self.exception_signal_count += 1
            self._mark_check()
        if name_l.startswith("self.assert") or name_l == "assert":
            self._mark_check()
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        v = node.value
        if v in (0, 1, -1, "", None):
            self.boundary_signal_count += 1
        self.generic_visit(node)

    def visit_List(self, node: ast.List) -> Any:
        if len(node.elts) == 0:
            self.boundary_signal_count += 1
        self.generic_visit(node)

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        if len(node.elts) == 0:
            self.boundary_signal_count += 1
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> Any:
        if len(node.keys) == 0:
            self.boundary_signal_count += 1
        self.generic_visit(node)


BOUNDARY_KEYWORDS = re.compile(r"\b(empty|none|null|zero|negative|min|max|boundary|invalid)\b", re.I)


def extract_test_logic_proxies(test_path: Path) -> Dict[str, Any]:
    source = test_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {
            "DDR_or_BRC": None,
            "TBC": None,
            "boundary_coverage": None,
            "exception_path_coverage": None,
            "num_test_functions": None,
            "num_test_functions_with_checks": None,
            "boundary_signal_count": None,
            "exception_signal_count": None,
        }

    visitor = TestHeuristicVisitor(source)
    visitor.visit(tree)
    keyword_hits = len(BOUNDARY_KEYWORDS.findall(source))
    boundary_signal_count = visitor.boundary_signal_count + keyword_hits
    exception_signal_count = visitor.exception_signal_count
    num_test_functions = len(visitor.test_function_names)
    num_test_functions_with_checks = visitor.test_functions_with_checks

    tbc = None
    if num_test_functions > 0:
        tbc = round(num_test_functions_with_checks / num_test_functions, 6)

    boundary_cov = round(min(boundary_signal_count / 3.0, 1.0), 6)
    exception_cov = round(min(exception_signal_count / 1.0, 1.0), 6)

    return {
        "DDR_or_BRC": None,  # filled later from branch coverage as BRC proxy
        "TBC": tbc,
        "boundary_coverage": boundary_cov,
        "exception_path_coverage": exception_cov,
        "num_test_functions": num_test_functions,
        "num_test_functions_with_checks": num_test_functions_with_checks,
        "boundary_signal_count": boundary_signal_count,
        "exception_signal_count": exception_signal_count,
    }


def main() -> None:
    if not EXEC_PATH.exists():
        raise FileNotFoundError(f"missing execution results: {EXEC_PATH}")

    exec_rows = load_jsonl(EXEC_PATH)
    out: List[Dict[str, Any]] = []
    status_counter = Counter()
    parse_counter = Counter()

    for row in exec_rows:
        work_dir = Path(row["work_dir"])
        coverage_stdout_path = work_dir / "coverage_stdout.txt"
        coverage_stderr_path = work_dir / "coverage_stderr.txt"

        rec: Dict[str, Any] = {
            **base_record(),
            "candidate_id": row["candidate_id"],
            "task_id": row["task_id"],
            "source": row["source"],
            "difficulty_bucket": row.get("difficulty_bucket", "unknown"),
            "test_source_type": row.get("test_source_type", "unknown"),
            "runner_mode": row.get("runner_mode"),
            "coverage_status": None,
            "coverage_error_type": None,
            "coverage_error_msg": None,
            "coverage_runtime_sec": None,
            "line_coverage": None,
            "branch_coverage": None,
            "DDR_or_BRC": None,
            "TBC": None,
            "boundary_coverage": None,
            "exception_path_coverage": None,
            "num_test_functions": None,
            "num_test_functions_with_checks": None,
            "boundary_signal_count": None,
            "exception_signal_count": None,
            "covered_lines": [],
            "missing_lines": [],
            "covered_branches": [],
            "missing_branches": [],
            "coverage_file_key": None,
            "coverage_raw_path": None,
            "coverage_stdout_path": str(coverage_stdout_path),
            "coverage_stderr_path": str(coverage_stderr_path),
        }

        if not row.get("execution_pass", False):
            rec["coverage_status"] = "skipped_execution_failed"
            out.append(rec)
            status_counter[rec["coverage_status"]] += 1
            continue

        test_path = Path(row["test_path"])
        stdout_text, stderr_text = "", ""

        try:
            coverage_json_data, runtime_sec, stdout_text, stderr_text = run_coverage_for_candidate(
                work_dir=work_dir,
                test_path=test_path,
                runner_mode=row.get("runner_mode"),
            )
            parsed = parse_solution_coverage(coverage_json_data, work_dir)
            proxy = extract_test_logic_proxies(test_path)
            ddr_or_brc = parsed["branch_coverage"] if parsed["branch_coverage"] is not None else parsed["line_coverage"]

            rec.update({
                "coverage_status": "ok",
                "coverage_runtime_sec": runtime_sec,
                "line_coverage": parsed["line_coverage"],
                "branch_coverage": parsed["branch_coverage"],
                "DDR_or_BRC": ddr_or_brc,
                "TBC": proxy["TBC"],
                "boundary_coverage": proxy["boundary_coverage"],
                "exception_path_coverage": proxy["exception_path_coverage"],
                "num_test_functions": proxy["num_test_functions"],
                "num_test_functions_with_checks": proxy["num_test_functions_with_checks"],
                "boundary_signal_count": proxy["boundary_signal_count"],
                "exception_signal_count": proxy["exception_signal_count"],
                "covered_lines": parsed["covered_lines"],
                "missing_lines": parsed["missing_lines"],
                "covered_branches": parsed["covered_branches"],
                "missing_branches": parsed["missing_branches"],
                "coverage_file_key": parsed["coverage_file_key"],
                "coverage_raw_path": parsed["coverage_json_path"],
            })
            parse_counter["coverage_parse_ok"] += 1
        except subprocess.TimeoutExpired as e:
            stdout_text = safe_text(getattr(e, "stdout", ""))
            stderr_text = safe_text(getattr(e, "stderr", "")) or str(e)
            rec.update({
                "coverage_status": "tool_error",
                "coverage_error_type": "TimeoutExpired",
                "coverage_error_msg": clip_text(stderr_text or stdout_text or str(e)),
                "coverage_runtime_sec": float(COVERAGE_TIMEOUT_SEC),
            })
            parse_counter["coverage_parse_fail"] += 1
        except Exception as e:
            stderr_text = str(e)
            rec.update({
                "coverage_status": "tool_error",
                "coverage_error_type": type(e).__name__,
                "coverage_error_msg": clip_text(stderr_text),
            })
            parse_counter["coverage_parse_fail"] += 1
        finally:
            coverage_stdout_path.write_text(stdout_text, encoding="utf-8")
            coverage_stderr_path.write_text(stderr_text, encoding="utf-8")
            out.append(rec)
            status_counter[rec["coverage_status"]] += 1

    write_jsonl(out, OUT_PATH)
    manifest = {
        "run_id": RUN_ID,
        "script_version": SCRIPT_VERSION,
        "input_path": str(EXEC_PATH),
        "output_path": str(OUT_PATH),
        "num_rows": len(out),
        "status_counts": dict(status_counter),
        "parse_counts": dict(parse_counter),
        "timeout_sec": COVERAGE_TIMEOUT_SEC,
        "derived_metrics": {
            "DDR_or_BRC": "branch_coverage as BRC proxy, fallback to line_coverage",
            "TBC": "checked_test_functions / test_functions from AST heuristic",
            "boundary_coverage": "normalized boundary-signal heuristic from test code",
            "exception_path_coverage": "normalized exception-path heuristic from test code",
        },
    }
    write_json(manifest, MANIFEST_PATH)

    print(f"wrote {len(out)} rows to {OUT_PATH}")
    print(f"status_counts={dict(status_counter)}")
    print(f"parse_counts={dict(parse_counter)}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
