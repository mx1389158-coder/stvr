from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.interimschema import base_record, stage_file, stage_manifest_file  # noqa: E402


SCRIPT_VERSION = "run_external_mutation_smoke_v1"
MAX_MUTANTS_PER_CANDIDATE = int(os.environ.get("UTPLM_EXTERNAL_MAX_MUTANTS", "8"))
TIMEOUT_SEC = int(os.environ.get("UTPLM_EXTERNAL_MUTATION_TIMEOUT_SEC", "20"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sanitize_fs_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "__", name).strip("._")
    return safe[:180] or "candidate"


def build_test_file_code(candidate_test_code: str) -> str:
    code = (candidate_test_code or "").strip()
    if not code:
        return "from solution_under_test import *\n\nraise ValueError('empty test code')\n"
    if "import solution_under_test" in code or "from solution_under_test" in code:
        return code + "\n"
    return "from solution_under_test import *\n\n" + code + "\n"


def run_pytest(work_dir: Path) -> tuple[bool, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    old_pythonpath = env.get("PYTHONPATH", "")
    prefix = f"{work_dir}{os.pathsep}{REPO_ROOT}"
    env["PYTHONPATH"] = f"{prefix}{os.pathsep}{old_pythonpath}" if old_pythonpath else prefix
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider", "-o", "cache_dir=/tmp", "test_candidate.py"],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SEC,
        env=env,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return proc.returncode == 0, text[-1200:]


class MutationCollector(ast.NodeVisitor):
    def __init__(self, target_name: str) -> None:
        self.target_name = target_name
        self.paths: list[tuple[int, str]] = []
        self._in_target = False
        self._counter = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prev = self._in_target
        self._in_target = node.name == self.target_name
        self.generic_visit(node)
        self._in_target = prev

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        prev = self._in_target
        self._in_target = node.name == self.target_name
        self.generic_visit(node)
        self._in_target = prev

    def generic_visit(self, node: ast.AST) -> None:
        if self._in_target and is_mutable_node(node):
            self.paths.append((self._counter, type(node).__name__))
            self._counter += 1
        super().generic_visit(node)


def is_mutable_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Compare) and node.ops:
        return type(node.ops[0]) in COMPARE_SWAPS
    if isinstance(node, ast.BinOp):
        return type(node.op) in BINOP_SWAPS
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool):
            return True
        if isinstance(node.value, int) and node.value in (-1, 0, 1):
            return True
        if isinstance(node.value, str) and node.value == "":
            return True
    return False


COMPARE_SWAPS: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}

BINOP_SWAPS: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
}


class ApplyOneMutation(ast.NodeTransformer):
    def __init__(self, target_name: str, wanted_index: int) -> None:
        self.target_name = target_name
        self.wanted_index = wanted_index
        self._in_target = False
        self._counter = 0
        self.applied = False
        self.mutation_kind: str | None = None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        prev = self._in_target
        self._in_target = node.name == self.target_name
        node = self.generic_visit(node)
        self._in_target = prev
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        prev = self._in_target
        self._in_target = node.name == self.target_name
        node = self.generic_visit(node)
        self._in_target = prev
        return node

    def _maybe_mutate_current(self, node: ast.AST) -> bool:
        if not self._in_target or not is_mutable_node(node):
            return False
        current = self._counter
        self._counter += 1
        return current == self.wanted_index and not self.applied

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        if self._maybe_mutate_current(node) and node.ops:
            old = type(node.ops[0])
            new = COMPARE_SWAPS.get(old)
            if new:
                node.ops[0] = new()
                self.applied = True
                self.mutation_kind = f"compare_{old.__name__}_to_{new.__name__}"
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if self._maybe_mutate_current(node):
            old = type(node.op)
            new = BINOP_SWAPS.get(old)
            if new:
                node.op = new()
                self.applied = True
                self.mutation_kind = f"binop_{old.__name__}_to_{new.__name__}"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self._maybe_mutate_current(node):
            if isinstance(node.value, bool):
                self.applied = True
                self.mutation_kind = "constant_bool_flip"
                return ast.copy_location(ast.Constant(value=not node.value), node)
            if isinstance(node.value, int) and node.value in (-1, 0, 1):
                self.applied = True
                self.mutation_kind = "constant_small_int_shift"
                return ast.copy_location(ast.Constant(value=node.value + 1), node)
            if isinstance(node.value, str) and node.value == "":
                self.applied = True
                self.mutation_kind = "constant_empty_string_to_x"
                return ast.copy_location(ast.Constant(value="x"), node)
        return node


def make_mutants(solution_code: str, entry_point: str, max_mutants: int) -> list[tuple[str, str]]:
    try:
        tree = ast.parse(solution_code)
    except SyntaxError:
        return []
    collector = MutationCollector(entry_point)
    collector.visit(tree)
    mutants: list[tuple[str, str]] = []
    for index, _kind in collector.paths[:max_mutants]:
        mutant_tree = copy.deepcopy(tree)
        mutator = ApplyOneMutation(entry_point, index)
        mutant_tree = mutator.visit(mutant_tree)
        ast.fix_missing_locations(mutant_tree)
        if not mutator.applied:
            continue
        try:
            mutants.append((ast.unparse(mutant_tree) + "\n", mutator.mutation_kind or "unknown"))
        except Exception:
            continue
    return mutants


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("UTPLM_RUN_ID", "smoke_v1"))
    parser.add_argument("--project_root", default=os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm"))
    parser.add_argument("--max_mutants", type=int, default=MAX_MUTANTS_PER_CANDIDATE)
    args = parser.parse_args()

    os.environ["UTPLM_RUN_ID"] = args.run_id
    os.environ["UTPLM_PROJECT_ROOT"] = args.project_root

    rows = load_jsonl(stage_file("generated_candidates"))
    exec_rows = {row.get("candidate_id"): row for row in load_jsonl(stage_file("execution_results"))}
    out_path = stage_file("mutation_results")
    manifest_path = stage_manifest_file("mutation_results")
    tmp_root = Path(args.project_root) / "tmp" / "external_mutation_smoke" / args.run_id
    tmp_root.mkdir(parents=True, exist_ok=True)

    out: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    start = time.time()

    for row in rows:
        candidate_id = row["candidate_id"]
        entry_point = str(row.get("entry_point") or "")
        exec_row = exec_rows.get(candidate_id, {})
        rec = {
            **base_record(SCRIPT_VERSION),
            "candidate_id": candidate_id,
            "task_id": row.get("task_id"),
            "source": row.get("source"),
            "difficulty_bucket": row.get("difficulty_bucket", "unknown"),
            "test_source_type": row.get("test_source_type", "model_generated"),
            "mutation_status": None,
            "mutation_score": None,
            "mutants_total": 0,
            "mutants_killed": 0,
            "mutants_survived": 0,
            "mutants_timeout_or_error": 0,
            "mutation_scope": "target_function_ast_smoke",
            "mutation_error": None,
        }

        if exec_row and not bool(exec_row.get("execution_pass")):
            rec["mutation_status"] = "skipped_execution_failed"
            out.append(rec)
            status_counts[rec["mutation_status"]] = status_counts.get(rec["mutation_status"], 0) + 1
            continue

        mutants = make_mutants(str(row.get("canonical_solution") or ""), entry_point, args.max_mutants)
        if not mutants:
            rec["mutation_status"] = "tool_error"
            rec["mutation_error"] = "no_mutants_generated"
            out.append(rec)
            status_counts[rec["mutation_status"]] = status_counts.get(rec["mutation_status"], 0) + 1
            continue

        work_dir = tmp_root / sanitize_fs_name(candidate_id)
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "test_candidate.py").write_text(build_test_file_code(str(row.get("candidate_test_code") or "")), encoding="utf-8")

        killed = 0
        survived = 0
        errors = 0
        for mutant_code, _kind in mutants:
            (work_dir / "solution_under_test.py").write_text(mutant_code, encoding="utf-8")
            try:
                passed, _text = run_pytest(work_dir)
            except subprocess.TimeoutExpired:
                errors += 1
                continue
            if passed:
                survived += 1
            else:
                killed += 1

        total = killed + survived + errors
        rec["mutation_status"] = "ok" if total else "tool_error"
        rec["mutants_total"] = total
        rec["mutants_killed"] = killed
        rec["mutants_survived"] = survived
        rec["mutants_timeout_or_error"] = errors
        rec["mutation_score"] = round(killed / total, 4) if total else None
        out.append(rec)
        status_counts[rec["mutation_status"]] = status_counts.get(rec["mutation_status"], 0) + 1

    write_jsonl(out, out_path)
    write_json(
        {
            "script_version": SCRIPT_VERSION,
            "run_id": args.run_id,
            "n_input_candidates": len(rows),
            "n_output_rows": len(out),
            "status_counts": status_counts,
            "max_mutants_per_candidate": args.max_mutants,
            "runtime_sec": round(time.time() - start, 3),
            "output_path": str(out_path),
        },
        manifest_path,
    )
    print(json.dumps({"n_output_rows": len(out), "status_counts": status_counts, "output_path": str(out_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
