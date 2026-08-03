from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.common.interimschema import stage_file, stage_manifest_file, base_record, FAILURE_CATEGORIES
except Exception:
    from src.common.interimschema import stage_file, stage_manifest_file, base_record  # type: ignore
    FAILURE_CATEGORIES = []

EXEC_PATH = stage_file("execution_results")
STATIC_PATH = stage_file("static_features")
COV_PATH = stage_file("coverage_results")
MUT_PATH = stage_file("mutation_results")
GENERATED_CANDIDATES_PATH = stage_file("generated_candidates")
OUT_PATH = stage_file("failure_taxonomy_logs")
MANIFEST_PATH = stage_manifest_file("failure_taxonomy_logs")

SCRIPT_VERSION = "build_failure_taxonomy_v6"
TAXONOMY_RULE_VERSION = "ftx_v3"
MAX_EVIDENCE_CHARS = 1200

LINE_COV_GAP_TH = 0.80
BRANCH_COV_GAP_TH = 0.80
MUTATION_MAJOR_SURVIVAL_MIN = 1

EXCEPTION_HINT_PATTERN = re.compile(
    r"\b(raise|raises|raised|error|exception|invalid|illegal|fail|fails|failure|overflow|underflow|zerodivision|typeerror|valueerror|keyerror|indexerror|runtimeerror)\b",
    flags=re.IGNORECASE,
)

EXPANDED_FAILURE_CATEGORIES = {
    "syntax_error",
    "execution_timeout",
    "import_or_env_error",
    "assertion_weak_or_missing",
    "test_runtime_failure",
    "execution_behavior_mismatch",
    "execution_assertion_mismatch",
    "execution_call_contract_error",
    "invalid_candidate_redefines_target",
    "mutation_unavailable",
    "exception_path_gap",
    "boundary_case_gap",
    "branch_distinguishing_gap",
    "mutation_timeout",
    "mutation_pipeline_unstable",
    "mutation_survival_major",
    "other",
}

_FAILURE_CATEGORIES_SET = set(FAILURE_CATEGORIES) | EXPANDED_FAILURE_CATEGORIES


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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


def normalize_evidence(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s[:MAX_EVIDENCE_CHARS] if s else None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, list):
        return [normalize_evidence(x) for x in value]
    if isinstance(value, dict):
        return {str(k): normalize_evidence(v) for k, v in value.items()}
    s = str(value).strip()
    return s[:MAX_EVIDENCE_CHARS] if s else None


def ensure_valid_category(category: str) -> str:
    return category if category in _FAILURE_CATEGORIES_SET else "other"


def safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


def base_failure_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **base_record(SCRIPT_VERSION),
        "candidate_id": row.get("candidate_id"),
        "task_id": row.get("task_id"),
        "source": row.get("source"),
        "difficulty_bucket": row.get("difficulty_bucket", "unknown"),
        "test_source_type": row.get("test_source_type", "unknown"),
    }


def make_log(
    base: Dict[str, Any],
    *,
    stage: str,
    failure_category: str,
    failure_subtype: str,
    evidence: Any,
    auto_detected: bool,
    needs_manual_review: bool,
    rule_strength: str,
    severity: str,
    rule_id: str,
    provenance: str,
) -> Dict[str, Any]:
    return {
        **base,
        "stage": stage,
        "failure_category": ensure_valid_category(failure_category),
        "failure_subtype": failure_subtype,
        "evidence": normalize_evidence(evidence),
        "auto_detected": bool(auto_detected),
        "needs_manual_review": bool(needs_manual_review),
        "taxonomy_rule_version": TAXONOMY_RULE_VERSION,
        "rule_id": rule_id,
        "rule_strength": rule_strength,
        "severity": severity,
        "provenance": provenance,
    }


def dedupe_and_sort_logs(logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for row in logs:
        fingerprint = json.dumps(
            {
                "candidate_id": row.get("candidate_id"),
                "stage": row.get("stage"),
                "failure_category": row.get("failure_category"),
                "failure_subtype": row.get("failure_subtype"),
                "evidence": row.get("evidence"),
                "rule_id": row.get("rule_id"),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        if fingerprint not in seen:
            seen.add(fingerprint)
            deduped.append(row)
    return sorted(
        deduped,
        key=lambda x: (
            str(x.get("candidate_id") or ""),
            str(x.get("stage") or ""),
            str(x.get("failure_category") or ""),
            str(x.get("failure_subtype") or ""),
            str(x.get("rule_id") or ""),
        ),
    )


def join_text_parts(*parts: Any) -> str:
    chunks: List[str] = []
    for part in parts:
        if part is None:
            continue
        if isinstance(part, list):
            chunks.extend(str(x) for x in part if x is not None)
        else:
            chunks.append(str(part))
    return "\n".join(chunks)


def infer_exception_relevance(candidate_row: Dict[str, Any], static_row: Optional[Dict[str, Any]]) -> bool:
    text = join_text_parts(
        candidate_row.get("prompt"),
        candidate_row.get("canonical_solution"),
        candidate_row.get("base_tests"),
        candidate_row.get("candidate_test_code"),
    )
    if EXCEPTION_HINT_PATTERN.search(text):
        return True
    if static_row:
        if safe_int(static_row.get("num_pytest_raises", 0)) and safe_int(static_row.get("num_pytest_raises", 0)) > 0:
            return True
        if safe_int(static_row.get("num_try_except", 0)) and safe_int(static_row.get("num_try_except", 0)) > 0:
            return True
        if safe_int(static_row.get("num_raise_statements", 0)) and safe_int(static_row.get("num_raise_statements", 0)) > 0:
            return True
    return False


def build_execution_failure_logs(exec_rows: List[Dict[str, Any]], candidate_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    for row in exec_rows:
        base = base_failure_record(row)
        status = row.get("execution_status")
        exception_type = str(row.get("exception_type") or "").strip()
        traceback_snippet = row.get("traceback_snippet") or row.get("stderr")
        candidate_row = candidate_index.get(row.get("candidate_id"), {})

        if int(candidate_row.get("redefines_target_flag", 0) or 0) > 0:
            logs.append(
                make_log(
                    base,
                    stage="execution",
                    failure_category="invalid_candidate_redefines_target",
                    failure_subtype="redefines_target_symbol",
                    evidence={
                        "entry_point": candidate_row.get("entry_point"),
                        "redefines_target_symbol": candidate_row.get("redefines_target_symbol"),
                        "candidate_validity_status": candidate_row.get("candidate_validity_status"),
                    },
                    auto_detected=True,
                    needs_manual_review=False,
                    rule_strength="strong",
                    severity="major",
                    rule_id="exec.invalid_candidate_redefines_target",
                    provenance="generated_candidates",
                )
            )

        if status == "syntax_error":
            logs.append(
                make_log(
                    base,
                    stage="syntax",
                    failure_category="syntax_error",
                    failure_subtype="parse_failed",
                    evidence=traceback_snippet,
                    auto_detected=True,
                    needs_manual_review=False,
                    rule_strength="strong",
                    severity="major",
                    rule_id="exec.syntax_error",
                    provenance="execution",
                )
            )
        elif status == "timeout":
            logs.append(
                make_log(
                    base,
                    stage="execution",
                    failure_category="execution_timeout",
                    failure_subtype="subprocess_timeout",
                    evidence=traceback_snippet,
                    auto_detected=True,
                    needs_manual_review=False,
                    rule_strength="strong",
                    severity="major",
                    rule_id="exec.timeout",
                    provenance="execution",
                )
            )
        elif status == "infra_error":
            logs.append(
                make_log(
                    base,
                    stage="execution",
                    failure_category="import_or_env_error",
                    failure_subtype=exception_type or "infra_error",
                    evidence=traceback_snippet,
                    auto_detected=True,
                    needs_manual_review=True,
                    rule_strength="strong",
                    severity="major",
                    rule_id="exec.infra_error",
                    provenance="execution",
                )
            )
        elif status == "failed":
            if exception_type == "AssertionError":
                failure_category = "execution_assertion_mismatch"
                failure_subtype = "assertion_error"
            elif exception_type in {"TypeError", "ValueError", "IndexError", "KeyError", "AttributeError"}:
                failure_category = "execution_call_contract_error"
                failure_subtype = exception_type or "call_contract_error"
            else:
                failure_category = "test_runtime_failure"
                failure_subtype = exception_type or "execution_failed"

            logs.append(
                make_log(
                    base,
                    stage="execution",
                    failure_category=failure_category,
                    failure_subtype=failure_subtype,
                    evidence=traceback_snippet,
                    auto_detected=True,
                    needs_manual_review=True,
                    rule_strength="moderate",
                    severity="major",
                    rule_id="exec.failed",
                    provenance="execution",
                )
            )
    return logs


def build_static_and_coverage_failure_logs(
    exec_rows: List[Dict[str, Any]],
    static_rows: List[Dict[str, Any]],
    cov_rows: List[Dict[str, Any]],
    candidate_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    exec_index = {row["candidate_id"]: row for row in exec_rows if "candidate_id" in row}
    cov_index = {row["candidate_id"]: row for row in cov_rows if "candidate_id" in row}
    candidate_index = {row["candidate_id"]: row for row in candidate_rows if "candidate_id" in row}

    for srow in static_rows:
        cid = srow.get("candidate_id")
        if cid is None:
            continue
        exec_row = exec_index.get(cid)
        if exec_row is None:
            continue
        base = base_failure_record(exec_row)
        crow = cov_index.get(cid, {})
        candidate_row = candidate_index.get(cid, {})

        num_asserts = safe_int(srow.get("num_asserts")) or 0
        num_try_except = safe_int(srow.get("num_try_except")) or 0
        has_pytest_raises = bool(srow.get("has_pytest_raises", False))
        exception_relevant = infer_exception_relevance(candidate_row, srow)

        if num_asserts == 0:
            logs.append(
                make_log(
                    base,
                    stage="static_structure",
                    failure_category="assertion_weak_or_missing",
                    failure_subtype="no_assertions",
                    evidence={
                        "num_asserts": num_asserts,
                        "test_style": srow.get("test_style"),
                        "num_test_functions": srow.get("num_test_functions"),
                    },
                    auto_detected=True,
                    needs_manual_review=True,
                    rule_strength="strong",
                    severity="major",
                    rule_id="static.no_assertions",
                    provenance="static_features",
                )
            )

        if exception_relevant and not has_pytest_raises and num_try_except == 0:
            logs.append(
                make_log(
                    base,
                    stage="static_structure",
                    failure_category="exception_path_gap",
                    failure_subtype="no_exception_checks",
                    evidence={
                        "num_try_except": num_try_except,
                        "has_pytest_raises": has_pytest_raises,
                        "num_pytest_raises": srow.get("num_pytest_raises"),
                    },
                    auto_detected=True,
                    needs_manual_review=True,
                    rule_strength="heuristic",
                    severity="moderate",
                    rule_id="static.exception_path_gap",
                    provenance="static_features",
                )
            )

        cov_status = crow.get("coverage_status")
        line_cov = safe_float(crow.get("line_coverage"))
        branch_cov = safe_float(crow.get("branch_coverage"))
        if cov_status == "ok" and line_cov is not None and line_cov < LINE_COV_GAP_TH:
            logs.append(
                make_log(
                    base,
                    stage="coverage",
                    failure_category="boundary_case_gap",
                    failure_subtype="low_line_coverage",
                    evidence={"line_coverage": line_cov, "threshold": LINE_COV_GAP_TH},
                    auto_detected=True,
                    needs_manual_review=True,
                    rule_strength="heuristic",
                    severity="moderate",
                    rule_id="coverage.low_line_coverage",
                    provenance="coverage_results",
                )
            )
        if cov_status == "ok" and branch_cov is not None and branch_cov < BRANCH_COV_GAP_TH:
            logs.append(
                make_log(
                    base,
                    stage="coverage",
                    failure_category="branch_distinguishing_gap",
                    failure_subtype="low_branch_coverage",
                    evidence={"branch_coverage": branch_cov, "threshold": BRANCH_COV_GAP_TH},
                    auto_detected=True,
                    needs_manual_review=True,
                    rule_strength="heuristic",
                    severity="moderate",
                    rule_id="coverage.low_branch_coverage",
                    provenance="coverage_results",
                )
            )
        if cov_status == "tool_error":
            logs.append(
                make_log(
                    base,
                    stage="coverage",
                    failure_category="other",
                    failure_subtype="coverage_tool_error",
                    evidence={
                        "coverage_error_type": crow.get("coverage_error_type"),
                        "coverage_error_msg": crow.get("coverage_error_msg"),
                    },
                    auto_detected=True,
                    needs_manual_review=True,
                    rule_strength="moderate",
                    severity="minor",
                    rule_id="coverage.tool_error",
                    provenance="coverage_results",
                )
            )
    return logs


def build_mutation_failure_logs(exec_rows: List[Dict[str, Any]], mut_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    exec_index = {row["candidate_id"]: row for row in exec_rows if "candidate_id" in row}

    for row in mut_rows:
        base_row = exec_index.get(row.get("candidate_id"), row)
        base = base_failure_record(base_row)
        status = row.get("mutation_status")
        error_type = row.get("mutation_error_type")
        error_msg = row.get("mutation_error_msg")

        if status == "timeout":
            logs.append(
                make_log(
                    base,
                    stage="mutation",
                    failure_category="mutation_timeout",
                    failure_subtype="mutation_tool_timeout",
                    evidence=error_msg,
                    auto_detected=True,
                    needs_manual_review=False,
                    rule_strength="strong",
                    severity="major",
                    rule_id="mutation.timeout",
                    provenance="mutation_results",
                )
            )
        elif status == "tool_error":
            logs.append(
                make_log(
                    base,
                    stage="mutation",
                    failure_category="mutation_unavailable",
                    failure_subtype=error_msg or error_type or "mutation_tool_error",
                    evidence={
                        "mutation_error_type": error_type,
                        "mutation_error_msg": error_msg,
                        "stderr": row.get("mutation_stderr"),
                        "stdout": row.get("mutation_stdout"),
                        "cmd": row.get("mutation_cmd"),
                        "exit_code": row.get("mutation_exit_code"),
                        "work_dir": row.get("work_dir"),
                        "runner_mode": row.get("runner_mode"),
                    },
                    auto_detected=True,
                    needs_manual_review=True,
                    rule_strength="strong",
                    severity="moderate",
                    rule_id="mutation.tool_error",
                    provenance="mutation_results",
                )
            )
        elif status == "ok":
            survived = safe_int(row.get("mutants_survived")) or 0
            if survived >= MUTATION_MAJOR_SURVIVAL_MIN:
                logs.append(
                    make_log(
                        base,
                        stage="mutation",
                        failure_category="mutation_survival_major",
                        failure_subtype="survived_mutants_present",
                        evidence={
                            "mutants_total": row.get("mutants_total"),
                            "mutants_killed": row.get("mutants_killed"),
                            "mutants_survived": survived,
                            "mutation_score": row.get("mutation_score"),
                            "survived_mutant_ids": row.get("survived_mutant_ids", []),
                        },
                        auto_detected=True,
                        needs_manual_review=True,
                        rule_strength="strong",
                        severity="major",
                        rule_id="mutation.survival_major",
                        provenance="mutation_results",
                    )
                )
    return logs


def main() -> None:
    exec_rows = load_jsonl(EXEC_PATH)
    static_rows = load_jsonl(STATIC_PATH)
    cov_rows = load_jsonl(COV_PATH)
    mut_rows = load_jsonl(MUT_PATH)
    candidate_rows = load_jsonl(GENERATED_CANDIDATES_PATH)

    candidate_index = {row["candidate_id"]: row for row in candidate_rows if "candidate_id" in row}

    logs: List[Dict[str, Any]] = []
    logs.extend(build_execution_failure_logs(exec_rows, candidate_index))
    logs.extend(build_static_and_coverage_failure_logs(exec_rows, static_rows, cov_rows, candidate_rows))
    logs.extend(build_mutation_failure_logs(exec_rows, mut_rows))

    logs = dedupe_and_sort_logs(logs)
    write_jsonl(logs, OUT_PATH)

    stage_counter = Counter(row["stage"] for row in logs)
    category_counter = Counter(row["failure_category"] for row in logs)
    review_counter = Counter("needs_manual_review" if row.get("needs_manual_review") else "auto_only" for row in logs)

    manifest = {
        "run_id": logs[0].get("run_id") if logs else None,
        "script_version": SCRIPT_VERSION,
        "taxonomy_rule_version": TAXONOMY_RULE_VERSION,
        "input_paths": {
            "generated_candidates": str(GENERATED_CANDIDATES_PATH),
            "execution_results": str(EXEC_PATH),
            "static_features": str(STATIC_PATH),
            "coverage_results": str(COV_PATH),
            "mutation_results": str(MUT_PATH),
        },
        "output_path": str(OUT_PATH),
        "num_rows": len(logs),
        "stage_counts": dict(stage_counter),
        "category_counts": dict(category_counter),
        "review_counts": dict(review_counter),
        "notes": [
            "This version separates execution assertion mismatch, call-contract error, and invalid candidate redefinition.",
            "Mutation tool_error is mapped to mutation_unavailable rather than a logic-quality failure.",
            "Static and coverage heuristics remain supportive signals and should be combined with manual review when needed.",
        ],
    }
    write_json(manifest, MANIFEST_PATH)

    print(f"wrote {len(logs)} rows to {OUT_PATH}")
    print(f"stage_counts={dict(stage_counter)}")
    print(f"category_counts={dict(category_counter)}")
    print(f"review_counts={dict(review_counter)}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()
