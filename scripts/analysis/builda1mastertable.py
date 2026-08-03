from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.interimschema import stage_file

SCRIPT_VERSION = "build_a1_master_table_v4"

# =========================================================
# 输出路径改到 UTPLM_PROJECT_ROOT/outputs/summaries/a1
# =========================================================
PROJECT_ROOT = Path(os.environ["UTPLM_PROJECT_ROOT"]).resolve()
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "a1autometricsmastertable.csv"
OUT_JSONL = OUT_DIR / "a1autometricsmastertable.jsonl"
OUT_MANIFEST = OUT_DIR / "a1autometricsmastertablemanifest.json"


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


def as_df(path_key: str) -> pd.DataFrame:
    path = stage_file(path_key)
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(load_jsonl(path))


def write_jsonl_from_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            obj = row.where(pd.notna(row), None).to_dict()
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# =========================================================
# 校验 / 工具
# =========================================================
def validate_unique_candidate_id(df: pd.DataFrame, table_name: str) -> None:
    if df.empty:
        return
    if "candidate_id" not in df.columns:
        raise ValueError(f"[{table_name}] missing required column: candidate_id")

    dup_mask = df["candidate_id"].duplicated(keep=False)
    if dup_mask.any():
        dup_df = df.loc[dup_mask, ["candidate_id"]].copy()
        dup_counts = dup_df["candidate_id"].value_counts().head(20).to_dict()
        raise ValueError(
            f"[{table_name}] candidate_id is not unique. "
            f"Top duplicate counts: {dup_counts}"
        )


def stable_sample_id(candidate_id: Any) -> str:
    s = str(candidate_id)
    digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"a1_{digest}"


def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_column(df: pd.DataFrame, col: str, default: Any = None) -> None:
    if col not in df.columns:
        df[col] = default


def collect_run_ids(df: pd.DataFrame) -> List[str]:
    if df.empty or "run_id" not in df.columns:
        return []
    vals = (
        df["run_id"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    return sorted(vals)


def validate_run_id_consistency(tables: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    run_id_map: Dict[str, List[str]] = {}
    for table_name, df in tables.items():
        run_ids = collect_run_ids(df)
        run_id_map[table_name] = run_ids

        if len(run_ids) > 1:
            raise ValueError(
                f"[{table_name}] multiple run_id values detected: {run_ids}"
            )

    non_empty_singletons = {
        name: ids[0]
        for name, ids in run_id_map.items()
        if len(ids) == 1
    }

    if non_empty_singletons:
        unique_values = sorted(set(non_empty_singletons.values()))
        if len(unique_values) > 1:
            raise ValueError(
                "run_id mismatch across input tables: "
                f"{non_empty_singletons}"
            )

    return run_id_map


# =========================================================
# failure taxonomy 聚合
# =========================================================
FAILURE_CATEGORY_PRIORITY = {
    "mutation_survival_major": 100,
    "branch_distinguishing_gap": 95,
    "exception_path_gap": 90,
    "boundary_case_gap": 85,
    "assertion_weak_or_missing": 80,
    "execution_assertion_mismatch": 78,
    "execution_call_contract_error": 77,
    "invalid_candidate_redefines_target": 76,
    "execution_behavior_mismatch": 75,
    "test_runtime_failure": 70,
    "mutation_unavailable": 56,
    "syntax_error": 65,
    "execution_timeout": 60,
    "mutation_timeout": 55,
    "mutation_pipeline_unstable": 50,
    "import_or_env_error": 40,
    "other": 0,
}

SEVERITY_PRIORITY = {
    "critical": 4,
    "major": 3,
    "moderate": 2,
    "minor": 1,
    "info": 0,
}

RULE_STRENGTH_PRIORITY = {
    "strong": 3,
    "moderate": 2,
    "heuristic": 1,
    "weak": 0,
}

STAGE_PRIORITY = {
    "mutation": 4,
    "static_structure": 3,
    "coverage": 2,
    "execution": 1,
    "syntax": 0,
}


def choose_primary_failure_row(group: pd.DataFrame) -> pd.Series:
    def rank_row(row: pd.Series) -> Tuple[int, int, int, int, str]:
        category = str(row.get("failure_category") or "other")
        severity = str(row.get("severity") or "")
        strength = str(row.get("rule_strength") or "")
        stage = str(row.get("stage") or "")

        return (
            FAILURE_CATEGORY_PRIORITY.get(category, 0),
            SEVERITY_PRIORITY.get(severity, 0),
            RULE_STRENGTH_PRIORITY.get(strength, 0),
            STAGE_PRIORITY.get(stage, 0),
            str(row.get("rule_id") or ""),
        )

    ranked = sorted(
        [row for _, row in group.iterrows()],
        key=rank_row,
        reverse=True,
    )
    return ranked[0]


def aggregate_failure_taxonomy(fta: pd.DataFrame) -> pd.DataFrame:
    if fta.empty:
        return pd.DataFrame(columns=["candidate_id"])

    rows: List[Dict[str, Any]] = []

    for cid, g in fta.groupby("candidate_id", sort=False):
        primary = choose_primary_failure_row(g)

        categories = (
            list(dict.fromkeys(g["failure_category"].dropna().astype(str).tolist()))
            if "failure_category" in g.columns else []
        )
        stages = (
            list(dict.fromkeys(g["stage"].dropna().astype(str).tolist()))
            if "stage" in g.columns else []
        )
        severities = (
            list(dict.fromkeys(g["severity"].dropna().astype(str).tolist()))
            if "severity" in g.columns else []
        )
        strengths = (
            list(dict.fromkeys(g["rule_strength"].dropna().astype(str).tolist()))
            if "rule_strength" in g.columns else []
        )

        rows.append(
            {
                "candidate_id": cid,
                "failure_log_count": int(len(g)),
                "primary_failure_category": primary.get("failure_category"),
                "primary_failure_subtype": primary.get("failure_subtype"),
                "primary_failure_stage": primary.get("stage"),
                "primary_failure_severity": primary.get("severity"),
                "primary_failure_rule_strength": primary.get("rule_strength"),
                "primary_failure_rule_id": primary.get("rule_id"),
                "failure_stage_list": "|".join(stages),
                "failure_category_list": "|".join(categories),
                "failure_severity_list": "|".join(severities),
                "failure_rule_strength_list": "|".join(strengths),
                "has_boundary_case_gap": int("boundary_case_gap" in categories),
                "has_branch_distinguishing_gap": int("branch_distinguishing_gap" in categories),
                "has_exception_path_gap": int("exception_path_gap" in categories),
                "has_mutation_survival_major": int("mutation_survival_major" in categories),
                "has_assertion_weak_or_missing": int("assertion_weak_or_missing" in categories),
                "has_execution_assertion_mismatch": int("execution_assertion_mismatch" in categories),
                "has_execution_call_contract_error": int("execution_call_contract_error" in categories),
                "has_invalid_candidate_redefines_target": int("invalid_candidate_redefines_target" in categories),
                "mutation_unavailable": int("mutation_unavailable" in categories),
            }
        )

    out = pd.DataFrame(rows)
    validate_unique_candidate_id(out, "failure_taxonomy_aggregated")
    return out


# =========================================================
# merge 逻辑
# =========================================================
def merge_non_overlapping(
    base: pd.DataFrame,
    other: pd.DataFrame,
    table_name: str,
) -> pd.DataFrame:
    if other.empty:
        return base

    validate_unique_candidate_id(other, table_name)

    keep_cols = ["candidate_id"] + [
        c for c in other.columns
        if c != "candidate_id" and c not in base.columns
    ]
    return base.merge(other[keep_cols], on="candidate_id", how="left")


# =========================================================
# 主流程
# =========================================================
def main() -> None:
    cand = as_df("generated_candidates")
    exe = as_df("execution_results")
    cov = as_df("coverage_results")
    mut = as_df("mutation_results")
    sta = as_df("static_features")
    fta_raw = as_df("failure_taxonomy_logs")

    if cand.empty:
        raise SystemExit("generated_candidates is empty or missing")

    validate_unique_candidate_id(cand, "generated_candidates")
    validate_unique_candidate_id(exe, "execution_results")
    validate_unique_candidate_id(cov, "coverage_results")
    validate_unique_candidate_id(mut, "mutation_results")
    validate_unique_candidate_id(sta, "static_features")

    run_id_map = validate_run_id_consistency(
        {
            "generated_candidates": cand,
            "execution_results": exe,
            "coverage_results": cov,
            "mutation_results": mut,
            "static_features": sta,
            "failure_taxonomy_logs": fta_raw,
        }
    )

    fta_agg = aggregate_failure_taxonomy(fta_raw)

    df = cand.copy()
    df = merge_non_overlapping(df, exe, "execution_results")
    df = merge_non_overlapping(df, cov, "coverage_results")
    df = merge_non_overlapping(df, mut, "mutation_results")
    df = merge_non_overlapping(df, sta, "static_features")
    df = merge_non_overlapping(df, fta_agg, "failure_taxonomy_aggregated")

    # -----------------------------------------------------
    # stable sample_id
    # -----------------------------------------------------
    df["sample_id"] = df["candidate_id"].map(stable_sample_id)

    # -----------------------------------------------------
    # syntax / ast / execution 语义分离
    # -----------------------------------------------------
    ensure_column(df, "ast_parse_pass", None)

    if "syntax_pass" not in df.columns:
        if "execution_status" in df.columns:
            df["syntax_pass"] = (df["execution_status"] != "syntax_error").astype("Int64")
        elif "ast_parse_pass" in df.columns:
            df["syntax_pass"] = df["ast_parse_pass"]
        else:
            df["syntax_pass"] = None

    if "execution_pass" not in df.columns:
        if "execution_status" in df.columns:
            df["execution_pass"] = (df["execution_status"] == "passed").astype("Int64")
        else:
            df["execution_pass"] = None

    # -----------------------------------------------------
    # 可选指标映射（只做语义合理的兼容，不伪造）
    # -----------------------------------------------------
    optional_passthrough = {
        "DDR_or_BRC": ["DDR_or_BRC", "ddr_brc", "BRC", "branch_related_coverage"],
        "TBC": ["TBC", "tbc"],
        "boundary_coverage": ["boundary_coverage", "Boundary_Coverage"],
        "exception_path_coverage": ["exception_path_coverage", "Exception_path_Coverage"],
    }

    for target, candidates in optional_passthrough.items():
        if target not in df.columns:
            df[target] = None
        src = first_existing_col(df, candidates)
        if src is not None and src != target:
            df[target] = df[src]

    # -----------------------------------------------------
    # 默认列补齐
    # -----------------------------------------------------
    default_cols = {
        "mutation_score": None,
        "line_coverage": None,
        "branch_coverage": None,
        "num_asserts": None,
        "assert_density": None,
        "avg_asserts_per_test_function": None,
        "num_exception_checks": None,
        "num_try_except": None,
        "num_pytest_raises": None,
        "num_bare_expression_calls": None,
        "logical_nesting_depth": None,
        "test_style": None,
        "failure_log_count": 0,
        "primary_failure_category": None,
        "primary_failure_subtype": None,
        "primary_failure_stage": None,
        "primary_failure_severity": None,
        "primary_failure_rule_strength": None,
        "primary_failure_rule_id": None,
        "failure_stage_list": "",
        "failure_category_list": "",
        "failure_severity_list": "",
        "failure_rule_strength_list": "",
        "has_boundary_case_gap": 0,
        "has_branch_distinguishing_gap": 0,
        "has_exception_path_gap": 0,
        "has_mutation_survival_major": 0,
        "has_assertion_weak_or_missing": 0,
        "has_execution_assertion_mismatch": 0,
        "has_execution_call_contract_error": 0,
        "has_invalid_candidate_redefines_target": 0,
        "mutation_unavailable": 0,
        "invalid_candidate_flag": 0,
        "candidate_validity_status": None,
        "invalid_candidate_reason": None,
        "redefines_target_flag": 0,
    }
    for col, default in default_cols.items():
        ensure_column(df, col, default)

    # -----------------------------------------------------
    # 对 failure aggregate 列做缺失值收口：
    # 没有 failure log 的样本，不应该表现成 NaN，而应该是 0 / ""
    # -----------------------------------------------------
    fill_zero_cols = [
        "failure_log_count",
        "has_boundary_case_gap",
        "has_branch_distinguishing_gap",
        "has_exception_path_gap",
        "has_mutation_survival_major",
        "has_assertion_weak_or_missing",
        "has_execution_assertion_mismatch",
        "has_execution_call_contract_error",
        "has_invalid_candidate_redefines_target",
        "mutation_unavailable",
        "invalid_candidate_flag",
        "redefines_target_flag",
    ]
    for col in fill_zero_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("Int64")

    fill_empty_string_cols = [
        "failure_stage_list",
        "failure_category_list",
        "failure_severity_list",
        "failure_rule_strength_list",
    ]
    for col in fill_empty_string_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # -----------------------------------------------------
    # 输出列：优先保留正式分析所需字段
    # -----------------------------------------------------
    wanted = [
        # ids / metadata
        "sample_id",
        "candidate_id",
        "task_id",
        "source",
        "difficulty_bucket",
        "test_source_type",
        "candidate_test_origin_detail",
        "run_id",

        # prompt / content
        "prompt",
        "candidate_test_code",

        # syntax / execution
        "ast_parse_pass",
        "syntax_pass",
        "execution_pass",
        "execution_status",
        "exception_type",
        "candidate_validity_status",
        "invalid_candidate_flag",
        "invalid_candidate_reason",
        "redefines_target_flag",

        # automatic metrics
        "mutation_score",
        "line_coverage",
        "branch_coverage",
        "DDR_or_BRC",
        "TBC",
        "boundary_coverage",
        "exception_path_coverage",

        # static features
        "num_asserts",
        "assert_density",
        "avg_asserts_per_test_function",
        "num_exception_checks",
        "num_try_except",
        "num_pytest_raises",
        "num_bare_expression_calls",
        "logical_nesting_depth",
        "test_style",

        # failure taxonomy aggregate
        "failure_log_count",
        "primary_failure_category",
        "primary_failure_subtype",
        "primary_failure_stage",
        "primary_failure_severity",
        "primary_failure_rule_strength",
        "primary_failure_rule_id",
        "failure_stage_list",
        "failure_category_list",
        "failure_severity_list",
        "failure_rule_strength_list",
        "has_boundary_case_gap",
        "has_branch_distinguishing_gap",
        "has_exception_path_gap",
        "has_mutation_survival_major",
        "has_assertion_weak_or_missing",
        "has_execution_assertion_mismatch",
        "has_execution_call_contract_error",
        "has_invalid_candidate_redefines_target",
        "mutation_unavailable",
        "invalid_candidate_flag",
        "redefines_target_flag",
    ]

    existing = [c for c in wanted if c in df.columns]
    out = df[existing].copy()

    sort_cols = [c for c in ["source", "difficulty_bucket", "task_id", "candidate_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    write_jsonl_from_df(out, OUT_JSONL)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "repo_root": str(REPO_ROOT),
        "input_paths": {
            "generated_candidates": str(stage_file("generated_candidates")),
            "execution_results": str(stage_file("execution_results")),
            "coverage_results": str(stage_file("coverage_results")),
            "mutation_results": str(stage_file("mutation_results")),
            "static_features": str(stage_file("static_features")),
            "failure_taxonomy_logs": str(stage_file("failure_taxonomy_logs")),
        },
        "output_paths": {
            "csv": str(OUT_CSV),
            "jsonl": str(OUT_JSONL),
            "manifest": str(OUT_MANIFEST),
        },
        "row_count": int(len(out)),
        "column_count": int(len(out.columns)),
        "columns": list(out.columns),
        "run_id_check": run_id_map,
        "notes": {
            "sample_id": "stable hash-based ID derived from candidate_id",
            "syntax_pass": "explicit syntax pass if available; otherwise inferred from execution_status or ast_parse_pass",
            "ast_parse_pass": "AST parsing success from static feature extraction",
            "execution_pass": "1 only when execution_status == 'passed'",
            "primary_failure_selection": "selected by category priority, then severity, rule strength, and stage priority",
            "boundary_coverage": "only filled from explicitly named boundary-coverage fields; never fabricated from line_coverage",
            "exception_path_coverage": "only filled from explicitly named exception-path fields; never fabricated",
            "failure_aggregate_fill_policy": "samples with no matched failure-taxonomy row use 0 for has_* flags and failure_log_count, and empty string for failure list fields",
            "candidate_validity_fields": "carried from generated_candidates and useful for filtering invalid generated tests such as target redefinition",
            "mutation_unavailable": "1 when mutation pipeline is unavailable/tool_error; should not be interpreted as a logic-quality failure by itself",
        },
    }
    write_json(manifest, OUT_MANIFEST)

    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_JSONL}")
    print(f"[ok] wrote {OUT_MANIFEST}")
    print(f"[rows] {len(out)}")
    print(f"[cols] {list(out.columns)}")


if __name__ == "__main__":
    main()