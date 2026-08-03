from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
a1_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV = a1_DIR / "a1autometricsmastertable.csv"
SCRIPT_VERSION = "build_a3_positive_scoring_table_v1"

SOURCE_ORDER = ["mbpp", "humaneval"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

HIGH_PER_BUCKET = int(os.environ.get("a3_HIGH_PER_BUCKET", "4"))
MEDIUM_PER_BUCKET = int(os.environ.get("a3_MEDIUM_PER_BUCKET", "4"))
WEAK_PER_BUCKET = int(os.environ.get("a3_WEAK_PER_BUCKET", "4"))

STRICT_POS_MIN_MUTATION = float(os.environ.get("a3_STRICT_POS_MIN_MUTATION", "0.90"))
STRICT_POS_MIN_LINE_COV = float(os.environ.get("a3_STRICT_POS_MIN_LINE_COV", "0.90"))
STRICT_POS_MIN_BRANCH_COV = float(os.environ.get("a3_STRICT_POS_MIN_BRANCH_COV", "0.90"))
STRICT_POS_MIN_NUM_ASSERTS = int(os.environ.get("a3_STRICT_POS_MIN_NUM_ASSERTS", "2"))

RELAXED_POS_MIN_MUTATION = float(os.environ.get("a3_RELAXED_POS_MIN_MUTATION", "0.72"))
RELAXED_POS_MIN_LINE_COV = float(os.environ.get("a3_RELAXED_POS_MIN_LINE_COV", "0.80"))
RELAXED_POS_MIN_BRANCH_COV = float(os.environ.get("a3_RELAXED_POS_MIN_BRANCH_COV", "0.70"))
RELAXED_POS_MIN_NUM_ASSERTS = int(os.environ.get("a3_RELAXED_POS_MIN_NUM_ASSERTS", "1"))
RELAXED_POS_MAX_SOFT_GAPS = int(os.environ.get("a3_RELAXED_POS_MAX_SOFT_GAPS", "1"))


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_bool_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).gt(0).astype(np.int8)


def zscore(s: pd.Series) -> pd.Series:
    x = safe_numeric(s)
    std = x.std()
    if pd.isna(std) or float(std) == 0.0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - x.mean()) / std


def safe_value_counts(s: pd.Series | None) -> Dict[str, int]:
    if s is None or len(s) == 0:
        return {}
    s2 = s.astype("object").where(~pd.isna(s), "NA")
    vc = s2.value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.items()}


def infer_current_run_id(df: pd.DataFrame) -> str:
    if "run_id" not in df.columns:
        return "unknown_run"
    vals = df["run_id"].dropna().astype(str).str.strip()
    vals = vals[vals != ""].unique()
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 0:
        return "unknown_run"
    return "mixed_runs"


def coalesce_duplicate_bool_col(df: pd.DataFrame, base_col: str) -> pd.DataFrame:
    dup_cols = [c for c in df.columns if c == base_col or c.startswith(base_col + ".")]
    if not dup_cols:
        return df
    merged = pd.concat(
        [pd.to_numeric(df[c], errors="coerce") for c in dup_cols],
        axis=1,
    ).max(axis=1, skipna=True)
    df[base_col] = merged.fillna(0).astype(np.int8)
    for c in dup_cols:
        if c != base_col:
            df.drop(columns=[c], inplace=True)
    return df


def ensure_default_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    defaults = {
        "source": "unknown",
        "difficulty_bucket": "unknown",
        "syntax_pass": 0,
        "execution_pass": 0,
        "invalid_candidate_flag": 0,
        "redefines_target_flag": 0,
        "mutation_unavailable": 0,
        "has_mutation_survival_major": 0,
        "has_exception_path_gap": 0,
        "has_boundary_case_gap": 0,
        "has_branch_distinguishing_gap": 0,
        "has_assertion_weak_or_missing": 0,
        "has_execution_assertion_mismatch": 0,
        "has_execution_call_contract_error": 0,
        "has_invalid_candidate_redefines_target": 0,
        "num_asserts": 0,
        "assert_density": 0.0,
        "avg_asserts_per_test_function": 0.0,
        "num_exception_checks": 0,
        "num_bare_expression_calls": 0,
        "logical_nesting_depth": 0,
        "line_coverage": np.nan,
        "branch_coverage": np.nan,
        "mutation_score": np.nan,
        "boundary_coverage": np.nan,
        "exception_path_coverage": np.nan,
        "failure_log_count": 0,
        "coverage_status": None,
        "mutation_status": None,
        "test_style": None,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
    return out


def bucket_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["source", "difficulty_bucket", "task_id", "candidate_id"] if c in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return df.reset_index(drop=True)


def concat_nonempty(parts: List[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    valid = [x for x in parts if x is not None and not x.empty]
    if not valid:
        return pd.DataFrame(columns=columns)
    return bucket_sort(pd.concat(valid, ignore_index=True))


def select_high_pool(hard_df: pd.DataFrame, relaxed_df: pd.DataFrame, k: int) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    parts: List[pd.DataFrame] = []
    report: List[Dict[str, Any]] = []

    hard_df = hard_df.sort_values(
        ["positive_teaching_score", "mutation_score", "branch_coverage", "candidate_id"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    relaxed_df = relaxed_df.sort_values(
        ["positive_teaching_score", "mutation_score", "branch_coverage", "candidate_id"],
        ascending=[False, False, False, True],
        kind="stable",
    )

    for src in SOURCE_ORDER:
        for diff in DIFFICULTY_ORDER:
            h = hard_df[(hard_df["source"].astype(str) == src) & (hard_df["difficulty_bucket"].astype(str) == diff)].copy()
            r = relaxed_df[(relaxed_df["source"].astype(str) == src) & (relaxed_df["difficulty_bucket"].astype(str) == diff)].copy()

            chosen_h = h.head(k).copy()
            chosen_h_ids = set(chosen_h["candidate_id"].astype(str).tolist())

            remain = max(0, k - len(chosen_h))
            chosen_r = r[~r["candidate_id"].astype(str).isin(chosen_h_ids)].head(remain).copy()

            if not chosen_h.empty:
                chosen_h["positive_tier"] = "high_quality"
                chosen_h["positive_keep_rule"] = "strict_hard"
            if not chosen_r.empty:
                chosen_r["positive_tier"] = "high_quality"
                chosen_r["positive_keep_rule"] = "relaxed_fallback"

            chosen = concat_nonempty([chosen_h, chosen_r], list(hard_df.columns))
            if not chosen.empty:
                parts.append(chosen)

            report.append(
                {
                    "source": src,
                    "difficulty_bucket": diff,
                    "hard_available": int(len(h)),
                    "relaxed_available": int(len(r)),
                    "selected_from_hard": int(len(chosen_h)),
                    "selected_from_relaxed": int(len(chosen_r)),
                    "selected_total": int(len(chosen)),
                    "target": int(k),
                    "shortfall": int(max(0, k - len(chosen))),
                }
            )

    return concat_nonempty(parts, list(hard_df.columns)), report


def select_medium_pool(df: pd.DataFrame, exclude_ids: set[str], k: int) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    parts: List[pd.DataFrame] = []
    report: List[Dict[str, Any]] = []

    base = df[~df["candidate_id"].astype(str).isin(exclude_ids)].copy()

    for src in SOURCE_ORDER:
        for diff in DIFFICULTY_ORDER:
            g = base[(base["source"].astype(str) == src) & (base["difficulty_bucket"].astype(str) == diff)].copy()

            if g.empty:
                report.append(
                    {
                        "source": src,
                        "difficulty_bucket": diff,
                        "available": 0,
                        "selected_total": 0,
                        "target": int(k),
                        "shortfall": int(k),
                    }
                )
                continue

            med = g["positive_teaching_score"].median()
            g["__dist_to_median__"] = (g["positive_teaching_score"] - med).abs()

            chosen = g.sort_values(
                ["__dist_to_median__", "candidate_id"],
                ascending=[True, True],
                kind="stable",
            ).head(k).copy()

            chosen["positive_tier"] = "medium_quality"
            chosen["positive_keep_rule"] = "median_band"
            chosen = chosen.drop(columns="__dist_to_median__")
            parts.append(chosen)

            report.append(
                {
                    "source": src,
                    "difficulty_bucket": diff,
                    "available": int(len(g)),
                    "selected_total": int(len(chosen)),
                    "target": int(k),
                    "shortfall": int(max(0, k - len(chosen))),
                }
            )

    return concat_nonempty(parts, list(df.columns)), report


def select_weak_pool(df: pd.DataFrame, exclude_ids: set[str], k: int) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    parts: List[pd.DataFrame] = []
    report: List[Dict[str, Any]] = []

    base = df[~df["candidate_id"].astype(str).isin(exclude_ids)].copy()

    for src in SOURCE_ORDER:
        for diff in DIFFICULTY_ORDER:
            g = base[(base["source"].astype(str) == src) & (base["difficulty_bucket"].astype(str) == diff)].copy()

            chosen = g.sort_values(
                ["positive_teaching_score", "candidate_id"],
                ascending=[True, True],
                kind="stable",
            ).head(k).copy()

            if not chosen.empty:
                chosen["positive_tier"] = "weak_quality"
                chosen["positive_keep_rule"] = "bottom_band"
                parts.append(chosen)

            report.append(
                {
                    "source": src,
                    "difficulty_bucket": diff,
                    "available": int(len(g)),
                    "selected_total": int(len(chosen)),
                    "target": int(k),
                    "shortfall": int(max(0, k - len(chosen))),
                }
            )

    return concat_nonempty(parts, list(df.columns)), report


def summarize_by_tier(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "positive_teaching_score",
        "teaching_proxy_score",
        "mutation_score",
        "line_coverage",
        "branch_coverage",
        "boundary_coverage",
        "exception_path_coverage",
        "num_asserts",
        "assert_density",
        "num_exception_checks",
        "failure_log_count",
        "pos_soft_gap_count",
    ]

    rows: List[Dict[str, Any]] = []
    for tier, sub in df.groupby("positive_tier", dropna=False):
        row: Dict[str, Any] = {"positive_tier": str(tier), "n_rows": int(len(sub))}
        for m in metrics:
            x = pd.to_numeric(sub[m], errors="coerce")
            row[f"{m}_mean"] = float(x.mean()) if x.notna().any() else None
            row[f"{m}_median"] = float(x.median()) if x.notna().any() else None
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"master table not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    for base in ["invalid_candidate_flag", "redefines_target_flag"]:
        df = coalesce_duplicate_bool_col(df, base)

    df = ensure_default_columns(df)
    current_run_id = infer_current_run_id(df)
    safe_run = str(current_run_id).replace("/", "_").replace(" ", "_")

    if "source" in df.columns:
        df["source"] = pd.Categorical(df["source"].astype(str), categories=SOURCE_ORDER, ordered=True)
    if "difficulty_bucket" in df.columns:
        df["difficulty_bucket"] = pd.Categorical(df["difficulty_bucket"].astype(str), categories=DIFFICULTY_ORDER, ordered=True)

    bool_cols = [
        "syntax_pass",
        "execution_pass",
        "invalid_candidate_flag",
        "redefines_target_flag",
        "mutation_unavailable",
        "has_mutation_survival_major",
        "has_exception_path_gap",
        "has_boundary_case_gap",
        "has_branch_distinguishing_gap",
        "has_assertion_weak_or_missing",
        "has_execution_assertion_mismatch",
        "has_execution_call_contract_error",
        "has_invalid_candidate_redefines_target",
    ]
    for col in bool_cols:
        df[col] = safe_bool_series(df[col])

    numeric_cols = [
        "mutation_score",
        "line_coverage",
        "branch_coverage",
        "boundary_coverage",
        "exception_path_coverage",
        "num_asserts",
        "assert_density",
        "avg_asserts_per_test_function",
        "num_exception_checks",
        "num_bare_expression_calls",
        "logical_nesting_depth",
        "failure_log_count",
    ]
    for col in numeric_cols:
        df[col] = safe_numeric(df[col])

    usable_exec_mask = (
        (df["syntax_pass"] == 1)
        & (df["execution_pass"] == 1)
        & (df["invalid_candidate_flag"] == 0)
        & (df["redefines_target_flag"] == 0)
    )

    if "coverage_status" in df.columns and df["coverage_status"].notna().any():
        coverage_ok = df["coverage_status"].astype(str).eq("ok")
    else:
        coverage_ok = df["line_coverage"].notna() & df["branch_coverage"].notna()

    if "mutation_status" in df.columns and df["mutation_status"].notna().any():
        mutation_ok = df["mutation_status"].astype(str).eq("ok")
    else:
        mutation_ok = df["mutation_score"].notna()

    usable_eval_mask = (
        usable_exec_mask
        & coverage_ok
        & mutation_ok
        & (df["mutation_unavailable"] == 0)
    )

    pos_soft_gap_cols = [
        "has_exception_path_gap",
        "has_boundary_case_gap",
        "has_branch_distinguishing_gap",
    ]
    df["pos_soft_gap_count"] = sum(safe_bool_series(df[c]) for c in pos_soft_gap_cols)

    df["teaching_proxy_score"] = (
        0.35 * zscore(df["assert_density"].fillna(0))
        + 0.20 * zscore(df["num_asserts"].fillna(0))
        + 0.15 * zscore(df["num_exception_checks"].fillna(0))
        + 0.10 * zscore(df["avg_asserts_per_test_function"].fillna(0))
        - 0.10 * zscore(df["num_bare_expression_calls"].fillna(0))
        - 0.10 * zscore(df["logical_nesting_depth"].fillna(0))
        - 0.15 * zscore(df["failure_log_count"].fillna(0))
    )

    df["positive_teaching_score"] = (
        0.35 * zscore(df["mutation_score"].fillna(0))
        + 0.15 * zscore(df["branch_coverage"].fillna(0))
        + 0.10 * zscore(df["line_coverage"].fillna(0))
        + 0.10 * zscore(df["boundary_coverage"].fillna(0))
        + 0.10 * zscore(df["exception_path_coverage"].fillna(0))
        + 0.20 * df["teaching_proxy_score"]
        - 0.20 * zscore(df["pos_soft_gap_count"].fillna(0))
        - 0.15 * zscore(df["failure_log_count"].fillna(0))
    )

    pos_critical_zero_mask = (
        (df["has_mutation_survival_major"] == 0)
        & (df["has_assertion_weak_or_missing"] == 0)
        & (df["has_execution_assertion_mismatch"] == 0)
        & (df["has_execution_call_contract_error"] == 0)
        & (df["has_invalid_candidate_redefines_target"] == 0)
        & (df["invalid_candidate_flag"] == 0)
        & (df["redefines_target_flag"] == 0)
        & (df["mutation_unavailable"] == 0)
    )

    strict_high_mask = (
        usable_eval_mask
        & pos_critical_zero_mask
        & (df["mutation_score"] >= STRICT_POS_MIN_MUTATION)
        & (df["line_coverage"] >= STRICT_POS_MIN_LINE_COV)
        & (df["branch_coverage"] >= STRICT_POS_MIN_BRANCH_COV)
        & (df["num_asserts"] >= STRICT_POS_MIN_NUM_ASSERTS)
        & (df["pos_soft_gap_count"] == 0)
    )

    relaxed_high_mask = (
        usable_eval_mask
        & pos_critical_zero_mask
        & (df["mutation_score"] >= RELAXED_POS_MIN_MUTATION)
        & (df["line_coverage"] >= RELAXED_POS_MIN_LINE_COV)
        & (df["branch_coverage"] >= RELAXED_POS_MIN_BRANCH_COV)
        & (df["num_asserts"] >= RELAXED_POS_MIN_NUM_ASSERTS)
        & (df["pos_soft_gap_count"] <= RELAXED_POS_MAX_SOFT_GAPS)
    )

    hard_high_candidates = bucket_sort(df[strict_high_mask].copy())
    relaxed_high_candidates = bucket_sort(df[relaxed_high_mask].copy())
    viable_positive_pool = bucket_sort(df[usable_eval_mask].copy())

    high_pool, high_bucket_report = select_high_pool(
        hard_high_candidates,
        relaxed_high_candidates,
        HIGH_PER_BUCKET,
    )
    selected_high_ids = set(high_pool["candidate_id"].astype(str).tolist()) if not high_pool.empty else set()

    medium_pool, medium_bucket_report = select_medium_pool(
        viable_positive_pool,
        selected_high_ids,
        MEDIUM_PER_BUCKET,
    )
    selected_medium_ids = set(medium_pool["candidate_id"].astype(str).tolist()) if not medium_pool.empty else set()

    weak_pool, weak_bucket_report = select_weak_pool(
        viable_positive_pool,
        selected_high_ids | selected_medium_ids,
        WEAK_PER_BUCKET,
    )

    scored = viable_positive_pool.copy()
    tier_map: Dict[str, str] = {}
    keep_rule_map: Dict[str, str] = {}

    for sub in [high_pool, medium_pool, weak_pool]:
        if sub.empty:
            continue
        for _, row in sub[["candidate_id", "positive_tier", "positive_keep_rule"]].iterrows():
            cid = str(row["candidate_id"])
            tier_map[cid] = str(row["positive_tier"])
            keep_rule_map[cid] = str(row["positive_keep_rule"])

    scored["positive_tier"] = scored["candidate_id"].astype(str).map(tier_map).fillna("unselected")
    scored["positive_keep_rule"] = scored["candidate_id"].astype(str).map(keep_rule_map).fillna("not_selected")
    scored = bucket_sort(scored)

    selected_all = concat_nonempty(
        [high_pool, medium_pool, weak_pool],
        list(viable_positive_pool.columns),
    )
    summary_df = summarize_by_tier(selected_all)

    out_scored = OUT_DIR / f"a3positivescoredtable{saferun}.csv"
    out_high = OUT_DIR / f"a3positivehighqualitypool{saferun}.csv"
    out_medium = OUT_DIR / f"a3positivemediumqualitypool{saferun}.csv"
    out_weak = OUT_DIR / f"a3positiveweakqualitypool{saferun}.csv"
    out_summary = OUT_DIR / f"a3positivetiersummary{saferun}.csv"
    out_report = OUT_DIR / f"a3positiverulereport{saferun}.json"

    scored.to_csv(out_scored, index=False, encoding="utf-8")
    high_pool.to_csv(out_high, index=False, encoding="utf-8")
    medium_pool.to_csv(out_medium, index=False, encoding="utf-8")
    weak_pool.to_csv(out_weak, index=False, encoding="utf-8")
    summary_df.to_csv(out_summary, index=False, encoding="utf-8")

    report = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "input_csv": str(IN_CSV),
        "current_run_id": current_run_id,
        "scoring_formula": {
            "positive_teaching_score": [
                "0.35*z(mutation_score)",
                "0.15*z(branch_coverage)",
                "0.10*z(line_coverage)",
                "0.10*z(boundary_coverage)",
                "0.10*z(exception_path_coverage)",
                "0.20*teaching_proxy_score",
                "-0.20*z(pos_soft_gap_count)",
                "-0.15*z(failure_log_count)",
            ],
            "teaching_proxy_score": [
                "0.35*z(assert_density)",
                "0.20*z(num_asserts)",
                "0.15*z(num_exception_checks)",
                "0.10*z(avg_asserts_per_test_function)",
                "-0.10*z(num_bare_expression_calls)",
                "-0.10*z(logical_nesting_depth)",
                "-0.15*z(failure_log_count)",
            ],
        },
        "tier_rules": {
            "high_quality": {
                "strict": {
                    "usable_eval": True,
                    "critical_flags_zero": True,
                    "mutation_score_min": STRICT_POS_MIN_MUTATION,
                    "line_coverage_min": STRICT_POS_MIN_LINE_COV,
                    "branch_coverage_min": STRICT_POS_MIN_BRANCH_COV,
                    "num_asserts_min": STRICT_POS_MIN_NUM_ASSERTS,
                    "pos_soft_gap_count_eq": 0,
                },
                "relaxed_fallback": {
                    "usable_eval": True,
                    "critical_flags_zero": True,
                    "mutation_score_min": RELAXED_POS_MIN_MUTATION,
                    "line_coverage_min": RELAXED_POS_MIN_LINE_COV,
                    "branch_coverage_min": RELAXED_POS_MIN_BRANCH_COV,
                    "num_asserts_min": RELAXED_POS_MIN_NUM_ASSERTS,
                    "pos_soft_gap_count_le": RELAXED_POS_MAX_SOFT_GAPS,
                },
                "selection": f"top {HIGH_PER_BUCKET} per source x difficulty bucket by positive_teaching_score",
            },
            "medium_quality": {
                "pool": "usable_eval_pool after removing selected high_quality",
                "selection": f"pick {MEDIUM_PER_BUCKET} per source x difficulty nearest to bucket median positive_teaching_score",
            },
            "weak_quality": {
                "pool": "usable_eval_pool after removing selected high_quality and medium_quality",
                "selection": f"pick bottom {WEAK_PER_BUCKET} per source x difficulty by positive_teaching_score",
            },
        },
        "counts": {
            "total_rows": int(len(df)),
            "usable_eval_pool": int(len(viable_positive_pool)),
            "strict_high_candidates": int(len(hard_high_candidates)),
            "relaxed_high_candidates": int(len(relaxed_high_candidates)),
            "high_quality_pool": int(len(high_pool)),
            "medium_quality_pool": int(len(medium_pool)),
            "weak_quality_pool": int(len(weak_pool)),
        },
        "distribution": {
            "high_quality_source_counts": safe_value_counts(high_pool.get("source")),
            "high_quality_difficulty_counts": safe_value_counts(high_pool.get("difficulty_bucket")),
            "medium_quality_source_counts": safe_value_counts(medium_pool.get("source")),
            "medium_quality_difficulty_counts": safe_value_counts(medium_pool.get("difficulty_bucket")),
            "weak_quality_source_counts": safe_value_counts(weak_pool.get("source")),
            "weak_quality_difficulty_counts": safe_value_counts(weak_pool.get("difficulty_bucket")),
        },
        "bucket_report": {
            "high_quality_pool": high_bucket_report,
            "medium_quality_pool": medium_bucket_report,
            "weak_quality_pool": weak_bucket_report,
        },
        "output_paths": {
            "scored_table": str(out_scored),
            "high_quality_pool": str(out_high),
            "medium_quality_pool": str(out_medium),
            "weak_quality_pool": str(out_weak),
            "tier_summary": str(out_summary),
            "report": str(out_report),
        },
    }

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_scored}")
    print(f"[ok] wrote {out_high}")
    print(f"[ok] wrote {out_medium}")
    print(f"[ok] wrote {out_weak}")
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {out_report}")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
