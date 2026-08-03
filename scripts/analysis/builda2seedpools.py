from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
a1_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV = a1_DIR / "a1autometricsmastertable.csv"

SCRIPT_VERSION = "build_a2_seed_pools_v1"

SOURCE_ORDER = ["mbpp", "humaneval"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

POS_PER_BUCKET = int(os.environ.get("a2_POS_PER_BUCKET", "8"))
NEG_PER_BUCKET = int(os.environ.get("a2_NEG_PER_BUCKET", "6"))
SPOTCHECK_PER_BUCKET = int(os.environ.get("a2_SPOTCHECK_PER_BUCKET", "3"))
BOUNDARY_SPOTCHECK_N = int(os.environ.get("a2_BOUNDARY_SPOTCHECK_N", "6"))

# ---------- hard positive thresholds ----------
STRICT_POS_MIN_MUTATION = float(os.environ.get("a2_STRICT_POS_MIN_MUTATION", "0.90"))
STRICT_POS_MIN_LINE_COV = float(os.environ.get("a2_STRICT_POS_MIN_LINE_COV", "0.90"))
STRICT_POS_MIN_BRANCH_COV = float(os.environ.get("a2_STRICT_POS_MIN_BRANCH_COV", "0.90"))
STRICT_POS_MIN_NUM_ASSERTS = int(os.environ.get("a2_STRICT_POS_MIN_NUM_ASSERTS", "2"))

# ---------- relaxed fallback thresholds ----------
RELAXED_POS_MIN_MUTATION = float(os.environ.get("a2_RELAXED_POS_MIN_MUTATION", "0.72"))
RELAXED_POS_MIN_LINE_COV = float(os.environ.get("a2_RELAXED_POS_MIN_LINE_COV", "0.80"))
RELAXED_POS_MIN_BRANCH_COV = float(os.environ.get("a2_RELAXED_POS_MIN_BRANCH_COV", "0.70"))
RELAXED_POS_MIN_NUM_ASSERTS = int(os.environ.get("a2_RELAXED_POS_MIN_NUM_ASSERTS", "1"))
RELAXED_POS_MAX_SOFT_GAPS = int(os.environ.get("a2_RELAXED_POS_MAX_SOFT_GAPS", "1"))


# =========================================================
# helpers
# =========================================================
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
    vals = (
        df["run_id"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    vals = vals[vals != ""].unique()
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 0:
        return "unknown_run"
    return "mixed_runs"


def validate_input(df: pd.DataFrame) -> None:
    required = {"candidate_id", "task_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"master table 缺少必要字段: {sorted(missing)}")

    if df["candidate_id"].duplicated().any():
        dupes = df.loc[df["candidate_id"].duplicated(keep=False), "candidate_id"].unique().tolist()
        raise ValueError(f"master table 中 candidate_id 重复，例如: {dupes[:10]}")


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
        "line_coverage": np.nan,
        "branch_coverage": np.nan,
        "mutation_score": np.nan,
        "boundary_coverage": np.nan,
        "exception_path_coverage": np.nan,
        "failure_log_count": 0,
        "coverage_status": None,
        "mutation_status": None,
        "test_style": None,
        "candidate_validity_status": None,
        "invalid_candidate_reason": None,
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


def pick_spotcheck_mixed_per_bucket(
    df: pd.DataFrame,
    k: int,
    score_col: str,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    df_sorted = df.sort_values(
        [score_col, "candidate_id"],
        ascending=[not higher_is_better, True],
        kind="stable",
    )

    parts = []
    for _, group in df_sorted.groupby(["source", "difficulty_bucket"], observed=True):
        if len(group) <= k:
            parts.append(group)
        elif k >= 3:
            parts.append(pd.concat([group.head(k - 1), group.iloc[[-1]]]))
        else:
            parts.append(group.head(k))

    if not parts:
        return pd.DataFrame(columns=df.columns)

    return bucket_sort(pd.concat(parts, ignore_index=True))


def select_positive_anchor_pool(
    hard_df: pd.DataFrame,
    relaxed_df: pd.DataFrame,
    k: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    先从 hard positive 里取，不足时再从 relaxed fallback 里补。
    """
    if hard_df.empty and relaxed_df.empty:
        return pd.DataFrame(), []

    hard_df = hard_df.sort_values(
        ["positive_anchor_score", "mutation_score", "branch_coverage", "line_coverage", "candidate_id"],
        ascending=[False, False, False, False, True],
        kind="stable",
    )
    relaxed_df = relaxed_df.sort_values(
        ["positive_anchor_score", "mutation_score", "branch_coverage", "line_coverage", "candidate_id"],
        ascending=[False, False, False, False, True],
        kind="stable",
    )

    selected_parts: List[pd.DataFrame] = []
    bucket_report: List[Dict[str, Any]] = []

    all_buckets = []
    for src in SOURCE_ORDER:
        for diff in DIFFICULTY_ORDER:
            all_buckets.append((src, diff))

    for src, diff in all_buckets:
        h = hard_df[(hard_df["source"].astype(str) == src) & (hard_df["difficulty_bucket"].astype(str) == diff)].copy()
        r = relaxed_df[(relaxed_df["source"].astype(str) == src) & (relaxed_df["difficulty_bucket"].astype(str) == diff)].copy()

        chosen_h = h.head(k).copy()
        chosen_ids = set(chosen_h["candidate_id"].astype(str).tolist())

        remaining = max(0, k - len(chosen_h))
        if remaining > 0:
            r = r[~r["candidate_id"].astype(str).isin(chosen_ids)].copy()
            chosen_r = r.head(remaining).copy()
        else:
            chosen_r = pd.DataFrame(columns=r.columns)

        if len(chosen_h) > 0:
            chosen_h["positive_anchor_tier"] = "strict_hard"
        if len(chosen_r) > 0:
            chosen_r["positive_anchor_tier"] = "relaxed_fallback"

        chosen = pd.concat([chosen_h, chosen_r], ignore_index=True)
        if len(chosen) > 0:
            selected_parts.append(chosen)

        bucket_report.append(
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

    if not selected_parts:
        return pd.DataFrame(), bucket_report

    selected = bucket_sort(pd.concat(selected_parts, ignore_index=True))
    return selected, bucket_report


def select_topk_per_bucket(
    df: pd.DataFrame,
    k: int,
    sort_cols: List[str],
    ascending: List[bool],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    if df.empty:
        return df.copy(), []

    df_sorted = df.sort_values(sort_cols, ascending=ascending, kind="stable")
    grouped = df_sorted.groupby(["source", "difficulty_bucket"], observed=True)

    take = grouped.head(k).copy()
    bucket_report = [
        {
            "source": str(src),
            "difficulty_bucket": str(diff),
            "available": int(len(group)),
            "selected": int(min(k, len(group))),
            "target": int(k),
            "shortfall": int(max(0, k - len(group))),
        }
        for (src, diff), group in grouped
    ]

    return bucket_sort(take), bucket_report


# =========================================================
# main
# =========================================================
def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"master table not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    validate_input(df)
    df = ensure_default_columns(df)

    current_run_id = infer_current_run_id(df)
    safe_run = str(current_run_id).replace("/", "_").replace(" ", "_")

    # 保持排序稳定
    if "source" in df.columns:
        df["source"] = pd.Categorical(df["source"], categories=SOURCE_ORDER, ordered=True)
    if "difficulty_bucket" in df.columns:
        df["difficulty_bucket"] = pd.Categorical(df["difficulty_bucket"], categories=DIFFICULTY_ORDER, ordered=True)

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
        "failure_log_count",
    ]
    for col in numeric_cols:
        df[col] = safe_numeric(df[col])

    # -------------------------
    # 1) usable pools
    # -------------------------
    usable_exec_mask = (
        (df["syntax_pass"] == 1)
        & (df["execution_pass"] == 1)
        & (df["invalid_candidate_flag"] == 0)
        & (df["redefines_target_flag"] == 0)
    )
    usable_exec_pool = bucket_sort(df[usable_exec_mask].copy())

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
    usable_eval_pool = bucket_sort(df[usable_eval_mask].copy())

    # 通用逻辑信号计数
    neg_logic_signal_cols = [
        "has_mutation_survival_major",
        "has_exception_path_gap",
        "has_boundary_case_gap",
        "has_branch_distinguishing_gap",
        "has_assertion_weak_or_missing",
        "has_execution_assertion_mismatch",
        "has_execution_call_contract_error",
    ]
    df["neg_logic_signal_count"] = sum(safe_bool_series(df[c]) for c in neg_logic_signal_cols)

    pos_soft_gap_cols = [
        "has_exception_path_gap",
        "has_boundary_case_gap",
        "has_branch_distinguishing_gap",
    ]
    df["pos_soft_gap_count"] = sum(safe_bool_series(df[c]) for c in pos_soft_gap_cols)

    # -------------------------
    # 2) positive anchors
    # -------------------------
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

    hard_positive_mask = (
        usable_eval_mask
        & pos_critical_zero_mask
        & (df["mutation_score"] >= STRICT_POS_MIN_MUTATION)
        & (df["line_coverage"] >= STRICT_POS_MIN_LINE_COV)
        & (df["branch_coverage"] >= STRICT_POS_MIN_BRANCH_COV)
        & (df["num_asserts"] >= STRICT_POS_MIN_NUM_ASSERTS)
        & (df["pos_soft_gap_count"] == 0)
    )

    relaxed_positive_mask = (
        usable_eval_mask
        & pos_critical_zero_mask
        & (df["mutation_score"] >= RELAXED_POS_MIN_MUTATION)
        & (df["line_coverage"] >= RELAXED_POS_MIN_LINE_COV)
        & (df["branch_coverage"] >= RELAXED_POS_MIN_BRANCH_COV)
        & (df["num_asserts"] >= RELAXED_POS_MIN_NUM_ASSERTS)
        & (df["pos_soft_gap_count"] <= RELAXED_POS_MAX_SOFT_GAPS)
    )

    hard_positive_candidates = df[hard_positive_mask].copy()
    relaxed_positive_candidates = df[relaxed_positive_mask].copy()

    # positive anchor score：保留 mutation 为核心，但不再纯硬阈值
    for sub in [hard_positive_candidates, relaxed_positive_candidates]:
        if not sub.empty:
            sub["positive_anchor_score"] = (
                0.45 * zscore(sub["mutation_score"])
                + 0.20 * zscore(sub["branch_coverage"])
                + 0.15 * zscore(sub["line_coverage"])
                + 0.10 * zscore(sub["boundary_coverage"].fillna(0))
                + 0.10 * zscore(sub["exception_path_coverage"].fillna(0))
                + 0.10 * zscore(sub["assert_density"].fillna(0))
                + 0.05 * zscore(sub["num_asserts"].fillna(0))
                - 0.20 * zscore(sub["failure_log_count"].fillna(0))
                - 0.25 * zscore(sub["pos_soft_gap_count"].fillna(0))
            )
        else:
            sub["positive_anchor_score"] = pd.Series(dtype=float)

    positive_anchor_pool, positive_bucket_report = select_positive_anchor_pool(
        hard_positive_candidates,
        relaxed_positive_candidates,
        k=POS_PER_BUCKET,
    )

    # -------------------------
    # 3) natural mechanistic negatives
    # -------------------------
    natural_negative_mask = (
        usable_eval_mask
        & (df["invalid_candidate_flag"] == 0)
        & (df["redefines_target_flag"] == 0)
        & (df["mutation_unavailable"] == 0)
        & (df["neg_logic_signal_count"] >= 1)
    )
    natural_negative_candidates = df[natural_negative_mask].copy()

    # 避免正负池明显重叠
    positive_ids = set(positive_anchor_pool["candidate_id"].astype(str).tolist()) if not positive_anchor_pool.empty else set()
    if positive_ids:
        natural_negative_candidates = natural_negative_candidates[
            ~natural_negative_candidates["candidate_id"].astype(str).isin(positive_ids)
        ].copy()

    # negative score：核心是逻辑失真强，而不是语法/执行坏
    if not natural_negative_candidates.empty:
        natural_negative_candidates["natural_negative_score"] = (
            1.20 * safe_bool_series(natural_negative_candidates["has_mutation_survival_major"])
            + 0.90 * safe_bool_series(natural_negative_candidates["has_exception_path_gap"])
            + 0.90 * safe_bool_series(natural_negative_candidates["has_boundary_case_gap"])
            + 0.90 * safe_bool_series(natural_negative_candidates["has_branch_distinguishing_gap"])
            + 0.90 * safe_bool_series(natural_negative_candidates["has_assertion_weak_or_missing"])
            + 0.90 * safe_bool_series(natural_negative_candidates["has_execution_assertion_mismatch"])
            + 0.90 * safe_bool_series(natural_negative_candidates["has_execution_call_contract_error"])
            + 0.35 * zscore(natural_negative_candidates["failure_log_count"].fillna(0))
            - 0.80 * zscore(natural_negative_candidates["mutation_score"].fillna(0))
            - 0.20 * zscore(natural_negative_candidates["branch_coverage"].fillna(0))
            - 0.15 * zscore(natural_negative_candidates["boundary_coverage"].fillna(0))
            - 0.15 * zscore(natural_negative_candidates["exception_path_coverage"].fillna(0))
        )
        natural_negative_candidates["neg_source"] = "natural_mechanistic_negative"
    else:
        natural_negative_candidates["natural_negative_score"] = pd.Series(dtype=float)
        natural_negative_candidates["neg_source"] = pd.Series(dtype=object)

    natural_negative_pool, negative_bucket_report = select_topk_per_bucket(
        natural_negative_candidates,
        k=NEG_PER_BUCKET,
        sort_cols=[
            "natural_negative_score",
            "neg_logic_signal_count",
            "mutation_score",
            "failure_log_count",
            "candidate_id",
        ],
        ascending=[False, False, True, False, True],
    )

    # -------------------------
    # 4) spotcheck outputs
    # -------------------------
    pos_spotcheck = pick_spotcheck_mixed_per_bucket(
        positive_anchor_pool,
        SPOTCHECK_PER_BUCKET,
        "positive_anchor_score",
        higher_is_better=True,
    )

    neg_spotcheck = pick_spotcheck_mixed_per_bucket(
        natural_negative_pool,
        SPOTCHECK_PER_BUCKET,
        "natural_negative_score",
        higher_is_better=True,
    )

    selected_ids = set(positive_anchor_pool["candidate_id"].astype(str)) | set(natural_negative_pool["candidate_id"].astype(str))
    boundary_candidates = usable_eval_pool[~usable_eval_pool["candidate_id"].astype(str).isin(selected_ids)].copy()

    if not boundary_candidates.empty:
        boundary_candidates["boundary_probe_score"] = (
            zscore(boundary_candidates["mutation_score"].fillna(0))
            + zscore(boundary_candidates["branch_coverage"].fillna(0))
            + zscore(boundary_candidates["line_coverage"].fillna(0))
            + zscore(boundary_candidates["assert_density"].fillna(0))
            - zscore(boundary_candidates["failure_log_count"].fillna(0))
        )
        boundary_spotcheck = bucket_sort(
            boundary_candidates.sort_values(
                ["boundary_probe_score", "candidate_id"],
                ascending=[False, True],
                kind="stable",
            ).head(BOUNDARY_SPOTCHECK_N).copy()
        )
    else:
        boundary_spotcheck = pd.DataFrame(columns=df.columns)

    # -------------------------
    # 5) outputs
    # -------------------------
    out_usable_exec = OUT_DIR / f"a2usableexecpool{saferun}.csv"
    out_usable_eval = OUT_DIR / f"a2usableevalpool{saferun}.csv"
    out_positive = OUT_DIR / f"a2positiveanchorpool{saferun}.csv"
    out_negative = OUT_DIR / f"a2naturalmechanisticnegativepool{saferun}.csv"
    out_pos_spot = OUT_DIR / f"a2spotcheckpositive{saferun}.csv"
    out_neg_spot = OUT_DIR / f"a2spotchecknegative{saferun}.csv"
    out_boundary_spot = OUT_DIR / f"a2spotcheckboundary{saferun}.csv"
    out_report = OUT_DIR / f"a2seedpoolreport{saferun}.json"

    usable_exec_pool.to_csv(out_usable_exec, index=False, encoding="utf-8")
    usable_eval_pool.to_csv(out_usable_eval, index=False, encoding="utf-8")
    positive_anchor_pool.to_csv(out_positive, index=False, encoding="utf-8")
    natural_negative_pool.to_csv(out_negative, index=False, encoding="utf-8")
    pos_spotcheck.to_csv(out_pos_spot, index=False, encoding="utf-8")
    neg_spotcheck.to_csv(out_neg_spot, index=False, encoding="utf-8")
    boundary_spotcheck.to_csv(out_boundary_spot, index=False, encoding="utf-8")

    report = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "input_csv": str(IN_CSV),
        "current_run_id": current_run_id,
        "positive_rule": {
            "strict_thresholds": {
                "mutation_score_min": STRICT_POS_MIN_MUTATION,
                "line_coverage_min": STRICT_POS_MIN_LINE_COV,
                "branch_coverage_min": STRICT_POS_MIN_BRANCH_COV,
                "num_asserts_min": STRICT_POS_MIN_NUM_ASSERTS,
                "require_soft_gap_count_eq_0": True,
            },
            "relaxed_fallback": {
                "mutation_score_min": RELAXED_POS_MIN_MUTATION,
                "line_coverage_min": RELAXED_POS_MIN_LINE_COV,
                "branch_coverage_min": RELAXED_POS_MIN_BRANCH_COV,
                "num_asserts_min": RELAXED_POS_MIN_NUM_ASSERTS,
                "max_soft_gap_count": RELAXED_POS_MAX_SOFT_GAPS,
            },
        },
        "negative_rule": {
            "base_requirements": [
                "usable_eval",
                "invalid_candidate_flag == 0",
                "redefines_target_flag == 0",
                "mutation_unavailable == 0",
                "neg_logic_signal_count >= 1",
            ],
            "neg_logic_signal_cols": neg_logic_signal_cols,
        },
        "recommended_quota": {
            "positive_per_bucket": POS_PER_BUCKET,
            "negative_per_bucket": NEG_PER_BUCKET,
            "spotcheck_per_bucket_positive": SPOTCHECK_PER_BUCKET,
            "spotcheck_per_bucket_negative": SPOTCHECK_PER_BUCKET,
            "spotcheck_boundary_global": BOUNDARY_SPOTCHECK_N,
        },
        "output_paths": {
            "usable_exec_pool": str(out_usable_exec),
            "usable_eval_pool": str(out_usable_eval),
            "positive_anchor_pool": str(out_positive),
            "natural_mechanistic_negative_pool": str(out_negative),
            "spotcheck_positive": str(out_pos_spot),
            "spotcheck_negative": str(out_neg_spot),
            "spotcheck_boundary": str(out_boundary_spot),
            "report": str(out_report),
        },
        "counts": {
            "total_rows": int(len(df)),
            "usable_exec_pool": int(len(usable_exec_pool)),
            "usable_eval_pool": int(len(usable_eval_pool)),
            "hard_positive_candidates": int(len(hard_positive_candidates)),
            "relaxed_positive_candidates": int(len(relaxed_positive_candidates)),
            "positive_anchor_pool": int(len(positive_anchor_pool)),
            "natural_mechanistic_negative_candidates": int(len(natural_negative_candidates)),
            "natural_mechanistic_negative_pool": int(len(natural_negative_pool)),
            "spotcheck_positive": int(len(pos_spotcheck)),
            "spotcheck_negative": int(len(neg_spotcheck)),
            "spotcheck_boundary": int(len(boundary_spotcheck)),
        },
        "distribution": {
            "positive_anchor_source_counts": safe_value_counts(positive_anchor_pool.get("source")),
            "positive_anchor_difficulty_counts": safe_value_counts(positive_anchor_pool.get("difficulty_bucket")),
            "negative_source_counts": safe_value_counts(natural_negative_pool.get("source")),
            "negative_difficulty_counts": safe_value_counts(natural_negative_pool.get("difficulty_bucket")),
            "positive_anchor_tier_counts": safe_value_counts(positive_anchor_pool.get("positive_anchor_tier")),
        },
        "bucket_report": {
            "positive_anchor_pool": positive_bucket_report,
            "natural_mechanistic_negative_pool": negative_bucket_report,
        },
    }

    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_usable_exec}")
    print(f"[ok] wrote {out_usable_eval}")
    print(f"[ok] wrote {out_positive}")
    print(f"[ok] wrote {out_negative}")
    print(f"[ok] wrote {out_pos_spot}")
    print(f"[ok] wrote {out_neg_spot}")
    print(f"[ok] wrote {out_boundary_spot}")
    print(f"[ok] wrote {out_report}")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()