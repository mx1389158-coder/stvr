# scripts/analysis/analyzea1failureconsistency.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

SCRIPT_VERSION = "analyze_a1_failure_consistency_v4"

PROJECT_ROOT = Path(os.environ["UTPLM_PROJECT_ROOT"]).resolve()
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
INP = OUT_DIR / "a1sampledcandidatesv1.csv"
OUT = OUT_DIR / "a1failureconsistencysummary.csv"
OUT_MANIFEST = OUT_DIR / "a1failureconsistencysummarymanifest.json"

GROUP_ORDER = [
    "high_quality",
    "medium_quality",
    "low_quality",
    "mechanistic_negative",
    "ALL",
]
GROUP_ORDER_MAP = {g: i for i, g in enumerate(GROUP_ORDER)}


# =========================================================
# 工具函数
# =========================================================
def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in pilot pack: {missing}")


def validate_unique(df: pd.DataFrame, col: str, table_name: str) -> None:
    if col not in df.columns:
        raise ValueError(f"[{table_name}] missing required column: {col}")

    dup_mask = df[col].duplicated(keep=False)
    if dup_mask.any():
        dup_counts = df.loc[dup_mask, col].value_counts().head(20).to_dict()
        raise ValueError(f"[{table_name}] {col} is not unique. Top duplicates: {dup_counts}")


def safe_stat(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def ordered_group_sort_key(group_name: str) -> int:
    return GROUP_ORDER_MAP.get(group_name, len(GROUP_ORDER))


def normalize_group_series(series: pd.Series) -> pd.Series:
    return series.fillna("NA").astype(str).str.strip().replace("", "NA")


def build_flag_summary_rows(
    df: pd.DataFrame,
    group_col: str,
    flag_cols: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for flag in flag_cols:
        raw_val = pd.to_numeric(df[flag], errors="coerce")

        # 明确区分 missing 与 positive
        is_valid = raw_val.notna()
        is_pos = (raw_val > 0).fillna(False).astype(np.int8)

        tmp = pd.DataFrame({
            "group": df[group_col],
            "valid": is_valid,
            "pos": is_pos,
        })

        grouped = tmp.groupby("group", dropna=False, sort=False)
        positive_groups: List[str] = []

        for group_name, sub in grouped:
            group_name_norm = "NA" if pd.isna(group_name) else str(group_name)

            total_n = int(sub.shape[0])
            valid_n = int(sub["valid"].sum())
            pos_n = int(sub["pos"].sum())

            if pos_n > 0:
                positive_groups.append(group_name_norm)

            rows.append({
                "group": group_name_norm,
                "flag": flag,
                "n_total": total_n,
                "n_non_missing": valid_n,
                "missing_rate": safe_stat(1 - (valid_n / total_n)) if total_n > 0 else None,
                "count": pos_n,
                "rate": safe_stat(pos_n / valid_n) if valid_n > 0 else None,
                "is_flag_present_in_group": int(pos_n > 0),
            })

        # 全体统计
        total_n_all = int(tmp.shape[0])
        valid_n_all = int(tmp["valid"].sum())
        pos_n_all = int(tmp["pos"].sum())

        rows.append({
            "group": "ALL",
            "flag": flag,
            "n_total": total_n_all,
            "n_non_missing": valid_n_all,
            "missing_rate": safe_stat(1 - (valid_n_all / total_n_all)) if total_n_all > 0 else None,
            "count": pos_n_all,
            "rate": safe_stat(pos_n_all / valid_n_all) if valid_n_all > 0 else None,
            "is_flag_present_in_group": int(pos_n_all > 0),
            "positive_groups_count": len(positive_groups),
            "support_groups": "|".join(positive_groups),
        })

    return rows


# =========================================================
# 主逻辑
# =========================================================
def main() -> None:
    if not INP.exists():
        raise FileNotFoundError(f"Pilot pack not found: {INP}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INP)

    ensure_required_columns(df, ["sample_id", "candidate_id", "a1_group"])
    validate_unique(df, "sample_id", "a1_sampled_candidates_v1")
    validate_unique(df, "candidate_id", "a1_sampled_candidates_v1")

    df["a1_group"] = normalize_group_series(df["a1_group"])

    flag_cols = [
        "has_boundary_case_gap",
        "has_branch_distinguishing_gap",
        "has_exception_path_gap",
        "has_mutation_survival_major",
        "has_assertion_weak_or_missing",
    ]

    existing_flag_cols = [c for c in flag_cols if c in df.columns]

    available_flag_cols: List[str] = []
    omitted_all_missing_flag_cols: List[str] = []
    flags_with_any_positive: List[str] = []

    for c in existing_flag_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().any():
            available_flag_cols.append(c)
            if (x > 0).any():
                flags_with_any_positive.append(c)
        else:
            omitted_all_missing_flag_cols.append(c)

    rows = build_flag_summary_rows(
        df=df,
        group_col="a1_group",
        flag_cols=available_flag_cols,
    )

    out = pd.DataFrame(rows)

    if not out.empty:
        out["_group_order"] = out["group"].map(ordered_group_sort_key)
        out = (
            out.sort_values(["flag", "_group_order", "group"], kind="stable")
            .drop(columns=["_group_order"])
            .reset_index(drop=True)
        )

    out.to_csv(OUT, index=False, encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "input_path": str(INP),
        "output_path": str(OUT),
        "output_manifest_path": str(OUT_MANIFEST),
        "row_count": int(len(out)),
        "input_sample_count": int(len(df)),
        "group_order": GROUP_ORDER,
        "group_counts": {
            str(k): int(v)
            for k, v in df["a1_group"].value_counts(dropna=False).to_dict().items()
        },
        "available_flag_cols": available_flag_cols,
        "omitted_all_missing_flag_cols": omitted_all_missing_flag_cols,
        "flags_with_any_positive": flags_with_any_positive,
        "notes": [
            "This is a pilot failure-consistency summary for a1_v1.",
            "a1_group is a pilot organizational label, not the final paper label.",
            "Only flags with at least one non-missing value are summarized.",
            "Flags that are entirely zero are still kept, because zero prevalence is itself informative.",
            "Count and rate are computed over non-missing flag observations; missing values are excluded from prevalence calculation.",
            "Current results should not be over-interpreted as formal a1 stratified evidence.",
            "Current smoke_v1 pilot may collapse into a skewed split rather than a full four-level stratification.",
        ],
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(out.to_string(index=False))
    print(f"[ok] wrote {OUT}")
    print(f"[ok] wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()