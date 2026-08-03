# scripts/analysis/analyzea1groupdifferences.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

SCRIPT_VERSION = "analyze_a1_group_differences_v4"

PROJECT_ROOT = Path(os.environ["UTPLM_PROJECT_ROOT"]).resolve()
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
INP = OUT_DIR / "a1sampledcandidatesv1.csv"
OUT = OUT_DIR / "a1groupdifferencesummary.csv"
OUT_MANIFEST = OUT_DIR / "a1groupdifferencesummarymanifest.json"

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


def build_metric_summary_rows(
    df: pd.DataFrame,
    group_col: str,
    metric_cols: List[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for metric in metric_cols:
        metric_series = pd.to_numeric(df[metric], errors="coerce")
        tmp = pd.DataFrame({
            group_col: df[group_col],
            "__metric__": metric_series,
        })

        grouped = tmp.groupby(group_col, dropna=False, sort=False)["__metric__"]

        for group_name, series in grouped:
            valid = series.dropna()
            total_n = int(series.shape[0])
            valid_n = int(valid.shape[0])

            rows.append({
                "group": "NA" if pd.isna(group_name) else str(group_name),
                "metric": metric,
                "n_total": total_n,
                "n_non_missing": valid_n,
                "missing_rate": safe_stat(1 - (valid_n / total_n)) if total_n > 0 else None,
                "mean": safe_stat(valid.mean()) if valid_n > 0 else None,
                "median": safe_stat(valid.median()) if valid_n > 0 else None,
                "std": safe_stat(valid.std()) if valid_n > 1 else None,
                "min": safe_stat(valid.min()) if valid_n > 0 else None,
                "max": safe_stat(valid.max()) if valid_n > 0 else None,
            })

        # 全体统计
        valid_all = metric_series.dropna()
        total_n_all = int(metric_series.shape[0])
        valid_n_all = int(valid_all.shape[0])

        rows.append({
            "group": "ALL",
            "metric": metric,
            "n_total": total_n_all,
            "n_non_missing": valid_n_all,
            "missing_rate": safe_stat(1 - (valid_n_all / total_n_all)) if total_n_all > 0 else None,
            "mean": safe_stat(valid_all.mean()) if valid_n_all > 0 else None,
            "median": safe_stat(valid_all.median()) if valid_n_all > 0 else None,
            "std": safe_stat(valid_all.std()) if valid_n_all > 1 else None,
            "min": safe_stat(valid_all.min()) if valid_n_all > 0 else None,
            "max": safe_stat(valid_all.max()) if valid_n_all > 0 else None,
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

    metric_cols = [
        "mutation_score",
        "line_coverage",
        "branch_coverage",
        "DDR_or_BRC",
        "TBC",
        "boundary_coverage",
        "exception_path_coverage",
        "failure_log_count",
        "num_asserts",
        "assert_density",
        "num_exception_checks",
        "logical_nesting_depth",
    ]

    existing_metric_cols = [c for c in metric_cols if c in df.columns]

    available_metric_cols: List[str] = []
    omitted_all_missing_metric_cols: List[str] = []

    # 只做一遍 numeric coercion 判定是否全空
    for c in existing_metric_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().any():
            available_metric_cols.append(c)
        else:
            omitted_all_missing_metric_cols.append(c)

    rows = build_metric_summary_rows(
        df=df,
        group_col="a1_group",
        metric_cols=available_metric_cols,
    )

    out = pd.DataFrame(rows)

    if not out.empty:
        out["_group_order"] = out["group"].map(ordered_group_sort_key)
        out = (
            out.sort_values(["metric", "_group_order", "group"], kind="stable")
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
            for k, v in df["a1_group"].fillna("NA").value_counts(dropna=False).to_dict().items()
        },
        "available_metric_cols": available_metric_cols,
        "omitted_all_missing_metric_cols": omitted_all_missing_metric_cols,
        "notes": [
            "This is a pilot group-difference summary for a1_v1.",
            "a1_group is a pilot organizational label, not the final paper label.",
            "Metrics that exist but are entirely missing are omitted from the summary table.",
            "Current results should not be over-interpreted as formal a1 stratified evidence.",
            "Current smoke_v1 pilot may collapse into a skewed split rather than a full four-level stratification.",
        ],
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(out.head(30).to_string(index=False))
    print(f"[ok] wrote {OUT}")
    print(f"[ok] wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()