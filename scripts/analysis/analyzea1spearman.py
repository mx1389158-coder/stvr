# scripts/analysis/analyzea1spearman.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_VERSION = "analyze_a1_spearman_v5"

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"

MASTER = OUT_DIR / "a1autometricsmastertable.csv"
R1 = OUT_DIR / "a1annotationpackrater1filled.csv"
R2 = OUT_DIR / "a1annotationpackrater2filled.csv"

OUT = OUT_DIR / "a1spearmanresults.csv"
OUT_MANIFEST = OUT_DIR / "a1spearmanresultsmanifest.json"


# =========================================================
# 工具函数
# =========================================================
def ensure_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def ensure_required_columns(df: pd.DataFrame, required: List[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{table_name}] missing required columns: {missing}")


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


def normalize_numeric_cols(df: pd.DataFrame, cols: List[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def normalize_applicability_col(series: pd.Series) -> pd.Series:
    """
    归一化为:
    - 1: applicable
    - 0: not applicable / N/A
    - NaN: unknown / empty
    """
    mapping = {
        "1": 1.0, "applicable": 1.0, "yes": 1.0, "true": 1.0,
        "0": 0.0, "n/a": 0.0, "na": 0.0, "not_applicable": 0.0,
        "not applicable": 0.0, "no": 0.0, "false": 0.0,
    }
    s = series.astype(str).str.strip().str.lower()
    return pd.to_numeric(s.map(mapping), errors="coerce")


def compute_official_a1_score_and_denominator(df: pd.DataFrame, suffix: str) -> Tuple[pd.Series, pd.Series]:
    base_dims = [
        f"assertion_effectiveness{suffix}",
        f"boundary_condition_checking{suffix}",
        f"branch_distinguishing_ability{suffix}",
        f"fault_revealing_potential{suffix}",
    ]
    eh_col = f"exception_path_handling{suffix}"
    app_col = f"exception_path_applicability{suffix}"

    for c in base_dims + [eh_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required rubric dimension column: {c}")

    base_sum = df[base_dims].fillna(0).sum(axis=1)
    base_count = df[base_dims].notna().sum(axis=1)

    eh_val = pd.to_numeric(df[eh_col], errors="coerce")
    if app_col in df.columns:
        app_val = normalize_applicability_col(df[app_col])
        eh_valid = app_val.where(app_val.notna(), eh_val.notna().astype(float))
    else:
        eh_valid = eh_val.notna().astype(float)

    numerator = base_sum + eh_val.where(eh_valid == 1, 0).fillna(0)
    denominator = base_count + (eh_valid == 1).astype(float)

    score = (10 * numerator / (2 * denominator)).where(denominator > 0)
    return score, denominator


def choose_metric_cols(df: pd.DataFrame, metric_cols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    existing_cols = [c for c in metric_cols if c in df.columns]

    if not existing_cols:
        return [], [c for c in metric_cols if c not in df.columns], []

    tmp_df = df[existing_cols].apply(pd.to_numeric, errors="coerce")
    valid_counts = tmp_df.notna().sum()
    nunique_counts = tmp_df.nunique(dropna=True)

    available = []
    omitted_all_missing = []
    omitted_constant = []

    for c in metric_cols:
        if c not in existing_cols:
            continue
        if valid_counts[c] == 0:
            omitted_all_missing.append(c)
        elif nunique_counts[c] <= 1:
            omitted_constant.append(c)
        else:
            available.append(c)

    return available, omitted_all_missing, omitted_constant


def compute_spearman_rows(
    df: pd.DataFrame,
    target_col: str,
    metric_cols: List[str],
    target_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    y = pd.to_numeric(df[target_col], errors="coerce")
    target_non_missing_n = int(y.notna().sum())

    if target_non_missing_n < 3:
        return rows

    for c in metric_cols:
        if c not in df.columns:
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & y.notna()
        overlap_n = int(mask.sum())

        if overlap_n >= 3:
            xv = x[mask]
            yv = y[mask]

            if xv.nunique() <= 1 or yv.nunique() <= 1:
                rho, p = np.nan, np.nan
            else:
                res = spearmanr(xv, yv)
                rho, p = res.correlation, res.pvalue

            rows.append({
                "target": target_name,
                "metric": c,
                "n_overlap": overlap_n,
                "target_non_missing_n": target_non_missing_n,
                "metric_non_missing_n": int(x.notna().sum()),
                "rho": safe_stat(rho),
                "p_value": safe_stat(p),
            })

    return rows


# =========================================================
# 主逻辑
# =========================================================
def main() -> None:
    ensure_file(MASTER, "MASTER")
    ensure_file(R1, "R1")
    ensure_file(R2, "R2")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER)
    r1 = pd.read_csv(R1)
    r2 = pd.read_csv(R2)

    ensure_required_columns(master, ["sample_id"], "MASTER")
    ensure_required_columns(r1, ["sample_id"], "R1")
    ensure_required_columns(r2, ["sample_id"], "R2")

    validate_unique(master, "sample_id", "MASTER")
    validate_unique(r1, "sample_id", "R1")
    validate_unique(r2, "sample_id", "R2")

    main_dim_cols = [
        "assertion_effectiveness",
        "boundary_condition_checking",
        "exception_path_handling",
        "branch_distinguishing_ability",
        "fault_revealing_potential",
    ]
    ensure_required_columns(r1, main_dim_cols, "R1")
    ensure_required_columns(r2, main_dim_cols, "R2")

    optional_cols = [
        "exception_path_applicability",
        "overall_total_0_10",
        "teaching_value",
    ]

    normalize_numeric_cols(r1, main_dim_cols + ["overall_total_0_10", "teaching_value"])
    normalize_numeric_cols(r2, main_dim_cols + ["overall_total_0_10", "teaching_value"])

    keep_r1 = ["sample_id"] + main_dim_cols + [c for c in optional_cols if c in r1.columns]
    keep_r2 = ["sample_id"] + main_dim_cols + [c for c in optional_cols if c in r2.columns]

    ann = r1[keep_r1].merge(
        r2[keep_r2],
        on="sample_id",
        how="inner",
        suffixes=("_r1", "_r2"),
    )

    ann["official_a1_score_r1"], ann["applicable_dims_r1"] = compute_official_a1_score_and_denominator(ann, "_r1")
    ann["official_a1_score_r2"], ann["applicable_dims_r2"] = compute_official_a1_score_and_denominator(ann, "_r2")
    ann["official_a1_score_mean"] = ann[["official_a1_score_r1", "official_a1_score_r2"]].mean(axis=1)
    ann["applicable_dims_mean"] = ann[["applicable_dims_r1", "applicable_dims_r2"]].mean(axis=1)

    if "overall_total_0_10_r1" in ann.columns and "overall_total_0_10_r2" in ann.columns:
        ann["holistic_total_mean"] = ann[["overall_total_0_10_r1", "overall_total_0_10_r2"]].mean(axis=1)
    else:
        ann["holistic_total_mean"] = np.nan

    df = master.merge(
        ann[["sample_id", "official_a1_score_mean", "holistic_total_mean", "applicable_dims_mean"]],
        on="sample_id",
        how="inner",
    )

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

    available_metric_cols, omitted_all_missing, omitted_constant = choose_metric_cols(df, metric_cols)

    rows = []
    rows.extend(compute_spearman_rows(df, "official_a1_score_mean", available_metric_cols, "official_a1_score_mean"))
    rows.extend(compute_spearman_rows(df, "holistic_total_mean", available_metric_cols, "holistic_total_mean"))

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["target", "rho"], ascending=[True, False], kind="stable").reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["target", "metric", "n_overlap", "target_non_missing_n", "metric_non_missing_n", "rho", "p_value"])

    out.to_csv(OUT, index=False, encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "input_paths": {
            "master": str(MASTER),
            "rater1": str(R1),
            "rater2": str(R2),
        },
        "output_path": str(OUT),
        "output_manifest_path": str(OUT_MANIFEST),
        "master_sample_count": int(len(master)),
        "rater1_sample_count": int(len(r1)),
        "rater2_sample_count": int(len(r2)),
        "merged_annotation_count": int(len(ann)),
        "merged_master_count": int(len(df)),
        "main_dimension_cols": main_dim_cols,
        "optional_cols_detected_r1": [c for c in optional_cols if c in r1.columns],
        "optional_cols_detected_r2": [c for c in optional_cols if c in r2.columns],
        "metric_cols_requested": metric_cols,
        "metric_cols_used": available_metric_cols,
        "omitted_all_missing_metric_cols": omitted_all_missing,
        "omitted_constant_metric_cols": omitted_constant,
        "targets_generated": sorted(out["target"].dropna().unique().tolist()) if not out.empty else [],
        "score_coverage": {
            "official_a1_score_mean_non_missing": int(df["official_a1_score_mean"].notna().sum()),
            "holistic_total_mean_non_missing": int(df["holistic_total_mean"].notna().sum()),
            "applicable_dims_mean_non_missing": int(df["applicable_dims_mean"].notna().sum()),
        },
        "notes": [
            "Official a1 score is computed from the five main rubric dimensions.",
            "Exception-Path Handling is excluded from the denominator when marked not applicable.",
            "overall_total_0_10 is treated as an auxiliary holistic score, not the official a1 main score.",
            "Metrics that are entirely missing or constant are omitted from the Spearman table.",
            "Constant metrics are omitted because rank correlation is undefined or uninformative under zero variance.",
            "This is a pilot Spearman analysis for a1_v1 and should not be over-interpreted as final construct-validation evidence.",
        ],
    }

    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(out.to_string(index=False))
    print(f"[ok] wrote {OUT}")
    print(f"[ok] wrote {OUT_MANIFEST}")


if __name__ == "__main__":
    main()