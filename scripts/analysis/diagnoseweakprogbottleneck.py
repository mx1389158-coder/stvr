from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("BOTTLENECK_TAG", "a17bpoolv1")

a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
a3_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3"
a4_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a4"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "b1_diag_bottleneck"
OUT_DIR.mkdir(parents=True, exist_ok=True)

a3_WEAK = Path(os.environ.get("a3_WEAK_CSV", str(a3_DIR / f"a3positiveweakqualitypool{TAG}.csv"))).resolve()
a2_PROGNEG = Path(os.environ.get("a2_PROGNEG_CSV", str(a2_DIR / "a2prognegautometricsmastertable.csv"))).resolve()
a4_PAIR_TABLE = Path(os.environ.get("a4_PAIR_TABLE_CSV", str(a4_DIR / f"a4pairsurfacetable{TAG}.csv"))).resolve()

SCRIPT_VERSION = "diagnose_weak_prog_bottleneck_v1"


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def safe_int(x) -> int:
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass
    return int(x)


def main():
    for p, name in [
        (a3_WEAK, "a3_WEAK"),
        (a2_PROGNEG, "a2_PROGNEG"),
        (a4_PAIR_TABLE, "a4_PAIR_TABLE"),
    ]:
        require_file(p, name)

    weak = pd.read_csv(a3_WEAK)
    prog = pd.read_csv(a2_PROGNEG)
    a4 = pd.read_csv(a4_PAIR_TABLE)

    for df in [weak, prog]:
        for col in ["candidate_id", "task_id", "source", "difficulty_bucket"]:
            if col not in df.columns:
                raise ValueError(f"missing required column {col}")

    weak["candidate_id"] = weak["candidate_id"].astype(str)
    weak["task_id"] = weak["task_id"].astype(str)
    weak["source"] = weak["source"].astype(str)
    weak["difficulty_bucket"] = weak["difficulty_bucket"].astype(str)

    prog["candidate_id"] = prog["candidate_id"].astype(str)
    prog["task_id"] = prog["task_id"].astype(str)
    prog["source"] = prog["source"].astype(str)
    prog["difficulty_bucket"] = prog["difficulty_bucket"].astype(str)

    a4["candidate_id_pos"] = a4["candidate_id_pos"].astype(str)
    a4["candidate_id_neg"] = a4["candidate_id_neg"].astype(str)
    a4["task_id"] = a4["task_id"].astype(str)
    a4["pos_tier"] = a4["pos_tier"].astype(str)
    a4["neg_source"] = a4["neg_source"].astype(str)
    a4["surface_hard_fail_count"] = pd.to_numeric(a4["surface_hard_fail_count"], errors="coerce").fillna(999)
    a4["surface_mild_fail_count"] = pd.to_numeric(a4["surface_mild_fail_count"], errors="coerce").fillna(999)

    # -------- task-level weak counts --------
    weak_task = (
        weak.groupby(["task_id", "source", "difficulty_bucket"], dropna=False)
        .size()
        .reset_index(name="weak_pos_n")
    )

    prog_task = (
        prog.groupby(["task_id", "source", "difficulty_bucket"], dropna=False)
        .size()
        .reset_index(name="progneg_n")
    )

    task_df = pd.merge(
        weak_task,
        prog_task,
        on=["task_id", "source", "difficulty_bucket"],
        how="outer",
    )
    task_df["weak_pos_n"] = task_df["weak_pos_n"].fillna(0).astype(int)
    task_df["progneg_n"] = task_df["progneg_n"].fillna(0).astype(int)
    task_df["raw_cross_pairs"] = task_df["weak_pos_n"] * task_df["progneg_n"]

    # -------- a4 pair-table view for weak + progneg --------
    sub = a4[
        (a4["pos_tier"] == "weak_quality")
        & (a4["neg_source"] == "programmatic_negative")
    ].copy()

    # pull source/difficulty from positive side if present
    src_col = "source_pos" if "source_pos" in sub.columns else None
    diff_col = "difficulty_bucket_pos" if "difficulty_bucket_pos" in sub.columns else None

    group_cols = ["task_id"]
    if src_col:
        group_cols.append(src_col)
    if diff_col:
        group_cols.append(diff_col)

    rows: List[Dict] = []
    if len(sub) > 0:
        for keys, g in sub.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            rec: Dict = {"task_id": str(keys[0])}
            if src_col:
                rec["source"] = str(keys[1])
            if diff_col:
                rec["difficulty_bucket"] = str(keys[-1])

            rec["a4_pair_rows_total"] = int(len(g))
            rec["hard0_rows"] = int((g["surface_hard_fail_count"] == 0).sum())
            rec["mild_le_1_rows"] = int(((g["surface_hard_fail_count"] == 0) & (g["surface_mild_fail_count"] <= 1)).sum())
            rec["mild_le_2_rows"] = int(((g["surface_hard_fail_count"] == 0) & (g["surface_mild_fail_count"] <= 2)).sum())
            rec["mild_le_3_rows"] = int(((g["surface_hard_fail_count"] == 0) & (g["surface_mild_fail_count"] <= 3)).sum())
            rows.append(rec)

    a4_task = pd.DataFrame(rows)
    if a4_task.empty:
        a4_task = pd.DataFrame(columns=[
            "task_id", "source", "difficulty_bucket",
            "a4_pair_rows_total", "hard0_rows",
            "mild_le_1_rows", "mild_le_2_rows", "mild_le_3_rows",
        ])

    merged = pd.merge(
        task_df,
        a4_task,
        on=["task_id", "source", "difficulty_bucket"],
        how="left",
    )
    fill_zero_cols = [
        "a4_pair_rows_total", "hard0_rows",
        "mild_le_1_rows", "mild_le_2_rows", "mild_le_3_rows",
    ]
    for c in fill_zero_cols:
        if c not in merged.columns:
            merged[c] = 0
        merged[c] = merged[c].fillna(0).astype(int)

    def classify(row):
        weak_n = safe_int(row["weak_pos_n"])
        prog_n = safe_int(row["progneg_n"])
        raw_pairs = safe_int(row["raw_cross_pairs"])
        hard0 = safe_int(row["hard0_rows"])
        mild1 = safe_int(row["mild_le_1_rows"])
        mild3 = safe_int(row["mild_le_3_rows"])

        if weak_n == 0 and prog_n == 0:
            return "neither_side_present"
        if weak_n == 0:
            return "no_weak_positive"
        if prog_n == 0:
            return "no_programmatic_negative"
        if raw_pairs == 0:
            return "no_same_task_overlap"
        if hard0 == 0:
            return "blocked_before_or_at_hard_surface"
        if mild1 == 0 and mild3 > 0:
            return "blocked_by_mild_surface_only"
        if mild1 > 0:
            return "survives_current_surface_filter"
        return "other"

    merged["bottleneck_class"] = merged.apply(classify, axis=1)

    # bucket summary
    bucket = (
        merged.groupby(["source", "difficulty_bucket"], dropna=False)[
            ["weak_pos_n", "progneg_n", "raw_cross_pairs", "hard0_rows", "mild_le_1_rows", "mild_le_2_rows", "mild_le_3_rows"]
        ]
        .sum()
        .reset_index()
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "tag": TAG,
        "inputs": {
            "a3_weak": str(a3_WEAK),
            "a2_progneg": str(a2_PROGNEG),
            "a4_pair_table": str(a4_PAIR_TABLE),
        },
        "overall_counts": {
            "weak_total_rows": int(len(weak)),
            "weak_unique_tasks": int(weak["task_id"].nunique()),
            "progneg_total_rows": int(len(prog)),
            "progneg_unique_tasks": int(prog["task_id"].nunique()),
            "tasks_with_weak_and_prog_overlap": int(((merged["weak_pos_n"] > 0) & (merged["progneg_n"] > 0)).sum()),
            "raw_cross_pairs_total": int(merged["raw_cross_pairs"].sum()),
            "a4_pair_rows_total": int(merged["a4_pair_rows_total"].sum()),
            "hard0_rows_total": int(merged["hard0_rows"].sum()),
            "mild_le_1_rows_total": int(merged["mild_le_1_rows"].sum()),
            "mild_le_2_rows_total": int(merged["mild_le_2_rows"].sum()),
            "mild_le_3_rows_total": int(merged["mild_le_3_rows"].sum()),
        },
        "bottleneck_class_counts": {
            str(k): int(v) for k, v in merged["bottleneck_class"].value_counts(dropna=False).to_dict().items()
        },
    }

    out_task = OUT_DIR / f"weakprogbottlenecktasktable{TAG}.csv"
    out_bucket = OUT_DIR / f"weakprogbottleneckbucketsummary{TAG}.csv"
    out_summary = OUT_DIR / f"weakprogbottlenecksummary{TAG}.json"

    merged = merged.sort_values(
        ["raw_cross_pairs", "mild_le_1_rows", "task_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)

    bucket = bucket.sort_values(
        ["source", "difficulty_bucket"],
        kind="stable",
    ).reset_index(drop=True)

    merged.to_csv(out_task, index=False, encoding="utf-8")
    bucket.to_csv(out_bucket, index=False, encoding="utf-8")
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_task}")
    print(f"[ok] wrote {out_bucket}")
    print(f"[ok] wrote {out_summary}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
