from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("TAG", "a17bpoolv1")

a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
a3M_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3_margin_v3"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2_strength_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIGH_V3_CSV = a3M_DIR / f"a3positivehighqualitypoolmarginv3{TAG}.csv"
LOW_V3_CSV = a3M_DIR / f"a3positiveweakqualitypoolmarginv3{TAG}.csv"
PROGNEG_CSV = a2_DIR / "a2prognegautometricsmastertable.csv"

SCRIPT_VERSION = "build_a2_progneg_strength_split_margin_v3_v1"


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def main():
    for p, name in [
        (HIGH_V3_CSV, "HIGH_V3_CSV"),
        (LOW_V3_CSV, "LOW_V3_CSV"),
        (PROGNEG_CSV, "PROGNEG_CSV"),
    ]:
        require_file(p, name)

    high_v3 = pd.read_csv(HIGH_V3_CSV)
    low_v3 = pd.read_csv(LOW_V3_CSV)
    prog = pd.read_csv(PROGNEG_CSV)

    for df in [high_v3, low_v3, prog]:
        if "task_id" in df.columns:
            df["task_id"] = df["task_id"].astype(str)
        if "candidate_id" in df.columns:
            df["candidate_id"] = df["candidate_id"].astype(str)

    target_tasks = sorted(set(high_v3["task_id"].tolist()) | set(low_v3["task_id"].tolist()))

    for c in ["syntax_pass", "execution_pass", "invalid_candidate_flag", "redefines_target_flag", "mutation_unavailable"]:
        if c not in prog.columns:
            prog[c] = 0
        prog[c] = pd.to_numeric(prog[c], errors="coerce").fillna(0)

    valid = prog[
        (prog["task_id"].isin(target_tasks))
        & (prog["syntax_pass"] == 1)
        & (prog["execution_pass"] == 1)
        & (prog["invalid_candidate_flag"] == 0)
        & (prog["redefines_target_flag"] == 0)
        & (prog["mutation_unavailable"] == 0)
    ].copy()

    logic_cols = [
        "has_mutation_survival_major",
        "has_exception_path_gap",
        "has_boundary_case_gap",
        "has_branch_distinguishing_gap",
        "has_assertion_weak_or_missing",
    ]
    for c in logic_cols:
        if c not in valid.columns:
            valid[c] = 0
        valid[c] = pd.to_numeric(valid[c], errors="coerce").fillna(0)

    if "mutation_score" not in valid.columns:
        valid["mutation_score"] = None
    valid["mutation_score"] = pd.to_numeric(valid["mutation_score"], errors="coerce")

    if "failure_log_count" not in valid.columns:
        valid["failure_log_count"] = 0
    valid["failure_log_count"] = pd.to_numeric(valid["failure_log_count"], errors="coerce").fillna(0)

    valid["logic_flag_count"] = valid[logic_cols].sum(axis=1)
    valid["low_mutation_bonus"] = (valid["mutation_score"].fillna(1.0) <= 0.85).astype(int)
    valid["failure_log_bonus"] = (valid["failure_log_count"] >= 2).astype(int)
    valid["diagnostic_score"] = valid["logic_flag_count"] + valid["low_mutation_bonus"] + valid["failure_log_bonus"]

    strong_parts: List[pd.DataFrame] = []
    ordinary_parts: List[pd.DataFrame] = []
    singleton_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for task_id, g in valid.groupby("task_id", sort=True):
        g = g.sort_values(
            ["diagnostic_score", "logic_flag_count", "failure_log_count", "mutation_score", "candidate_id"],
            ascending=[False, False, False, True, True],
            kind="stable",
        ).reset_index(drop=True)

        n = len(g)
        if n == 1:
            singleton_rows.append({
                "task_id": task_id,
                "candidate_id": str(g.iloc[0]["candidate_id"]),
                "diagnostic_score": float(g.iloc[0]["diagnostic_score"]),
            })
            report_rows.append({
                "task_id": task_id,
                "valid_progneg_n": int(n),
                "strong_n": 0,
                "ordinary_n": 0,
                "singleton_n": 1,
                "strong_score_mean": None,
                "ordinary_score_mean": None,
            })
            continue

        strong_n = math.ceil(n / 2)
        ordinary_n = n - strong_n

        strong = g.head(strong_n).copy()
        ordinary = g.tail(ordinary_n).copy()

        strong["progneg_strength_tier"] = "strong_progneg"
        ordinary["progneg_strength_tier"] = "ordinary_progneg"

        strong_parts.append(strong)
        ordinary_parts.append(ordinary)

        report_rows.append({
            "task_id": task_id,
            "valid_progneg_n": int(n),
            "strong_n": int(len(strong)),
            "ordinary_n": int(len(ordinary)),
            "singleton_n": 0,
            "strong_score_mean": float(strong["diagnostic_score"].mean()) if len(strong) else None,
            "ordinary_score_mean": float(ordinary["diagnostic_score"].mean()) if len(ordinary) else None,
        })

    strong_df = pd.concat(strong_parts, ignore_index=True) if strong_parts else pd.DataFrame()
    ordinary_df = pd.concat(ordinary_parts, ignore_index=True) if ordinary_parts else pd.DataFrame()
    report_df = pd.DataFrame(report_rows)
    singleton_df = pd.DataFrame(singleton_rows)

    all_csv = OUT_DIR / f"a2prognegvalidonmarginv3tasks{TAG}.csv"
    strong_csv = OUT_DIR / f"a2prognegstrongmarginv3{TAG}.csv"
    ordinary_csv = OUT_DIR / f"a2prognegordinarymarginv3{TAG}.csv"
    report_csv = OUT_DIR / f"a2prognegstrengthreportmarginv3{TAG}.csv"
    singleton_csv = OUT_DIR / f"a2prognegsingletontasksmarginv3{TAG}.csv"
    manifest = OUT_DIR / f"a2prognegstrengthmanifestmarginv3{TAG}.json"

    valid.to_csv(all_csv, index=False, encoding="utf-8")
    strong_df.to_csv(strong_csv, index=False, encoding="utf-8")
    ordinary_df.to_csv(ordinary_csv, index=False, encoding="utf-8")
    report_df.to_csv(report_csv, index=False, encoding="utf-8")
    singleton_df.to_csv(singleton_csv, index=False, encoding="utf-8")

    obj = {
        "script_version": SCRIPT_VERSION,
        "tag": TAG,
        "counts": {
            "target_task_union_n": int(len(target_tasks)),
            "valid_progneg_rows_on_target_tasks": int(len(valid)),
            "valid_progneg_unique_tasks_on_target_tasks": int(valid["task_id"].nunique()) if len(valid) else 0,
            "strong_progneg_rows": int(len(strong_df)),
            "ordinary_progneg_rows": int(len(ordinary_df)),
            "singleton_task_n": int(len(singleton_df)),
        },
        "outputs": {
            "all_csv": str(all_csv),
            "strong_csv": str(strong_csv),
            "ordinary_csv": str(ordinary_csv),
            "report_csv": str(report_csv),
            "singleton_csv": str(singleton_csv),
            "manifest": str(manifest),
        },
        "notes": [
            "This diagnostic fixes task coverage and overlap, and only varies programmatic-negative strength.",
            "Singleton tasks are excluded from strong-vs-ordinary contrast because they cannot populate both tiers.",
        ],
    }

    manifest.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {all_csv}")
    print(f"[ok] wrote {strong_csv}")
    print(f"[ok] wrote {ordinary_csv}")
    print(f"[ok] wrote {report_csv}")
    print(f"[ok] wrote {singleton_csv}")
    print(f"[ok] wrote {manifest}")
    print(json.dumps(obj["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
