from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("TAG", "a17bpoolv1")
WEAK_PER_BUCKET = int(os.environ.get("a3_WEAK_PER_BUCKET", "4"))

a3_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3"
a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORED_CSV = a3_DIR / f"a3positivescoredtable{TAG}.csv"
HIGH_CSV = a3_DIR / f"a3positivehighqualitypool{TAG}.csv"
MEDIUM_CSV = a3_DIR / f"a3positivemediumqualitypool{TAG}.csv"
PROGNEG_CSV = a2_DIR / "a2prognegautometricsmastertable.csv"

SOURCE_ORDER = ["mbpp", "humaneval"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def safe_bool_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).gt(0).astype(np.int8)


def bucket_sort(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = [c for c in ["source", "difficulty_bucket", "task_id", "candidate_id"] if c in df.columns]
    return df.sort_values(sort_cols, kind="stable").reset_index(drop=True)


def main() -> None:
    for p, name in [
        (SCORED_CSV, "SCORED_CSV"),
        (HIGH_CSV, "HIGH_CSV"),
        (MEDIUM_CSV, "MEDIUM_CSV"),
        (PROGNEG_CSV, "PROGNEG_CSV"),
    ]:
        require_file(p, name)

    scored = pd.read_csv(SCORED_CSV)
    high = pd.read_csv(HIGH_CSV)
    medium = pd.read_csv(MEDIUM_CSV)
    progneg = pd.read_csv(PROGNEG_CSV)

    scored["candidate_id"] = scored["candidate_id"].astype(str)
    high["candidate_id"] = high["candidate_id"].astype(str)
    medium["candidate_id"] = medium["candidate_id"].astype(str)
    progneg["task_id"] = progneg["task_id"].astype(str)

    selected_ids = set(high["candidate_id"].tolist()) | set(medium["candidate_id"].tolist())

    for c in ["syntax_pass", "execution_pass", "invalid_candidate_flag", "redefines_target_flag", "mutation_unavailable"]:
        if c not in progneg.columns:
            progneg[c] = 0
        progneg[c] = pd.to_numeric(progneg[c], errors="coerce").fillna(0)

    progneg_valid = progneg[
        (progneg["syntax_pass"] == 1)
        & (progneg["execution_pass"] == 1)
        & (progneg["invalid_candidate_flag"] == 0)
        & (progneg["redefines_target_flag"] == 0)
        & (progneg["mutation_unavailable"] == 0)
    ].copy()

    covered_tasks = set(progneg_valid["task_id"].astype(str).tolist())

    pool = scored[~scored["candidate_id"].isin(selected_ids)].copy()

    if "positive_teaching_score" not in pool.columns:
        raise ValueError("positive_teaching_score missing in scored table")

    pool["task_id"] = pool["task_id"].astype(str)
    pool["source"] = pool["source"].astype(str)
    pool["difficulty_bucket"] = pool["difficulty_bucket"].astype(str)
    pool["has_progneg_overlap"] = pool["task_id"].isin(covered_tasks).astype(int)

    parts: List[pd.DataFrame] = []
    report_rows: List[Dict[str, Any]] = []

    for src in SOURCE_ORDER:
        for diff in DIFFICULTY_ORDER:
            g = pool[(pool["source"] == src) & (pool["difficulty_bucket"] == diff)].copy()

            g_overlap = g[g["has_progneg_overlap"] == 1].copy()
            g_fallback = g[g["has_progneg_overlap"] == 0].copy()

            g_overlap = g_overlap.sort_values(
                ["positive_teaching_score", "candidate_id"],
                ascending=[True, True],
                kind="stable",
            )
            g_fallback = g_fallback.sort_values(
                ["positive_teaching_score", "candidate_id"],
                ascending=[True, True],
                kind="stable",
            )

            take_overlap = g_overlap.head(WEAK_PER_BUCKET).copy()
            remain = max(0, WEAK_PER_BUCKET - len(take_overlap))
            taken_ids = set(take_overlap["candidate_id"].astype(str).tolist())
            take_fallback = g_fallback[~g_fallback["candidate_id"].astype(str).isin(taken_ids)].head(remain).copy()

            chosen = pd.concat([take_overlap, take_fallback], ignore_index=True)
            if not chosen.empty:
                chosen["positive_tier"] = "weak_quality_v2_overlap"
                chosen["positive_keep_rule"] = "weak_overlap_first"
                parts.append(chosen)

            report_rows.append(
                {
                    "source": src,
                    "difficulty_bucket": diff,
                    "candidate_pool_n": int(len(g)),
                    "overlap_candidate_n": int(len(g_overlap)),
                    "fallback_candidate_n": int(len(g_fallback)),
                    "selected_from_overlap": int(len(take_overlap)),
                    "selected_from_fallback": int(len(take_fallback)),
                    "selected_total": int(len(chosen)),
                    "target": int(WEAK_PER_BUCKET),
                    "shortfall": int(max(0, WEAK_PER_BUCKET - len(chosen))),
                }
            )

    out = bucket_sort(pd.concat(parts, ignore_index=True)) if parts else pd.DataFrame()
    out_csv = OUT_DIR / f"a3positiveweakqualitypoolv2overlap{TAG}.csv"
    report_csv = OUT_DIR / f"a3positiveweakqualitypoolv2overlapreport{TAG}.csv"
    manifest = OUT_DIR / f"a3positiveweakqualitypoolv2overlapmanifest{TAG}.json"

    out.to_csv(out_csv, index=False, encoding="utf-8")
    pd.DataFrame(report_rows).to_csv(report_csv, index=False, encoding="utf-8")

    obj = {
        "tag": TAG,
        "weak_per_bucket": WEAK_PER_BUCKET,
        "inputs": {
            "scored_csv": str(SCORED_CSV),
            "high_csv": str(HIGH_CSV),
            "medium_csv": str(MEDIUM_CSV),
            "progneg_csv": str(PROGNEG_CSV),
        },
        "counts": {
            "covered_progneg_tasks": int(len(covered_tasks)),
            "selected_weak_v2_rows": int(len(out)),
            "selected_weak_v2_unique_tasks": int(out["task_id"].nunique()) if len(out) else 0,
        },
        "outputs": {
            "weak_v2_csv": str(out_csv),
            "report_csv": str(report_csv),
            "manifest": str(manifest),
        },
        "notes": [
            "weak v2 prioritizes tasks already covered by valid programmatic negatives.",
            "fallback to ordinary weakest candidates only if overlap-first selection is insufficient.",
        ],
    }
    manifest.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {report_csv}")
    print(f"[ok] wrote {manifest}")
    print(json.dumps(obj["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
