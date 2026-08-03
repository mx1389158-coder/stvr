from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("TAG", "a17bpoolv1")
HIGH_PER_BUCKET = int(os.environ.get("a3_MARGIN_V3_HIGH_PER_BUCKET", "3"))
LOW_PER_BUCKET = int(os.environ.get("a3_MARGIN_V3_LOW_PER_BUCKET", "3"))

a3_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3"
a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3_margin_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCORED_CSV = a3_DIR / f"a3positivescoredtable{TAG}.csv"
PROGNEG_CSV = a2_DIR / "a2prognegautometricsmastertable.csv"

SOURCE_ORDER = ["mbpp", "humaneval"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def main():
    for p, name in [
        (SCORED_CSV, "SCORED_CSV"),
        (PROGNEG_CSV, "PROGNEG_CSV"),
    ]:
        require_file(p, name)

    scored = pd.read_csv(SCORED_CSV)
    progneg = pd.read_csv(PROGNEG_CSV)

    scored["candidate_id"] = scored["candidate_id"].astype(str)
    scored["task_id"] = scored["task_id"].astype(str)
    scored["source"] = scored["source"].astype(str)
    scored["difficulty_bucket"] = scored["difficulty_bucket"].astype(str)

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

    if "positive_teaching_score" not in scored.columns:
        raise ValueError("positive_teaching_score missing in scored table")

    # 只在已有 progneg 覆盖的任务上选，固定 overlap
    pool = scored[scored["task_id"].isin(covered_tasks)].copy()

    # 去掉当前 scored 里已经被标成 unselected? 不需要，只要 top/bottom 真能拉开即可
    # 但为了避免明显的中间层混入，可优先用 selected + unselected 整池中按 score 极端挑
    high_parts: List[pd.DataFrame] = []
    low_parts: List[pd.DataFrame] = []
    report_rows: List[Dict[str, Any]] = []

    for src in SOURCE_ORDER:
        for diff in DIFFICULTY_ORDER:
            g = pool[(pool["source"] == src) & (pool["difficulty_bucket"] == diff)].copy()

            g_sorted_low = g.sort_values(
                ["positive_teaching_score", "candidate_id"],
                ascending=[True, True],
                kind="stable",
            )
            g_sorted_high = g.sort_values(
                ["positive_teaching_score", "candidate_id"],
                ascending=[False, True],
                kind="stable",
            )

            high_take = g_sorted_high.head(HIGH_PER_BUCKET).copy()
            high_ids = set(high_take["candidate_id"].astype(str).tolist())

            low_take = g_sorted_low[~g_sorted_low["candidate_id"].astype(str).isin(high_ids)].head(LOW_PER_BUCKET).copy()

            if not high_take.empty:
                high_take["positive_tier"] = "high_quality_margin_v3"
                high_take["positive_keep_rule"] = "top_overlap_margin_v3"
                high_parts.append(high_take)

            if not low_take.empty:
                low_take["positive_tier"] = "weak_quality_margin_v3"
                low_take["positive_keep_rule"] = "bottom_overlap_margin_v3"
                low_parts.append(low_take)

            report_rows.append(
                {
                    "source": src,
                    "difficulty_bucket": diff,
                    "covered_pool_n": int(len(g)),
                    "high_selected": int(len(high_take)),
                    "low_selected": int(len(low_take)),
                    "high_target": int(HIGH_PER_BUCKET),
                    "low_target": int(LOW_PER_BUCKET),
                    "high_score_mean": float(high_take["positive_teaching_score"].mean()) if len(high_take) else None,
                    "low_score_mean": float(low_take["positive_teaching_score"].mean()) if len(low_take) else None,
                }
            )

    high_df = pd.concat(high_parts, ignore_index=True) if high_parts else pd.DataFrame()
    low_df = pd.concat(low_parts, ignore_index=True) if low_parts else pd.DataFrame()

    high_df = high_df.sort_values(["source", "difficulty_bucket", "task_id", "candidate_id"], kind="stable").reset_index(drop=True)
    low_df = low_df.sort_values(["source", "difficulty_bucket", "task_id", "candidate_id"], kind="stable").reset_index(drop=True)

    report_df = pd.DataFrame(report_rows)
    manifest = {
        "tag": TAG,
        "high_per_bucket": HIGH_PER_BUCKET,
        "low_per_bucket": LOW_PER_BUCKET,
        "inputs": {
            "scored_csv": str(SCORED_CSV),
            "progneg_csv": str(PROGNEG_CSV),
        },
        "counts": {
            "covered_progneg_tasks": int(len(covered_tasks)),
            "high_margin_v3_rows": int(len(high_df)),
            "high_margin_v3_unique_tasks": int(high_df["task_id"].nunique()) if len(high_df) else 0,
            "low_margin_v3_rows": int(len(low_df)),
            "low_margin_v3_unique_tasks": int(low_df["task_id"].nunique()) if len(low_df) else 0,
        },
        "outputs": {
            "high_csv": str(OUT_DIR / f"a3positivehighqualitypoolmarginv3{TAG}.csv"),
            "low_csv": str(OUT_DIR / f"a3positiveweakqualitypoolmarginv3{TAG}.csv"),
            "report_csv": str(OUT_DIR / f"a3positivemarginv3report{TAG}.csv"),
            "manifest": str(OUT_DIR / f"a3positivemarginv3manifest{TAG}.json"),
        },
        "notes": [
            "Margin-v3 fixes overlap by selecting only from tasks already covered by valid programmatic negatives.",
            "It increases high/low margin by taking stronger top and weaker bottom candidates within the covered pool.",
        ],
    }

    high_csv = Path(manifest["outputs"]["high_csv"])
    low_csv = Path(manifest["outputs"]["low_csv"])
    report_csv = Path(manifest["outputs"]["report_csv"])
    manifest_path = Path(manifest["outputs"]["manifest"])

    high_df.to_csv(high_csv, index=False, encoding="utf-8")
    low_df.to_csv(low_csv, index=False, encoding="utf-8")
    report_df.to_csv(report_csv, index=False, encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {high_csv}")
    print(f"[ok] wrote {low_csv}")
    print(f"[ok] wrote {report_csv}")
    print(f"[ok] wrote {manifest_path}")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
