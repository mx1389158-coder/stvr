from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("TAG", "a17bpoolv1")

a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
a3_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3"
a3M_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3_margin_v3"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2_targeted_margin_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIGH_V3_CSV = a3M_DIR / f"a3positivehighqualitypoolmarginv3{TAG}.csv"
LOW_V3_CSV = a3M_DIR / f"a3positiveweakqualitypoolmarginv3{TAG}.csv"
PROGNEG_CSV = a2_DIR / "a2prognegautometricsmastertable.csv"
a2_POS_ANCHOR_CSV = a2_DIR / f"a2positiveanchorpool{TAG}.csv"
a3_HIGH_CSV = a3_DIR / f"a3positivehighqualitypool{TAG}.csv"

SCRIPT_VERSION = "build_targeted_progneg_anchor_subset_margin_v3_v1"


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def main():
    for p, name in [
        (HIGH_V3_CSV, "HIGH_V3_CSV"),
        (LOW_V3_CSV, "LOW_V3_CSV"),
        (PROGNEG_CSV, "PROGNEG_CSV"),
        (a2_POS_ANCHOR_CSV, "a2_POS_ANCHOR_CSV"),
        (a3_HIGH_CSV, "a3_HIGH_CSV"),
    ]:
        require_file(p, name)

    high_v3 = pd.read_csv(HIGH_V3_CSV)
    low_v3 = pd.read_csv(LOW_V3_CSV)
    prog = pd.read_csv(PROGNEG_CSV)
    pos_anchor = pd.read_csv(a2_POS_ANCHOR_CSV)
    a3_high = pd.read_csv(a3_HIGH_CSV)

    for df in [high_v3, low_v3, prog, pos_anchor, a3_high]:
        if "task_id" in df.columns:
            df["task_id"] = df["task_id"].astype(str)

    for c in ["syntax_pass", "execution_pass", "invalid_candidate_flag", "redefines_target_flag", "mutation_unavailable"]:
        if c not in prog.columns:
            prog[c] = 0
        prog[c] = pd.to_numeric(prog[c], errors="coerce").fillna(0)

    prog_valid = prog[
        (prog["syntax_pass"] == 1)
        & (prog["execution_pass"] == 1)
        & (prog["invalid_candidate_flag"] == 0)
        & (prog["redefines_target_flag"] == 0)
        & (prog["mutation_unavailable"] == 0)
    ].copy()

    covered_tasks = set(prog_valid["task_id"].astype(str).tolist())
    target_tasks = sorted(set(high_v3["task_id"].astype(str).tolist()) | set(low_v3["task_id"].astype(str).tolist()))
    missing_tasks = [t for t in target_tasks if t not in covered_tasks]

    pos_anchor["task_id"] = pos_anchor["task_id"].astype(str)
    a3_high["task_id"] = a3_high["task_id"].astype(str)
    high_v3["task_id"] = high_v3["task_id"].astype(str)

    # 优先从 a2 positive anchor pool 里拿
    primary = pos_anchor[pos_anchor["task_id"].isin(missing_tasks)].copy()
    primary["anchor_source"] = "a2_positive_anchor_pool"

    remain_1 = sorted(set(missing_tasks) - set(primary["task_id"].tolist()))
    fallback_v3 = high_v3[high_v3["task_id"].isin(remain_1)].copy()
    fallback_v3["anchor_source"] = "a3_margin_v3_high_pool"

    remain_2 = sorted(set(remain_1) - set(fallback_v3["task_id"].tolist()))
    fallback_high = a3_high[a3_high["task_id"].isin(remain_2)].copy()
    fallback_high["anchor_source"] = "a3_high_quality_pool"

    subset = pd.concat([primary, fallback_v3, fallback_high], ignore_index=True)
    if "candidate_id" in subset.columns:
        subset["candidate_id"] = subset["candidate_id"].astype(str)

    subset = subset.sort_values(["task_id", "candidate_id"], kind="stable").drop_duplicates(subset=["task_id"], keep="first")

    unresolved = sorted(set(missing_tasks) - set(subset["task_id"].astype(str).tolist()))

    out_anchor = OUT_DIR / f"a2targetedprogneganchorsubsetmarginv3{TAG}.csv"
    out_missing = OUT_DIR / f"a2targetedprognegmissingtasksmarginv3{TAG}.csv"
    out_manifest = OUT_DIR / f"a2targetedprogneganchorsubsetmarginv3manifest{TAG}.json"

    subset.to_csv(out_anchor, index=False, encoding="utf-8")
    pd.DataFrame({"task_id": missing_tasks}).to_csv(out_missing, index=False, encoding="utf-8")

    obj = {
        "script_version": SCRIPT_VERSION,
        "tag": TAG,
        "inputs": {
            "high_v3_csv": str(HIGH_V3_CSV),
            "low_v3_csv": str(LOW_V3_CSV),
            "progneg_csv": str(PROGNEG_CSV),
            "a2_pos_anchor_csv": str(a2_POS_ANCHOR_CSV),
            "a3_high_csv": str(a3_HIGH_CSV),
        },
        "counts": {
            "target_task_union_n": int(len(target_tasks)),
            "progneg_valid_task_n": int(len(covered_tasks)),
            "missing_tasks_needing_targeted_progneg": int(len(missing_tasks)),
            "targeted_anchor_subset_rows": int(len(subset)),
            "unresolved_missing_tasks_without_anchor": int(len(unresolved)),
        },
        "unresolved_missing_tasks": unresolved,
        "outputs": {
            "targeted_anchor_subset_csv": str(out_anchor),
            "missing_tasks_csv": str(out_missing),
            "manifest": str(out_manifest),
        },
    }

    out_manifest.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_anchor}")
    print(f"[ok] wrote {out_missing}")
    print(f"[ok] wrote {out_manifest}")
    print(json.dumps(obj["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
