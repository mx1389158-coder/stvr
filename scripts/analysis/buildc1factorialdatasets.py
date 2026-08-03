from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("TAG", "a17bpoolv1")

# 新增：高负/低负阈值
c1_ENABLE_NEG_SCORE_FILTER = int(os.environ.get("c1_ENABLE_NEG_SCORE_FILTER", "1"))
c1_HIGH_NEG_MIN_SCORE = float(os.environ.get("c1_HIGH_NEG_MIN_SCORE", "1"))
c1_LOW_NEG_MAX_SCORE = float(os.environ.get("c1_LOW_NEG_MAX_SCORE", "0"))

a1_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2_strength_v4"
a3_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3_margin_v3"
a4_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a4"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "c1" / TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

a1_MASTER = a1_DIR / "a1autometricsmastertable.csv"

HIGH_POS = a3_DIR / f"a3positivehighqualitypoolmarginv3{TAG}.csv"
LOW_POS = a3_DIR / f"a3positiveweakqualitypoolmarginv3{TAG}.csv"

HIGH_NEG = a2_DIR / f"a2prognegstrongmarginv3{TAG}.csv"
LOW_NEG = a2_DIR / f"a2prognegordinarymarginv3{TAG}.csv"

a4_HIGHNEG = a4_DIR / f"a4pairsurfacefiltered{TAG}progstrongv4.csv"
a4_LOWNEG = a4_DIR / f"a4pairsurfacefiltered{TAG}progordinaryv4.csv"

SCRIPT_VERSION = "build_c1_factorial_datasets_v2"


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def choose_prompt_col(df: pd.DataFrame) -> str:
    for col in ["prompt", "problem", "task_description", "text"]:
        if col in df.columns:
            return col
    raise ValueError("No prompt-like column found.")


def safe_str(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v)


def format_prompt(row: pd.Series, prompt_col: str) -> str:
    task_text = safe_str(row.get(prompt_col))
    entry_point = safe_str(row.get("entry_point"))
    signature = safe_str(row.get("signature"))
    task_type = safe_str(row.get("task_type"))

    return (
        "Task description:\n"
        f"{task_text}\n\n"
        f"Entry point: {entry_point}\n"
        f"Signature (if available): {signature}\n"
        f"Task type (if available): {task_type}\n\n"
        "Write pytest-style Python unit tests for the target above.\n\n"
        "Hard requirements:\n"
        "1. Output only executable Python test code.\n"
        "2. Use pytest-style test functions named like test_*.\n"
        "3. Every test function must contain at least one real behavioral check:\n"
        "   - either an assert statement\n"
        "   - or a pytest.raises(...) block when exception behavior is clearly supported.\n"
        "4. Do not write bare calls, print statements, or checks without assertions.\n"
        "5. Do not redefine, reimplement, alias, wrap, or shadow the target function/class.\n"
        "6. Do not copy the solution into the test file.\n"
        "7. Do not assume unsupported semantics. Only test behavior supported by the task description.\n"
        "8. Prefer 3 to 6 concise test functions.\n"
        "9. Include meaningful checks for normal behavior and supported edge cases.\n"
        "10. Include exception-related checks only when clearly justified by the task description.\n"
        "11. Avoid helper functions, wrappers, aliases, or fixtures unless strictly necessary and clearly harmless.\n"
    )


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def prep_pos_pool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    prompt_col = choose_prompt_col(df)
    keep = [
        "candidate_id", "task_id", "source", "difficulty_bucket",
        "candidate_test_code", "entry_point", "signature", "task_type",
        "positive_teaching_score", prompt_col
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    if prompt_col != "prompt":
        out = out.rename(columns={prompt_col: "prompt"})
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["task_id"] = out["task_id"].astype(str)
    out["positive_teaching_score"] = pd.to_numeric(out["positive_teaching_score"], errors="coerce")
    return out


def prep_neg_pool(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [
        "candidate_id", "task_id", "source", "difficulty_bucket",
        "candidate_test_code", "entry_point", "signature", "task_type",
        "diagnostic_score"
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()
    out["candidate_id"] = out["candidate_id"].astype(str)
    out["task_id"] = out["task_id"].astype(str)
    if "diagnostic_score" not in out.columns:
        out["diagnostic_score"] = None
    out["diagnostic_score"] = pd.to_numeric(out["diagnostic_score"], errors="coerce").fillna(0)
    return out


def prep_pairs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["candidate_id_pos"] = df["candidate_id_pos"].astype(str)
    df["candidate_id_neg"] = df["candidate_id_neg"].astype(str)
    df["task_id"] = df["task_id"].astype(str)
    return df


def attach_metadata(
    pair_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    group_code: str,
) -> pd.DataFrame:
    pos_meta = pos_df.rename(columns={
        "candidate_id": "candidate_id_pos",
        "candidate_test_code": "chosen",
        "positive_teaching_score": "pos_teaching_score",
        "prompt": "prompt",
        "source": "source",
        "difficulty_bucket": "difficulty_bucket",
    })
    neg_meta = neg_df.rename(columns={
        "candidate_id": "candidate_id_neg",
        "candidate_test_code": "rejected",
        "diagnostic_score": "neg_diagnostic_score",
    })

    use_cols_pos = [c for c in [
        "candidate_id_pos", "task_id", "source", "difficulty_bucket",
        "chosen", "prompt", "entry_point", "signature", "task_type", "pos_teaching_score"
    ] if c in pos_meta.columns]

    use_cols_neg = [c for c in [
        "candidate_id_neg", "rejected", "neg_diagnostic_score"
    ] if c in neg_meta.columns]

    out = pair_df.merge(pos_meta[use_cols_pos], on=["candidate_id_pos", "task_id"], how="left")
    out = out.merge(neg_meta[use_cols_neg], on="candidate_id_neg", how="left")
    out["group_code"] = group_code
    out["pos_teaching_score"] = pd.to_numeric(out["pos_teaching_score"], errors="coerce")
    out["neg_diagnostic_score"] = pd.to_numeric(out["neg_diagnostic_score"], errors="coerce").fillna(0)
    return out


def build_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    for _, r in df.iterrows():
        chosen = safe_str(r.get("chosen"))
        rejected = safe_str(r.get("rejected"))
        if not chosen.strip() or not rejected.strip():
            continue
        rows.append({
            "group": safe_str(r.get("group_code")),
            "task_id": safe_str(r.get("task_id")),
            "prompt": format_prompt(r, "prompt"),
            "chosen": chosen,
            "rejected": rejected,
            "chosen_candidate_id": safe_str(r.get("candidate_id_pos")),
            "rejected_candidate_id": safe_str(r.get("candidate_id_neg")),
            "source": safe_str(r.get("source")),
            "difficulty_bucket": safe_str(r.get("difficulty_bucket")),
            "pos_teaching_score": r.get("pos_teaching_score"),
            "neg_diagnostic_score": r.get("neg_diagnostic_score"),
        })
    return rows


def filter_by_neg_strength(groups: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    if not c1_ENABLE_NEG_SCORE_FILTER:
        return groups

    out = {}
    for gname, gdf in groups.items():
        g = gdf.copy()
        if gname in {"LH", "HH"}:
            g = g[g["neg_diagnostic_score"] >= c1_HIGH_NEG_MIN_SCORE].copy()
        elif gname in {"LL", "HL"}:
            g = g[g["neg_diagnostic_score"] <= c1_LOW_NEG_MAX_SCORE].copy()
        out[gname] = g.reset_index(drop=True)
    return out


def sort_for_sampling(gname: str, sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()

    if gname == "LL":
        by = ["neg_diagnostic_score", "pos_teaching_score", "task_id", "candidate_id_pos", "candidate_id_neg"]
        asc = [True, True, True, True, True]
    elif gname == "HL":
        by = ["neg_diagnostic_score", "pos_teaching_score", "task_id", "candidate_id_pos", "candidate_id_neg"]
        asc = [True, False, True, True, True]
    elif gname == "LH":
        by = ["neg_diagnostic_score", "pos_teaching_score", "task_id", "candidate_id_pos", "candidate_id_neg"]
        asc = [False, True, True, True, True]
    elif gname == "HH":
        by = ["neg_diagnostic_score", "pos_teaching_score", "task_id", "candidate_id_pos", "candidate_id_neg"]
        asc = [False, False, True, True, True]
    else:
        by = ["task_id", "candidate_id_pos", "candidate_id_neg"]
        asc = [True, True, True]

    return sub.sort_values(by, ascending=asc, kind="stable").reset_index(drop=True)


def main() -> None:
    for p, name in [
        (a1_MASTER, "a1_MASTER"),
        (HIGH_POS, "HIGH_POS"),
        (LOW_POS, "LOW_POS"),
        (HIGH_NEG, "HIGH_NEG"),
        (LOW_NEG, "LOW_NEG"),
        (a4_HIGHNEG, "a4_HIGHNEG"),
        (a4_LOWNEG, "a4_LOWNEG"),
    ]:
        require_file(p, name)

    high_pos = prep_pos_pool(HIGH_POS)
    low_pos = prep_pos_pool(LOW_POS)
    high_neg = prep_neg_pool(HIGH_NEG)
    low_neg = prep_neg_pool(LOW_NEG)

    a4_highneg = prep_pairs(a4_HIGHNEG)
    a4_lowneg = prep_pairs(a4_LOWNEG)

    LL = a4_lowneg[
        (a4_lowneg["pos_tier"].astype(str) == "weak_quality") &
        (a4_lowneg["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    HL = a4_lowneg[
        (a4_lowneg["pos_tier"].astype(str) == "high_quality") &
        (a4_lowneg["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    LH = a4_highneg[
        (a4_highneg["pos_tier"].astype(str) == "weak_quality") &
        (a4_highneg["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    HH = a4_highneg[
        (a4_highneg["pos_tier"].astype(str) == "high_quality") &
        (a4_highneg["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    LL = attach_metadata(LL, low_pos, low_neg, "LL")
    HL = attach_metadata(HL, high_pos, low_neg, "HL")
    LH = attach_metadata(LH, low_pos, high_neg, "LH")
    HH = attach_metadata(HH, high_pos, high_neg, "HH")

    groups = {"LL": LL, "HL": HL, "LH": LH, "HH": HH}
    groups = filter_by_neg_strength(groups)

    raw_counts = []
    for gname, gdf in groups.items():
        tmp = gdf.groupby(["source", "difficulty_bucket"], dropna=False).size().reset_index(name="n_raw")
        tmp["group"] = gname
        raw_counts.append(tmp)
    raw_counts_df = pd.concat(raw_counts, ignore_index=True) if raw_counts else pd.DataFrame()

    bucket_base = (
        raw_counts_df.pivot_table(
            index=["source", "difficulty_bucket"],
            columns="group",
            values="n_raw",
            fill_value=0,
            aggfunc="sum"
        )
        .reset_index()
    )

    for gname in ["LL", "HL", "LH", "HH"]:
        if gname not in bucket_base.columns:
            bucket_base[gname] = 0

    bucket_base["bucket_target"] = bucket_base[["LL", "HL", "LH", "HH"]].min(axis=1).astype(int)
    bucket_base = bucket_base.sort_values(["source", "difficulty_bucket"], kind="stable").reset_index(drop=True)

    sampled = {}
    final_counts = []

    for gname, gdf in groups.items():
        parts = []
        for _, row in bucket_base.iterrows():
            src = row["source"]
            diff = row["difficulty_bucket"]
            n = int(row["bucket_target"])
            if n <= 0:
                continue
            sub = gdf[(gdf["source"] == src) & (gdf["difficulty_bucket"] == diff)].copy()
            sub = sort_for_sampling(gname, sub).head(n)
            parts.append(sub)
        out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        sampled[gname] = out

        cnt = out.groupby(["source", "difficulty_bucket"], dropna=False).size().reset_index(name="n_final")
        cnt["group"] = gname
        final_counts.append(cnt)

    final_counts_df = pd.concat(final_counts, ignore_index=True) if final_counts else pd.DataFrame()

    bucket_final = (
        final_counts_df.pivot_table(
            index=["source", "difficulty_bucket"],
            columns="group",
            values="n_final",
            fill_value=0,
            aggfunc="sum"
        )
        .reset_index()
    )
    for gname in ["LL", "HL", "LH", "HH"]:
        if gname not in bucket_final.columns:
            bucket_final[gname] = 0

    group_summary_rows = []
    for gname, gdf in sampled.items():
        group_summary_rows.append({
            "group": gname,
            "n_pairs_final": int(len(gdf)),
            "unique_tasks": int(gdf["task_id"].nunique()) if len(gdf) else 0,
            "pos_teaching_score_mean": float(pd.to_numeric(gdf["pos_teaching_score"], errors="coerce").dropna().mean()) if len(gdf) else None,
            "neg_diagnostic_score_mean": float(pd.to_numeric(gdf["neg_diagnostic_score"], errors="coerce").dropna().mean()) if len(gdf) else None,
            "neg_diagnostic_score_min": float(pd.to_numeric(gdf["neg_diagnostic_score"], errors="coerce").dropna().min()) if len(gdf) else None,
            "neg_diagnostic_score_max": float(pd.to_numeric(gdf["neg_diagnostic_score"], errors="coerce").dropna().max()) if len(gdf) else None,
        })
    group_summary_df = pd.DataFrame(group_summary_rows)

    outputs = {}
    for gname, gdf in sampled.items():
        jsonl_path = OUT_DIR / f"c1pref{gname}.jsonl"
        rows = build_rows(gdf)
        write_jsonl(rows, jsonl_path)
        outputs[gname] = str(jsonl_path)

    raw_counts_path = OUT_DIR / "c1bucketrawcounts.csv"
    bucket_targets_path = OUT_DIR / "c1buckettargets.csv"
    bucket_final_path = OUT_DIR / "c1bucketfinalcounts.csv"
    group_summary_path = OUT_DIR / "c1groupsamplingsummary.csv"
    manifest_path = OUT_DIR / "c1datasetmanifest.json"

    raw_counts_df.to_csv(raw_counts_path, index=False, encoding="utf-8")
    bucket_base.to_csv(bucket_targets_path, index=False, encoding="utf-8")
    bucket_final.to_csv(bucket_final_path, index=False, encoding="utf-8")
    group_summary_df.to_csv(group_summary_path, index=False, encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "tag": TAG,
        "neg_score_filter": {
            "enabled": int(c1_ENABLE_NEG_SCORE_FILTER),
            "high_neg_min_score": float(c1_HIGH_NEG_MIN_SCORE),
            "low_neg_max_score": float(c1_LOW_NEG_MAX_SCORE),
        },
        "group_definition": {
            "LL": "low_positive + low_negative",
            "HL": "high_positive + low_negative",
            "LH": "low_positive + high_negative",
            "HH": "high_positive + high_negative",
        },
        "counts": {r["group"]: int(r["n_pairs_final"]) for r in group_summary_rows},
        "outputs": outputs,
        "reports": {
            "raw_counts": str(raw_counts_path),
            "bucket_targets": str(bucket_targets_path),
            "bucket_final_counts": str(bucket_final_path),
            "group_sampling_summary": str(group_summary_path),
            "manifest": str(manifest_path),
        },
        "notes": [
            "c1 v2 adds explicit negative-strength filtering and bucket-internal score-aware sampling.",
            "High-negative groups are sampled from higher neg_diagnostic_score pairs; low-negative groups from lower-score pairs.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {manifest_path}")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
