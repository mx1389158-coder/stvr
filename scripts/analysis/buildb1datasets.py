from __future__ import annotations

import json
import os
from itertools import cycle, islice
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("b1_TAG", "a17bpoolv1")

a1_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
a4_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a4"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "b1" / TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

a1_MASTER = a1_DIR / "a1autometricsmastertable.csv"
a2_PROGNEG = a2_DIR / "a2prognegautometricsmastertable.csv"
a4_FILTERED = a4_DIR / f"a4pairsurfacefiltered{TAG}.csv"

SCRIPT_VERSION = "build_b1_datasets_v1"


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
    if pd.isna(v):
        return ""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    for p, name in [
        (a1_MASTER, "a1_MASTER"),
        (a2_PROGNEG, "a2_PROGNEG"),
        (a4_FILTERED, "a4_FILTERED"),
    ]:
        require_file(p, name)

    a1 = pd.read_csv(a1_MASTER)
    progneg = pd.read_csv(a2_PROGNEG)
    filtered = pd.read_csv(a4_FILTERED)

    prompt_col_a1 = choose_prompt_col(a1)
    prompt_col_prog = choose_prompt_col(progneg)

    a1["candidate_id"] = a1["candidate_id"].astype(str)
    progneg["candidate_id"] = progneg["candidate_id"].astype(str)

    keep_cols = [
        "candidate_id",
        "task_id",
        "source",
        "difficulty_bucket",
        "candidate_test_code",
        "entry_point",
        "signature",
        "task_type",
        "canonical_solution",
        "imports",
        "base_tests",
        "eval_assets",
        prompt_col_a1,
    ]
    keep_cols_a1 = [c for c in keep_cols if c in a1.columns]
    keep_cols_prog = [c if c != prompt_col_a1 else prompt_col_prog for c in keep_cols if c != "candidate_id"]

    a1_small = a1[keep_cols_a1].copy()
    if prompt_col_a1 != "prompt":
        a1_small = a1_small.rename(columns={prompt_col_a1: "prompt"})
    else:
        a1_small = a1_small.rename(columns={"prompt": "prompt"})

    prog_keep = [
        "candidate_id", "task_id", "source", "difficulty_bucket",
        "candidate_test_code", "entry_point", "signature", "task_type",
        "canonical_solution", "imports", "base_tests", "eval_assets", prompt_col_prog,
    ]
    prog_keep = [c for c in prog_keep if c in progneg.columns]
    prog_small = progneg[prog_keep].copy()
    if prompt_col_prog != "prompt":
        prog_small = prog_small.rename(columns={prompt_col_prog: "prompt"})

    cand_union = pd.concat([a1_small, prog_small], ignore_index=True)
    cand_union = cand_union.sort_values(["candidate_id"], kind="stable").drop_duplicates(subset=["candidate_id"], keep="first")
    cand_union["candidate_id"] = cand_union["candidate_id"].astype(str)

    filtered["candidate_id_pos"] = filtered["candidate_id_pos"].astype(str)
    filtered["candidate_id_neg"] = filtered["candidate_id_neg"].astype(str)

    high_pairs = filtered[
        (filtered["pos_tier"].astype(str) == "high_quality")
        & (filtered["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    low_pairs = filtered[
        (filtered["pos_tier"].astype(str) == "weak_quality")
    ].copy()

    high_pairs = high_pairs.sort_values(["task_id", "candidate_id_pos", "candidate_id_neg"], kind="stable").reset_index(drop=True)
    low_pairs = low_pairs.sort_values(["task_id", "candidate_id_pos", "candidate_id_neg"], kind="stable").reset_index(drop=True)

    target_n = min(len(high_pairs), len(low_pairs))
    if target_n <= 0:
        raise RuntimeError(f"No usable b1 pairs. high_pairs={len(high_pairs)}, low_pairs={len(low_pairs)}")

    high_pairs = high_pairs.head(target_n).copy()
    low_pairs = low_pairs.head(target_n).copy()

    cand_index = cand_union.set_index("candidate_id", drop=False)

    def build_pref_rows(pair_df: pd.DataFrame, group_name: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, r in pair_df.iterrows():
            pos_id = str(r["candidate_id_pos"])
            neg_id = str(r["candidate_id_neg"])

            if pos_id not in cand_index.index or neg_id not in cand_index.index:
                continue

            pos = cand_index.loc[pos_id]
            neg = cand_index.loc[neg_id]

            prompt_text = format_prompt(pos, "prompt")
            chosen = safe_str(pos.get("candidate_test_code"))
            rejected = safe_str(neg.get("candidate_test_code"))

            if not chosen.strip() or not rejected.strip():
                continue

            rows.append(
                {
                    "group": group_name,
                    "task_id": safe_str(r.get("task_id")),
                    "prompt": prompt_text,
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_candidate_id": pos_id,
                    "rejected_candidate_id": neg_id,
                    "source": safe_str(pos.get("source")),
                    "difficulty_bucket": safe_str(pos.get("difficulty_bucket")),
                }
            )
        return rows

    high_pref_rows = build_pref_rows(high_pairs, "dpo_high_quality")
    low_pref_rows = build_pref_rows(low_pairs, "dpo_low_quality")

    if len(high_pref_rows) == 0 or len(low_pref_rows) == 0:
        raise RuntimeError("After join, b1 preference dataset is empty.")

    # ---------- SFT rows ----------
    high_pos_ids = [row["chosen_candidate_id"] for row in high_pref_rows]
    high_pos_unique = []
    seen = set()
    for cid in high_pos_ids:
        if cid not in seen:
            seen.add(cid)
            high_pos_unique.append(cid)

    sft_base_rows: List[Dict[str, Any]] = []
    for cid in high_pos_unique:
        row = cand_index.loc[cid]
        response = safe_str(row.get("candidate_test_code"))
        if not response.strip():
            continue
        prompt_text = format_prompt(row, "prompt")
        sft_base_rows.append(
            {
                "prompt": prompt_text,
                "response": response,
                "text": prompt_text + "\n\n" + response,
                "candidate_id": cid,
                "task_id": safe_str(row.get("task_id")),
                "source": safe_str(row.get("source")),
                "difficulty_bucket": safe_str(row.get("difficulty_bucket")),
            }
        )

    if len(sft_base_rows) == 0:
        raise RuntimeError("SFT base rows are empty.")

    sft_rows = list(islice(cycle(sft_base_rows), target_n))

    # ---------- label shuffle (for D1 later) ----------
    shuffle_rows: List[Dict[str, Any]] = []
    for i, row in enumerate(high_pref_rows):
        swapped = (i % 2 == 1)
        shuffle_rows.append(
            {
                **row,
                "chosen": row["rejected"] if swapped else row["chosen"],
                "rejected": row["chosen"] if swapped else row["rejected"],
                "chosen_candidate_id": row["rejected_candidate_id"] if swapped else row["chosen_candidate_id"],
                "rejected_candidate_id": row["chosen_candidate_id"] if swapped else row["rejected_candidate_id"],
                "label_shuffled": int(swapped),
            }
        )

    out_sft = OUT_DIR / "b1sfttrain.jsonl"
    out_pref_high = OUT_DIR / "b1prefhighquality.jsonl"
    out_pref_low = OUT_DIR / "b1preflowquality.jsonl"
    out_pref_shuffle = OUT_DIR / "b1prefhighqualitylabelshuffle.jsonl"
    out_manifest = OUT_DIR / "b1datasetmanifest.json"

    write_jsonl(sft_rows, out_sft)
    write_jsonl(high_pref_rows, out_pref_high)
    write_jsonl(low_pref_rows, out_pref_low)
    write_jsonl(shuffle_rows, out_pref_shuffle)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "tag": TAG,
        "inputs": {
            "a1_master": str(a1_MASTER),
            "a2_progneg": str(a2_PROGNEG),
            "a4_filtered": str(a4_FILTERED),
        },
        "outputs": {
            "sft_train": str(out_sft),
            "pref_high_quality": str(out_pref_high),
            "pref_low_quality": str(out_pref_low),
            "pref_high_quality_label_shuffle": str(out_pref_shuffle),
            "manifest": str(out_manifest),
        },
        "counts": {
            "high_pairs_available_after_a4": int(len(filtered[
                (filtered["pos_tier"].astype(str) == "high_quality")
                & (filtered["neg_source"].astype(str) == "programmatic_negative")
            ])),
            "low_pairs_available_after_a4": int(len(filtered[
                (filtered["pos_tier"].astype(str) == "weak_quality")
            ])),
            "target_pair_count_balanced": int(target_n),
            "pref_high_quality_rows": int(len(high_pref_rows)),
            "pref_low_quality_rows": int(len(low_pref_rows)),
            "sft_rows": int(len(sft_rows)),
            "sft_unique_positive_candidates": int(len(high_pos_unique)),
        },
        "notes": [
            "b1-min balances high-quality and low-quality preference datasets to the same pair count.",
            "high-quality preference uses high_quality + programmatic_negative after a4 filtering.",
            "low-quality preference uses weak_quality pairs after a4 filtering.",
            "SFT repeats high-quality chosen samples cyclically to match target pair count.",
        ],
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_sft}")
    print(f"[ok] wrote {out_pref_high}")
    print(f"[ok] wrote {out_pref_low}")
    print(f"[ok] wrote {out_pref_shuffle}")
    print(f"[ok] wrote {out_manifest}")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
