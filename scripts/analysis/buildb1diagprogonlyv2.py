from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("TAG", "a17bpoolv1")

a1_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
a4_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a4"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "b1diagprogonlyv2" / TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

a1_MASTER = a1_DIR / "a1autometricsmastertable.csv"
a2_PROGNEG = a2_DIR / "a2prognegautometricsmastertable.csv"
a4_FILTERED = a4_DIR / f"a4pairsurfacefiltered{TAG}.csv"

SCRIPT_VERSION = "build_b1diagprogonlyv2"


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

    keep_cols = [
        "candidate_id",
        "task_id",
        "source",
        "difficulty_bucket",
        "candidate_test_code",
        "entry_point",
        "signature",
        "task_type",
        prompt_col_a1,
    ]
    keep_cols_a1 = [c for c in keep_cols if c in a1.columns]
    a1_small = a1[keep_cols_a1].copy()
    if prompt_col_a1 != "prompt":
        a1_small = a1_small.rename(columns={prompt_col_a1: "prompt"})

    prog_keep = [
        "candidate_id", "task_id", "source", "difficulty_bucket",
        "candidate_test_code", "entry_point", "signature", "task_type", prompt_col_prog,
    ]
    prog_keep = [c for c in prog_keep if c in progneg.columns]
    prog_small = progneg[prog_keep].copy()
    if prompt_col_prog != "prompt":
        prog_small = prog_small.rename(columns={prompt_col_prog: "prompt"})

    cand_union = pd.concat([a1_small, prog_small], ignore_index=True)
    cand_union["candidate_id"] = cand_union["candidate_id"].astype(str)
    cand_union = cand_union.sort_values(["candidate_id"], kind="stable").drop_duplicates(subset=["candidate_id"], keep="first")
    cand_index = cand_union.set_index("candidate_id", drop=False)

    filtered["candidate_id_pos"] = filtered["candidate_id_pos"].astype(str)
    filtered["candidate_id_neg"] = filtered["candidate_id_neg"].astype(str)

    high_prog = filtered[
        (filtered["pos_tier"].astype(str) == "high_quality")
        & (filtered["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    low_prog = filtered[
        (filtered["pos_tier"].astype(str) == "weak_quality")
        & (filtered["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    high_prog = high_prog.sort_values(["task_id", "candidate_id_pos", "candidate_id_neg"], kind="stable").reset_index(drop=True)
    low_prog = low_prog.sort_values(["task_id", "candidate_id_pos", "candidate_id_neg"], kind="stable").reset_index(drop=True)

    high_raw = len(high_prog)
    low_raw = len(low_prog)
    target_n = min(high_raw, low_raw)

    if target_n <= 0:
        raise RuntimeError(f"No usable prog-only v2 pairs. high={high_raw}, low={low_raw}")

    high_prog = high_prog.head(target_n).copy()
    low_prog = low_prog.head(target_n).copy()

    def build_pref_rows(pair_df: pd.DataFrame, group_name: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for _, r in pair_df.iterrows():
            pos_id = str(r["candidate_id_pos"])
            neg_id = str(r["candidate_id_neg"])

            if pos_id not in cand_index.index or neg_id not in cand_index.index:
                continue

            pos = cand_index.loc[pos_id]
            neg = cand_index.loc[neg_id]

            chosen = safe_str(pos.get("candidate_test_code"))
            rejected = safe_str(neg.get("candidate_test_code"))
            if not chosen.strip() or not rejected.strip():
                continue

            rows.append(
                {
                    "group": group_name,
                    "task_id": safe_str(r.get("task_id")),
                    "prompt": format_prompt(pos, "prompt"),
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_candidate_id": pos_id,
                    "rejected_candidate_id": neg_id,
                    "source": safe_str(pos.get("source")),
                    "difficulty_bucket": safe_str(pos.get("difficulty_bucket")),
                    "neg_source": "programmatic_negative",
                }
            )
        return rows

    high_rows = build_pref_rows(high_prog, "dpo_high_progonly_v2")
    low_rows = build_pref_rows(low_prog, "dpo_low_progonly_v2")

    out_high = OUT_DIR / "b1diagv2prefhighprogonly.jsonl"
    out_low = OUT_DIR / "b1diagv2preflowprogonly.jsonl"
    out_manifest = OUT_DIR / "b1diagv2manifest.json"

    write_jsonl(high_rows, out_high)
    write_jsonl(low_rows, out_low)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "tag": TAG,
        "inputs": {
            "a1_master": str(a1_MASTER),
            "a2_progneg": str(a2_PROGNEG),
            "a4_filtered": str(a4_FILTERED),
        },
        "counts": {
            "high_prog_raw_after_weak_v2": int(high_raw),
            "low_prog_raw_after_weak_v2": int(low_raw),
            "target_pair_count_balanced": int(target_n),
            "pref_high_progonly_v2_rows": int(len(high_rows)),
            "pref_low_progonly_v2_rows": int(len(low_rows)),
        },
        "outputs": {
            "pref_high_progonly_v2": str(out_high),
            "pref_low_progonly_v2": str(out_low),
            "manifest": str(out_manifest),
        },
        "notes": [
            "b1 fixed-source v2 uses weak pool v2 (overlap-aware) and keeps negative source fixed to programmatic_negative.",
        ],
    }

    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_high}")
    print(f"[ok] wrote {out_low}")
    print(f"[ok] wrote {out_manifest}")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
