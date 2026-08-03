from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
TAG = os.environ.get("TAG", "a17bpoolv1")

a1_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
a2S_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2_strength_v4"
a4_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a4"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "b1diaga2strengthv4" / TAG
OUT_DIR.mkdir(parents=True, exist_ok=True)

a1_MASTER = a1_DIR / "a1autometricsmastertable.csv"
STRONG_PROGNEG = a2S_DIR / f"a2prognegstrongmarginv3{TAG}.csv"
ORD_PROGNEG = a2S_DIR / f"a2prognegordinarymarginv3{TAG}.csv"
a4_STRONG = a4_DIR / f"a4pairsurfacefiltereda17bpoolv1progstrongv4.csv"
a4_ORD = a4_DIR / f"a4pairsurfacefiltereda17bpoolv1progordinaryv4.csv"

SCRIPT_VERSION = "build_b1diaga2strengthv4"


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
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    for p, name in [
        (a1_MASTER, "a1_MASTER"),
        (STRONG_PROGNEG, "STRONG_PROGNEG"),
        (ORD_PROGNEG, "ORD_PROGNEG"),
        (a4_STRONG, "a4_STRONG"),
        (a4_ORD, "a4_ORD"),
    ]:
        require_file(p, name)

    a1 = pd.read_csv(a1_MASTER)
    strong = pd.read_csv(STRONG_PROGNEG)
    ordinary = pd.read_csv(ORD_PROGNEG)
    a4_strong = pd.read_csv(a4_STRONG)
    a4_ord = pd.read_csv(a4_ORD)

    prompt_col_a1 = choose_prompt_col(a1)

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
    a1_small = a1[[c for c in keep_cols if c in a1.columns]].copy()
    if prompt_col_a1 != "prompt":
        a1_small = a1_small.rename(columns={prompt_col_a1: "prompt"})

    def prep_prog(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ["candidate_id","task_id","source","difficulty_bucket","candidate_test_code","entry_point","signature","task_type","prompt"] if c in df.columns]
        out = df[cols].copy()
        if "prompt" not in out.columns:
            out["prompt"] = ""
        return out

    strong_small = prep_prog(strong)
    ord_small = prep_prog(ordinary)

    cand_union = pd.concat([a1_small, strong_small, ord_small], ignore_index=True)
    cand_union["candidate_id"] = cand_union["candidate_id"].astype(str)
    cand_union = cand_union.sort_values(["candidate_id"], kind="stable").drop_duplicates(subset=["candidate_id"], keep="first")
    cand_index = cand_union.set_index("candidate_id", drop=False)

    for df in [a4_strong, a4_ord]:
        df["candidate_id_pos"] = df["candidate_id_pos"].astype(str)
        df["candidate_id_neg"] = df["candidate_id_neg"].astype(str)

    strong_pairs = a4_strong[
        (a4_strong["pos_tier"].astype(str) == "high_quality")
        & (a4_strong["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    ordinary_pairs = a4_ord[
        (a4_ord["pos_tier"].astype(str) == "high_quality")
        & (a4_ord["neg_source"].astype(str) == "programmatic_negative")
    ].copy()

    strong_pairs = strong_pairs.sort_values(["task_id", "candidate_id_pos", "candidate_id_neg"], kind="stable").reset_index(drop=True)
    ordinary_pairs = ordinary_pairs.sort_values(["task_id", "candidate_id_pos", "candidate_id_neg"], kind="stable").reset_index(drop=True)

    target_n = min(len(strong_pairs), len(ordinary_pairs))
    if target_n <= 0:
        raise RuntimeError(f"No usable strong-vs-ordinary pairs. strong={len(strong_pairs)}, ordinary={len(ordinary_pairs)}")

    strong_pairs = strong_pairs.head(target_n).copy()
    ordinary_pairs = ordinary_pairs.head(target_n).copy()

    def build_rows(pair_df: pd.DataFrame, group_name: str) -> List[Dict[str, Any]]:
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

            rows.append({
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
            })
        return rows

    strong_rows = build_rows(strong_pairs, "dpo_high_progstrong_v4")
    ordinary_rows = build_rows(ordinary_pairs, "dpo_high_progordinary_v4")

    out_strong = OUT_DIR / "b1diaga2strengthprefhighprogstrongv4.jsonl"
    out_ord = OUT_DIR / "b1diaga2strengthprefhighprogordinaryv4.jsonl"
    out_manifest = OUT_DIR / "b1diaga2strengthmanifestv4.json"

    write_jsonl(strong_rows, out_strong)
    write_jsonl(ordinary_rows, out_ord)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "tag": TAG,
        "counts": {
            "strong_pairs_raw": int(len(strong_pairs)),
            "ordinary_pairs_raw": int(len(ordinary_pairs)),
            "target_pair_count_balanced": int(target_n),
            "pref_high_progstrong_v4_rows": int(len(strong_rows)),
            "pref_high_progordinary_v4_rows": int(len(ordinary_rows)),
        },
        "outputs": {
            "strong_jsonl": str(out_strong),
            "ordinary_jsonl": str(out_ord),
            "manifest": str(out_manifest),
        },
        "notes": [
            "This diagnostic fixes positive quality at current high_margin_v3 and varies only programmatic-negative strength."
        ],
    }

    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_strong}")
    print(f"[ok] wrote {out_ord}")
    print(f"[ok] wrote {out_manifest}")
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
