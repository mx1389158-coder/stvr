from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List


GROUPS = ["HH", "HL", "LH", "LL"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_str(value: Any) -> str:
    return "" if value is None else str(value)


def code_len(row: Dict[str, Any], key: str = "rejected") -> int:
    return len(safe_str(row.get(key)).split())


def load_groups(input_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for group in GROUPS:
        path = input_dir / f"{group}.jsonl"
        if path.exists():
            rows = read_jsonl(path)
            for row in rows:
                row.setdefault("cell", group)
            out[group] = rows
    if "HH" not in out or not out["HH"]:
        raise RuntimeError(f"HH source data missing or empty under {input_dir}")
    return out


def build_chosen_only_sft(hh_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in hh_rows:
        prompt = safe_str(row.get("prompt"))
        chosen = safe_str(row.get("chosen"))
        if not prompt.strip() or not chosen.strip():
            continue
        rows.append(
            {
                "control_group": "chosen_only_sft_HH",
                "prompt": prompt,
                "response": chosen,
                "text": prompt + "\n\n" + chosen,
                "task_id": safe_str(row.get("task_id")),
                "source": safe_str(row.get("source")),
                "difficulty_bucket": safe_str(row.get("difficulty_bucket")),
                "chosen_candidate_id": safe_str(row.get("chosen_candidate_id")),
                "source_cell": safe_str(row.get("cell", "HH")),
            }
        )
    return rows


def build_label_shuffled(hh_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for i, row in enumerate(hh_rows):
        swapped = i % 2 == 1
        chosen = row["rejected"] if swapped else row["chosen"]
        rejected = row["chosen"] if swapped else row["rejected"]
        chosen_id = row.get("rejected_candidate_id") if swapped else row.get("chosen_candidate_id")
        rejected_id = row.get("chosen_candidate_id") if swapped else row.get("rejected_candidate_id")
        rows.append(
            {
                **row,
                "control_group": "label_shuffled_dpo_HH",
                "chosen": chosen,
                "rejected": rejected,
                "chosen_candidate_id": safe_str(chosen_id),
                "rejected_candidate_id": safe_str(rejected_id),
                "label_shuffled": int(swapped),
                "source_cell": safe_str(row.get("cell", "HH")),
            }
        )
    return rows


def random_negative_pool(groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for group, rows in groups.items():
        for row in rows:
            if safe_str(row.get("neg_kind")) != "ordinary":
                continue
            rejected = safe_str(row.get("rejected"))
            if not rejected.strip():
                continue
            pool.append(
                {
                    "rejected": rejected,
                    "rejected_candidate_id": safe_str(row.get("rejected_candidate_id")),
                    "task_id": safe_str(row.get("task_id")),
                    "source": safe_str(row.get("source")),
                    "difficulty_bucket": safe_str(row.get("difficulty_bucket")),
                    "cell": group,
                    "length_words": code_len(row, "rejected"),
                }
            )
    if not pool:
        raise RuntimeError("No ordinary rejected candidates found for random-negative control.")
    return pool


def choose_surface_matched_negative(
    anchor: Dict[str, Any],
    pool: List[Dict[str, Any]],
    rng: random.Random,
) -> Dict[str, Any]:
    original_id = safe_str(anchor.get("rejected_candidate_id"))
    anchor_len = code_len(anchor, "rejected")
    source = safe_str(anchor.get("source"))
    difficulty = safe_str(anchor.get("difficulty_bucket"))

    candidates = [
        row
        for row in pool
        if row["source"] == source
        and row["difficulty_bucket"] == difficulty
        and row["rejected_candidate_id"] != original_id
    ]
    if not candidates:
        candidates = [row for row in pool if row["source"] == source and row["rejected_candidate_id"] != original_id]
    if not candidates:
        candidates = [row for row in pool if row["rejected_candidate_id"] != original_id]
    if not candidates:
        candidates = list(pool)

    candidates = sorted(
        candidates,
        key=lambda row: (
            abs(int(row["length_words"]) - anchor_len),
            row["source"],
            row["difficulty_bucket"],
            row["task_id"],
            row["rejected_candidate_id"],
        ),
    )
    top_k = max(1, min(5, len(candidates)))
    return rng.choice(candidates[:top_k])


def build_random_negative_dpo(
    hh_rows: List[Dict[str, Any]],
    pool: List[Dict[str, Any]],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for row in hh_rows:
        neg = choose_surface_matched_negative(row, pool, rng)
        rows.append(
            {
                **row,
                "control_group": "surface_matched_random_negative_dpo_HH",
                "rejected": neg["rejected"],
                "rejected_candidate_id": neg["rejected_candidate_id"],
                "random_negative_source_cell": neg["cell"],
                "random_negative_task_id": neg["task_id"],
                "random_negative_length_words": neg["length_words"],
                "anchor_rejected_candidate_id": safe_str(row.get("rejected_candidate_id")),
                "anchor_rejected_length_words": code_len(row, "rejected"),
                "source_cell": safe_str(row.get("cell", "HH")),
            }
        )
    return rows


def summarize(rows: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    source_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}
    tasks = set()
    for row in rows:
        source = safe_str(row.get("source", "NA"))
        difficulty = safe_str(row.get("difficulty_bucket", "NA"))
        source_counts[source] = source_counts.get(source, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        if row.get("task_id") is not None:
            tasks.add(safe_str(row.get("task_id")))
    return {
        "kind": kind,
        "rows": len(rows),
        "unique_tasks": len(tasks),
        "source_counts": source_counts,
        "difficulty_counts": difficulty_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="/root/autodl-tmp/utplm/data/processed/c1formalv2c")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/utplm/data/processed/c1controlsv1")
    parser.add_argument("--random_seed", type=int, default=20260706)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_groups(input_dir)
    hh_rows = groups["HH"]
    pool = random_negative_pool(groups)

    sft_rows = build_chosen_only_sft(hh_rows)
    shuffled_rows = build_label_shuffled(hh_rows)
    random_neg_rows = build_random_negative_dpo(hh_rows, pool, args.random_seed)

    out_sft = output_dir / "chosenonlysfthh.jsonl"
    out_shuffle = output_dir / "labelshuffleddpohh.jsonl"
    out_random = output_dir / "surfacematchedrandomnegativedpohh.jsonl"
    out_manifest = output_dir / "manifest.json"

    write_jsonl(sft_rows, out_sft)
    write_jsonl(shuffled_rows, out_shuffle)
    write_jsonl(random_neg_rows, out_random)

    manifest = {
        "script": "buildc1controldatasets.py",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "random_seed": args.random_seed,
        "anchor_condition": "HH",
        "outputs": {
            "chosen_only_sft_HH": str(out_sft),
            "label_shuffled_dpo_HH": str(out_shuffle),
            "surface_matched_random_negative_dpo_HH": str(out_random),
            "manifest": str(out_manifest),
        },
        "input_counts": {group: len(rows) for group, rows in groups.items()},
        "ordinary_negative_pool_rows": len(pool),
        "summaries": {
            "chosen_only_sft_HH": summarize(sft_rows, "sft"),
            "label_shuffled_dpo_HH": summarize(shuffled_rows, "dpo"),
            "surface_matched_random_negative_dpo_HH": summarize(random_neg_rows, "dpo"),
        },
        "notes": [
            "Controls anchor on the HH condition so chosen samples and prompts match the high-quality/high-information DPO condition.",
            "chosen_only_sft_HH exposes only HH chosen responses.",
            "label_shuffled_dpo_HH flips every second HH pair.",
            "surface_matched_random_negative_dpo_HH keeps the HH prompt/chosen response and replaces rejected with an ordinary negative matched by source, difficulty, and nearby token length where possible.",
        ],
    }
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
