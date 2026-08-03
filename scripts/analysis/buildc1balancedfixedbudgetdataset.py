from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


GROUPS = ["LL", "LH", "HL", "HH"]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_key(row: Dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source", "")),
        str(row.get("difficulty_bucket", "")),
        str(row.get("task_id", "")),
        str(row.get("chosen_candidate_id", "")),
        str(row.get("rejected_candidate_id", "")),
    )


def allocate_source_targets(reference_rows: List[Dict[str, Any]], target_n: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in reference_rows:
        source = str(row.get("source", "NA"))
        counts[source] = counts.get(source, 0) + 1
    total = sum(counts.values())
    if total == 0:
        return {}

    raw = {k: (v / total) * target_n for k, v in counts.items()}
    out = {k: int(raw[k]) for k in raw}
    while sum(out.values()) < target_n:
        k = max(raw, key=lambda x: (raw[x] - out[x], raw[x], x))
        out[k] += 1
    return out


def select_diverse_within_source(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    ordered = sorted(rows, key=stable_key)
    rng = random.Random(seed)
    by_diff: Dict[str, List[Dict[str, Any]]] = {}
    for row in ordered:
        by_diff.setdefault(str(row.get("difficulty_bucket", "NA")), []).append(row)
    for diff_rows in by_diff.values():
        rng.shuffle(diff_rows)

    selected: List[Dict[str, Any]] = []
    # Round-robin over difficulty buckets so the sample is not dominated by one bucket.
    while len(selected) < n and by_diff:
        for diff in sorted(list(by_diff)):
            if len(selected) >= n:
                break
            bucket = by_diff.get(diff, [])
            if bucket:
                selected.append(bucket.pop(0))
            if not bucket:
                by_diff.pop(diff, None)
    return selected


def select_rows(
    rows: List[Dict[str, Any]],
    target_n: int,
    seed: int,
    source_targets: Dict[str, int] | None = None,
) -> List[Dict[str, Any]]:
    if len(rows) < target_n:
        raise ValueError(f"Cannot sample {target_n} rows from only {len(rows)} rows.")
    if len(rows) == target_n:
        return sorted(rows, key=stable_key)

    if not source_targets:
        ordered = sorted(rows, key=stable_key)
        rng = random.Random(seed)
        sampled = rng.sample(ordered, target_n)
        return sorted(sampled, key=stable_key)

    selected: List[Dict[str, Any]] = []
    used_ids: set[int] = set()
    for source, n in sorted(source_targets.items()):
        source_rows = [r for r in rows if str(r.get("source", "NA")) == source]
        take = min(n, len(source_rows))
        part = select_diverse_within_source(source_rows, take, seed + len(source))
        selected.extend(part)
        used_ids.update(id(r) for r in part)

    if len(selected) < target_n:
        rest = [r for r in rows if id(r) not in used_ids]
        selected.extend(select_diverse_within_source(rest, target_n - len(selected), seed + 999))

    return sorted(selected[:target_n], key=stable_key)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", "NA"))
        diff = str(row.get("difficulty_bucket", "NA"))
        source_counts[source] = source_counts.get(source, 0) + 1
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
    return {
        "n": len(rows),
        "unique_tasks": len({str(r.get("task_id", "")) for r in rows}),
        "source_counts": source_counts,
        "difficulty_counts": difficulty_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/root/autodl-tmp/utplm/data/processed/c1formalv2c",
    )
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/utplm/data/processed/c1balancedfixedbudgetv1",
    )
    parser.add_argument("--target_per_cell", type=int, default=7)
    parser.add_argument("--fixed_downsampling_seed", "--sample_seed", dest="fixed_downsampling_seed", type=int, default=20260624)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "script": "buildc1balancedfixedbudgetdataset.py",
        "purpose": "Balanced/fixed-budget sensitivity dataset for c1/c2 internal validity.",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_per_cell": args.target_per_cell,
        "fixed_downsampling_seed": args.fixed_downsampling_seed,
        "groups": {},
        "notes": [
            "LL and LH already contain 7 rows in c1formalv2c and are retained when target_per_cell=7.",
            "HL and HH are deterministically downsampled with the fixed downsampling seed.",
            "Training should use a fixed max_steps/optimizer-update budget rather than equal epochs.",
        ],
    }

    reference_rows = read_jsonl(input_dir / "ll.jsonl")
    source_targets = allocate_source_targets(reference_rows, args.target_per_cell)
    manifest["source_targets"] = source_targets

    for group in GROUPS:
        src = input_dir / f"{group}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)
        rows = read_jsonl(src)
        selected = select_rows(rows, args.target_per_cell, args.fixed_downsampling_seed, source_targets)
        out_path = output_dir / f"{group}.jsonl"
        write_jsonl(selected, out_path)
        manifest["groups"][group] = {
            "input_count": len(rows),
            "output_count": len(selected),
            "output_path": str(out_path),
            **summarize(selected),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
