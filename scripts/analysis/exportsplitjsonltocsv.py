from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.input_jsonl))
    df = pd.DataFrame(rows)

    keep_cols = [
        "task_id",
        "source",
        "difficulty_bucket",
        "prompt",
        "entry_point",
        "signature",
        "task_type",
        "canonical_solution",
        "imports",
        "base_tests",
        "eval_assets",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    print("[ok] wrote", out, "rows=", len(df))


if __name__ == "__main__":
    main()
