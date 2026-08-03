from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


CORE_METRICS = [
    "execution_pass_rate",
    "mutation_score_mean",
    "boundary_coverage_mean",
    "failure_log_count_mean",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/root/autodl-tmp/utplm")
    parser.add_argument("--input_eval_root", default="outputs/evaluations/b1")
    parser.add_argument("--run_prefix", default="c1bal_")
    parser.add_argument("--output_dir", default="outputs/evaluations/c1balancedfixedbudgetv1")
    args = parser.parse_args()

    root = Path(args.project_root)
    eval_root = root / args.input_eval_root
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for summary_path in sorted(eval_root.glob(f"{args.run_prefix}*/summary.json")):
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        if obj.get("group") in {"LL", "LH", "HL", "HH"}:
            rows.append(obj)

    if not rows:
        raise SystemExit(f"No balanced c1 summary files found under {eval_root} with prefix {args.run_prefix}")

    df = pd.DataFrame(rows)
    all_csv = out_dir / "c1balancedallrunsummaries.csv"
    core_csv = out_dir / "c1balancedcoregroupaggregate.csv"
    df.to_csv(all_csv, index=False, encoding="utf-8")

    agg_rows = []
    for (group, split_name), sub in df.groupby(["group", "split_name"], dropna=False):
        row: Dict[str, Any] = {"group": group, "split_name": split_name, "n_runs": int(len(sub))}
        for metric in CORE_METRICS:
            x = pd.to_numeric(sub[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(x.mean()) if len(x) else None
            row[f"{metric}_std"] = float(x.std(ddof=1)) if len(x) > 1 else 0.0 if len(x) == 1 else None
            if len(x) > 1:
                ci = 1.96 * x.std(ddof=1) / (len(x) ** 0.5)
                row[f"{metric}_ci95_low"] = float(x.mean() - ci)
                row[f"{metric}_ci95_high"] = float(x.mean() + ci)
            elif len(x) == 1:
                row[f"{metric}_ci95_low"] = float(x.mean())
                row[f"{metric}_ci95_high"] = float(x.mean())
            else:
                row[f"{metric}_ci95_low"] = None
                row[f"{metric}_ci95_high"] = None
        agg_rows.append(row)

    agg = pd.DataFrame(agg_rows).sort_values(["split_name", "group"])
    agg.to_csv(core_csv, index=False, encoding="utf-8")

    manifest = {
        "script": "aggregatec1balancedfixedbudgetresults.py",
        "run_prefix": args.run_prefix,
        "input_eval_root": str(eval_root),
        "output_all_runs": str(all_csv),
        "output_core_aggregate": str(core_csv),
        "core_metrics": CORE_METRICS,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
