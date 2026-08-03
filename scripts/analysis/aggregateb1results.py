from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path("/root/autodl-tmp/utplm/outputs/evaluations/b1")
OUT_DIR = ROOT
OUT_CSV = OUT_DIR / "b1allrunsummaries.csv"
OUT_AGG = OUT_DIR / "b1groupaggregate.csv"

rows: List[Dict] = []
for p in ROOT.glob("*/summary.json"):
    obj = json.loads(p.read_text(encoding="utf-8"))
    rows.append(obj)

if not rows:
    raise SystemExit("No b1 summary.json found.")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False, encoding="utf-8")

metric_cols = [
    "execution_pass_rate",
    "mutation_score_mean",
    "line_coverage_mean",
    "branch_coverage_mean",
    "DDR_or_BRC_mean",
    "TBC_mean",
    "boundary_coverage_mean",
    "exception_path_coverage_mean",
    "num_asserts_mean",
    "assert_density_mean",
    "failure_log_count_mean",
]

agg_rows = []
for (group, split_name), sub in df.groupby(["group", "split_name"], dropna=False):
    row = {"group": group, "split_name": split_name, "n_runs": int(len(sub))}
    for m in metric_cols:
        x = pd.to_numeric(sub[m], errors="coerce").dropna()
        row[f"{m}_mean"] = float(x.mean()) if len(x) else None
        row[f"{m}_std"] = float(x.std()) if len(x) > 1 else None
    agg_rows.append(row)

agg = pd.DataFrame(agg_rows)
agg.to_csv(OUT_AGG, index=False, encoding="utf-8")

print("[ok] wrote", OUT_CSV)
print("[ok] wrote", OUT_AGG)
print(agg.to_string(index=False))
