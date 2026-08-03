from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


SCRIPT_VERSION = "summarize_external_failure_audit_v1"

MANUAL_COLUMNS = [
    "manual_validity",
    "manual_failure_type",
    "auto_category_agree",
    "reveals_stricter_behavior",
]


def count_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    values = df[col].fillna("").astype(str).str.strip()
    values = values.where(values != "", "(blank)")
    out = values.value_counts(dropna=False).rename_axis(col).reset_index(name="count")
    out["percent"] = (out["count"] / len(df) * 100).round(1) if len(df) else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit_csv", default="/root/autodl-tmp/utplm/outputs/audits/external89failureauditpack.csv")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    audit_csv = Path(args.audit_csv).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else audit_csv.parent / "external89failureauditsummary"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(audit_csv)
    missing = [col for col in MANUAL_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing manual audit columns: {missing}")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "audit_csv": str(audit_csv),
        "out_dir": str(out_dir),
        "n_rows": int(len(df)),
        "manual_columns": MANUAL_COLUMNS,
        "blank_counts": {col: int(df[col].fillna("").astype(str).str.strip().eq("").sum()) for col in MANUAL_COLUMNS},
        "outputs": {},
    }

    for col in MANUAL_COLUMNS:
        table = count_table(df, col)
        out_path = out_dir / f"{col}counts.csv"
        table.to_csv(out_path, index=False)
        manifest["outputs"][col] = str(out_path)

    cross_cols = ["group", "external_project", "auto_failure_category", *MANUAL_COLUMNS]
    available = [col for col in cross_cols if col in df.columns]
    cross = df[available].copy()
    cross_path = out_dir / "annotatedcasescompact.csv"
    cross.to_csv(cross_path, index=False)
    manifest["outputs"]["annotated_cases_compact"] = str(cross_path)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
