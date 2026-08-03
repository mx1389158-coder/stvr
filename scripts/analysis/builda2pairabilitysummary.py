from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


SCRIPT_VERSION = "build_a2_pairability_summary_v1"


FLAG_COLS = [
    "has_mutation_survival_major",
    "has_branch_distinguishing_gap",
    "has_exception_path_gap",
    "has_boundary_case_gap",
    "has_assertion_weak_or_missing",
    "has_execution_assertion_mismatch",
    "has_execution_call_contract_error",
    "has_invalid_candidate_redefines_target",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build a2 pair-first source comparison, pair table, and pairability summary."
    )
    ap.add_argument(
        "--project_root",
        default=os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm"),
        help="Project root. Defaults to UTPLM_PROJECT_ROOT or /root/autodl-tmp/utplm",
    )
    ap.add_argument(
        "--tag",
        default=os.environ.get("a2_TAG", "a17bpoolv1"),
        help="Run tag used to infer default input filenames, e.g. a17bpoolv1",
    )
    ap.add_argument(
        "--positive_csv",
        default=None,
        help="Optional explicit path to positive anchor pool csv",
    )
    ap.add_argument(
        "--natural_negative_csv",
        default=None,
        help="Optional explicit path to natural mechanistic negative pool csv",
    )
    ap.add_argument(
        "--programmatic_negative_csv",
        default=None,
        help="Optional explicit path to programmatic negative master table csv",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Optional explicit output directory. Defaults to <project_root>/outputs/summaries/a2",
    )
    return ap.parse_args()


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    defaults = {
        "candidate_id": "",
        "task_id": "",
        "source": "unknown",
        "difficulty_bucket": "unknown",
        "syntax_pass": 0,
        "execution_pass": 0,
        "invalid_candidate_flag": 0,
        "redefines_target_flag": 0,
        "mutation_unavailable": 0,
        "mutation_score": np.nan,
        "line_coverage": np.nan,
        "branch_coverage": np.nan,
        "boundary_coverage": np.nan,
        "exception_path_coverage": np.nan,
        "failure_log_count": 0,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v

    for c in FLAG_COLS:
        if c not in df.columns:
            df[c] = 0

    bool_like_cols = [
        "syntax_pass",
        "execution_pass",
        "invalid_candidate_flag",
        "redefines_target_flag",
        "mutation_unavailable",
    ] + FLAG_COLS

    for c in bool_like_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    numeric_cols = [
        "mutation_score",
        "line_coverage",
        "branch_coverage",
        "boundary_coverage",
        "exception_path_coverage",
        "failure_log_count",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["candidate_id", "task_id", "source", "difficulty_bucket"]:
        df[c] = df[c].astype(str)

    return df


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def safe_mean(series: pd.Series) -> Optional[float]:
    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if len(s) == 0:
        return None
    return float(s.mean())


def build_candidate_source_comparison(neg_all: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for src, sub in neg_all.groupby("neg_source", dropna=False):
        rows.append(
            {
                "neg_source": src,
                "n_candidates": int(len(sub)),
                "syntax_pass_rate": float(sub["syntax_pass"].mean()),
                "execution_pass_rate": float(sub["execution_pass"].mean()),
                "invalid_rate": float(sub["invalid_candidate_flag"].mean()),
                "redefines_target_rate": float(sub["redefines_target_flag"].mean()),
                "mutation_unavailable_rate": float(sub["mutation_unavailable"].mean()),
                "mutation_score_mean": safe_mean(sub["mutation_score"]),
                "branch_coverage_mean": safe_mean(sub["branch_coverage"]),
                "boundary_coverage_mean": safe_mean(sub["boundary_coverage"]),
                "exception_path_coverage_mean": safe_mean(sub["exception_path_coverage"]),
                "failure_log_count_mean": safe_mean(sub["failure_log_count"].fillna(0)),
                "logic_signal_count_mean": safe_mean(sub["logic_signal_count"].fillna(0)),
            }
        )

    return pd.DataFrame(rows)


def build_pair_table(pos: pd.DataFrame, neg_all: pd.DataFrame) -> pd.DataFrame:
    pairs = pos.merge(
        neg_all,
        on="task_id",
        how="inner",
        suffixes=("_pos", "_neg"),
    )

    pairs["pair_exec_valid"] = (
        pairs["syntax_pass_neg"].eq(1)
        & pairs["execution_pass_neg"].eq(1)
        & pairs["invalid_candidate_flag_neg"].eq(0)
        & pairs["redefines_target_flag_neg"].eq(0)
        & pairs["mutation_unavailable_neg"].eq(0)
    ).astype(int)

    pairs["delta_mutation"] = pairs["mutation_score_pos"] - pairs["mutation_score_neg"]
    pairs["delta_branch"] = pairs["branch_coverage_pos"] - pairs["branch_coverage_neg"]
    pairs["delta_boundary"] = pairs["boundary_coverage_pos"] - pairs["boundary_coverage_neg"]
    pairs["delta_exception"] = pairs["exception_path_coverage_pos"] - pairs["exception_path_coverage_neg"]

    pairs["pair_logic_valid"] = (
        pairs["pair_exec_valid"].eq(1)
        & (
            pairs["delta_mutation"].fillna(-999).gt(0)
            | pairs["logic_signal_count_neg"].fillna(0).gt(0)
            | pairs["failure_log_count_neg"].fillna(0).gt(0)
        )
    ).astype(int)

    pairs["high_info_negative_keep"] = (
        pairs["pair_exec_valid"].eq(1)
        & pairs["delta_mutation"].fillna(-999).gt(0)
        & pairs["logic_signal_count_neg"].fillna(0).ge(1)
    ).astype(int)

    return pairs


def build_pairability_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for src, sub in pairs.groupby("neg_source", dropna=False):
        rows.append(
            {
                "neg_source": src,
                "n_pairs_total": int(len(sub)),
                "n_pairs_exec_valid": int(sub["pair_exec_valid"].sum()),
                "n_pairs_logic_valid": int(sub["pair_logic_valid"].sum()),
                "n_pairs_high_info_keep": int(sub["high_info_negative_keep"].sum()),
                "pair_exec_valid_rate": float(sub["pair_exec_valid"].mean()) if len(sub) else None,
                "pair_logic_valid_rate": float(sub["pair_logic_valid"].mean()) if len(sub) else None,
                "pair_high_info_keep_rate": float(sub["high_info_negative_keep"].mean()) if len(sub) else None,
                "delta_mutation_mean": safe_mean(sub["delta_mutation"]),
                "delta_branch_mean": safe_mean(sub["delta_branch"]),
                "delta_boundary_mean": safe_mean(sub["delta_boundary"]),
                "delta_exception_mean": safe_mean(sub["delta_exception"]),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "outputs" / "summaries" / "a2")
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag

    positive_csv = (
        Path(args.positive_csv).resolve()
        if args.positive_csv
        else out_dir / f"a2positiveanchorpool{tag}.csv"
    )
    natural_negative_csv = (
        Path(args.natural_negative_csv).resolve()
        if args.natural_negative_csv
        else out_dir / f"a2naturalmechanisticnegativepool{tag}.csv"
    )
    programmatic_negative_csv = (
        Path(args.programmatic_negative_csv).resolve()
        if args.programmatic_negative_csv
        else out_dir / "a2prognegautometricsmastertable.csv"
    )

    require_file(positive_csv, "positive_csv")
    require_file(natural_negative_csv, "natural_negative_csv")
    require_file(programmatic_negative_csv, "programmatic_negative_csv")

    pos = pd.read_csv(positive_csv)
    natneg = pd.read_csv(natural_negative_csv)
    progneg = pd.read_csv(programmatic_negative_csv)

    pos = ensure_cols(pos)
    natneg = ensure_cols(natneg)
    progneg = ensure_cols(progneg)

    # 统一逻辑信号计数
    for df in [pos, natneg, progneg]:
        df["logic_signal_count"] = sum(df[c] for c in FLAG_COLS)

    natneg["neg_source"] = "natural_mechanistic_negative"
    progneg["neg_source"] = "programmatic_negative"

    neg_all = pd.concat([natneg, progneg], ignore_index=True)

    cand_cmp = build_candidate_source_comparison(neg_all)
    pair_table = build_pair_table(pos, neg_all)
    pair_summary = build_pairability_summary(pair_table)

    out_candidate_source_comparison = out_dir / "a2negativesourcecomparison.csv"
    out_pair_table = out_dir / "a2pairtableall.csv"
    out_pairability_summary = out_dir / "a2pairabilitysummary.csv"
    out_manifest = out_dir / "a2pairabilitymanifest.json"

    cand_cmp.to_csv(out_candidate_source_comparison, index=False, encoding="utf-8")
    pair_table.to_csv(out_pair_table, index=False, encoding="utf-8")
    pair_summary.to_csv(out_pairability_summary, index=False, encoding="utf-8")

    manifest = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(root),
        "tag": tag,
        "inputs": {
            "positive_pool_path": str(positive_csv),
            "natural_negative_pool_path": str(natural_negative_csv),
            "programmatic_negative_pool_path": str(programmatic_negative_csv),
        },
        "outputs": {
            "candidate_source_comparison": str(out_candidate_source_comparison),
            "pair_table_all": str(out_pair_table),
            "pairability_summary": str(out_pairability_summary),
            "manifest": str(out_manifest),
        },
        "flag_cols": FLAG_COLS,
        "rules": {
            "pair_exec_valid": [
                "syntax_pass_neg == 1",
                "execution_pass_neg == 1",
                "invalid_candidate_flag_neg == 0",
                "redefines_target_flag_neg == 0",
                "mutation_unavailable_neg == 0",
            ],
            "pair_logic_valid": [
                "pair_exec_valid == 1",
                "delta_mutation > 0 OR logic_signal_count_neg > 0 OR failure_log_count_neg > 0",
            ],
            "high_info_negative_keep": [
                "pair_exec_valid == 1",
                "delta_mutation > 0",
                "logic_signal_count_neg >= 1",
            ],
        },
        "notes": [
            "a2-min follows pair-first validation.",
            "Pool overlap is not used as the sole progress gate.",
            "high_info_negative_keep is an initial proxy rule, not the final frozen a2 rule.",
        ],
    }

    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {out_candidate_source_comparison}")
    print(f"[ok] wrote {out_pair_table}")
    print(f"[ok] wrote {out_pairability_summary}")
    print(f"[ok] wrote {out_manifest}")
    print()
    print(pair_summary.to_string(index=False))


if __name__ == "__main__":
    main()