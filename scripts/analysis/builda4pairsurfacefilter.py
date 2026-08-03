from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
a3_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3"
a4_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a4"
a4_DIR.mkdir(parents=True, exist_ok=True)

RUN_TAG = os.environ.get("a4_RUN_TAG", "a17bpoolv1")

SURFACE_CSV = Path(os.environ.get(
    "a4_SURFACE_CSV",
    str(a4_DIR / f"a4surfacefeaturetable{RUNTAG}.csv")
)).resolve()

a3_HIGH_CSV = Path(os.environ.get(
    "a4_a3_HIGH_CSV",
    str(a3_DIR / f"a3positivehighqualitypool{RUNTAG}.csv")
)).resolve()

a3_WEAK_CSV = Path(os.environ.get(
    "a4_a3_WEAK_CSV",
    str(a3_DIR / f"a3positiveweakqualitypool{RUNTAG}.csv")
)).resolve()

a2_NATNEG_CSV = Path(os.environ.get(
    "a4_a2_NATNEG_CSV",
    str(a2_DIR / f"a2naturalmechanisticnegativepool{RUNTAG}.csv")
)).resolve()

a2_PROGNEG_CSV = Path(os.environ.get(
    "a4_a2_PROGNEG_CSV",
    str(a2_DIR / "a2prognegautometricsmastertable.csv")
)).resolve()

SCRIPT_VERSION = "build_a4_pair_surface_filter_v1"

MAX_TOKEN_RATIO = float(os.environ.get("a4_MAX_TOKEN_RATIO", "2.0"))
MAX_TOKEN_DIFF = int(os.environ.get("a4_MAX_TOKEN_DIFF", "80"))
MAX_ASSERT_DIFF = int(os.environ.get("a4_MAX_ASSERT_DIFF", "4"))
MAX_TEST_FN_DIFF = int(os.environ.get("a4_MAX_TEST_FN_DIFF", "2"))
MAX_COMMENT_RATIO_DIFF = float(os.environ.get("a4_MAX_COMMENT_RATIO_DIFF", "0.25"))
MAX_ASSERT_DENSITY_DIFF = float(os.environ.get("a4_MAX_ASSERT_DENSITY_DIFF", "0.25"))
MAX_STRUCTURAL_COMPLEXITY_DIFF = float(os.environ.get("a4_MAX_STRUCTURAL_COMPLEXITY_DIFF", "2.0"))
ALLOW_MILD_FAIL_COUNT = int(os.environ.get("a4_ALLOW_MILD_FAIL_COUNT", "1"))


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_bool_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).gt(0).astype(np.int8)


def load_surface(surface_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(surface_csv)
    needed = [
        "candidate_id",
        "task_id",
        "syntax_pass",
        "execution_pass",
        "char_length",
        "nonempty_line_count",
        "token_length",
        "num_asserts",
        "num_test_functions",
        "comment_ratio",
        "assert_density",
        "surface_structural_complexity",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"surface table missing columns: {missing}")

    for c in needed:
        if c != "candidate_id" and c != "task_id":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["candidate_id"] = df["candidate_id"].astype(str)
    df["task_id"] = df["task_id"].astype(str)
    return df


def load_pool(path: Path, label_col: str, label_val: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "candidate_id" not in df.columns or "task_id" not in df.columns:
        raise ValueError(f"{path} missing candidate_id/task_id")
    df["candidate_id"] = df["candidate_id"].astype(str)
    df["task_id"] = df["task_id"].astype(str)
    df[label_col] = label_val
    return df


def add_surface(df: pd.DataFrame, surface: pd.DataFrame, side: str) -> pd.DataFrame:
    surf = surface.copy()
    rename = {c: f"{c}_{side}" for c in surf.columns if c != "candidate_id"}
    surf = surf.rename(columns=rename)
    out = df.merge(surf, on="candidate_id", how="left")
    return out


def build_pairs(
    pos_pool: pd.DataFrame,
    neg_pool: pd.DataFrame,
    surface: pd.DataFrame,
    pos_tier: str,
    neg_source: str,
) -> pd.DataFrame:
    pos = pos_pool[["candidate_id", "task_id"]].copy()
    pos["pos_tier"] = pos_tier
    pos = pos.rename(columns={"candidate_id": "candidate_id_pos"})

    neg = neg_pool[["candidate_id", "task_id"]].copy()
    neg["neg_source"] = neg_source
    neg = neg.rename(columns={"candidate_id": "candidate_id_neg"})

    pairs = pos.merge(
        neg,
        on="task_id",
        how="inner",
    )

    pos_surface = surface.copy().rename(
        columns={
            "candidate_id": "candidate_id_pos",
            **{c: f"{c}_pos" for c in surface.columns if c not in ["candidate_id", "task_id"]},
        }
    )

    neg_surface = surface.copy().rename(
        columns={
            "candidate_id": "candidate_id_neg",
            **{c: f"{c}_neg" for c in surface.columns if c not in ["candidate_id", "task_id"]},
        }
    )

    pairs = pairs.merge(
        pos_surface,
        on=["candidate_id_pos", "task_id"],
        how="left",
    )

    pairs = pairs.merge(
        neg_surface,
        on=["candidate_id_neg", "task_id"],
        how="left",
    )

    return pairs


def compute_pair_surface_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in [
        "token_length_pos", "token_length_neg",
        "num_asserts_pos", "num_asserts_neg",
        "num_test_functions_pos", "num_test_functions_neg",
        "comment_ratio_pos", "comment_ratio_neg",
        "assert_density_pos", "assert_density_neg",
        "surface_structural_complexity_pos", "surface_structural_complexity_neg",
        "syntax_pass_pos", "syntax_pass_neg",
        "execution_pass_pos", "execution_pass_neg",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["delta_token_length"] = out["token_length_pos"] - out["token_length_neg"]
    out["abs_delta_token_length"] = out["delta_token_length"].abs()

    token_min = out[["token_length_pos", "token_length_neg"]].min(axis=1).replace(0, np.nan)
    token_max = out[["token_length_pos", "token_length_neg"]].max(axis=1)
    out["token_length_ratio"] = token_max / token_min

    out["abs_delta_num_asserts"] = (out["num_asserts_pos"] - out["num_asserts_neg"]).abs()
    out["abs_delta_num_test_functions"] = (out["num_test_functions_pos"] - out["num_test_functions_neg"]).abs()
    out["abs_delta_comment_ratio"] = (out["comment_ratio_pos"] - out["comment_ratio_neg"]).abs()
    out["abs_delta_assert_density"] = (out["assert_density_pos"] - out["assert_density_neg"]).abs()
    out["abs_delta_surface_structural_complexity"] = (
        out["surface_structural_complexity_pos"] - out["surface_structural_complexity_neg"]
    ).abs()

    out["hard_fail_exec_or_syntax"] = (
        (out["syntax_pass_pos"] != 1)
        | (out["syntax_pass_neg"] != 1)
        | (out["execution_pass_pos"] != 1)
        | (out["execution_pass_neg"] != 1)
    ).astype(int)

    out["hard_fail_length"] = (
        (out["token_length_ratio"] > MAX_TOKEN_RATIO)
        | (out["abs_delta_token_length"] > MAX_TOKEN_DIFF)
    ).astype(int)

    out["hard_fail_test_fn"] = (out["abs_delta_num_test_functions"] > MAX_TEST_FN_DIFF).astype(int)
    out["hard_fail_comment_ratio"] = (out["abs_delta_comment_ratio"] > MAX_COMMENT_RATIO_DIFF).astype(int)

    out["mild_fail_asserts"] = (out["abs_delta_num_asserts"] > MAX_ASSERT_DIFF).astype(int)
    out["mild_fail_assert_density"] = (out["abs_delta_assert_density"] > MAX_ASSERT_DENSITY_DIFF).astype(int)
    out["mild_fail_structure"] = (
        out["abs_delta_surface_structural_complexity"] > MAX_STRUCTURAL_COMPLEXITY_DIFF
    ).astype(int)

    out["surface_hard_fail_count"] = (
        out["hard_fail_exec_or_syntax"]
        + out["hard_fail_length"]
        + out["hard_fail_test_fn"]
        + out["hard_fail_comment_ratio"]
    )

    out["surface_mild_fail_count"] = (
        out["mild_fail_asserts"]
        + out["mild_fail_assert_density"]
        + out["mild_fail_structure"]
    )

    out["surface_filter_keep"] = (
        (out["surface_hard_fail_count"] == 0)
        & (out["surface_mild_fail_count"] <= ALLOW_MILD_FAIL_COUNT)
    ).astype(int)

    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for (pos_tier, neg_source), sub in df.groupby(["pos_tier", "neg_source"], dropna=False):
        rows.append(
            {
                "pos_tier": str(pos_tier),
                "neg_source": str(neg_source),
                "n_pairs_before": int(len(sub)),
                "n_pairs_after": int(sub["surface_filter_keep"].sum()),
                "keep_rate": float(sub["surface_filter_keep"].mean()) if len(sub) else None,
                "hard_fail_exec_or_syntax_rate": float(sub["hard_fail_exec_or_syntax"].mean()) if len(sub) else None,
                "hard_fail_length_rate": float(sub["hard_fail_length"].mean()) if len(sub) else None,
                "hard_fail_test_fn_rate": float(sub["hard_fail_test_fn"].mean()) if len(sub) else None,
                "hard_fail_comment_ratio_rate": float(sub["hard_fail_comment_ratio"].mean()) if len(sub) else None,
                "mild_fail_asserts_rate": float(sub["mild_fail_asserts"].mean()) if len(sub) else None,
                "mild_fail_assert_density_rate": float(sub["mild_fail_assert_density"].mean()) if len(sub) else None,
                "mild_fail_structure_rate": float(sub["mild_fail_structure"].mean()) if len(sub) else None,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    for path, name in [
        (SURFACE_CSV, "SURFACE_CSV"),
        (a3_HIGH_CSV, "a3_HIGH_CSV"),
        (a3_WEAK_CSV, "a3_WEAK_CSV"),
        (a2_NATNEG_CSV, "a2_NATNEG_CSV"),
        (a2_PROGNEG_CSV, "a2_PROGNEG_CSV"),
    ]:
        require_file(path, name)

    surface = load_surface(SURFACE_CSV)

    a3_high = load_pool(a3_HIGH_CSV, "pos_tier", "high_quality")
    a3_weak = load_pool(a3_WEAK_CSV, "pos_tier", "weak_quality")
    natneg = load_pool(a2_NATNEG_CSV, "neg_source", "natural_mechanistic_negative")
    progneg = load_pool(a2_PROGNEG_CSV, "neg_source", "programmatic_negative")

    pair_parts: List[pd.DataFrame] = []
    pair_parts.append(build_pairs(a3_high, natneg, surface, "high_quality", "natural_mechanistic_negative"))
    pair_parts.append(build_pairs(a3_high, progneg, surface, "high_quality", "programmatic_negative"))
    pair_parts.append(build_pairs(a3_weak, natneg, surface, "weak_quality", "natural_mechanistic_negative"))
    pair_parts.append(build_pairs(a3_weak, progneg, surface, "weak_quality", "programmatic_negative"))

    pair_df = pd.concat(pair_parts, ignore_index=True)
    pair_df = compute_pair_surface_flags(pair_df)
    summary_df = build_summary(pair_df)
    filtered_df = pair_df[pair_df["surface_filter_keep"] == 1].copy()

    out_pair_table = a4_DIR / f"a4pairsurfacetable{RUNTAG}.csv"
    out_summary = a4_DIR / f"a4pairsurfacesummary{RUNTAG}.csv"
    out_filtered = a4_DIR / f"a4pairsurfacefiltered{RUNTAG}.csv"
    out_rules = a4_DIR / f"a4surfacefilterrules{RUNTAG}.json"

    pair_df.to_csv(out_pair_table, index=False, encoding="utf-8")
    summary_df.to_csv(out_summary, index=False, encoding="utf-8")
    filtered_df.to_csv(out_filtered, index=False, encoding="utf-8")

    rules = {
        "script_version": SCRIPT_VERSION,
        "run_tag": RUN_TAG,
        "inputs": {
            "surface_csv": str(SURFACE_CSV),
            "a3_high_csv": str(a3_HIGH_CSV),
            "a3_weak_csv": str(a3_WEAK_CSV),
            "a2_natneg_csv": str(a2_NATNEG_CSV),
            "a2_progneg_csv": str(a2_PROGNEG_CSV),
        },
        "thresholds": {
            "MAX_TOKEN_RATIO": MAX_TOKEN_RATIO,
            "MAX_TOKEN_DIFF": MAX_TOKEN_DIFF,
            "MAX_ASSERT_DIFF": MAX_ASSERT_DIFF,
            "MAX_TEST_FN_DIFF": MAX_TEST_FN_DIFF,
            "MAX_COMMENT_RATIO_DIFF": MAX_COMMENT_RATIO_DIFF,
            "MAX_ASSERT_DENSITY_DIFF": MAX_ASSERT_DENSITY_DIFF,
            "MAX_STRUCTURAL_COMPLEXITY_DIFF": MAX_STRUCTURAL_COMPLEXITY_DIFF,
            "ALLOW_MILD_FAIL_COUNT": ALLOW_MILD_FAIL_COUNT,
        },
        "hard_rules": [
            "both pos and neg must be syntax_pass == 1 and execution_pass == 1",
            "token_length_ratio <= MAX_TOKEN_RATIO",
            "abs_delta_token_length <= MAX_TOKEN_DIFF",
            "abs_delta_num_test_functions <= MAX_TEST_FN_DIFF",
            "abs_delta_comment_ratio <= MAX_COMMENT_RATIO_DIFF",
        ],
        "mild_rules": [
            "abs_delta_num_asserts <= MAX_ASSERT_DIFF",
            "abs_delta_assert_density <= MAX_ASSERT_DENSITY_DIFF",
            "abs_delta_surface_structural_complexity <= MAX_STRUCTURAL_COMPLEXITY_DIFF",
        ],
        "keep_rule": "surface_hard_fail_count == 0 AND surface_mild_fail_count <= ALLOW_MILD_FAIL_COUNT",
        "outputs": {
            "pair_surface_table": str(out_pair_table),
            "pair_surface_summary": str(out_summary),
            "pair_surface_filtered": str(out_filtered),
            "rules": str(out_rules),
        },
    }

    out_rules.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_pair_table}")
    print(f"[ok] wrote {out_summary}")
    print(f"[ok] wrote {out_filtered}")
    print(f"[ok] wrote {out_rules}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
