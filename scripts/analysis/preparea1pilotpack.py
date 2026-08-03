from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV = OUT_DIR / "a1autometricsmastertable.csv"
OUT_POOL = OUT_DIR / "a1samplingpool.csv"
OUT_SAMPLE = OUT_DIR / "a1sampledcandidatesv1.csv"
OUT_REPORT = OUT_DIR / "a1samplingreportv1.json"

SCRIPT_VERSION = "prepare_a1_pilot_pack_v6_revised"

# 采样模式：
#   stratified   -> 默认，按 a1_group × source × difficulty_bucket 分层抽样
#   all_eligible -> 保留全部可标注样本
SAMPLE_MODE = os.environ.get("a1_PILOT_SAMPLE_MODE", "stratified").strip().lower()
RANDOM_SEED = int(os.environ.get("a1_PILOT_RANDOM_SEED", "42"))
PER_CELL = int(os.environ.get("a1_PILOT_PER_CELL", "4"))

# 为了保证可标注性，只有满足这些条件的样本才进入 sampled_candidates
REQUIRE_EXECUTABLE_FOR_ANNOTATION = True


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(np.int8)
    x = pd.to_numeric(s, errors="coerce").fillna(0)
    return (x > 0).astype(np.int8)


def zscore(s: pd.Series) -> pd.Series:
    x = safe_numeric(s)
    std = x.std()
    if pd.isna(std) or float(std) == 0.0:
        return pd.Series(np.zeros(len(x)), index=x.index, dtype=float)
    return (x - x.mean()) / std


def safe_value_counts(s: Optional[pd.Series]) -> Dict[str, int]:
    if s is None or len(s) == 0:
        return {}
    vc = s.fillna("NA").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def count_non_missing(s: pd.Series) -> int:
    return int(safe_numeric(s).notna().sum())


def safe_stat(x: Any) -> Any:
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x)
    return x


def infer_current_run_id(df: pd.DataFrame) -> str:
    if "run_id" not in df.columns:
        return "unknown_run"
    vals = (
        df["run_id"]
        .dropna()
        .astype(str)
        .map(str.strip)
        .loc[lambda s: s != ""]
        .unique()
        .tolist()
    )
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 0:
        return "unknown_run"
    return "mixed_runs"


def choose_prompt_col(df: pd.DataFrame) -> str:
    for col in ["prompt", "problem", "task_description", "text"]:
        if col in df.columns:
            return col
    raise ValueError("No prompt-like column found in master table.")


def stable_sample_id(candidate_id: Any) -> str:
    s = str(candidate_id).strip()
    digest = hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]
    return f"a1_{digest}"


def ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"candidate_id", "task_id", "candidate_test_code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"master table 缺少必要字段: {sorted(missing)}")

    prompt_col = choose_prompt_col(df)
    if df[prompt_col].fillna("").astype(str).str.strip().eq("").all():
        raise ValueError(f"prompt-like 列 {prompt_col} 全为空，无法生成 a1 标注包输入。")


def ensure_default_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    defaults = {
        "run_id": None,
        "source": "unknown",
        "difficulty_bucket": "unknown",
        "syntax_pass": 0,
        "execution_pass": 0,
        "invalid_candidate_flag": 0,
        "redefines_target_flag": 0,
        "mutation_unavailable": 0,
        "failure_log_count": 0,
        "mutation_score": np.nan,
        "line_coverage": np.nan,
        "branch_coverage": np.nan,
        "boundary_coverage": np.nan,
        "exception_path_coverage": np.nan,
        "num_asserts": 0,
        "assert_density": 0.0,
        "test_style": None,
        "primary_failure_category": None,
    }

    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default

    signal_cols = [
        "has_mutation_survival_major",
        "has_branch_distinguishing_gap",
        "has_exception_path_gap",
        "has_boundary_case_gap",
        "has_assertion_weak_or_missing",
        "has_execution_assertion_mismatch",
        "has_execution_call_contract_error",
        "has_invalid_candidate_redefines_target",
    ]
    for col in signal_cols:
        if col not in out.columns:
            out[col] = 0

    return out


def build_logic_signal_count(df: pd.DataFrame) -> pd.Series:
    signal_cols = [
        "has_mutation_survival_major",
        "has_branch_distinguishing_gap",
        "has_exception_path_gap",
        "has_boundary_case_gap",
        "has_assertion_weak_or_missing",
        "has_execution_assertion_mismatch",
        "has_execution_call_contract_error",
        "has_invalid_candidate_redefines_target",
    ]
    parts = [safe_bool_series(df[c]) for c in signal_cols]
    mat = pd.concat(parts, axis=1)
    return mat.sum(axis=1).astype(int)


def select_stratified_sample(df: pd.DataFrame, per_cell: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rng = np.random.RandomState(seed)
    work = df.copy()

    group_cols = [c for c in ["a1_group", "source", "difficulty_bucket"] if c in work.columns]
    if not group_cols:
        return work.copy()

    sampled_parts: List[pd.DataFrame] = []
    for _, sub in work.groupby(group_cols, dropna=False, observed=False):
        if len(sub) <= per_cell:
            sampled_parts.append(sub.copy())
        else:
            idx = rng.choice(sub.index.to_numpy(), size=per_cell, replace=False)
            sampled_parts.append(sub.loc[idx].copy())

    out = pd.concat(sampled_parts, axis=0, ignore_index=True)
    sort_cols = [c for c in ["a1_group", "source", "difficulty_bucket", "task_id", "candidate_id"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return out


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"master table not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)
    ensure_required_columns(df)
    df = ensure_default_columns(df)

    current_run_id = infer_current_run_id(df)
    prompt_col = choose_prompt_col(df)

    # 统一文本字段
    df["candidate_id"] = df["candidate_id"].astype(str).str.strip()
    df["task_id"] = df["task_id"].astype(str).str.strip()
    df["candidate_test_code"] = df["candidate_test_code"].fillna("").astype(str)
    df[prompt_col] = df[prompt_col].fillna("").astype(str)

    # 生成 sample_id，供 builda1annotationpack.py 直接使用
    if "sample_id" not in df.columns:
        df["sample_id"] = df["candidate_id"].map(stable_sample_id)

    if df["sample_id"].duplicated().any():
        dupes = df.loc[df["sample_id"].duplicated(keep=False), "sample_id"].tolist()[:10]
        raise ValueError(f"sample_id 发生重复，例如: {dupes}")

    # 基础可标注性标记
    syntax_ok = safe_bool_series(df["syntax_pass"])
    execution_ok = safe_bool_series(df["execution_pass"])
    invalid_candidate = safe_bool_series(df["invalid_candidate_flag"])
    redefines_target = safe_bool_series(df["redefines_target_flag"])
    mutation_unavailable = safe_bool_series(df["mutation_unavailable"])

    nonempty_code = df["candidate_test_code"].str.strip().ne("")
    nonempty_prompt = df[prompt_col].str.strip().ne("")

    df["annotation_text_ready"] = (nonempty_code & nonempty_prompt).astype(np.int8)

    df["usable_exec_flag"] = (
        (syntax_ok == 1) &
        (execution_ok == 1) &
        (invalid_candidate == 0) &
        (redefines_target == 0) &
        (df["annotation_text_ready"] == 1)
    ).astype(np.int8)

    coverage_present = (
        safe_numeric(df["line_coverage"]).notna() &
        safe_numeric(df["branch_coverage"]).notna()
    ).astype(np.int8)

    mutation_present = (
        safe_numeric(df["mutation_score"]).notna() &
        (mutation_unavailable == 0)
    ).astype(np.int8)

    df["usable_eval_flag"] = (
        (df["usable_exec_flag"] == 1) &
        (coverage_present == 1) &
        (mutation_present == 1)
    ).astype(np.int8)

    # logic failure proxy
    df["logic_signal_count"] = build_logic_signal_count(df)
    df["has_logic_failure_signal"] = (df["logic_signal_count"] > 0).astype(np.int8)

    # provisional quality score: 只用于 pilot 分层，不当最终论文标签
    df["provisional_quality_score"] = (
        1.00 * zscore(df["mutation_score"].fillna(0))
        + 0.60 * zscore(df["branch_coverage"].fillna(0))
        + 0.40 * zscore(df["line_coverage"].fillna(0))
        + 0.50 * zscore(df["boundary_coverage"].fillna(0))
        + 0.50 * zscore(df["exception_path_coverage"].fillna(0))
        + 0.20 * zscore(df["num_asserts"].fillna(0))
        + 0.20 * zscore(df["assert_density"].fillna(0))
        - 0.60 * zscore(df["failure_log_count"].fillna(0))
        - 0.80 * zscore(df["logic_signal_count"].fillna(0))
    )

    # proxy group:
    # 1) 不适合逻辑标注的样本先单独标出来
    # 2) 可执行且存在逻辑失败信号的，暂归 mechanistic_negative
    # 3) 其他按 provisional score 三分位切 high/medium/low
    base_mask = (df["usable_exec_flag"] == 1) & (df["has_logic_failure_signal"] == 0)
    base_scores = df.loc[base_mask, "provisional_quality_score"].dropna()

    if len(base_scores) > 0:
        q1 = float(base_scores.quantile(0.33))
        q2 = float(base_scores.quantile(0.67))
    else:
        q1, q2 = 0.0, 0.0

    score = safe_numeric(df["provisional_quality_score"])

    df["a1_group"] = np.where(
        df["usable_exec_flag"] == 0,
        "not_annotation_ready",
        np.where(
            df["has_logic_failure_signal"] == 1,
            "mechanistic_negative",
            np.where(
                score.isna(),
                "medium_quality",
                np.where(
                    score >= q2,
                    "high_quality",
                    np.where(score >= q1, "medium_quality", "low_quality"),
                ),
            ),
        ),
    )

    # 全量 sampling_pool：保留全部，供后续分析和审计
    pool = df.copy()
    pool_sort_cols = [c for c in ["source", "difficulty_bucket", "task_id", "candidate_id"] if c in pool.columns]
    if pool_sort_cols:
        pool = pool.sort_values(pool_sort_cols, kind="stable").reset_index(drop=True)

    # sampled_candidates：默认只保留“适合做 a1 逻辑标注”的样本
    if REQUIRE_EXECUTABLE_FOR_ANNOTATION:
        eligible = df.loc[df["usable_exec_flag"] == 1].copy()
    else:
        eligible = df.loc[df["annotation_text_ready"] == 1].copy()

    if SAMPLE_MODE == "all_eligible":
        sampled = eligible.copy()
    elif SAMPLE_MODE == "stratified":
        sampled = select_stratified_sample(eligible, per_cell=PER_CELL, seed=RANDOM_SEED)
    else:
        raise ValueError(f"Unsupported a1_PILOT_SAMPLE_MODE: {SAMPLE_MODE}")

    sample_sort_cols = [c for c in ["a1_group", "source", "difficulty_bucket", "task_id", "candidate_id"] if c in sampled.columns]
    if sample_sort_cols:
        sampled = sampled.sort_values(sample_sort_cols, kind="stable").reset_index(drop=True)

    # 输出
    pool.to_csv(OUT_POOL, index=False, encoding="utf-8")
    sampled.to_csv(OUT_SAMPLE, index=False, encoding="utf-8")

    score_series = safe_numeric(sampled["provisional_quality_score"]) if "provisional_quality_score" in sampled.columns else pd.Series(dtype=float)

    report: Dict[str, Any] = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "current_run_id": current_run_id,
        "sample_mode": SAMPLE_MODE,
        "random_seed": RANDOM_SEED,
        "per_cell": PER_CELL if SAMPLE_MODE == "stratified" else None,
        "master_table_path": str(IN_CSV),
        "sampling_pool_path": str(OUT_POOL),
        "pilot_pack_path": str(OUT_SAMPLE),
        "prompt_col": prompt_col,
        "total_pool_size": int(len(pool)),
        "eligible_size": int(len(eligible)),
        "pilot_pack_size": int(len(sampled)),
        "quantiles": {
            "q1": q1,
            "q2": q2,
        },
        "pool_group_counts": safe_value_counts(pool["a1_group"]),
        "eligible_group_counts": safe_value_counts(eligible["a1_group"]),
        "pilot_group_counts": safe_value_counts(sampled["a1_group"]),
        "source_counts": safe_value_counts(sampled["source"]) if "source" in sampled.columns else {},
        "difficulty_counts": safe_value_counts(sampled["difficulty_bucket"]) if "difficulty_bucket" in sampled.columns else {},
        "test_style_counts": safe_value_counts(sampled["test_style"]) if "test_style" in sampled.columns else {},
        "primary_failure_counts": safe_value_counts(sampled["primary_failure_category"]) if "primary_failure_category" in sampled.columns else {},
        "usable_exec_count": int(df["usable_exec_flag"].sum()),
        "usable_eval_count": int(df["usable_eval_flag"].sum()),
        "mechanistic_negative_count_in_pool": int((pool["a1_group"] == "mechanistic_negative").sum()),
        "mechanistic_negative_count_in_sample": int((sampled["a1_group"] == "mechanistic_negative").sum()),
        "mutation_unavailable_count": int(pd.to_numeric(pool["mutation_unavailable"], errors="coerce").fillna(0).sum()) if "mutation_unavailable" in pool.columns else 0,
        "metric_non_missing_counts_in_sample": {
            "mutation_score": count_non_missing(sampled["mutation_score"]) if "mutation_score" in sampled.columns else 0,
            "line_coverage": count_non_missing(sampled["line_coverage"]) if "line_coverage" in sampled.columns else 0,
            "branch_coverage": count_non_missing(sampled["branch_coverage"]) if "branch_coverage" in sampled.columns else 0,
            "boundary_coverage": count_non_missing(sampled["boundary_coverage"]) if "boundary_coverage" in sampled.columns else 0,
            "exception_path_coverage": count_non_missing(sampled["exception_path_coverage"]) if "exception_path_coverage" in sampled.columns else 0,
            "num_asserts": count_non_missing(sampled["num_asserts"]) if "num_asserts" in sampled.columns else 0,
            "assert_density": count_non_missing(sampled["assert_density"]) if "assert_density" in sampled.columns else 0,
        },
        "score_summary_in_sample": {
            "provisional_quality_score_mean": safe_stat(score_series.mean()) if len(score_series) > 0 else None,
            "provisional_quality_score_std": safe_stat(score_series.std()) if len(score_series) > 0 else None,
            "provisional_quality_score_min": safe_stat(score_series.min()) if len(score_series) > 0 else None,
            "provisional_quality_score_max": safe_stat(score_series.max()) if len(score_series) > 0 else None,
        },
        "notes": [
            "sampling_pool 保留全量，用于审计与后续分析。",
            "sampled_candidates 只面向当前 a1 标注包构建。",
            "a1_group 与 provisional_quality_score 仅用于 pilot 组织，不直接当论文最终标签。",
            "mechanistic_negative 是 pilot 代理组，不等于 a2 最终正式负样本定义。",
            "默认排除不适合做逻辑标注的样本（例如语法失败、执行失败、重定义 target、空代码、空 prompt）。",
        ],
    }

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {OUT_POOL}")
    print(f"[ok] wrote {OUT_SAMPLE}")
    print(f"[ok] wrote {OUT_REPORT}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()