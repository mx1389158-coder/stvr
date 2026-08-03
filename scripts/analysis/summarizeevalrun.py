from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(x) == 0:
        return None
    return float(x.mean())


def valid_only_mean(df: pd.DataFrame, col: str):
    if col not in df.columns or "execution_pass" not in df.columns:
        return None
    valid = pd.to_numeric(df["execution_pass"], errors="coerce").fillna(0) == 1
    x = pd.to_numeric(df.loc[valid, col], errors="coerce").dropna()
    if len(x) == 0:
        return None
    return float(x.mean())


def pass_adjusted_mean(df: pd.DataFrame, col: str):
    if col not in df.columns or "execution_pass" not in df.columns:
        return None
    passed = pd.to_numeric(df["execution_pass"], errors="coerce").fillna(0)
    values = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return float((passed * values).mean())


def safe_merge_on_candidate(
    base_df: pd.DataFrame,
    sub_df: pd.DataFrame,
    *,
    stage_name: str,
) -> pd.DataFrame:
    if sub_df.empty or "candidate_id" not in sub_df.columns:
        return base_df

    # 这些列是重复的元信息，不从后续 stage 再 merge 进来
    meta_drop = {
        "task_id",
        "source",
        "difficulty_bucket",
        "run_id",
        "script_version",
        "timestamp",
        "test_source_type",
    }

    keep_cols = ["candidate_id"]
    for c in sub_df.columns:
        if c == "candidate_id":
            continue
        if c in meta_drop:
            continue
        if c in base_df.columns:
            continue
        keep_cols.append(c)

    if keep_cols == ["candidate_id"]:
        return base_df

    sub_use = sub_df[keep_cols].copy()

    # 防御：candidate_id 若有重复，取第一条
    sub_use = sub_use.drop_duplicates(subset=["candidate_id"], keep="first")

    out = base_df.merge(sub_use, on="candidate_id", how="left")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm"))
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--group", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--split_name", required=True)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    run_id = args.run_id

    generated = pd.DataFrame(load_jsonl(root / "data" / "interim" / "generatedcandidates" / run_id / "generatedcandidates.jsonl"))
    if generated.empty:
        legacy = root / "data" / "interim" / "generatedcandidates" / run_id / "smokecandidates.jsonl"
        generated = pd.DataFrame(load_jsonl(legacy))

    execution = pd.DataFrame(load_jsonl(root / "data" / "interim" / "executionresults" / run_id / "executionresults.jsonl"))
    coverage = pd.DataFrame(load_jsonl(root / "data" / "interim" / "coverageresults" / run_id / "coverageresults.jsonl"))
    mutation = pd.DataFrame(load_jsonl(root / "data" / "interim" / "mutationresults" / run_id / "mutationresults.jsonl"))
    static = pd.DataFrame(load_jsonl(root / "data" / "interim" / "staticfeatures" / run_id / "staticfeatures.jsonl"))
    failure_logs = pd.DataFrame(load_jsonl(root / "data" / "interim" / "failuretaxonomylogs" / run_id / "failuretaxonomylogs.jsonl"))

    if generated.empty:
        raise RuntimeError(f"generated candidates missing for run_id={run_id}")

    base_cols = [c for c in ["candidate_id", "task_id", "source", "difficulty_bucket"] if c in generated.columns]
    df = generated[base_cols].copy()

    df = safe_merge_on_candidate(df, execution, stage_name="execution")
    df = safe_merge_on_candidate(df, coverage, stage_name="coverage")
    df = safe_merge_on_candidate(df, mutation, stage_name="mutation")
    df = safe_merge_on_candidate(df, static, stage_name="static")

    if not failure_logs.empty and "candidate_id" in failure_logs.columns:
        cnt = failure_logs.groupby("candidate_id").size().reset_index(name="failure_log_count")
        df = df.merge(cnt, on="candidate_id", how="left")
    else:
        df["failure_log_count"] = 0

    if "execution_pass" not in df.columns:
        if "execution_status" in df.columns:
            df["execution_pass"] = (df["execution_status"] == "passed").astype(int)
        else:
            df["execution_pass"] = 0

    df["execution_pass_x_mutation_score"] = (
        pd.to_numeric(df["execution_pass"], errors="coerce").fillna(0)
        * pd.to_numeric(df.get("mutation_score", 0), errors="coerce").fillna(0)
    )
    df["execution_pass_x_boundary_coverage"] = (
        pd.to_numeric(df["execution_pass"], errors="coerce").fillna(0)
        * pd.to_numeric(df.get("boundary_coverage", 0), errors="coerce").fillna(0)
    )

    summary = {
        "group": args.group,
        "seed": args.seed,
        "split_name": args.split_name,
        "run_id": run_id,
        "n_candidates": int(len(df)),
        "execution_pass_rate": float(pd.to_numeric(df["execution_pass"], errors="coerce").fillna(0).mean()),
        "valid_test_rate": float(pd.to_numeric(df["execution_pass"], errors="coerce").fillna(0).mean()),
        "mutation_score_mean": safe_mean(df, "mutation_score"),
        "valid_tests_only_mutation_score_mean": valid_only_mean(df, "mutation_score"),
        "execution_pass_x_mutation_score_mean": pass_adjusted_mean(df, "mutation_score"),
        "all_attempt_failure_adjusted_mutation_score_mean": pass_adjusted_mean(df, "mutation_score"),
        "line_coverage_mean": safe_mean(df, "line_coverage"),
        "branch_coverage_mean": safe_mean(df, "branch_coverage"),
        "DDR_or_BRC_mean": safe_mean(df, "DDR_or_BRC"),
        "TBC_mean": safe_mean(df, "TBC"),
        "boundary_coverage_mean": safe_mean(df, "boundary_coverage"),
        "valid_tests_only_boundary_coverage_mean": valid_only_mean(df, "boundary_coverage"),
        "execution_pass_x_boundary_coverage_mean": pass_adjusted_mean(df, "boundary_coverage"),
        "exception_path_coverage_mean": safe_mean(df, "exception_path_coverage"),
        "num_asserts_mean": safe_mean(df, "num_asserts"),
        "assert_density_mean": safe_mean(df, "assert_density"),
        "failure_log_count_mean": safe_mean(df, "failure_log_count"),
    }

    out_dir = root / "outputs" / "evaluations" / "b1" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df.to_csv(out_dir / "percandidate.csv", index=False, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
