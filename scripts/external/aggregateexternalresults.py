from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_VERSION = "aggregate_external_results_v2_project_bootstrap"

METRICS = [
    ("execution_pass_rate", "execution_pass"),
    ("mutation_score_mean", "mutation_score"),
    ("num_asserts_mean", "num_asserts"),
    ("assert_density_mean", "assert_density"),
    ("failure_log_count_mean", "failure_log_count"),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_summary(root: Path, run_id: str) -> dict[str, Any]:
    summary_path = root / "outputs" / "evaluations" / "b1" / run_id / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary for {run_id}: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def parse_run_spec(value: str) -> tuple[str, str, str]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"Run spec must be group:seed:run_id, got {value!r}")
    return parts[0], parts[1], parts[2]


def parse_eval_assets(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        obj = json.loads(value)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def infer_project(row: pd.Series) -> str:
    for col in ["external_project", "source"]:
        value = row.get(col)
        if isinstance(value, str) and value.strip():
            value = value.strip()
            return value.removeprefix("external_")
    assets = parse_eval_assets(row.get("eval_assets"))
    value = assets.get("external_project")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown"


def generated_enrichment(root: Path, run_id: str) -> pd.DataFrame:
    gen_path = root / "data" / "interim" / "generatedcandidates" / run_id / "generatedcandidates.jsonl"
    rows = load_jsonl(gen_path)
    if not rows:
        legacy = root / "data" / "interim" / "generatedcandidates" / run_id / "smokecandidates.jsonl"
        rows = load_jsonl(legacy)
    if not rows:
        return pd.DataFrame(columns=["candidate_id"])

    out = []
    for row in rows:
        assets = parse_eval_assets(row.get("eval_assets"))
        project = assets.get("external_project") or row.get("external_project") or str(row.get("source", "")).removeprefix("external_")
        out.append(
            {
                "candidate_id": row.get("candidate_id"),
                "external_project": project or "unknown",
                "external_project_commit": assets.get("external_project_commit") or row.get("external_project_commit"),
                "external_module": assets.get("module") or row.get("external_module"),
                "decoding_config_name": row.get("decoding_config_name") or row.get("candidate_test_origin_detail"),
                "prompt_variant_name": row.get("prompt_variant_name"),
                "task_id_generated": row.get("task_id"),
            }
        )
    return pd.DataFrame(out).drop_duplicates(subset=["candidate_id"], keep="first")


def load_run_candidates(root: Path, group: str, seed: str, run_id: str) -> pd.DataFrame:
    per_candidate_path = root / "outputs" / "evaluations" / "b1" / run_id / "percandidate.csv"
    if not per_candidate_path.exists():
        raise FileNotFoundError(f"Missing per-candidate table for {run_id}: {per_candidate_path}")
    df = pd.read_csv(per_candidate_path)
    enrich = generated_enrichment(root, run_id)
    if not enrich.empty:
        df = df.merge(enrich, on="candidate_id", how="left")

    df["group"] = group
    df["seed"] = seed
    df["run_id"] = run_id
    if "external_project" not in df.columns:
        df["external_project"] = df.apply(infer_project, axis=1)
    else:
        df["external_project"] = df["external_project"].fillna(df.apply(infer_project, axis=1))
    if "decoding_config_name" not in df.columns:
        df["decoding_config_name"] = df.get("candidate_test_origin_detail", "unknown")
    df["decoding_config_name"] = df["decoding_config_name"].fillna("unknown")
    if "failure_log_count" not in df.columns:
        df["failure_log_count"] = 0
    if "execution_pass" not in df.columns:
        df["execution_pass"] = 0
    for _out_col, source_col in METRICS:
        if source_col in df.columns:
            df[source_col] = pd.to_numeric(df[source_col], errors="coerce")
    return df


def summarize_candidates(df: pd.DataFrame, dims: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = df.groupby(dims, dropna=False) if dims else [((), df)]
    for key, sub in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        rec: dict[str, Any] = {dim: value for dim, value in zip(dims, key)}
        rec["n_candidates"] = int(len(sub))
        rec["n_tasks"] = int(sub["task_id"].nunique()) if "task_id" in sub.columns else None
        rec["n_projects"] = int(sub["external_project"].nunique()) if "external_project" in sub.columns else None
        for out_col, source_col in METRICS:
            values = pd.to_numeric(sub[source_col], errors="coerce") if source_col in sub.columns else pd.Series(dtype=float)
            rec[out_col] = float(values.mean()) if values.notna().any() else None
            rec[f"{out_col}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def aggregate_run_summaries(root: Path, specs: list[tuple[str, str, str]]) -> pd.DataFrame:
    rows = []
    for group, seed, run_id in specs:
        row = load_summary(root, run_id)
        row["group"] = group
        row["seed"] = seed
        row["run_id"] = run_id
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_group_summaries(run_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [out_col for out_col, _source_col in METRICS]
    rows = []
    for group, sub in run_df.groupby("group", dropna=False):
        rec: dict[str, Any] = {"group": group, "n_runs": int(len(sub))}
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            rec[f"{col}_mean"] = float(values.mean()) if values.notna().any() else None
            rec[f"{col}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def load_failure_logs(root: Path, specs: list[tuple[str, str, str]], candidate_meta: pd.DataFrame) -> pd.DataFrame:
    meta_cols = ["candidate_id", "external_project"]
    meta = candidate_meta[meta_cols].drop_duplicates("candidate_id") if not candidate_meta.empty else pd.DataFrame(columns=meta_cols)
    rows = []
    for group, seed, run_id in specs:
        path = root / "data" / "interim" / "failuretaxonomylogs" / run_id / "failuretaxonomylogs.jsonl"
        for row in load_jsonl(path):
            row["group"] = group
            row["seed"] = seed
            row["run_id"] = run_id
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["group", "seed", "run_id", "candidate_id", "external_project", "failure_category"])
    df = pd.DataFrame(rows)
    if not meta.empty:
        df = df.merge(meta, on="candidate_id", how="left", suffixes=("", "_meta"))
        if "external_project_meta" in df.columns:
            df["external_project"] = df.get("external_project").fillna(df["external_project_meta"])
            df = df.drop(columns=["external_project_meta"])
    if "external_project" not in df.columns:
        df["external_project"] = "unknown"
    return df


def count_table(df: pd.DataFrame, dims: list[str], value_col: str, count_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=dims + [value_col, count_col])
    return df.groupby(dims + [value_col], dropna=False).size().reset_index(name=count_col)


def project_variance(project_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, sub in project_summary.groupby("group", dropna=False):
        rec: dict[str, Any] = {"group": group, "n_projects": int(sub["external_project"].nunique())}
        for out_col, _source_col in METRICS:
            values = pd.to_numeric(sub[out_col], errors="coerce")
            rec[f"{out_col}_project_mean"] = float(values.mean()) if values.notna().any() else None
            rec[f"{out_col}_project_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
            rec[f"{out_col}_project_min"] = float(values.min()) if values.notna().any() else None
            rec[f"{out_col}_project_max"] = float(values.max()) if values.notna().any() else None
        rows.append(rec)
    return pd.DataFrame(rows)


def bootstrap_cluster_effects(
    df: pd.DataFrame,
    *,
    baseline_group: str,
    cluster_col: str,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    if cluster_col not in df.columns:
        return pd.DataFrame()
    clusters = sorted(x for x in df[cluster_col].dropna().unique())
    groups = sorted(g for g in df["group"].dropna().unique() if g != baseline_group)
    rows: list[dict[str, Any]] = []
    if len(clusters) < 2 or not groups:
        return pd.DataFrame(columns=["group", "baseline_group", "metric", "cluster_unit", "observed_delta", "ci_low", "ci_high", "n_boot", "n_clusters", "bootstrap_seed"])

    cluster_frames = {cluster: df[df[cluster_col] == cluster] for cluster in clusters}
    rng = np.random.default_rng(seed)

    for metric_name, source_col in METRICS:
        if source_col not in df.columns:
            continue
        observed_by_group = df.groupby("group")[source_col].mean()
        baseline_observed = observed_by_group.get(baseline_group)
        for group in groups:
            group_observed = observed_by_group.get(group)
            observed_delta = None
            if pd.notna(group_observed) and pd.notna(baseline_observed):
                observed_delta = float(group_observed - baseline_observed)

            deltas: list[float] = []
            for _ in range(n_boot):
                sampled = rng.choice(clusters, size=len(clusters), replace=True)
                sample = pd.concat([cluster_frames[c] for c in sampled], ignore_index=True)
                means = sample.groupby("group")[source_col].mean()
                base_value = means.get(baseline_group)
                group_value = means.get(group)
                if pd.notna(base_value) and pd.notna(group_value):
                    deltas.append(float(group_value - base_value))
            rows.append(
                {
                    "group": group,
                    "baseline_group": baseline_group,
                    "metric": metric_name,
                    "cluster_unit": cluster_col,
                    "observed_delta": observed_delta,
                    "ci_low": float(np.quantile(deltas, 0.025)) if deltas else None,
                    "ci_high": float(np.quantile(deltas, 0.975)) if deltas else None,
                    "n_boot": n_boot,
                    "n_clusters": len(clusters),
                    "bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/root/autodl-tmp/utplm")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--run", action="append", required=True, help="group:seed:run_id")
    parser.add_argument("--baseline_group", default="base")
    parser.add_argument("--bootstrap_n", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=20260620)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "outputs" / "evaluations" / "external"
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [parse_run_spec(spec) for spec in args.run]

    run_df = aggregate_run_summaries(root, specs)
    run_df.to_csv(out_dir / "externalallrunsummaries.csv", index=False)
    group_agg = aggregate_group_summaries(run_df)
    group_agg.to_csv(out_dir / "externalgroupaggregate.csv", index=False)

    candidates = pd.concat([load_run_candidates(root, group, seed, run_id) for group, seed, run_id in specs], ignore_index=True)
    candidates.to_csv(out_dir / "externalallcandidates.csv", index=False)

    project_summary = summarize_candidates(candidates, ["group", "seed", "run_id", "external_project"])
    project_summary.to_csv(out_dir / "externalprojectbreakdown.csv", index=False)
    group_project_summary = summarize_candidates(candidates, ["group", "external_project"])
    group_project_summary.to_csv(out_dir / "externalgroupprojectaggregate.csv", index=False)
    project_variance(group_project_summary).to_csv(out_dir / "externalprojectvariance.csv", index=False)

    decoding_dims = ["group", "seed", "run_id", "external_project", "decoding_config_name"]
    summarize_candidates(candidates, decoding_dims).to_csv(out_dir / "externaldecodingbreakdown.csv", index=False)

    failure_logs = load_failure_logs(root, specs, candidates)
    failure_logs.to_csv(out_dir / "externalfailurelogsenriched.csv", index=False)
    count_table(failure_logs, ["group", "run_id", "external_project"], "failure_category", "count").to_csv(
        out_dir / "externalfailurecategorycounts.csv", index=False
    )
    count_table(candidates, ["group", "run_id", "external_project"], "mutation_status", "count").to_csv(
        out_dir / "externalmutationstatuscounts.csv", index=False
    )

    bootstrap_task = bootstrap_cluster_effects(
        candidates,
        baseline_group=args.baseline_group,
        cluster_col="task_id",
        n_boot=args.bootstrap_n,
        seed=args.bootstrap_seed,
    )
    bootstrap_task.to_csv(out_dir / "externalbootstraptaskcluster.csv", index=False)
    bootstrap_project = bootstrap_cluster_effects(
        candidates,
        baseline_group=args.baseline_group,
        cluster_col="external_project",
        n_boot=args.bootstrap_n,
        seed=args.bootstrap_seed + 1,
    )
    bootstrap_project.to_csv(out_dir / "externalbootstrapprojectcluster.csv", index=False)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(root),
        "out_dir": str(out_dir),
        "runs": args.run,
        "n_runs": len(specs),
        "n_candidates": int(len(candidates)),
        "n_tasks": int(candidates["task_id"].nunique()) if "task_id" in candidates.columns else None,
        "n_projects": int(candidates["external_project"].nunique()) if "external_project" in candidates.columns else None,
        "baseline_group": args.baseline_group,
        "bootstrap_n": args.bootstrap_n,
        "outputs": {
            "all_run_summaries": str(out_dir / "externalallrunsummaries.csv"),
            "group_aggregate": str(out_dir / "externalgroupaggregate.csv"),
            "all_candidates": str(out_dir / "externalallcandidates.csv"),
            "project_breakdown": str(out_dir / "externalprojectbreakdown.csv"),
            "group_project_aggregate": str(out_dir / "externalgroupprojectaggregate.csv"),
            "project_variance": str(out_dir / "externalprojectvariance.csv"),
            "decoding_breakdown": str(out_dir / "externaldecodingbreakdown.csv"),
            "failure_category_counts": str(out_dir / "externalfailurecategorycounts.csv"),
            "mutation_status_counts": str(out_dir / "externalmutationstatuscounts.csv"),
            "bootstrap_task_cluster": str(out_dir / "externalbootstraptaskcluster.csv"),
            "bootstrap_project_cluster": str(out_dir / "externalbootstrapprojectcluster.csv"),
        },
    }
    (out_dir / "externalmanifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
