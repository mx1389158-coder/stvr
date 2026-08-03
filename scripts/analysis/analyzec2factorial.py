from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

def infer_project_root() -> Path:
    env_root = os.environ.get("UTPLM_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "outputs").exists() and ((parent / "repo").exists() or (parent / "scripts").exists()):
            return parent
    return Path("/root/autodl-tmp/utplm")


PROJECT_ROOT = infer_project_root()
IN_CSV = PROJECT_ROOT / "outputs" / "evaluations" / "c1" / "c1allrunsummaries.csv"
OUT_DIR = PROJECT_ROOT / "outputs" / "evaluations" / "c2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    "execution_pass_rate",
    "mutation_score_mean",
    "boundary_coverage_mean",
    "failure_log_count_mean",
    "execution_pass_x_mutation_score_mean",
    "execution_pass_x_boundary_coverage_mean",
]

PLOT_METRICS = [
    "execution_pass_rate",
    "mutation_score_mean",
    "boundary_coverage_mean",
    "failure_log_count_mean",
]

BOOTSTRAP_N = 20000
BOOTSTRAP_SEED = 20260609

METRIC_LABELS = {
    "execution_pass_rate": "Execution pass rate",
    "mutation_score_mean": "Mutation score",
    "boundary_coverage_mean": "Boundary coverage",
    "failure_log_count_mean": "Failure log count",
    "execution_pass_x_mutation_score_mean": "Pass x mutation",
    "execution_pass_x_boundary_coverage_mean": "Pass x boundary",
}

ATTEMPT_LEVEL_SOURCES = {
    "execution_pass_x_mutation_score_mean": "mutation_score",
    "execution_pass_x_boundary_coverage_mean": "boundary_coverage",
}

POS_LABELS = {
    0: "Weak positive",
    1: "High-quality positive",
}

NEG_LABELS = {
    0: "Low-info negative",
    1: "High-info negative",
}

PLOT_COLORS = {
    0: "#3B5B92",
    1: "#8F4a3A",
}

PLOT_LINESTYLES = {
    0: "-",
    1: "--",
}

PLOT_MARKERS = {
    0: "o",
    1: "s",
}

GROUP_MAP = {
    "LL": {"pos_high": 0, "neg_high": 0},
    "HL": {"pos_high": 1, "neg_high": 0},
    "LH": {"pos_high": 0, "neg_high": 1},
    "HH": {"pos_high": 1, "neg_high": 1},
}


def resolve_per_candidate_path(run_id: str, group: str, seed: object) -> Path:
    run_ids = [run_id]
    if group in GROUP_MAP and pd.notna(seed):
        try:
            seed_int = int(seed)
        except (TypeError, ValueError):
            seed_int = None
        if seed_int is not None:
            run_ids.append(f"c1_{group}_seed{seed_int}_main")

    checked = []
    for candidate_run_id in dict.fromkeys(run_ids):
        per_candidate = PROJECT_ROOT / "outputs" / "evaluations" / "b1" / candidate_run_id / "percandidate.csv"
        checked.append(str(per_candidate))
        if per_candidate.exists():
            return per_candidate
    raise FileNotFoundError(
        f"missing per-candidate evaluation file for run_id={run_id}, group={group}, seed={seed}; "
        f"checked: {checked}"
    )


def compute_attempt_level_metrics(run_id: str, group: str, seed: object) -> Dict[str, float]:
    per_candidate = resolve_per_candidate_path(run_id, group, seed)

    required = {"execution_pass", *ATTEMPT_LEVEL_SOURCES.values()}
    df = pd.read_csv(per_candidate, usecols=lambda col: col in required)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {per_candidate}: {sorted(missing)}")

    execution = pd.to_numeric(df["execution_pass"], errors="coerce").fillna(0)
    out: Dict[str, float] = {}
    for metric, source_col in ATTEMPT_LEVEL_SOURCES.items():
        values = pd.to_numeric(df[source_col], errors="coerce").fillna(0)
        out[metric] = float((execution * values).mean())
    return out


def ensure_attempt_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for metric in ATTEMPT_LEVEL_SOURCES:
        if metric not in df.columns:
            df[metric] = np.nan

    cache: Dict[str, Dict[str, float]] = {}
    for idx, row in df.iterrows():
        missing_metrics = [metric for metric in ATTEMPT_LEVEL_SOURCES if pd.isna(row.get(metric))]
        if not missing_metrics:
            continue

        run_id = str(row.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(f"missing run_id for row requiring attempt-level metrics: index={idx}")
        cache_key = f"{run_id}|{row.get('group', '')}|{row.get('seed', '')}"
        if cache_key not in cache:
            cache[cache_key] = compute_attempt_level_metrics(run_id, str(row.get("group", "")).strip(), row.get("seed"))
        for metric in missing_metrics:
            df.at[idx, metric] = cache[cache_key][metric]

    return df


def bootstrap_seed_cluster_effects(
    df: pd.DataFrame,
    metrics: List[str],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """Bootstrap marginal factorial effects by resampling random seeds.

    The c1/c2 design evaluates all four factorial cells under the same seed
    values. Resampling seeds preserves this paired structure and avoids treating
    the 12 group-seed summaries as fully independent observations.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted(df["seed"].dropna().unique())
    rows: List[Dict] = []

    for metric in metrics:
        sub = df[["group", "seed", metric]].dropna().copy()
        value_by_group_seed = {
            group: sub[sub["group"] == group].set_index("seed")[metric].to_dict()
            for group in ["LL", "HL", "LH", "HH"]
        }

        def compute_effects(seed_sample):
            cell = {
                group: float(np.mean([value_by_group_seed[group][s] for s in seed_sample]))
                for group in ["LL", "HL", "LH", "HH"]
            }
            return {
                "pos_main": ((cell["HL"] + cell["HH"]) / 2.0) - ((cell["LL"] + cell["LH"]) / 2.0),
                "neg_main": ((cell["LH"] + cell["HH"]) / 2.0) - ((cell["LL"] + cell["HL"]) / 2.0),
                "interaction": cell["HH"] - cell["HL"] - cell["LH"] + cell["LL"],
                "HH_minus_LL": cell["HH"] - cell["LL"],
            }

        observed = compute_effects(seeds)
        boot = {name: [] for name in observed}

        for _ in range(n_boot):
            sampled_seeds = list(rng.choice(seeds, size=len(seeds), replace=True))
            effects = compute_effects(sampled_seeds)
            for name, value in effects.items():
                boot[name].append(value)

        for name, estimate in observed.items():
            arr = np.asarray(boot[name], dtype=float)
            rows.append({
                "metric": metric,
                "effect": name,
                "estimate": float(estimate),
                "ci95_low": float(np.quantile(arr, 0.025)),
                "ci95_high": float(np.quantile(arr, 0.975)),
                "bootstrap_n": n_boot,
                "bootstrap_seed": seed,
                "resampling_unit": "seed_cluster",
            })

    return pd.DataFrame(rows)


def configure_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#2f3542",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "grid.color": "#d9dee7",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.85,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def save_interaction_plot(means: pd.DataFrame, metric: str, out_dir: Path) -> None:
    plot_df = means.copy()
    metric_label = METRIC_LABELS.get(metric, metric.replace("_", " ").title())

    fig, ax = plt.subplots(figsize=(4.6, 3.55))
    for pos_level in [0, 1]:
        part = plot_df[plot_df["pos_high"] == pos_level].sort_values("neg_high")
        x = part["neg_high"].astype(int).to_numpy()
        y = part["mean"].astype(float).to_numpy()
        std = part["std"].fillna(0).astype(float).to_numpy()
        count = part["count"].astype(float).to_numpy()
        ci95 = 1.96 * std / (count ** 0.5)

        ax.errorbar(
            x,
            y,
            yerr=ci95,
            marker=PLOT_MARKERS[pos_level],
            linestyle=PLOT_LINESTYLES[pos_level],
            markersize=5,
            linewidth=2,
            capsize=3,
            capthick=1,
            color=PLOT_COLORS[pos_level],
            label=POS_LABELS[pos_level],
        )

    ax.set_xticks([0, 1], [NEG_LABELS[0], NEG_LABELS[1]])
    ax.set_xlabel("Negative factor")
    ax.set_ylabel(metric_label)
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.12)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
        columnspacing=1.4,
        handlelength=2.2,
    )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    legacy_png = out_dir / f"c2_interaction_{metric}.png"
    clean_stub = metric.replace("_mean", "").replace("_rate", "").replace("_count", "")
    clean_png = out_dir / f"fig3_interaction_{clean_stub}.png"
    clean_svg = out_dir / f"fig3_interaction_{clean_stub}.svg"
    fig.savefig(legacy_png)
    fig.savefig(clean_png)
    fig.savefig(clean_svg)
    plt.close(fig)


def save_combined_interaction_plot(all_means: Dict[str, pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.7))
    axes = axes.flatten()

    handles = []
    labels = []
    for ax, metric in zip(axes, PLOT_METRICS):
        plot_df = all_means[metric].copy()
        for pos_level in [0, 1]:
            part = plot_df[plot_df["pos_high"] == pos_level].sort_values("neg_high")
            x = part["neg_high"].astype(int).to_numpy()
            y = part["mean"].astype(float).to_numpy()
            std = part["std"].fillna(0).astype(float).to_numpy()
            count = part["count"].astype(float).to_numpy()
            ci95 = 1.96 * std / (count ** 0.5)
            line = ax.errorbar(
                x,
                y,
                yerr=ci95,
                marker=PLOT_MARKERS[pos_level],
                linestyle=PLOT_LINESTYLES[pos_level],
                markersize=5,
                linewidth=2,
                capsize=3,
                capthick=1,
                color=PLOT_COLORS[pos_level],
                label=POS_LABELS[pos_level],
            )
            if metric == METRICS[0]:
                handles.append(line)
                labels.append(POS_LABELS[pos_level])

        ax.set_xticks([0, 1], [NEG_LABELS[0], NEG_LABELS[1]])
        ax.set_xlabel("Negative factor")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric.replace("_", " ").title()))
        ax.grid(axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.12)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        columnspacing=1.8,
        handlelength=2.4,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_dir / "fig3c2interactions.png")
    fig.savefig(out_dir / "fig3c2interactions.svg")
    plt.close(fig)


def main():
    if not IN_CSV.exists():
        raise FileNotFoundError(f"missing input: {IN_CSV}")

    configure_plot_style()

    df = pd.read_csv(IN_CSV)
    df = df[df["split_name"] == "main_test"].copy()
    df = df[df["group"].isin(GROUP_MAP.keys())].copy()
    df = ensure_attempt_level_metrics(df)

    for g, bits in GROUP_MAP.items():
        df.loc[df["group"] == g, "pos_high"] = bits["pos_high"]
        df.loc[df["group"] == g, "neg_high"] = bits["neg_high"]

    df["pos_high"] = pd.to_numeric(df["pos_high"], errors="coerce").astype(int)
    df["neg_high"] = pd.to_numeric(df["neg_high"], errors="coerce").astype(int)

    coef_rows: List[Dict] = []
    anova_rows: List[Dict] = []
    effect_rows: List[Dict] = []
    mean_rows: List[Dict] = []
    plot_means: Dict[str, pd.DataFrame] = {}

    for metric in METRICS:
        y = pd.to_numeric(df[metric], errors="coerce")
        sub = df.loc[y.notna(), ["group", "seed", "pos_high", "neg_high", metric]].copy()

        if len(sub) == 0:
            continue

        model = smf.ols(f"{metric} ~ pos_high + neg_high + pos_high:neg_high", data=sub).fit()
        anova = sm.stats.anova_lm(model, typ=2)
        ci = model.conf_int()

        for term in model.params.index:
            coef_rows.append({
                "metric": metric,
                "term": term,
                "estimate": float(model.params[term]),
                "std_err": float(model.bse[term]),
                "t_value": float(model.tvalues[term]),
                "p_value": float(model.pvalues[term]),
                "ci95_low": float(ci.loc[term, 0]),
                "ci95_high": float(ci.loc[term, 1]),
            })

        ss_total = float(anova["sum_sq"].sum())
        for term, row in anova.iterrows():
            anova_rows.append({
                "metric": metric,
                "term": term,
                "sum_sq": float(row["sum_sq"]),
                "df": float(row["df"]),
                "F": float(row["F"]) if pd.notna(row["F"]) else None,
                "PR(>F)": float(row["PR(>F)"]) if pd.notna(row["PR(>F)"]) else None,
            })
            effect_rows.append({
                "metric": metric,
                "term": term,
                "eta_sq": float(row["sum_sq"] / ss_total) if ss_total > 0 else None,
            })

        means = (
            sub.groupby(["pos_high", "neg_high"], dropna=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        means["metric"] = metric
        mean_rows.extend(means.to_dict("records"))
        plot_means[metric] = means.copy()

        if metric in PLOT_METRICS:
            save_interaction_plot(means, metric, OUT_DIR)

    save_combined_interaction_plot(plot_means, OUT_DIR)

    coef_df = pd.DataFrame(coef_rows)
    anova_df = pd.DataFrame(anova_rows)
    effect_df = pd.DataFrame(effect_rows)
    means_df = pd.DataFrame(mean_rows)
    bootstrap_df = bootstrap_seed_cluster_effects(df, METRICS)

    coef_df.to_csv(OUT_DIR / "c2factorcoefficients.csv", index=False, encoding="utf-8")
    anova_df.to_csv(OUT_DIR / "c2anovatable.csv", index=False, encoding="utf-8")
    effect_df.to_csv(OUT_DIR / "c2effectsizes.csv", index=False, encoding="utf-8")
    means_df.to_csv(OUT_DIR / "c2groupmeans.csv", index=False, encoding="utf-8")
    bootstrap_df.to_csv(OUT_DIR / "c2bootstrapeffectsseedcluster.csv", index=False, encoding="utf-8")

    summary = {
        "input_csv": str(IN_CSV),
        "metrics": METRICS,
        "outputs": {
            "coefficients": str(OUT_DIR / "c2factorcoefficients.csv"),
            "anova": str(OUT_DIR / "c2anovatable.csv"),
            "effect_sizes": str(OUT_DIR / "c2effectsizes.csv"),
            "group_means": str(OUT_DIR / "c2groupmeans.csv"),
            "bootstrap_effects": str(OUT_DIR / "c2bootstrapeffectsseedcluster.csv"),
            "plots_dir": str(OUT_DIR),
        },
    }
    (OUT_DIR / "c2manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[ok] wrote", OUT_DIR / "c2factorcoefficients.csv")
    print("[ok] wrote", OUT_DIR / "c2anovatable.csv")
    print("[ok] wrote", OUT_DIR / "c2effectsizes.csv")
    print("[ok] wrote", OUT_DIR / "c2groupmeans.csv")
    print("[ok] wrote", OUT_DIR / "c2bootstrapeffectsseedcluster.csv")
    print("[ok] wrote", OUT_DIR / "c2manifest.json")


if __name__ == "__main__":
    main()
