from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path("/root/autodl-tmp/utplm")
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_POOL = PROJECT_ROOT / "data" / "splits" / "v1cleaned" / "trainpoolv1.jsonl"

# v2 输出
OUT_DIFF_JSONL = OUT_DIR / "difficultybucketsv2sourcewise.jsonl"
OUT_DIFF_REPORT = OUT_DIR / "difficultybucketsv2sourcewisereport.json"
OUT_CSV = OUT_DIR / "a1formaltasksubsetv2.csv"
OUT_JSON = OUT_DIR / "a1formaltasksubsetreportv2.json"

SEED = 42
VALID_SOURCES = {"mbpp", "humaneval"}
BUCKET_ORDER = ["easy", "medium", "hard"]

# v2 建议配额：每桶 18
TARGET_PER_BUCKET = 18


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_sourcewise_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """
    用 source 内 task_id 的排序位置做近似 source-wise bucket。
    这里不是“真实难度预测器”，而是为了在现有数据条件下，
    先把 mbpp / humaneval 各自稳定切成 easy / medium / hard，
    解决 v1 中跨 source bucket 失衡的问题。
    """
    out_parts = []

    for source, sub in df.groupby("source", observed=True):
        sub = sub.copy().sort_values("task_id", kind="stable").reset_index(drop=True)
        n = len(sub)
        if n == 0:
            continue

        ranks = pd.Series(range(n), index=sub.index, dtype=float)
        frac = (ranks + 0.5) / n

        def bucket_of(x: float) -> str:
            if x < 1 / 3:
                return "easy"
            elif x < 2 / 3:
                return "medium"
            else:
                return "hard"

        sub["difficulty_bucket"] = frac.map(bucket_of)
        out_parts.append(sub)

    out = pd.concat(out_parts, ignore_index=True) if out_parts else pd.DataFrame(columns=df.columns.tolist() + ["difficulty_bucket"])
    out["difficulty_bucket"] = pd.Categorical(out["difficulty_bucket"], categories=BUCKET_ORDER, ordered=True)
    return out


def main() -> None:
    if not TRAIN_POOL.exists():
        raise FileNotFoundError(f"missing input file: {TRAIN_POOL}")

    train = pd.DataFrame(load_jsonl(TRAIN_POOL))

    required_train_cols = {"task_id", "source", "prompt"}
    missing_train = required_train_cols - set(train.columns)
    if missing_train:
        raise ValueError(f"train_pool 缺少字段: {missing_train}")

    if train["task_id"].duplicated().any():
        dup_ids = train.loc[train["task_id"].duplicated(keep=False), "task_id"].tolist()[:10]
        raise ValueError(f"train_pool 中存在重复 task_id，例如: {dup_ids}")

    invalid_sources = sorted(set(train["source"].dropna().astype(str)) - VALID_SOURCES)
    if invalid_sources:
        raise ValueError(f"train_pool 中存在不支持的 source: {invalid_sources}")

    train = train[train["source"].isin(VALID_SOURCES)].copy()

    keep_cols = [
        c for c in [
            "task_id",
            "source",
            "prompt",
            "entry_point",
            "canonical_solution",
            "base_tests",
            "imports",
            "signature",
            "task_type",
            "eval_assets",
        ]
        if c in train.columns
    ]
    train = train[keep_cols].copy()

    # -------------------------------------------------
    # 1) source-wise difficulty buckets
    # -------------------------------------------------
    diff_df = infer_sourcewise_difficulty(train[["task_id", "source"]].copy())

    # 写出中间 difficulty 结果，便于审计与复现
    diff_out = diff_df[["task_id", "source", "difficulty_bucket"]].copy()
    with open(OUT_DIFF_JSONL, "w", encoding="utf-8") as f:
        for _, row in diff_out.iterrows():
            rec = {
                "task_id": row["task_id"],
                "source": row["source"],
                "difficulty_bucket": str(row["difficulty_bucket"]),
                "difficulty_bucket_version": "v2_sourcewise_proxy",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    diff_report = {
        "difficulty_bucket_version": "v2_sourcewise_proxy",
        "bucket_order": BUCKET_ORDER,
        "valid_sources": sorted(VALID_SOURCES),
        "counts": (
            diff_out.groupby(["source", "difficulty_bucket"], observed=False)
            .size()
            .rename("n")
            .reset_index()
            .to_dict(orient="records")
        ),
        "notes": [
            "Buckets are assigned source-wise rather than globally.",
            "This is a controlled balancing proxy for a1_v2 finalization.",
            "The immediate goal is to reduce structural shortfall across source × difficulty buckets.",
        ],
    }
    OUT_DIFF_REPORT.write_text(json.dumps(diff_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # -------------------------------------------------
    # 2) merge back and sample formal subset v2
    # -------------------------------------------------
    df = train.merge(
        diff_out[["task_id", "difficulty_bucket"]],
        on="task_id",
        how="left",
        validate="one_to_one",
    )

    missing_diff_mask = df["difficulty_bucket"].isna()
    if missing_diff_mask.any():
        missing_ids = df.loc[missing_diff_mask, "task_id"].tolist()
        raise ValueError(f"有 {missing_diff_mask.sum()} 个任务缺少 difficulty_bucket，例如: {missing_ids[:10]}")

    df["difficulty_bucket"] = pd.Categorical(df["difficulty_bucket"], categories=BUCKET_ORDER, ordered=True)

    sampled_parts = []
    bucket_report = []

    for source in ["mbpp", "humaneval"]:
        for difficulty in BUCKET_ORDER:
            group = df[(df["source"] == source) & (df["difficulty_bucket"] == difficulty)].copy()
            available = len(group)
            n_sample = min(TARGET_PER_BUCKET, available)

            if n_sample > 0:
                sampled = group.sample(n=n_sample, random_state=SEED)
                sampled_parts.append(sampled)

            bucket_report.append(
                {
                    "source": source,
                    "difficulty_bucket": difficulty,
                    "available": int(available),
                    "sampled": int(n_sample),
                    "target": int(TARGET_PER_BUCKET),
                    "shortfall": int(max(0, TARGET_PER_BUCKET - n_sample)),
                }
            )

    out = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame(columns=df.columns)
    out["subset_name"] = "a1_formal_task_subset_v2"
    out = out.sort_values(["source", "difficulty_bucket", "task_id"], kind="stable").reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    report = {
        "subset_name": "a1_formal_task_subset_v2",
        "difficulty_bucket_version": "v2_sourcewise_proxy",
        "target_per_bucket": TARGET_PER_BUCKET,
        "seed": SEED,
        "valid_sources": sorted(VALID_SOURCES),
        "bucket_order": BUCKET_ORDER,
        "total_sampled_tasks": int(len(out)),
        "bucket_report": bucket_report,
        "source_counts": out["source"].value_counts(dropna=False).to_dict() if "source" in out.columns else {},
        "difficulty_counts": out["difficulty_bucket"].value_counts(dropna=False).to_dict() if "difficulty_bucket" in out.columns else {},
        "sampled_task_ids_preview": out["task_id"].head(20).tolist() if "task_id" in out.columns else [],
        "notes": [
            "This is the v2 formal subset for a1 finalization.",
            "Source-wise difficulty bucketing is used to mitigate v1 structural imbalance.",
            "Target per bucket is reduced from 30 to 18 to make the bucket design feasible.",
            "main_test and capability_retention are intentionally excluded.",
        ],
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[ok] wrote {OUT_DIFF_JSONL}")
    print(f"[ok] wrote {OUT_DIFF_REPORT}")
    print(f"[ok] wrote {OUT_CSV}")
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[tasks] {len(out)}")
    print("\n[source x difficulty]")
    print(out.groupby(["source", "difficulty_bucket"], observed=False).size())


if __name__ == "__main__":
    main()