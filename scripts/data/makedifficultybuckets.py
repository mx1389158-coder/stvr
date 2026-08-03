import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
import os
from typing import Any, Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()

UNIFIED_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unifiedpool"
    / "unifiedtaskpoolv1.jsonl"
)

SPLIT_MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "v1cleaned"
    / "splitmanifestv1.json"
)

OUT_JSONL_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "difficultybuckets"
    / "difficultybucketsv1.jsonl"
)

OUT_REPORT_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "datasetversions"
    / "difficultybucketsv1report.json"
)

DIFFICULTY_VERSION = "v1"
DIFFICULTY_METHOD = "train_pool_fitted_weighted_zscore_quantile_binning"
SOURCE_POOL_VERSION = "unified_task_pool_v1"
SPLIT_MANIFEST_VERSION = "v1cleaned"

FEATURE_KEYS = [
    "prompt_length",
    "prompt_line_count",
    "loc",
    "arg_count",
    "base_test_count",
]

WEIGHTS = {
    "prompt_length": 0.30,
    "prompt_line_count": 0.15,
    "loc": 0.30,
    "arg_count": 0.10,
    "base_test_count": 0.15,
}

BUCKET_LABELS = ("easy", "medium", "hard")


# =========================================================
# 基础 IO
# =========================================================

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")

    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error("JSON 解析错误 - 文件: %s, 行号: %d, 错误: %s", path, line_num, e)
                    raise

    logger.info("已加载 JSONL: %d 条 - %s", len(items), path)
    return items


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("已写入 %d 条 JSONL: %s", len(items), path)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info("已写入 JSON: %s", path)


# =========================================================
# 校验
# =========================================================

def validate_weights(weights: Dict[str, float], feature_keys: List[str]) -> List[Dict[str, Any]]:
    errors: List[Dict[str, Any]] = []

    for key in feature_keys:
        if key not in weights:
            errors.append({"error_type": "missing_weight", "detail": key})

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-8:
        errors.append({"error_type": "weights_do_not_sum_to_one", "detail": total})

    return errors


def validate_unified_items(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if not items:
        errors.append({"task_id": None, "error_type": "empty_unifiedpool", "detail": None})
        return errors, warnings

    seen = set()
    for idx, item in enumerate(items, start=1):
        task_id = item.get("task_id", f"[index={idx}]")

        if task_id in seen:
            errors.append({"task_id": task_id, "error_type": "duplicate_task_id", "detail": None})
        seen.add(task_id)

        diff = item.get("difficulty_features")
        if not isinstance(diff, dict):
            errors.append({"task_id": task_id, "error_type": "missing_difficulty_features", "detail": None})
            continue

        for key in FEATURE_KEYS:
            if key not in diff:
                errors.append({"task_id": task_id, "error_type": f"missing_{key}", "detail": None})
            else:
                val = diff[key]
                if val is None:
                    warnings.append({"task_id": task_id, "warning_type": f"null_{key}", "detail": None})
                elif not isinstance(val, (int, float)):
                    errors.append({"task_id": task_id, "error_type": f"invalid_type_{key}", "detail": str(type(val))})

    return errors, warnings


def validate_split_manifest(
    manifest: Dict[str, Any],
    unified_items: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    required_keys = ["train_pool", "main_test", "capability_retention"]
    for key in required_keys:
        if key not in manifest:
            errors.append({"error_type": "missing_split_manifest_key", "detail": key})

    if errors:
        return errors, warnings

    for key in required_keys:
        if not isinstance(manifest[key], list):
            errors.append({"error_type": "split_ids_not_list", "detail": key})

    if errors:
        return errors, warnings

    train_set = set(manifest["train_pool"])
    main_set = set(manifest["main_test"])
    cap_set = set(manifest["capability_retention"])

    intersections = {
        "train_intersect_main": len(train_set & main_set),
        "train_intersect_capability": len(train_set & cap_set),
        "main_intersect_capability": len(main_set & cap_set)
    }

    for k, v in intersections.items():
        if v != 0:
            errors.append({"error_type": "split_overlap_detected", "detail": {k: v}})

    unified_ids = {x["task_id"] for x in unified_items}
    split_union = train_set | main_set | cap_set

    missing = unified_ids - split_union
    extra = split_union - unified_ids

    if missing:
        errors.append({"error_type": "split_manifest_missing_ids", "detail": sorted(list(missing))[:20]})

    if extra:
        errors.append({"error_type": "split_manifest_extra_ids", "detail": sorted(list(extra))[:20]})

    return errors, warnings


# =========================================================
# 特征拟合与打分
# =========================================================

def build_taskid_to_split(split_manifest: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for split_name in ["train_pool", "main_test", "capability_retention"]:
        for task_id in split_manifest.get(split_name, []):
            mapping[task_id] = split_name
    return mapping


def fit_feature_statistics(
    train_items: List[Dict[str, Any]],
    feature_keys: List[str]
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, Dict[str, int]]]:
    """
    仅用 train_pool 拟合中位数、均值、标准差。
    """
    medians: Dict[str, float] = {}
    stats: Dict[str, Dict[str, float]] = {}
    missing_report: Dict[str, Dict[str, int]] = {}

    train_features = {k: [] for k in feature_keys}
    for item in train_items:
        diff = item.get("difficulty_features") or {}
        for k in feature_keys:
            train_features[k].append(diff.get(k))

    for key in feature_keys:
        raw_vals = train_features[key]
        non_null_vals = [float(v) for v in raw_vals if isinstance(v, (int, float))]

        missing_count = len(raw_vals) - len(non_null_vals)
        missing_report[key] = {
            "train_total": len(raw_vals),
            "train_non_null": len(non_null_vals),
            "train_missing": missing_count
        }

        if not non_null_vals:
            med, mean, std = 0.0, 0.0, 1.0
        else:
            arr = np.array(non_null_vals, dtype=float)
            med = float(np.median(arr))
            mean = float(arr.mean())
            std = float(arr.std())
            if std == 0.0:
                std = 1.0

        medians[key] = med
        stats[key] = {
            "median": med,
            "mean": mean,
            "std": std
        }

    return medians, stats, missing_report


def assign_bucket(score: float, q1: float, q2: float) -> str:
    if score <= q1:
        return BUCKET_LABELS[0]
    elif score <= q2:
        return BUCKET_LABELS[1]
    return BUCKET_LABELS[2]


def compute_difficultybuckets(
    unified_items: List[Dict[str, Any]],
    split_manifest: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    核心逻辑：
    1. 仅用 train_pool 拟合中位数、均值、标准差、分位阈值。
    2. 将规则应用到全部样本。
    """
    taskid_to_split = build_taskid_to_split(split_manifest)

    train_items = [x for x in unified_items if taskid_to_split.get(x["task_id"]) == "train_pool"]
    if not train_items:
        raise ValueError("train_pool 为空，无法拟合难度分数。")

    medians, stats, missing_report = fit_feature_statistics(train_items, FEATURE_KEYS)

    score_params = {
        key: (stats[key]["mean"], stats[key]["std"], WEIGHTS[key])
        for key in FEATURE_KEYS
    }

    all_scored_data = []
    train_scores = []

    for item in unified_items:
        split = taskid_to_split.get(item["task_id"])
        if split is None:
            raise ValueError(f"任务 {item['task_id']} 未出现在 split_manifest 中")

        diff = item.get("difficulty_features") or {}
        values: Dict[str, float] = {}
        was_imputed: Dict[str, bool] = {}
        score = 0.0

        for key in FEATURE_KEYS:
            raw = diff.get(key)
            if raw is None or not isinstance(raw, (int, float)):
                val = medians[key]
                was_imputed[key] = True
            else:
                val = float(raw)
                was_imputed[key] = False

            values[key] = val
            mean, std, weight = score_params[key]
            score += weight * ((val - mean) / std)

        if split == "train_pool":
            train_scores.append(score)

        all_scored_data.append({
            "item": item,
            "values": values,
            "was_imputed": was_imputed,
            "score": score,
            "split": split
        })

    q1 = float(np.quantile(train_scores, 0.33))
    q2 = float(np.quantile(train_scores, 0.67))

    out: List[Dict[str, Any]] = []
    for data in all_scored_data:
        bucket = assign_bucket(data["score"], q1, q2)
        item = data["item"]
        out.append({
            "task_id": item["task_id"],
            "source": item["source"],
            "split": data["split"],
            "difficulty_bucket": bucket,
            "difficulty_score": float(data["score"]),
            "difficulty_features": data["values"],
            "imputation_flags": data["was_imputed"],
            "metadata": {
                "difficulty_version": DIFFICULTY_VERSION,
                "difficulty_method": DIFFICULTY_METHOD,
                "fitted_on": "train_pool",
                "source_pool_version": SOURCE_POOL_VERSION,
                "split_manifest_version": SPLIT_MANIFEST_VERSION,
                "weights": WEIGHTS,
                "quantiles": {"q1": q1, "q2": q2}
            }
        })

    fitting_info = {
        "feature_medians_from_train_pool": medians,
        "feature_stats_from_train_pool": stats,
        "missing_report": missing_report,
        "quantiles_from_train_pool": {"q1": q1, "q2": q2}
    }

    return out, fitting_info


# =========================================================
# 分布统计
# =========================================================

def build_bucket_distribution(items: List[Dict[str, Any]]) -> Dict[str, int]:
    return dict(Counter(x["difficulty_bucket"] for x in items))


def build_source_bucket_distribution(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out = defaultdict(Counter)
    for item in items:
        out[item["source"]][item["difficulty_bucket"]] += 1
    return {k: dict(v) for k, v in out.items()}


def build_split_bucket_distribution(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out = defaultdict(Counter)
    for item in items:
        split = item.get("split", "unknown")
        out[split][item["difficulty_bucket"]] += 1
    return {k: dict(v) for k, v in out.items()}


def build_source_split_bucket_distribution(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    out = defaultdict(lambda: defaultdict(Counter))
    for item in items:
        source = item["source"]
        split = item.get("split", "unknown")
        bucket = item["difficulty_bucket"]
        out[source][split][bucket] += 1

    return {
        source: {split: dict(counter) for split, counter in split_map.items()}
        for source, split_map in out.items()
    }


# =========================================================
# 报告
# =========================================================

def build_report(
    unified_items: List[Dict[str, Any]],
    bucket_items: List[Dict[str, Any]],
    validation_errors: List[Dict[str, Any]],
    validation_warnings: List[Dict[str, Any]],
    split_manifest_errors: List[Dict[str, Any]],
    split_manifest_warnings: List[Dict[str, Any]],
    fitting_info: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "difficulty_version": DIFFICULTY_VERSION,
        "difficulty_method": DIFFICULTY_METHOD,
        "source_pool_version": SOURCE_POOL_VERSION,
        "split_manifest_version": SPLIT_MANIFEST_VERSION,
        "input_paths": {
            "unifiedpool": str(UNIFIED_PATH),
            "split_manifest": str(SPLIT_MANIFEST_PATH)
        },
        "total_unified_items": len(unified_items),
        "total_bucket_items": len(bucket_items),

        "validation_error_count": len(validation_errors),
        "validation_warning_count": len(validation_warnings),
        "split_manifest_error_count": len(split_manifest_errors),
        "split_manifest_warning_count": len(split_manifest_warnings),

        "validation_errors": validation_errors[:50],
        "validation_warnings": validation_warnings[:50],
        "split_manifest_errors": split_manifest_errors[:50],
        "split_manifest_warnings": split_manifest_warnings[:50],

        "overall_source_distribution": dict(Counter(x["source"] for x in unified_items)),
        "overall_bucket_distribution": build_bucket_distribution(bucket_items),
        "source_bucket_distribution": build_source_bucket_distribution(bucket_items),
        "split_bucket_distribution": build_split_bucket_distribution(bucket_items),
        "source_split_bucket_distribution": build_source_split_bucket_distribution(bucket_items),

        "fitting_info": fitting_info,

        "note": (
            "当前 difficultybuckets_v1 使用 train_pool 拟合均值、标准差和分位阈值，"
            "再将该规则应用到 main_test 与 capability_retention。"
            "若后续发现 source 与 difficulty 强耦合，可在 v2 中升级为 source-aware 分桶。"
        )
    }


# =========================================================
# 主函数
# =========================================================

def main() -> None:
    logger.info("开始生成 difficultybuckets_v1 ...")

    weight_errors = validate_weights(WEIGHTS, FEATURE_KEYS)
    if weight_errors:
        raise ValueError(f"WEIGHTS 配置非法: {weight_errors}")

    unified_items = load_jsonl(UNIFIED_PATH)
    split_manifest = load_json(SPLIT_MANIFEST_PATH)

    validation_errors, validation_warnings = validate_unified_items(unified_items)
    split_manifest_errors, split_manifest_warnings = validate_split_manifest(split_manifest, unified_items)

    if validation_errors or split_manifest_errors:
        report = build_report(
            unified_items=unified_items,
            bucket_items=[],
            validation_errors=validation_errors,
            validation_warnings=validation_warnings,
            split_manifest_errors=split_manifest_errors,
            split_manifest_warnings=split_manifest_warnings,
            fitting_info={}
        )
        save_json(report, OUT_REPORT_PATH)
        raise ValueError(
            f"输入校验失败: unified_errors={len(validation_errors)}, "
            f"split_manifest_errors={len(split_manifest_errors)}"
        )

    bucket_items, fitting_info = compute_difficultybuckets(unified_items, split_manifest)

    save_jsonl(bucket_items, OUT_JSONL_PATH)

    report = build_report(
        unified_items=unified_items,
        bucket_items=bucket_items,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
        split_manifest_errors=split_manifest_errors,
        split_manifest_warnings=split_manifest_warnings,
        fitting_info=fitting_info
    )
    save_json(report, OUT_REPORT_PATH)

    logger.info("difficultybuckets_v1 生成完成。")
    logger.info("总任务数: %d", len(bucket_items))
    logger.info("整体难度分布: %s", build_bucket_distribution(bucket_items))


if __name__ == "__main__":
    main()
