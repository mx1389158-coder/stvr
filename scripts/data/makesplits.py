import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
import os
from typing import Any, Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()

IN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unifiedpool"
    / "unifiedtaskpoolv1.jsonl"
)

OUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "v1cleaned"
)

TRAIN_PATH = OUT_DIR / "trainpoolv1.jsonl"
MAIN_PATH = OUT_DIR / "maintestv1.jsonl"
CAP_PATH = OUT_DIR / "capabilityretentionv1.jsonl"
MANIFEST_PATH = OUT_DIR / "splitmanifestv1.json"

REPORT_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "datasetversions"
    / "splitmanifestv1report.json"
)

TRAIN_RATIO = 0.70
MAIN_RATIO = 0.20
CAPABILITY_RATIO = 0.10

SPLIT_SEED = 42

SUPPORTED_SOURCES = {"mbpp", "humaneval"}
SUPPORTED_MAIN_POOL_ROLE = "main_pool"


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
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    logger.info("已加载 unified pool: %d 条", len(items))
    return items


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

def validate_unifiedpool(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    针对切分阶段的最小必要校验：
    - task_id 唯一
    - source 合法
    - split 必须为 None
    - role 必须为 main_pool
    """
    seen_ids = set()
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if not items:
        errors.append({
            "index": None,
            "task_id": None,
            "error_type": "empty_unifiedpool",
            "detail": None
        })
        return errors, warnings

    for i, item in enumerate(items, start=1):
        task_id = item.get("task_id", f"[index={i}]")
        source = item.get("source")

        if task_id in seen_ids:
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "duplicate_task_id",
                "detail": None
            })
        seen_ids.add(task_id)

        if not isinstance(task_id, str) or not task_id.strip():
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "empty_task_id",
                "detail": None
            })

        if source not in SUPPORTED_SOURCES:
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "invalid_source",
                "detail": source
            })

        if item.get("split") is not None:
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "split_should_be_null_in_unifiedpool",
                "detail": item.get("split")
            })

        if item.get("role") != SUPPORTED_MAIN_POOL_ROLE:
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "role_should_be_main_pool_in_unifiedpool",
                "detail": item.get("role")
            })

        if not item.get("difficulty_features"):
            warnings.append({
                "index": i,
                "task_id": task_id,
                "warning_type": "missing_difficulty_features",
                "detail": None
            })

    return errors, warnings


# =========================================================
# 分层切分
# =========================================================

def stratified_split_by_source(
    items: List[Dict[str, Any]],
    train_ratio: float,
    main_ratio: float,
    capability_ratio: float,
    seed: int
) -> Tuple[List[str], List[str], List[str], Dict[str, Dict[str, int]], List[Dict[str, Any]]]:
    """
    仅按 source 做分层切分。
    当前 v1 是合理的最小方案。
    后续若需要可升级为 source × difficulty 的联合分层。
    """
    if abs(train_ratio + main_ratio + capability_ratio - 1.0) > 1e-8:
        raise ValueError("切分比例之和必须为 1.0")

    rng = random.Random(seed)
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    warnings: List[Dict[str, Any]] = []

    for item in items:
        by_source[item["source"]].append(item)

    train_ids: List[str] = []
    main_ids: List[str] = []
    cap_ids: List[str] = []
    source_summary: Dict[str, Dict[str, int]] = {}

    for source, group in by_source.items():
        group = group[:]
        rng.shuffle(group)

        n = len(group)
        n_train = int(n * train_ratio)
        n_main = int(n * main_ratio)
        n_cap = n - n_train - n_main

        train_group = group[:n_train]
        main_group = group[n_train:n_train + n_main]
        cap_group = group[n_train + n_main:]

        train_ids.extend([x["task_id"] for x in train_group])
        main_ids.extend([x["task_id"] for x in main_group])
        cap_ids.extend([x["task_id"] for x in cap_group])

        source_summary[source] = {
            "total": n,
            "train_pool": len(train_group),
            "main_test": len(main_group),
            "capability_retention": len(cap_group)
        }

        if len(train_group) == 0 or len(main_group) == 0 or len(cap_group) == 0:
            warnings.append({
                "source": source,
                "warning_type": "small_source_split_zero_count",
                "detail": source_summary[source]
            })

    return train_ids, main_ids, cap_ids, source_summary, warnings


# =========================================================
# 切分结果构造 (优化：O(1)哈希查询，取代O(N)列表遍历)
# =========================================================

def build_split_items(
    item_map: Dict[str, Dict[str, Any]],
    split_ids: List[str],
    split_name: str
) -> List[Dict[str, Any]]:
    role_map = {
        "train_pool": "main_pool",
        "main_test": "main_test",
        "capability_retention": "capability_retention"
    }
    target_role = role_map[split_name]

    out: List[Dict[str, Any]] = []
    # 按照 ID 字典序输出，保持一致性
    for tid in sorted(split_ids):
        # 性能优化：浅拷贝即可满足只改顶层 key 的需求，比 deepcopy 快数十倍
        new_item = item_map[tid].copy()
        new_item["split"] = split_name
        new_item["role"] = target_role
        out.append(new_item)

    return out


# =========================================================
# 交集与覆盖检查
# =========================================================

def check_disjoint(train_ids: List[str], main_ids: List[str], cap_ids: List[str]) -> Dict[str, int]:
    train_set, main_set, cap_set = set(train_ids), set(main_ids), set(cap_ids)
    return {
        "train_intersect_main": len(train_set & main_set),
        "train_intersect_capability": len(train_set & cap_set),
        "main_intersect_capability": len(main_set & cap_set)
    }


def check_full_coverage(
    all_items: List[Dict[str, Any]],
    train_ids: List[str],
    main_ids: List[str],
    cap_ids: List[str]
) -> Dict[str, Any]:
    all_ids = {x["task_id"] for x in all_items}
    split_union = set(train_ids) | set(main_ids) | set(cap_ids)
    missing = sorted(all_ids - split_union)
    extra = sorted(split_union - all_ids)

    return {
        "all_count": len(all_ids),
        "union_count": len(split_union),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_examples": missing[:20],
        "extra_examples": extra[:20]
    }


# =========================================================
# 分布统计 (优化：使用哈希映射取代全表遍历)
# =========================================================

def build_source_distribution_for_split(item_map: Dict[str, Dict[str, Any]], split_ids: List[str]) -> Dict[str, int]:
    return dict(Counter(item_map[tid]["source"] for tid in split_ids))


def build_difficulty_distribution_for_split(item_map: Dict[str, Dict[str, Any]], split_ids: List[str]) -> Dict[str, int]:
    counter = Counter()
    for tid in split_ids:
        bucket = item_map[tid].get("difficulty_bucket")
        bucket_key = "null" if bucket is None else str(bucket)
        counter[bucket_key] += 1
    return dict(counter)


# =========================================================
# Manifest / Report
# =========================================================

def build_manifest(
    train_ids: List[str],
    main_ids: List[str],
    cap_ids: List[str],
    source_summary: Dict[str, Dict[str, int]]
) -> Dict[str, Any]:
    return {
        "version": "v1cleaned",
        "seed": SPLIT_SEED,
        "source_pool_version": "unified_task_pool_v1",
        "input_file": str(IN_PATH),
        "output_files": {
            "train_pool": str(TRAIN_PATH),
            "main_test": str(MAIN_PATH),
            "capability_retention": str(CAP_PATH)
        },
        "ratios": {
            "train_pool": TRAIN_RATIO,
            "main_test": MAIN_RATIO,
            "capability_retention": CAPABILITY_RATIO
        },
        "source_summary": source_summary,
        "train_pool": train_ids,
        "main_test": main_ids,
        "capability_retention": cap_ids
    }


def build_report(
    items: List[Dict[str, Any]],
    item_map: Dict[str, Dict[str, Any]],
    errors: List[Dict[str, Any]],
    validation_warnings: List[Dict[str, Any]],
    split_warnings: List[Dict[str, Any]],
    train_ids: List[str],
    main_ids: List[str],
    cap_ids: List[str],
    source_summary: Dict[str, Dict[str, int]]
) -> Dict[str, Any]:
    overall_source_dist = dict(Counter(x["source"] for x in items))
    intersections = check_disjoint(train_ids, main_ids, cap_ids)
    coverage_check = check_full_coverage(items, train_ids, main_ids, cap_ids)

    return {
        "version": "v1cleaned",
        "input_file": str(IN_PATH),
        "seed": SPLIT_SEED,
        "ratios": {
            "train_pool": TRAIN_RATIO,
            "main_test": MAIN_RATIO,
            "capability_retention": CAPABILITY_RATIO
        },
        "total_items": len(items),
        "overall_source_distribution": overall_source_dist,
        "source_summary": source_summary,

        "train_pool_count": len(train_ids),
        "main_test_count": len(main_ids),
        "capability_retention_count": len(cap_ids),

        "train_pool_source_distribution": build_source_distribution_for_split(item_map, train_ids),
        "main_test_source_distribution": build_source_distribution_for_split(item_map, main_ids),
        "capability_retention_source_distribution": build_source_distribution_for_split(item_map, cap_ids),

        "train_pool_difficulty_distribution": build_difficulty_distribution_for_split(item_map, train_ids),
        "main_test_difficulty_distribution": build_difficulty_distribution_for_split(item_map, main_ids),
        "capability_retention_difficulty_distribution": build_difficulty_distribution_for_split(item_map, cap_ids),

        "intersections": intersections,
        "coverage_check": coverage_check,

        "validation_error_count": len(errors),
        "validation_warning_count": len(validation_warnings),
        "split_warning_count": len(split_warnings),

        "validation_errors": errors[:50],
        "validation_warnings": validation_warnings[:50],
        "split_warnings": split_warnings[:50],

        "note": (
            "当前 v1 切分只按 source 做分层。"
            "若后续发现 difficulty_bucket 分布差异明显，可在 v2_balanced 中升级为 source × difficulty 联合分层。"
        )
    }


# =========================================================
# 主函数
# =========================================================

def main() -> None:
    logger.info("开始执行 makesplits.py ...")
    items = load_jsonl(IN_PATH)

    errors, validation_warnings = validate_unifiedpool(items)
    
    # 构建全局映射表，将后续查询时间从 O(N^2) 降低为 O(N)
    item_map = {item["task_id"]: item for item in items}
    
    if len(item_map) != len(items):
        raise ValueError(
            f"item_map 长度与 items 不一致: len(item_map)={len(item_map)}, len(items)={len(items)}"
        )

    # 即使有错误，也先写出 report，方便排查
    if errors:
        report = build_report(
            items=items,
            item_map=item_map,
            errors=errors,
            validation_warnings=validation_warnings,
            split_warnings=[],
            train_ids=[],
            main_ids=[],
            cap_ids=[],
            source_summary={}
        )
        save_json(report, REPORT_PATH)
        logger.error("unified pool 校验失败，停止切分。")
        raise ValueError(f"统一主任务池校验失败，共 {len(errors)} 条错误，请先修复。")

    train_ids, main_ids, cap_ids, source_summary, split_warnings = stratified_split_by_source(
        items=items,
        train_ratio=TRAIN_RATIO,
        main_ratio=MAIN_RATIO,
        capability_ratio=CAPABILITY_RATIO,
        seed=SPLIT_SEED
    )

    intersections = check_disjoint(train_ids, main_ids, cap_ids)
    if any(v > 0 for v in intersections.values()):
        report = build_report(
            items=items,
            item_map=item_map,
            errors=[],
            validation_warnings=validation_warnings,
            split_warnings=split_warnings,
            train_ids=train_ids,
            main_ids=main_ids,
            cap_ids=cap_ids,
            source_summary=source_summary
        )
        save_json(report, REPORT_PATH)
        logger.error("切分结果存在交集，停止写出切分文件。")
        raise ValueError(f"切分结果存在交集: {intersections}")

    coverage_check = check_full_coverage(items, train_ids, main_ids, cap_ids)
    if coverage_check["missing_count"] > 0 or coverage_check["extra_count"] > 0:
        report = build_report(
            items=items,
            item_map=item_map,
            errors=[],
            validation_warnings=validation_warnings,
            split_warnings=split_warnings,
            train_ids=train_ids,
            main_ids=main_ids,
            cap_ids=cap_ids,
            source_summary=source_summary
        )
        save_json(report, REPORT_PATH)
        logger.error("切分结果未完整覆盖 unified pool，停止写出切分文件。")
        raise ValueError(f"切分覆盖异常: {coverage_check}")

    train_items = build_split_items(item_map, train_ids, "train_pool")
    main_items = build_split_items(item_map, main_ids, "main_test")
    cap_items = build_split_items(item_map, cap_ids, "capability_retention")

    manifest = build_manifest(
        train_ids=train_ids,
        main_ids=main_ids,
        cap_ids=cap_ids,
        source_summary=source_summary
    )
    report = build_report(
        items=items,
        item_map=item_map,
        errors=[],
        validation_warnings=validation_warnings,
        split_warnings=split_warnings,
        train_ids=train_ids,
        main_ids=main_ids,
        cap_ids=cap_ids,
        source_summary=source_summary
    )

    save_jsonl(train_items, TRAIN_PATH)
    save_jsonl(main_items, MAIN_PATH)
    save_jsonl(cap_items, CAP_PATH)
    save_json(manifest, MANIFEST_PATH)
    save_json(report, REPORT_PATH)

    logger.info("切分完成。")
    logger.info("train_pool: %d", len(train_items))
    logger.info("main_test: %d", len(main_items))
    logger.info("capability_retention: %d", len(cap_items))
    logger.info("来源分布: %s", source_summary)


if __name__ == "__main__":
    main()
