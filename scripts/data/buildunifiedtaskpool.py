import gzip
import json
import logging
import re
from collections import Counter
from pathlib import Path
import os
from typing import Optional, List, Dict, Any, Tuple


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "train_pool"

# 统一主任务池母体文件，不再放到 release_ready，避免语义过早
OUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "unifiedpool"
    / "unifiedtaskpoolv1.jsonl"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "manifests"
    / "datasetversions"
    / "unifiedtaskpoolv1report.json"
)

NORMALIZATION_VERSION = "v1"
SUPPORTED_SOURCES_THIS_SCRIPT = {"mbpp", "humaneval"}
SUPPORTED_TASK_TYPES_THIS_SCRIPT = {"function_level"}
VALID_BASE_TEST_SOURCES = {"original", "normalized", "mapped"}


# =========================================================
# 模块 1：基础工具
# =========================================================

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"已写入 {len(items)} 条记录到: {path}")


def save_json(obj: Dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logger.info(f"已写入 JSON 报告到: {path}")


def validate_required_string(obj: Dict[str, Any], field: str) -> bool:
    val = obj.get(field)
    return isinstance(val, str) and bool(val.strip())


# =========================================================
# 模块 2：签名与入口点提取
# =========================================================

def _signature_complete(line: str) -> bool:
    """
    判断签名行是否完整：
    - 括号平衡
    - 以冒号结尾
    """
    stripped = re.sub(r'["\'].*?["\']', '', line)
    balanced = stripped.count('(') <= stripped.count(')')
    ends_with_colon = line.rstrip().endswith(':')
    return balanced and ends_with_colon


def extract_signature_from_prompt(prompt: str) -> str:
    """
    从 prompt 中提取函数签名。
    支持多行签名。
    """
    if not prompt:
        return ""

    lines = prompt.splitlines()
    sig_lines = []
    in_sig = False

    for line in lines:
        stripped = line.strip()

        if not in_sig:
            if re.match(r"def\s+[a-zA-Z_]\w*\s*\(", stripped):
                in_sig = True
                sig_lines = [stripped]
                if _signature_complete(stripped):
                    return " ".join(sig_lines)
        else:
            sig_lines.append(stripped)
            if _signature_complete(stripped):
                return " ".join(sig_lines)

    return " ".join(sig_lines) if sig_lines else ""


def extract_signature_from_code(code: str, entry_point: str = "") -> str:
    """
    从代码中提取指定函数签名。
    支持多行参数列表。
    """
    if not code:
        return ""

    lines = code.splitlines()
    sig_lines = []
    in_sig = False
    paren_depth = 0

    if entry_point:
        start_pattern = re.compile(
            rf"^\s*def\s+{re.escape(entry_point)}\s*\("
        )
    else:
        start_pattern = re.compile(
            r"^\s*def\s+[a-zA-Z_]\w*\s*\("
        )

    for line in lines:
        if not in_sig:
            if start_pattern.match(line):
                in_sig = True
                sig_lines = [line.strip()]
                paren_depth = line.count('(') - line.count(')')
                if paren_depth <= 0 and _signature_complete(line.strip()):
                    return " ".join(sig_lines)
        else:
            sig_lines.append(line.strip())
            paren_depth += line.count('(') - line.count(')')
            if paren_depth <= 0 and _signature_complete(line.strip()):
                return " ".join(sig_lines)

    return " ".join(sig_lines) if sig_lines else ""


def extract_entry_point_from_code(code: str) -> str:
    if not code:
        return ""
    match = re.search(r"def\s+([a-zA-Z_]\w*)\s*\(", code)
    return match.group(1) if match else ""


# =========================================================
# 模块 3：基础度量
# =========================================================

def count_loc(code: str) -> Optional[int]:
    """
    统计有效代码行数（非空、非纯注释）。
    """
    if not code:
        return None
    lines = [
        x for x in code.splitlines()
        if x.strip() and not x.strip().startswith('#')
    ]
    return len(lines)


def count_prompt_line_count(prompt: str) -> int:
    if not prompt:
        return 0
    return len([x for x in prompt.splitlines() if x.strip()])


def count_arg_count_from_signature(signature: str) -> Optional[int]:
    """
    统计函数参数个数。
    仅在括号深度为 0 时统计逗号，避免类型注解中的逗号误判。
    """
    if not signature:
        return None

    start = signature.find('(')
    end = signature.rfind(')')
    if start == -1 or end == -1 or end <= start:
        return None

    inside = signature[start + 1: end].strip()
    if not inside:
        return 0

    depth = 0
    arg_count = 1
    has_content = False

    for char in inside:
        if char in ('(', '[', '{'):
            depth += 1
        elif char in (')', ']', '}'):
            depth -= 1
        elif char == ',' and depth == 0:
            arg_count += 1
        if char not in (' ', '\t', '\n'):
            has_content = True

    if not has_content:
        return 0

    # 尝试排除 self / cls
    first_arg = inside.split(',')[0].strip()
    if first_arg in ('self', 'cls'):
        arg_count -= 1

    return max(0, arg_count)


def detect_has_docstring_examples(prompt: str) -> Optional[bool]:
    """
    粗略判断 prompt 中是否包含 doctest / 示例。
    """
    if prompt is None:
        return None
    return ">>>" in prompt


# =========================================================
# 模块 4：tests / imports 归一化
# =========================================================

def normalize_imports(imports: Any) -> List[str]:
    """
    仅保留非空字符串 import，去掉两端空白。
    """
    if not isinstance(imports, list):
        return []
    cleaned = []
    for x in imports:
        if isinstance(x, str):
            s = x.strip()
            if s:
                cleaned.append(s)
    return cleaned


def normalize_base_tests(
    test_list: List[str],
    source: str = "original"
) -> List[Dict[str, Any]]:
    """
    过滤空字符串测试，转为结构化对象。
    """
    out = []
    idx = 1
    for test_code in test_list:
        if not test_code or not str(test_code).strip():
            continue
        out.append({
            "test_id": f"base_{idx}",
            "source": source,
            "code": str(test_code).strip()
        })
        idx += 1
    return out


# =========================================================
# 模块 5：difficulty_features 构造
# =========================================================

def build_difficulty_features(
    prompt: str,
    canonical_solution: str,
    signature: str,
    base_tests: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "prompt_length": len(prompt) if prompt else 0,               # 字符长度
        "prompt_line_count": count_prompt_line_count(prompt),
        "loc": count_loc(canonical_solution),
        "branch_count": None,                                        # 后续 AST 分析补全
        "has_exception_path": None,                                  # 后续 AST/静态分析补全
        "arg_count": count_arg_count_from_signature(signature),
        "base_test_count": len(base_tests),
        "cyclomatic_complexity": None,                               # 后续补
        "has_docstring_examples": detect_has_docstring_examples(prompt)
    }


# =========================================================
# 模块 6：eval_assets 构造
# =========================================================

def build_eval_assets() -> Dict[str, Any]:
    """
    当前 unified task pool v1 只声明基础测试存在；
    EvalPlus 映射后再回填 plus_tests / plus_tests_path / plus_tests_key。
    """
    return {
        "base_tests": True,
        "plus_tests": False,
        "plus_tests_path": None,
        "plus_tests_key": None
    }


# =========================================================
# 模块 7：单条样本归一化
# =========================================================

def normalize_mbpp_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    code = item.get("code", "")
    entry_point = extract_entry_point_from_code(code)
    signature = extract_signature_from_code(code, entry_point)

    imports = normalize_imports(item.get("test_imports", []))
    test_list = item.get("test_list", []) or []
    base_tests = normalize_base_tests(test_list, source="original")

    prompt = item.get("prompt", "")
    metadata = {
        "normalization_version": NORMALIZATION_VERSION,
        "source_file": "sanitized-mbpp.json",
        "source_path": str(RAW_ROOT / "mbpp" / "data" / "sanitized-mbpp.json"),
        "raw_record_index": idx
    }

    return {
        "task_id": f"mbpp_{idx:04d}",
        "source": "mbpp",
        "origin_id": str(item.get("task_id", idx)),
        "task_type": "function_level",
        "language": "python",

        "split": None,
        "role": "main_pool",

        "prompt": prompt,
        "entry_point": entry_point,
        "signature": signature,
        "canonical_solution": code,

        "context_code": "",
        "imports": imports,

        "base_tests": base_tests,
        "eval_assets": build_eval_assets(),

        "difficulty_bucket": None,
        "difficulty_features": build_difficulty_features(
            prompt=prompt,
            canonical_solution=code,
            signature=signature,
            base_tests=base_tests
        ),

        "metadata": metadata
    }


def normalize_humaneval_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    prompt = item.get("prompt", "")
    signature = extract_signature_from_prompt(prompt)
    canonical_solution = item.get("canonical_solution", "")

    raw_test = item.get("test", "")
    base_tests = normalize_base_tests(
        [raw_test] if raw_test else [],
        source="original"
    )

    metadata = {
        "normalization_version": NORMALIZATION_VERSION,
        "source_file": "HumanEval.jsonl.gz",
        "source_path": str(RAW_ROOT / "humaneval" / "data" / "HumanEval.jsonl.gz"),
        "raw_record_index": idx,
        "todo_import_extraction": (
            "若后续执行平台需要，可从 HumanEval prompt/test 中进一步抽取 imports。"
        )
    }

    return {
        "task_id": f"humaneval_{idx:04d}",
        "source": "humaneval",
        "origin_id": item.get("task_id", f"HumanEval/{idx}"),
        "task_type": "function_level",
        "language": "python",

        "split": None,
        "role": "main_pool",

        "prompt": prompt,
        "entry_point": item.get("entry_point", ""),
        "signature": signature,
        "canonical_solution": canonical_solution,

        "context_code": "",
        "imports": [],  # TODO: 如执行需要，可后续从 prompt/test 中抽取 imports

        "base_tests": base_tests,
        "eval_assets": build_eval_assets(),

        "difficulty_bucket": None,
        "difficulty_features": build_difficulty_features(
            prompt=prompt,
            canonical_solution=canonical_solution,
            signature=signature,
            base_tests=base_tests
        ),

        "metadata": metadata
    }


# =========================================================
# 模块 8：原始数据加载
# =========================================================

def load_mbpp() -> List[Dict[str, Any]]:
    path = RAW_ROOT / "mbpp" / "data" / "sanitized-mbpp.json"
    if not path.exists():
        raise FileNotFoundError(f"MBPP 数据文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(f"MBPP 原始数据: {len(data)} 条")
    return [normalize_mbpp_item(item, i + 1) for i, item in enumerate(data)]


def load_humaneval() -> List[Dict[str, Any]]:
    path = RAW_ROOT / "humaneval" / "data" / "HumanEval.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"HumanEval 数据文件不存在: {path}")

    data = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                item = json.loads(line)
                data.append(normalize_humaneval_item(item, i))

    logger.info(f"HumanEval 原始数据: {len(data)} 条")
    return data


# =========================================================
# 模块 9：校验
# =========================================================

def validate_items(
    items: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    返回 (errors, warnings)
    errors: 会影响 unified pool 合法性的错误
    warnings: 建议关注但不一定阻止写出
    """
    seen_ids: set = set()
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    for i, item in enumerate(items, start=1):
        task_id = item.get("task_id", f"[index={i}]")
        source = item.get("source")

        # 1. task_id 唯一
        if task_id in seen_ids:
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "duplicate_task_id",
                "detail": None
            })
        seen_ids.add(task_id)

        # 2. source 合法
        if source not in SUPPORTED_SOURCES_THIS_SCRIPT:
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "invalid_source",
                "detail": source
            })

        # 3. task_type 合法
        if item.get("task_type") not in SUPPORTED_TASK_TYPES_THIS_SCRIPT:
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "invalid_task_type",
                "detail": item.get("task_type")
            })

        # 4. split 必须是 null
        if item.get("split") is not None:
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "split_should_be_null",
                "detail": item.get("split")
            })

        # 5. role 必须是 main_pool
        if item.get("role") != "main_pool":
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "invalid_role",
                "detail": item.get("role")
            })

        # 6. 基本必填字符串字段
        for field in ["task_id", "source", "prompt"]:
            if not validate_required_string(item, field):
                errors.append({
                    "index": i,
                    "task_id": task_id,
                    "error_type": f"empty_{field}",
                    "detail": None
                })

        # 7. 当前 v1 主任务池（MBPP/HumanEval）要求 canonical_solution 非空
        if source in {"mbpp", "humaneval"}:
            if not validate_required_string(item, "canonical_solution"):
                errors.append({
                    "index": i,
                    "task_id": task_id,
                    "error_type": "empty_canonical_solution",
                    "detail": None
                })

        # 8. entry_point 规则：MBPP 为空报错；HumanEval 为空先警告
        if source == "mbpp":
            if not validate_required_string(item, "entry_point"):
                errors.append({
                    "index": i,
                    "task_id": task_id,
                    "error_type": "empty_entry_point",
                    "detail": None
                })
        elif source == "humaneval":
            if not validate_required_string(item, "entry_point"):
                warnings.append({
                    "index": i,
                    "task_id": task_id,
                    "warning_type": "empty_entry_point",
                    "detail": None
                })

        # 9. base_tests 非空（对当前主任务池来源）
        base_tests = item.get("base_tests", [])
        if source in {"mbpp", "humaneval"}:
            if not base_tests:
                errors.append({
                    "index": i,
                    "task_id": task_id,
                    "error_type": "empty_base_tests",
                    "detail": None
                })

        # 10. base_tests 内部结构合法性
        if not isinstance(base_tests, list):
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "base_tests_not_list",
                "detail": str(type(base_tests))
            })
        else:
            for j, bt in enumerate(base_tests, start=1):
                if not isinstance(bt, dict):
                    errors.append({
                        "index": i,
                        "task_id": task_id,
                        "error_type": "base_test_not_dict",
                        "detail": f"base_tests[{j}]"
                    })
                    continue

                if not validate_required_string(bt, "test_id"):
                    errors.append({
                        "index": i,
                        "task_id": task_id,
                        "error_type": "empty_base_test_id",
                        "detail": f"base_tests[{j}]"
                    })

                if not validate_required_string(bt, "source"):
                    errors.append({
                        "index": i,
                        "task_id": task_id,
                        "error_type": "empty_base_test_source",
                        "detail": f"base_tests[{j}]"
                    })
                else:
                    if bt["source"] not in VALID_BASE_TEST_SOURCES:
                        warnings.append({
                            "index": i,
                            "task_id": task_id,
                            "warning_type": "unknown_base_test_source",
                            "detail": f"base_tests[{j}]={bt['source']}"
                        })

                if not validate_required_string(bt, "code"):
                    errors.append({
                        "index": i,
                        "task_id": task_id,
                        "error_type": "empty_base_test_code",
                        "detail": f"base_tests[{j}]"
                    })

        # 11. difficulty_features 基本类型检查
        diff = item.get("difficulty_features", {})
        numeric_or_none_fields = [
            "prompt_length",
            "prompt_line_count",
            "loc",
            "branch_count",
            "arg_count",
            "base_test_count",
            "cyclomatic_complexity"
        ]
        for field in numeric_or_none_fields:
            val = diff.get(field)
            if val is not None and not isinstance(val, (int, float)):
                errors.append({
                    "index": i,
                    "task_id": task_id,
                    "error_type": f"invalid_difficulty_{field}",
                    "detail": val
                })

        bool_or_none_fields = [
            "has_exception_path",
            "has_docstring_examples"
        ]
        for field in bool_or_none_fields:
            val = diff.get(field)
            if val is not None and not isinstance(val, bool):
                errors.append({
                    "index": i,
                    "task_id": task_id,
                    "error_type": f"invalid_difficulty_{field}",
                    "detail": val
                })

        # 12. eval_assets 基本结构检查
        eval_assets = item.get("eval_assets", {})
        if not isinstance(eval_assets, dict):
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "invalid_eval_assets",
                "detail": str(type(eval_assets))
            })
        else:
            required_eval_keys = [
                "base_tests", "plus_tests", "plus_tests_path", "plus_tests_key"
            ]
            for key in required_eval_keys:
                if key not in eval_assets:
                    errors.append({
                        "index": i,
                        "task_id": task_id,
                        "error_type": f"missing_eval_assets_{key}",
                        "detail": None
                    })

        # 13. imports 必须是 list[str]
        imports = item.get("imports", [])
        if not isinstance(imports, list):
            errors.append({
                "index": i,
                "task_id": task_id,
                "error_type": "imports_not_list",
                "detail": str(type(imports))
            })
        else:
            for imp in imports:
                if not isinstance(imp, str):
                    errors.append({
                        "index": i,
                        "task_id": task_id,
                        "error_type": "imports_contains_non_string",
                        "detail": str(type(imp))
                    })
                    break

    return errors, warnings



# 模块 10：报告


def build_report(
    items: List[Dict[str, Any]],
    errors: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]]
) -> Dict[str, Any]:
    source_counter = Counter(item["source"] for item in items)
    task_type_counter = Counter(item["task_type"] for item in items)

    error_counter = Counter(e["error_type"] for e in errors)
    warning_counter = Counter(w["warning_type"] for w in warnings)

    report = {
        "normalization_version": NORMALIZATION_VERSION,
        "total_items": len(items),
        "output_path": str(OUT_PATH),
        "source_distribution": dict(source_counter),
        "task_type_distribution": dict(task_type_counter),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "error_type_distribution": dict(error_counter),
        "warning_type_distribution": dict(warning_counter),
        "supported_sources_this_script": sorted(list(SUPPORTED_SOURCES_THIS_SCRIPT)),
        "supported_task_types_this_script": sorted(list(SUPPORTED_TASK_TYPES_THIS_SCRIPT)),
        "note": (
            "当前 buildunifiedtaskpool.py 仅处理主任务池来源：MBPP 与 HumanEval。"
            "ClassEval 与 BugsInPy 将由后续独立导入脚本处理。"
        )
    }
    return report



# 主函数


def main():
    logger.info("开始构建统一任务池...")

    mbpp = load_mbpp()
    humaneval = load_humaneval()
    unified = mbpp + humaneval

    logger.info(f"合并后总数: {len(unified)} 条")

    errors, warnings = validate_items(unified)

    logger.info(f"MBPP:      {len(mbpp):>4d} 条")
    logger.info(f"HumanEval: {len(humaneval):>4d} 条")
    logger.info(f"合计:      {len(unified):>4d} 条")
    logger.info(f"校验错误:  {len(errors):>4d} 条")
    logger.info(f"校验警告:  {len(warnings):>4d} 条")

    if errors:
        logger.error("前 20 条错误：")
        for e in errors[:20]:
            logger.error(
                f"  [{e['index']:>4d}] {e['task_id']:>20s} "
                f"| {e['error_type']}"
                + (f" ({e['detail']})" if e['detail'] is not None else "")
            )

    if warnings:
        logger.warning("前 20 条警告：")
        for w in warnings[:20]:
            logger.warning(
                f"  [{w['index']:>4d}] {w['task_id']:>20s} "
                f"| {w['warning_type']}"
                + (f" ({w['detail']})" if w['detail'] is not None else "")
            )

    report = build_report(unified, errors, warnings)
    save_json(report, REPORT_PATH)

    if errors:
        logger.error("存在校验错误，停止写出 unifiedtaskpoolv1.jsonl")
        raise ValueError(f"统一任务池校验失败，共 {len(errors)} 条错误")

    save_jsonl(unified, OUT_PATH)


if __name__ == "__main__":
    main()
