from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(
    os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")
).resolve()

ALL_DECODING_PRESETS: List[Dict[str, Any]] = [
    {"name": "greedy", "do_sample": False, "temperature": None, "top_p": None, "seed": 11},
    {"name": "lowtemp_a", "do_sample": True, "temperature": 0.3, "top_p": 0.90, "seed": 22},
    {"name": "lowtemp_b", "do_sample": True, "temperature": 0.7, "top_p": 0.95, "seed": 33},
    {"name": "hightemp_a", "do_sample": True, "temperature": 0.90, "top_p": 0.95, "seed": 44},
    {"name": "hightemp_b", "do_sample": True, "temperature": 1.00, "top_p": 0.98, "seed": 55},
    {"name": "hightemp_c", "do_sample": True, "temperature": 1.10, "top_p": 0.98, "seed": 66},
]

ALL_PROMPT_VARIANTS: List[Dict[str, str]] = [
    {
        "name": "balanced",
        "focus": (
            "Focus on normal behavior plus at least one meaningful edge case. "
            "Every test function must contain at least one assertion."
        ),
    },
    {
        "name": "boundary",
        "focus": (
            "Emphasize boundary values, empty inputs, small corner cases, or shape/length-sensitive cases "
            "when those behaviors are supported by the task description. "
            "Every test function must contain at least one assertion."
        ),
    },
    {
        "name": "contract_exception",
        "focus": (
            "Emphasize contract-sensitive behaviors and exception-related checks only when clearly supported "
            "by the task description. Do not invent unsupported exceptions. "
            "Every test function must contain at least one assertion or one pytest.raises block."
        ),
    },
]

SYSTEM_PROMPT = (
    "You are an expert Python unit test writer.\n"
    "Write only executable Python test code.\n"
    "Do not output explanations.\n"
    "Do not output markdown fences.\n"
    "Use pytest-style test functions.\n"
    "Each test function must contain at least one real check: either an assert statement or pytest.raises(...).\n"
    "Do not write bare calls without checking behavior.\n"
    "Do not redefine, reimplement, wrap, alias, or shadow the target function or class under test.\n"
    "Do not invent unspecified semantics. Only test behaviors supported by the prompt or function contract.\n"
    "Do not introduce helper functions, wrappers, aliases, or fixtures unless strictly necessary and they do not alter the test target.\n"
    "Return only Python test code."
)

SCRIPT_VERSION = "generate_candidates_qwen_v6_shared_formal_final"
PROMPT_VERSION = "gen_prompt_v6_assert_strict"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")


def choose_prompt_col(df: pd.DataFrame) -> str:
    for col in ["prompt", "problem", "task_description", "text"]:
        if col in df.columns:
            return col
    raise ValueError("No prompt-like column found in input CSV.")


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    fenced_blocks = re.findall(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced_blocks:
        for block in fenced_blocks:
            if re.search(r"^\s*(?:async\s+)?def\s+test[_a-zA-Z0-9]*\s*\(", block, flags=re.MULTILINE):
                return block.strip()
        for block in fenced_blocks:
            if "assert " in block or "pytest.raises" in block:
                return block.strip()
        return fenced_blocks[0].strip()
    text = re.sub(r"^\s*```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def normalize_tokenizer_artifacts(text: str) -> str:
    return (
        (text or "")
        .replace("Ċ", "\n")
        .replace("Ġ", " ")
        .replace("ĉ", "\t")
    )


def strip_placeholder_imports(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not re.match(r"^\s*from\s+your_module\s+import\s+", line)
    )


def keep_probable_test_region(text: str) -> str:
    lines = text.splitlines()
    test_start = None
    for idx, line in enumerate(lines):
        if re.match(r"^\s*(?:async\s+)?def\s+test[_a-zA-Z0-9]*\s*\(", line):
            test_start = idx
            break
    if test_start is None:
        return text

    prelude = []
    for line in lines[:test_start]:
        if re.match(r"^\s*(import|from)\s+", line) and "your_module" not in line:
            prelude.append(line)
    body = lines[test_start:]
    return "\n".join(prelude + body).strip()


def strip_main_guard(text: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", line):
            return "\n".join(lines[:idx]).rstrip()
    return text


def has_test_signal(text: str) -> bool:
    return bool(re.search(r"^\s*(?:async\s+)?def\s+test[_a-zA-Z0-9]*\s*\(", text, flags=re.MULTILINE)) and (
        "assert " in text or "pytest.raises" in text
    )


def salvage_parseable_prefix(text: str) -> str:
    if not text.strip():
        return text
    try:
        ast.parse(text)
        return text
    except SyntaxError:
        pass

    lines = text.rstrip().splitlines()
    for end in range(len(lines) - 1, 0, -1):
        candidate = "\n".join(lines[:end]).rstrip()
        if not has_test_signal(candidate):
            continue
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue
    return text


def normalize_generated_code(text: str) -> str:
    text = normalize_tokenizer_artifacts(text)
    text = strip_code_fences(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = strip_placeholder_imports(text)
    text = keep_probable_test_region(text)
    text = strip_main_guard(text)
    text = salvage_parseable_prefix(text)
    return text.strip()


def clip_text(text: Optional[str], max_chars: int = 12000) -> Optional[str]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    return text[:max_chars]


def safe_scalar(row: pd.Series, col: str) -> Any:
    if col not in row:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    return val


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_device(model: AutoModelForCausalLM) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def get_pad_token_id(tokenizer: AutoTokenizer) -> int:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return tokenizer.pad_token_id
    if getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer.eos_token_id
    raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")


def count_nonempty_lines(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


def maybe_unlink(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )
        path.unlink()


def parse_name_csv(value: Optional[str]) -> Optional[Set[str]]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def filter_named_items(
    items: List[Dict[str, Any]],
    selected_names: Optional[Set[str]],
    item_type: str,
) -> List[Dict[str, Any]]:
    if selected_names is None:
        return items

    out = [x for x in items if str(x.get("name", "")).strip() in selected_names]
    if not out:
        valid = sorted(str(x.get("name", "")).strip() for x in items)
        raise ValueError(
            f"No valid {item_type} selected. "
            f"Requested={sorted(selected_names)}, valid={valid}"
        )
    return out


def build_variant_schedule(
    num_candidates_per_task: int,
    prompt_variants: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    if not prompt_variants:
        raise ValueError("prompt_variants must not be empty.")
    return [prompt_variants[i % len(prompt_variants)] for i in range(num_candidates_per_task)]


def build_user_prompt(row: pd.Series, prompt_col: str, variant: Dict[str, str]) -> str:
    task_text = str(row[prompt_col]) if pd.notna(row.get(prompt_col)) else ""
    entry_point = str(row["entry_point"]) if pd.notna(row.get("entry_point")) else ""
    signature = str(row["signature"]) if pd.notna(row.get("signature")) else ""
    task_type = str(row["task_type"]) if pd.notna(row.get("task_type")) else ""

    return (
        "Task description:\n"
        f"{task_text}\n\n"
        f"Entry point: {entry_point}\n"
        f"Signature (if available): {signature}\n"
        f"Task type (if available): {task_type}\n\n"
        "Write pytest-style Python unit tests for the target above.\n\n"
        "Hard requirements:\n"
        "1. Output only executable Python test code.\n"
        "2. Use pytest-style test functions named like test_*.\n"
        "3. Every test function must contain at least one real behavioral check:\n"
        "   - either an assert statement\n"
        "   - or a pytest.raises(...) block when exception behavior is clearly supported.\n"
        "4. Do not write bare calls, print statements, or checks without assertions.\n"
        "5. Do not redefine, reimplement, alias, wrap, or shadow the target function/class.\n"
        "6. Do not copy the solution into the test file.\n"
        "7. Do not assume unsupported semantics. Only test behavior supported by the task description.\n"
        "8. Prefer 3 to 6 concise test functions.\n"
        "9. Include meaningful checks for normal behavior and supported edge cases.\n"
        "10. Include exception-related checks only when clearly justified by the task description.\n"
        "11. Avoid helper functions, wrappers, aliases, or fixtures unless strictly necessary and clearly harmless.\n\n"
        f"Additional focus for this candidate: {variant['focus']}\n\n"
        "Return only Python test code."
    )


def build_messages(row: pd.Series, prompt_col: str, variant: Dict[str, str]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(row, prompt_col, variant)},
    ]


def build_generation_text(
    tokenizer: AutoTokenizer,
    row: pd.Series,
    prompt_col: str,
    variant: Dict[str, str],
) -> str:
    messages = build_messages(row, prompt_col, variant)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        f"[SYSTEM]\n{SYSTEM_PROMPT}\n\n"
        f"[USER]\n{build_user_prompt(row, prompt_col, variant)}\n\n"
        "[ASSISTANT]\n"
    )


def ast_parse_check(code: str) -> Dict[str, Any]:
    if not code.strip():
        return {
            "ast_parse_pass": False,
            "syntax_pass": False,
            "syntax_error_type": "empty_code",
            "syntax_error_message": "candidate_test_code is empty",
        }
    try:
        ast.parse(code)
        return {
            "ast_parse_pass": True,
            "syntax_pass": True,
            "syntax_error_type": None,
            "syntax_error_message": None,
        }
    except SyntaxError as e:
        return {
            "ast_parse_pass": False,
            "syntax_pass": False,
            "syntax_error_type": "syntax_error",
            "syntax_error_message": f"{e.__class__.__name__}: {e}",
        }


def _collect_defined_names(tree: ast.AST) -> Tuple[Set[str], Set[str]]:
    fn_names: Set[str] = set()
    class_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn_names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            class_names.add(node.name)
    return fn_names, class_names


def _is_pytest_raises(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "pytest"
        and node.func.attr == "raises"
    )


def _collect_assert_stats(tree: ast.AST) -> Dict[str, int]:
    num_assert_stmt = 0
    num_pytest_raises = 0
    num_test_functions = 0
    num_test_functions_with_checks = 0
    num_bare_calls = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            num_assert_stmt += 1
        elif isinstance(node, ast.With):
            for item in node.items:
                if _is_pytest_raises(item.context_expr):
                    num_pytest_raises += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            num_bare_calls += 1

    for node in getattr(tree, "body", []):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            num_test_functions += 1
            has_check = False
            for inner in ast.walk(node):
                if isinstance(inner, ast.Assert):
                    has_check = True
                    break
                if isinstance(inner, ast.With) and any(
                    _is_pytest_raises(item.context_expr) for item in inner.items
                ):
                    has_check = True
                    break
            if has_check:
                num_test_functions_with_checks += 1

    return {
        "num_assert_stmt": num_assert_stmt,
        "num_pytest_raises": num_pytest_raises,
        "num_test_functions": num_test_functions,
        "num_test_functions_with_checks": num_test_functions_with_checks,
        "num_bare_calls": num_bare_calls,
    }


def _collect_alias_names(tree: ast.AST) -> Set[str]:
    alias_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    alias_names.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                alias_names.add(node.target.id)
    return alias_names


def _collect_wrapper_risk(tree: ast.AST, entry_point: Optional[str]) -> bool:
    if not entry_point:
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    if inner.func.id == entry_point and node.name != entry_point:
                        return True
    return False


def _checks_outside_test_functions(tree: ast.AST) -> int:
    total_outside = 0
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assert):
            total_outside += 1
        elif isinstance(node, ast.With):
            if any(_is_pytest_raises(item.context_expr) for item in node.items):
                total_outside += 1
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            # bare top-level call is also suspicious
            total_outside += 1
    return total_outside


def analyze_candidate_validity(code: str, entry_point: Optional[str]) -> Dict[str, Any]:
    out = {
        "invalid_candidate_flag": 0,
        "candidate_validity_status": "ok",
        "redefines_target_flag": 0,
        "redefines_target_kind": None,
        "missing_behavioral_check_flag": 0,
        "suspicious_alias_or_wrapper_flag": 0,
        "analysis_num_assert_stmt": None,
        "analysis_num_pytest_raises": None,
        "analysis_num_test_functions": None,
        "analysis_num_test_functions_with_checks": None,
        "analysis_num_bare_calls": None,
        "has_pytest_style_test_function": 0,
        "all_checks_inside_test_functions_flag": 0,
    }

    if not code.strip():
        out["invalid_candidate_flag"] = 1
        out["candidate_validity_status"] = "empty_code"
        return out

    try:
        tree = ast.parse(code)
    except Exception:
        out["candidate_validity_status"] = "syntax_error"
        return out

    fn_names, class_names = _collect_defined_names(tree)
    stats = _collect_assert_stats(tree)
    alias_names = _collect_alias_names(tree)
    wrapper_risk = _collect_wrapper_risk(tree, entry_point)

    out["analysis_num_assert_stmt"] = stats["num_assert_stmt"]
    out["analysis_num_pytest_raises"] = stats["num_pytest_raises"]
    out["analysis_num_test_functions"] = stats["num_test_functions"]
    out["analysis_num_test_functions_with_checks"] = stats["num_test_functions_with_checks"]
    out["analysis_num_bare_calls"] = stats["num_bare_calls"]
    out["has_pytest_style_test_function"] = int(stats["num_test_functions"] > 0)

    total_checks = stats["num_assert_stmt"] + stats["num_pytest_raises"]
    outside_checks = _checks_outside_test_functions(tree)
    if total_checks > 0 and outside_checks == 0 and stats["num_test_functions_with_checks"] > 0:
        out["all_checks_inside_test_functions_flag"] = 1

    if entry_point:
        if entry_point in fn_names:
            out.update(
                {
                    "invalid_candidate_flag": 1,
                    "candidate_validity_status": "invalid_redefines_target",
                    "redefines_target_flag": 1,
                    "redefines_target_kind": "function",
                }
            )
            return out
        if entry_point in class_names:
            out.update(
                {
                    "invalid_candidate_flag": 1,
                    "candidate_validity_status": "invalid_redefines_target",
                    "redefines_target_flag": 1,
                    "redefines_target_kind": "class",
                }
            )
            return out
        if entry_point in alias_names or wrapper_risk:
            out["suspicious_alias_or_wrapper_flag"] = 1

    if stats["num_assert_stmt"] == 0 and stats["num_pytest_raises"] == 0:
        out["missing_behavioral_check_flag"] = 1
        out["candidate_validity_status"] = "weak_no_assert_like_check"

    return out


def make_candidate_id(
    task_id: str,
    preset_name: str,
    candidate_index: int,
    generator_tag: str,
    branch_tag: str,
) -> str:
    return f"{task_id}__{branch_tag}__{generator_tag}__{preset_name}__c{candidate_index:02d}"


def passthrough_fields(row: pd.Series, prompt_col: str) -> Dict[str, Any]:
    extra_cols = [
        "canonical_solution",
        "base_tests",
        "imports",
        "signature",
        "task_type",
        "eval_assets",
        "entry_point",
        prompt_col,
    ]
    return {col: safe_scalar(row, col) for col in extra_cols}


def get_generator_metadata(
    model_name_or_path: str,
    teacher_only_humaneval: bool,
) -> Dict[str, str]:
    name = Path(model_name_or_path).name.strip() or model_name_or_path.strip()
    lower = name.lower()

    if "deepseek" in lower and "coder" in lower:
        return {
            "generator_model": "DeepSeek-Coder-6.7B-Instruct",
            "generator_size": "6.7B",
            "generator_tag": "deepseek67b",
            "generator_role": "model_level_sensitivity_check",
            "candidate_source": "deepseek_model_sensitivity_generation",
        }

    if "14b" in lower and "qwen" in lower:
        return {
            "generator_model": "Qwen2.5-14B-Instruct",
            "generator_size": "14B",
            "generator_tag": "qwen14b",
            "generator_role": "teacher_only_data_synthesis",
            "candidate_source": (
                "teacher_only_humaneval_rescue"
                if teacher_only_humaneval
                else "teacher_only_candidate_generation"
            ),
        }

    if "7b" in lower and "qwen" in lower:
        return {
            "generator_model": "Qwen2.5-7B-Instruct",
            "generator_size": "7B",
            "generator_tag": "qwen7b",
            "generator_role": "student_primary_candidate_generation",
            "candidate_source": "shared_formal_primary_generation",
        }

    return {
        "generator_model": name,
        "generator_size": "unknown",
        "generator_tag": re.sub(r"[^a-z0-9]+", "", lower)[:16] or "unknown",
        "generator_role": "raw_candidate_generation",
        "candidate_source": "shared_formal_candidate_generation",
    }


def build_distribution_counter(df: pd.DataFrame, col: str) -> Dict[str, int]:
    if col not in df.columns:
        return {}
    vc = df[col].fillna("NA").astype(str).value_counts(dropna=False)
    return {str(k): int(v) for k, v in vc.to_dict().items()}


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_dir", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preset_names", type=str, default="")
    parser.add_argument("--prompt_variant_names", type=str, default="")
    parser.add_argument("--keep_raw_generation_text", action="store_true")
    parser.add_argument("--teacher_only_humaneval", action="store_true")
    parser.add_argument("--branch_tag", type=str, default="")
    args = parser.parse_args()

    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    selected_preset_names = parse_name_csv(args.preset_names)
    selected_prompt_variant_names = parse_name_csv(args.prompt_variant_names)

    decoding_presets = filter_named_items(
        ALL_DECODING_PRESETS,
        selected_preset_names,
        "decoding presets",
    )
    prompt_variants = filter_named_items(
        ALL_PROMPT_VARIANTS,
        selected_prompt_variant_names,
        "prompt variants",
    )

    branch_tag = (args.branch_tag or args.run_id).strip()
    branch_tag = re.sub(r"[^a-zA-Z0-9_\-]+", "_", branch_tag)

    out_dir = PROJECT_ROOT / "data" / "interim" / "generatedcandidates" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "generatedcandidates.jsonl"
    compat_jsonl = out_dir / "smokecandidates.jsonl"
    out_manifest = out_dir / "manifest.json"

    maybe_unlink(out_jsonl, overwrite=args.overwrite)
    maybe_unlink(compat_jsonl, overwrite=args.overwrite)
    maybe_unlink(out_manifest, overwrite=args.overwrite)

    df = load_csv(input_csv)
    ensure_required_columns(df, ["task_id", "canonical_solution"])
    prompt_col = choose_prompt_col(df)

    if df["task_id"].duplicated().any():
        dupes = df.loc[df["task_id"].duplicated(), "task_id"].astype(str).tolist()
        raise ValueError(
            f"Input CSV contains duplicated task_id, examples: {dupes[:10]}"
        )

    if df["canonical_solution"].isna().any():
        raise ValueError(
            f"Input CSV contains {df['canonical_solution'].isna().sum()} rows with missing canonical_solution"
        )

    if args.teacher_only_humaneval and "source" in df.columns:
        non_humaneval = df[df["source"].astype(str) != "humaneval"]
        if len(non_humaneval) > 0:
            raise ValueError(
                "--teacher_only_humaneval is set, but input CSV contains non-humaneval rows."
            )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    pad_token_id = get_pad_token_id(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    adapter_dir = args.adapter_dir.strip()
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    device = get_model_device(model)

    generator_meta = get_generator_metadata(
        args.model_name_or_path,
        teacher_only_humaneval=args.teacher_only_humaneval,
    )

    parse_ok_count = 0
    nonempty_count = 0
    generation_fail_count = 0
    invalid_candidate_count = 0
    redefines_target_count = 0
    weak_no_assert_like_count = 0
    suspicious_alias_or_wrapper_count = 0
    num_candidates = 0

    interrupted = False
    interrupt_reason: Optional[str] = None

    num_tasks = int(df["task_id"].nunique())
    total_candidates = num_tasks * len(decoding_presets)

    pbar = None
    variant_schedule = build_variant_schedule(
        len(decoding_presets),
        prompt_variants,
    )

    generation_status_counter: Counter[str] = Counter()
    validity_status_counter: Counter[str] = Counter()

    try:
        with open(out_jsonl, "w", encoding="utf-8") as fout:
            pbar = tqdm(
                total=total_candidates,
                desc=f"Generating {args.run_id}",
                unit="cand",
            )

            for _, row in df.iterrows():
                task_id = str(row["task_id"])
                source = safe_scalar(row, "source")
                difficulty_bucket = safe_scalar(row, "difficulty_bucket")
                entry_point = safe_scalar(row, "entry_point")
                source_task_group = f"{source}__{difficulty_bucket}"
                passthrough = passthrough_fields(row, prompt_col)

                for candidate_index, preset in enumerate(decoding_presets, start=1):
                    variant = variant_schedule[candidate_index - 1]
                    text = build_generation_text(
                        tokenizer,
                        row,
                        prompt_col,
                        variant,
                    )

                    inputs = tokenizer(text, return_tensors="pt")
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs.get("attention_mask")
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)

                    seed = int(preset["seed"])
                    set_seed(seed)

                    gen_kwargs = {
                        "input_ids": input_ids,
                        "max_new_tokens": int(args.max_new_tokens),
                        "do_sample": bool(preset["do_sample"]),
                        "pad_token_id": pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                        "use_cache": True,
                    }
                    if attention_mask is not None:
                        gen_kwargs["attention_mask"] = attention_mask

                    if preset["do_sample"]:
                        gen_kwargs["temperature"] = float(preset["temperature"])
                        gen_kwargs["top_p"] = float(preset["top_p"])

                    generation_status = "ok"
                    generation_error_type = None
                    generation_error_message = None
                    raw_generation_text = None
                    candidate_test_code = ""

                    try:
                        with torch.inference_mode():
                            outputs = model.generate(**gen_kwargs)

                        generated_ids = outputs[0][input_ids.shape[1]:]
                        raw_generation_text = tokenizer.decode(
                            generated_ids,
                            skip_special_tokens=True,
                        ).strip()
                        candidate_test_code = normalize_generated_code(raw_generation_text)

                    except KeyboardInterrupt:
                        interrupted = True
                        interrupt_reason = "KeyboardInterrupt"
                        raise
                    except Exception as e:
                        generation_status = "error"
                        generation_error_type = e.__class__.__name__
                        generation_error_message = f"{e.__class__.__name__}: {e}"
                        generation_fail_count += 1
                        raw_generation_text = None
                        candidate_test_code = ""

                    parse_info = ast_parse_check(candidate_test_code)
                    validity_info = analyze_candidate_validity(
                        candidate_test_code,
                        entry_point,
                    )

                    if parse_info["ast_parse_pass"]:
                        parse_ok_count += 1
                    if candidate_test_code.strip():
                        nonempty_count += 1
                    if validity_info["invalid_candidate_flag"]:
                        invalid_candidate_count += 1
                    if validity_info["redefines_target_flag"]:
                        redefines_target_count += 1
                    if validity_info["missing_behavioral_check_flag"]:
                        weak_no_assert_like_count += 1
                    if validity_info["suspicious_alias_or_wrapper_flag"]:
                        suspicious_alias_or_wrapper_count += 1

                    generation_status_counter[generation_status] += 1
                    validity_status_counter[
                        str(validity_info["candidate_validity_status"])
                    ] += 1

                    out_row = {
                        "run_id": args.run_id,
                        "script_version": SCRIPT_VERSION,
                        "candidate_id": make_candidate_id(
                            task_id,
                            preset["name"],
                            candidate_index,
                            generator_meta["generator_tag"],
                            branch_tag,
                        ),
                        "task_id": task_id,
                        "source": source,
                        "difficulty_bucket": difficulty_bucket,
                        "source_task_group": source_task_group,
                        "test_source_type": "model_generated",
                        "candidate_test_origin_detail": preset["name"],
                        "prompt_variant_name": variant["name"],
                        "prompt_variant_focus": variant["focus"],
                        "model_name_or_path": args.model_name_or_path,
                        "adapter_dir": adapter_dir or None,
                        "prompt_version": PROMPT_VERSION,
                        "decoding_config_name": preset["name"],
                        "temperature": preset["temperature"],
                        "top_p": preset["top_p"],
                        "seed": seed,
                        "candidate_index": candidate_index,
                        "generation_status": generation_status,
                        "generation_error_type": generation_error_type,
                        "generation_error_message": generation_error_message,
                        "raw_generation_text": (
                            clip_text(raw_generation_text)
                            if args.keep_raw_generation_text
                            else None
                        ),
                        "candidate_test_code": candidate_test_code,
                        "candidate_char_len": len(candidate_test_code),
                        "candidate_nonempty_line_count": count_nonempty_lines(candidate_test_code),
                        "branch_tag": branch_tag,
                        **generator_meta,
                        **parse_info,
                        **validity_info,
                        **passthrough,
                    }

                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    num_candidates += 1

                    if num_candidates % 10 == 0:
                        fout.flush()

                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "task": task_id,
                            "preset": preset["name"],
                            "ok": parse_ok_count,
                            "err": generation_fail_count,
                        }
                    )

            fout.flush()

    except KeyboardInterrupt:
        interrupted = True
        interrupt_reason = interrupt_reason or "KeyboardInterrupt"
        print(
            "\n[warn] Generation interrupted by user. Partial JSONL has been preserved.",
            file=sys.stderr,
        )

    finally:
        if pbar is not None:
            pbar.close()

        if out_jsonl.exists():
            shutil.copyfile(out_jsonl, compat_jsonl)

        manifest = {
            "script_version": SCRIPT_VERSION,
            "run_id": args.run_id,
            "branch_tag": branch_tag,
            "input_csv": str(input_csv),
            "output_jsonl": str(out_jsonl),
            "compat_jsonl": str(compat_jsonl),
            "model_name_or_path": args.model_name_or_path,
            "adapter_dir": adapter_dir or None,
            "generator_metadata": generator_meta,
            "num_tasks": num_tasks,
            "num_candidates": int(num_candidates),
            "num_presets_per_task": int(len(decoding_presets)),
            "expected_total_candidates": int(total_candidates),
            "prompt_col": prompt_col,
            "prompt_version": PROMPT_VERSION,
            "decoding_presets": decoding_presets,
            "prompt_variants": prompt_variants,
            "teacher_only_humaneval": bool(args.teacher_only_humaneval),
            "keep_raw_generation_text": bool(args.keep_raw_generation_text),
            "passthrough_fields": [
                "canonical_solution",
                "base_tests",
                "imports",
                "signature",
                "task_type",
                "eval_assets",
                "entry_point",
                prompt_col,
            ],
            "input_distribution": {
                "source_distribution": build_distribution_counter(df, "source"),
                "difficulty_distribution": build_distribution_counter(df, "difficulty_bucket"),
                "source_task_group_distribution": build_distribution_counter(
                    pd.DataFrame(
                        {
                            "source_task_group": (
                                df["source"].astype(str).fillna("NA")
                                + "__"
                                + df["difficulty_bucket"].astype(str).fillna("NA")
                            )
                            if "source" in df.columns and "difficulty_bucket" in df.columns
                            else pd.Series([], dtype=str)
                        }
                    ),
                    "source_task_group",
                ),
            },
            "summary": {
                "generation_error_count": int(generation_fail_count),
                "nonempty_candidate_count": int(nonempty_count),
                "ast_parse_ok_count": int(parse_ok_count),
                "invalid_candidate_count": int(invalid_candidate_count),
                "redefines_target_count": int(redefines_target_count),
                "weak_no_assert_like_count": int(weak_no_assert_like_count),
                "suspicious_alias_or_wrapper_count": int(suspicious_alias_or_wrapper_count),
                "generation_status_counts": dict(generation_status_counter),
                "candidate_validity_status_counts": dict(validity_status_counter),
            },
            "completed": not interrupted,
            "interrupted": interrupted,
            "interrupt_reason": interrupt_reason,
            "notes": [
                "This run generates model-based candidate tests.",
                "Prompt explicitly requires assert or pytest.raises in every test function.",
                "Prompt explicitly forbids redefining or reimplementing the target function/class.",
                "Prompt explicitly discourages helper wrappers and aliases.",
                "Execution-critical fields such as canonical_solution are passed through to generatedcandidates.jsonl.",
                "generatedcandidates.jsonl is the canonical output; smokecandidates.jsonl is kept for compatibility.",
                "Streaming JSONL writing is enabled.",
                "A tqdm progress bar is shown during generation.",
                "candidate_id includes branch_tag and generator_tag to avoid collisions across merged pools.",
            ],
        }

        write_manifest(out_manifest, manifest)

    print("[ok] wrote", out_jsonl)
    print("[ok] wrote", compat_jsonl)
    print("[ok] wrote", out_manifest)
    print("num_tasks =", num_tasks)
    print("num_candidates =", num_candidates)
    print("nonempty_candidate_count =", nonempty_count)
    print("ast_parse_ok_count =", parse_ok_count)
    print("invalid_candidate_count =", invalid_candidate_count)
    print("redefines_target_count =", redefines_target_count)
    print("weak_no_assert_like_count =", weak_no_assert_like_count)
    print("suspicious_alias_or_wrapper_count =", suspicious_alias_or_wrapper_count)
    print("generation_error_count =", generation_fail_count)
    print("completed =", not interrupted)

    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
