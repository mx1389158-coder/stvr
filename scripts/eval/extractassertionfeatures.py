from __future__ import annotations

import ast
import io
import json
import sys
import token
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common.interimschema import (
    stage_file,
    stage_manifest_file,
    base_record,
)

INPUT_PATH = stage_file("generated_candidates")
OUT_PATH = stage_file("static_features")
MANIFEST_PATH = stage_manifest_file("static_features")

SCRIPT_VERSION = "extract_assertion_features_v4"
MAX_ERROR_MSG_CHARS = 1000

IGNORED_TOKENS = {
    token.ENDMARKER,
    token.NEWLINE,
    token.NL,
    token.INDENT,
    token.DEDENT,
    getattr(token, "ENCODING", -1),
}


# =========================================================
# 基础 IO
# =========================================================

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def clip_text(text: Optional[str], max_chars: int = MAX_ERROR_MSG_CHARS) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    return text[:max_chars] if text else None


# =========================================================
# tokenize 级统计
# =========================================================

def compute_loc(code: str) -> int:
    return sum(1 for ln in code.splitlines() if ln.strip())


def compute_token_stats(code: str) -> Dict[str, Any]:
    token_length = 0
    num_comments = 0
    num_inline_comments = 0
    comment_lines: Set[int] = set()

    lines = code.splitlines()
    loc = sum(1 for ln in lines if ln.strip())

    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            tok_type = tok.type
            start_row = tok.start[0]
            start_col = tok.start[1]

            if tok_type == token.COMMENT:
                num_comments += 1
                comment_lines.add(start_row)

                line_text = lines[start_row - 1] if 1 <= start_row <= len(lines) else ""
                if line_text[:start_col].strip():
                    num_inline_comments += 1
                continue

            if tok_type in IGNORED_TOKENS:
                continue

            token_length += 1

    except tokenize.TokenError:
        token_length = len(code.split())

    return {
        "loc": loc,
        "token_length": token_length,
        "num_comments": num_comments,
        "num_comment_lines": len(comment_lines),
        "num_inline_comments": num_inline_comments,
        "comment_ratio": round(len(comment_lines) / max(1, loc), 4),
    }


# =========================================================
# AST 分析器
# =========================================================

class Analyzer(ast.NodeVisitor):
    """
    说明：
    1. ast_max_depth 表示 AST 树深度；
    2. logical_nesting_depth 表示控制流逻辑嵌套深度；
    3. max_nesting_depth 明确作为 logical_nesting_depth 的兼容别名，
       不再与 ast_max_depth 混同；
    4. num_imports_in_test 是规范字段；num_imports_in_candidate 仅作为
       兼容旧表结构保留，当前与 num_imports_in_test 等值。
    """

    def __init__(self) -> None:
        self.num_asserts = 0
        self.num_top_level_asserts = 0
        self.num_asserts_in_test_functions = 0
        self.num_bare_expression_calls = 0

        self.num_test_functions = 0
        self.num_async_test_functions = 0
        self.has_test_function = False

        self.num_try_except = 0
        self.num_raise_statements = 0

        self.has_pytest_raises = False
        self.num_pytest_raises = 0
        self.num_exception_checks = 0

        self.num_imports_in_test = 0

        self.ast_node_count = 0
        self.ast_max_depth = 0
        self._ast_depth = 0

        self.logical_nesting_depth = 0
        self._logical_depth = 0

        self._in_test_function = False

        self.pytest_aliases: Set[str] = {"pytest"}
        self.raises_aliases: Set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        self.ast_node_count += 1
        self._ast_depth += 1
        self.ast_max_depth = max(self.ast_max_depth, self._ast_depth)
        super().generic_visit(node)
        self._ast_depth -= 1

    def _visit_control_node(self, node: ast.AST) -> None:
        self._logical_depth += 1
        self.logical_nesting_depth = max(self.logical_nesting_depth, self._logical_depth)
        self.generic_visit(node)
        self._logical_depth -= 1

    def visit_Import(self, node: ast.Import) -> None:
        self.num_imports_in_test += 1
        for alias in node.names:
            if alias.name == "pytest":
                self.pytest_aliases.add(alias.asname or "pytest")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.num_imports_in_test += 1
        if node.module == "pytest":
            for alias in node.names:
                if alias.name == "raises":
                    self.raises_aliases.add(alias.asname or "raises")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        self.num_asserts += 1
        if self._in_test_function:
            self.num_asserts_in_test_functions += 1
        else:
            self.num_top_level_asserts += 1
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        self.num_raise_statements += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_test_fn = node.name.startswith("test_")
        prev = self._in_test_function
        if is_test_fn:
            self.num_test_functions += 1
            self.has_test_function = True
            self._in_test_function = True
        self.generic_visit(node)
        self._in_test_function = prev

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        is_test_fn = node.name.startswith("test_")
        prev = self._in_test_function
        if is_test_fn:
            self.num_test_functions += 1
            self.num_async_test_functions += 1
            self.has_test_function = True
            self._in_test_function = True
        self.generic_visit(node)
        self._in_test_function = prev

    def visit_If(self, node: ast.If) -> None:
        self._visit_control_node(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_control_node(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_control_node(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_control_node(node)

    def visit_Try(self, node: ast.Try) -> None:
        if node.handlers:
            self.num_try_except += 1
            self.num_exception_checks += 1
        self._visit_control_node(node)

    def visit_With(self, node: ast.With) -> None:
        pytest_raises_count = sum(
            1 for item in node.items if self._is_pytest_raises_expr(item.context_expr)
        )
        if pytest_raises_count > 0:
            self.has_pytest_raises = True
            self.num_pytest_raises += pytest_raises_count
            self.num_exception_checks += pytest_raises_count
        self._visit_control_node(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        pytest_raises_count = sum(
            1 for item in node.items if self._is_pytest_raises_expr(item.context_expr)
        )
        if pytest_raises_count > 0:
            self.has_pytest_raises = True
            self.num_pytest_raises += pytest_raises_count
            self.num_exception_checks += pytest_raises_count
        self._visit_control_node(node)

    def visit_Match(self, node: ast.Match) -> None:  # type: ignore[attr-defined]
        self._visit_control_node(node)

    def visit_Expr(self, node: ast.Expr) -> None:
        if isinstance(node.value, ast.Call):
            self.num_bare_expression_calls += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_pytest_raises_expr(node):
            self.has_pytest_raises = True
            self.num_pytest_raises += 1
            self.num_exception_checks += 1
        self.generic_visit(node)

    def _is_pytest_raises_expr(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False

        func = node.func
        if isinstance(func, ast.Attribute):
            return (
                isinstance(func.value, ast.Name)
                and func.value.id in self.pytest_aliases
                and func.attr == "raises"
            )

        if isinstance(func, ast.Name):
            return func.id in self.raises_aliases

        return False


# =========================================================
# 特征提取
# =========================================================

def infer_test_style(*, num_test_functions: int, num_top_level_asserts: int, num_asserts: int) -> str:
    if num_test_functions > 0 and num_top_level_asserts > 0:
        return "mixed"
    if num_test_functions > 0:
        return "pytest_function"
    if num_top_level_asserts > 0:
        return "top_level_assert"
    if num_asserts > 0:
        return "other_assertion_style"
    return "none"


def build_feature_record(row: Dict[str, Any]) -> Dict[str, Any]:
    code = row.get("candidate_test_code", "") or ""
    token_stats = compute_token_stats(code)
    loc = token_stats["loc"]

    rec = {
        **base_record(SCRIPT_VERSION),
        "candidate_id": row["candidate_id"],
        "task_id": row["task_id"],
        "source": row["source"],
        "difficulty_bucket": row.get("difficulty_bucket", "unknown"),
        "test_source_type": row.get("test_source_type", "unknown"),
        "candidate_test_origin_detail": row.get("candidate_test_origin_detail"),
        "smoke_strength": row.get("smoke_strength"),
        "ast_parse_pass": True,
        "ast_parse_error_type": None,
        "ast_parse_error_msg": None,
        **token_stats,
        "num_test_functions": 0,
        "num_async_test_functions": 0,
        "has_test_function": False,
        "num_asserts": 0,
        "has_assertions": False,
        "num_top_level_asserts": 0,
        "num_asserts_in_test_functions": 0,
        "has_top_level_assert": False,
        "avg_asserts_per_test_function": 0.0,
        "assert_density": 0.0,
        "num_try_except": 0,
        "num_raise_statements": 0,
        "has_pytest_raises": False,
        "num_pytest_raises": 0,
        "num_exception_checks": 0,
        "num_imports_in_test": 0,
        # 兼容旧字段：当前与 num_imports_in_test 等值
        "num_imports_in_candidate": 0,
        "ast_max_depth": 0,
        # 兼容旧字段：当前明确等于 logical_nesting_depth
        "max_nesting_depth": 0,
        "logical_nesting_depth": 0,
        "ast_node_count": 0,
        "num_bare_expression_calls": 0,
        "test_style": "none",
    }

    try:
        tree = ast.parse(code)
        analyzer = Analyzer()
        analyzer.visit(tree)

        avg_asserts = (
            round(
                analyzer.num_asserts_in_test_functions / max(1, analyzer.num_test_functions),
                4,
            )
            if analyzer.num_test_functions > 0
            else 0.0
        )

        rec.update(
            {
                "num_test_functions": analyzer.num_test_functions,
                "num_async_test_functions": analyzer.num_async_test_functions,
                "has_test_function": analyzer.has_test_function,
                "num_asserts": analyzer.num_asserts,
                "has_assertions": analyzer.num_asserts > 0,
                "num_top_level_asserts": analyzer.num_top_level_asserts,
                "num_asserts_in_test_functions": analyzer.num_asserts_in_test_functions,
                "has_top_level_assert": analyzer.num_top_level_asserts > 0,
                "avg_asserts_per_test_function": avg_asserts,
                "assert_density": round(analyzer.num_asserts / max(1, loc), 4),
                "num_try_except": analyzer.num_try_except,
                "num_raise_statements": analyzer.num_raise_statements,
                "has_pytest_raises": analyzer.has_pytest_raises,
                "num_pytest_raises": analyzer.num_pytest_raises,
                "num_exception_checks": analyzer.num_exception_checks,
                "num_imports_in_test": analyzer.num_imports_in_test,
                "num_imports_in_candidate": analyzer.num_imports_in_test,
                "ast_max_depth": analyzer.ast_max_depth,
                "max_nesting_depth": analyzer.logical_nesting_depth,
                "logical_nesting_depth": analyzer.logical_nesting_depth,
                "ast_node_count": analyzer.ast_node_count,
                "num_bare_expression_calls": analyzer.num_bare_expression_calls,
                "test_style": infer_test_style(
                    num_test_functions=analyzer.num_test_functions,
                    num_top_level_asserts=analyzer.num_top_level_asserts,
                    num_asserts=analyzer.num_asserts,
                ),
            }
        )

    except SyntaxError as e:
        rec.update(
            {
                "ast_parse_pass": False,
                "ast_parse_error_type": "SyntaxError",
                "ast_parse_error_msg": clip_text(str(e)),
            }
        )
    except Exception as e:
        rec.update(
            {
                "ast_parse_pass": False,
                "ast_parse_error_type": type(e).__name__,
                "ast_parse_error_msg": clip_text(str(e)),
            }
        )

    return rec


# =========================================================
# 主流程
# =========================================================

def main() -> None:
    rows = load_jsonl(INPUT_PATH)
    out: List[Dict[str, Any]] = [build_feature_record(row) for row in rows]

    write_jsonl(out, OUT_PATH)

    parse_counter = Counter(
        "ast_parse_pass" if row["ast_parse_pass"] else "ast_parse_fail"
        for row in out
    )
    style_counter = Counter(row["test_style"] for row in out)

    manifest = {
        "run_id": out[0].get("run_id") if out else None,
        "script_version": SCRIPT_VERSION,
        "input_path": str(INPUT_PATH),
        "output_path": str(OUT_PATH),
        "num_rows": len(out),
        "parse_counts": dict(parse_counter),
        "test_style_counts": dict(style_counter),
        "field_notes": {
            "num_imports_in_test": "canonical import count for candidate test code",
            "num_imports_in_candidate": "deprecated compatibility alias; currently equal to num_imports_in_test",
            "ast_max_depth": "AST tree depth",
            "logical_nesting_depth": "control-flow nesting depth",
            "max_nesting_depth": "deprecated compatibility alias; currently equal to logical_nesting_depth",
        },
    }
    write_json(manifest, MANIFEST_PATH)

    print(f"wrote {len(out)} rows to {OUT_PATH}")
    print(f"parse_counts={dict(parse_counter)}")
    print(f"test_style_counts={dict(style_counter)}")
    print(f"manifest={MANIFEST_PATH}")


if __name__ == "__main__":
    main()