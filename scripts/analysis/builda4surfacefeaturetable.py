from __future__ import annotations

import ast
import io
import json
import os
import token
import tokenize
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
RUN_TAG = os.environ.get("a4_RUN_TAG", "a17bpoolv1")

a1_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
a2_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a2"
a3_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a3"
OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

a1_MASTER = Path(os.environ.get(
    "a4_a1_MASTER_CSV",
    str(a1_DIR / "a1autometricsmastertable.csv")
)).resolve()

a2_NATNEG = Path(os.environ.get(
    "a4_a2_NATNEG_CSV",
    str(a2_DIR / f"a2naturalmechanisticnegativepool{RUNTAG}.csv")
)).resolve()

a2_PROGNEG = Path(os.environ.get(
    "a4_a2_PROGNEG_CSV",
    str(a2_DIR / "a2prognegautometricsmastertable.csv")
)).resolve()

a3_HIGH = Path(os.environ.get(
    "a4_a3_HIGH_CSV",
    str(a3_DIR / f"a3positivehighqualitypool{RUNTAG}.csv")
)).resolve()

a3_WEAK = Path(os.environ.get(
    "a4_a3_WEAK_CSV",
    str(a3_DIR / f"a3positiveweakqualitypool{RUNTAG}.csv")
)).resolve()

SCRIPT_VERSION = "build_a4_surface_feature_table_union_v1"

IGNORED_TOKENS = {
    token.ENDMARKER,
    token.NEWLINE,
    token.NL,
    token.INDENT,
    token.DEDENT,
    getattr(token, "ENCODING", -1),
}


def require_file(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_bool_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0).gt(0).astype(np.int8)


def infer_current_run_id(df: pd.DataFrame) -> str:
    if "run_id" not in df.columns:
        return "unknown_run"
    vals = df["run_id"].dropna().astype(str).str.strip()
    vals = vals[vals != ""].unique()
    if len(vals) == 1:
        return vals[0]
    if len(vals) == 0:
        return "unknown_run"
    return "mixed_runs"


def load_pool(path: Path, pool_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = ["candidate_id", "task_id", "candidate_test_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{pool_name} missing required columns: {missing}")

    defaults = {
        "source": "unknown",
        "difficulty_bucket": "unknown",
        "run_id": None,
        "syntax_pass": 0,
        "execution_pass": 0,
        "num_asserts": 0,
        "assert_density": np.nan,
        "logical_nesting_depth": np.nan,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v

    keep = [
        "candidate_id",
        "task_id",
        "source",
        "difficulty_bucket",
        "run_id",
        "candidate_test_code",
        "syntax_pass",
        "execution_pass",
        "num_asserts",
        "assert_density",
        "logical_nesting_depth",
    ]
    out = df[keep].copy()
    out["origin_pool"] = pool_name

    out["candidate_id"] = out["candidate_id"].astype(str)
    out["task_id"] = out["task_id"].astype(str)
    out["source"] = out["source"].astype(str)
    out["difficulty_bucket"] = out["difficulty_bucket"].astype(str)
    out["candidate_test_code"] = out["candidate_test_code"].fillna("").astype(str)

    out["syntax_pass"] = safe_bool_series(out["syntax_pass"])
    out["execution_pass"] = safe_bool_series(out["execution_pass"])
    out["num_asserts"] = safe_numeric(out["num_asserts"])
    out["assert_density"] = safe_numeric(out["assert_density"])
    out["logical_nesting_depth"] = safe_numeric(out["logical_nesting_depth"])

    return out


def compute_token_stats(code: str) -> Dict[str, Any]:
    token_length = 0
    num_comments = 0
    num_inline_comments = 0
    comment_lines: Set[int] = set()

    lines = code.splitlines()
    loc = sum(1 for ln in lines if ln.strip())
    nonempty_line_count = loc
    char_length = len(code)

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
        "char_length": char_length,
        "nonempty_line_count": nonempty_line_count,
        "token_length": token_length,
        "num_comments": num_comments,
        "num_comment_lines": len(comment_lines),
        "num_inline_comments": num_inline_comments,
        "comment_ratio": round(len(comment_lines) / max(1, loc), 4),
    }


class Analyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.num_test_functions = 0
        self.num_async_test_functions = 0
        self.logical_nesting_depth = 0
        self._logical_depth = 0
        self.ast_node_count = 0
        self.ast_max_depth = 0
        self._ast_depth = 0

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

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name.startswith("test"):
            self.num_test_functions += 1
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        if node.name.startswith("test"):
            self.num_async_test_functions += 1
            self.num_test_functions += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self._visit_control_node(node)

    def visit_For(self, node: ast.For) -> None:
        self._visit_control_node(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._visit_control_node(node)

    def visit_While(self, node: ast.While) -> None:
        self._visit_control_node(node)

    def visit_Try(self, node: ast.Try) -> None:
        self._visit_control_node(node)

    def visit_With(self, node: ast.With) -> None:
        self._visit_control_node(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._visit_control_node(node)


def analyze_code(code: str) -> Dict[str, Any]:
    code = (code or "").strip()
    token_stats = compute_token_stats(code)

    out = {
        **token_stats,
        "ast_parse_pass": 0,
        "num_test_functions": 0,
        "num_async_test_functions": 0,
        "ast_node_count": None,
        "ast_max_depth": None,
        "logical_nesting_depth_recomputed": None,
    }

    if not code:
        return out

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return out

    analyzer = Analyzer()
    analyzer.visit(tree)

    out.update(
        {
            "ast_parse_pass": 1,
            "num_test_functions": int(analyzer.num_test_functions),
            "num_async_test_functions": int(analyzer.num_async_test_functions),
            "ast_node_count": int(analyzer.ast_node_count),
            "ast_max_depth": int(analyzer.ast_max_depth),
            "logical_nesting_depth_recomputed": int(analyzer.logical_nesting_depth),
        }
    )
    return out


def dedupe_union(df: pd.DataFrame) -> pd.DataFrame:
    # 同一个 candidate_id 若出现多次，优先保留较“靠后”的 pool：
    # programmatic > natneg > a3_high > a3_weak > a1_master
    priority = {
        "a1_master": 0,
        "a3_weak": 1,
        "a3_high": 2,
        "a2_natneg": 3,
        "a2_progneg": 4,
    }
    df = df.copy()
    df["_priority"] = df["origin_pool"].map(priority).fillna(-1)
    df = df.sort_values(["candidate_id", "_priority"], ascending=[True, False], kind="stable")
    df = df.drop_duplicates(subset=["candidate_id"], keep="first").drop(columns="_priority")
    return df.reset_index(drop=True)


def main() -> None:
    for p, name in [
        (a1_MASTER, "a1_MASTER"),
        (a2_NATNEG, "a2_NATNEG"),
        (a2_PROGNEG, "a2_PROGNEG"),
        (a3_HIGH, "a3_HIGH"),
        (a3_WEAK, "a3_WEAK"),
    ]:
        require_file(p, name)

    parts = [
        load_pool(a1_MASTER, "a1_master"),
        load_pool(a3_HIGH, "a3_high"),
        load_pool(a3_WEAK, "a3_weak"),
        load_pool(a2_NATNEG, "a2_natneg"),
        load_pool(a2_PROGNEG, "a2_progneg"),
    ]

    union_df = pd.concat(parts, ignore_index=True)
    before_dedupe = len(union_df)
    union_df = dedupe_union(union_df)
    after_dedupe = len(union_df)

    current_run_id = infer_current_run_id(union_df)
    safe_run = RUN_TAG

    feature_rows: List[Dict[str, Any]] = []
    for _, row in union_df.iterrows():
        extra = analyze_code(str(row["candidate_test_code"]))
        feature_rows.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "task_id": str(row["task_id"]),
                "source": str(row["source"]),
                "difficulty_bucket": str(row["difficulty_bucket"]),
                "run_id": row.get("run_id"),
                "origin_pool": str(row["origin_pool"]),
                "syntax_pass": int(row["syntax_pass"]),
                "execution_pass": int(row["execution_pass"]),
                "num_asserts": row["num_asserts"],
                "assert_density": row["assert_density"],
                "logical_nesting_depth": row["logical_nesting_depth"],
                **extra,
            }
        )

    out = pd.DataFrame(feature_rows)

    out["surface_structural_complexity"] = out["logical_nesting_depth"]
    miss_mask = out["surface_structural_complexity"].isna()
    out.loc[miss_mask, "surface_structural_complexity"] = out.loc[miss_mask, "logical_nesting_depth_recomputed"]

    out = out.sort_values(
        [c for c in ["origin_pool", "source", "difficulty_bucket", "task_id", "candidate_id"] if c in out.columns],
        kind="stable",
    ).reset_index(drop=True)

    out_csv = OUT_DIR / f"a4surfacefeaturetableunion{saferun}.csv"
    out_report = OUT_DIR / f"a4surfacefeatureunionreport{saferun}.json"

    out.to_csv(out_csv, index=False, encoding="utf-8")

    report = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(PROJECT_ROOT),
        "run_tag": RUN_TAG,
        "inputs": {
            "a1_master": str(a1_MASTER),
            "a2_natneg": str(a2_NATNEG),
            "a2_progneg": str(a2_PROGNEG),
            "a3_high": str(a3_HIGH),
            "a3_weak": str(a3_WEAK),
        },
        "counts": {
            "union_rows_before_dedupe": int(before_dedupe),
            "union_rows_after_dedupe": int(after_dedupe),
            "output_rows": int(len(out)),
        },
        "origin_pool_counts": {
            str(k): int(v) for k, v in out["origin_pool"].value_counts(dropna=False).to_dict().items()
        },
        "columns": list(out.columns),
        "notes": [
            "This union surface table is built to cover all candidates that may enter a4/b1/c1/D2 pair builders.",
            "Programmatic negatives must be included; otherwise pair-level surface checks can systematically false-fail.",
        ],
        "output_paths": {
            "surface_feature_union_table": str(out_csv),
            "report": str(out_report),
        },
    }
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] wrote {out_csv}")
    print(f"[ok] wrote {out_report}")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()