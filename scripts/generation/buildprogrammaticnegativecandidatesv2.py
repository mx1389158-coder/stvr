from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


SCRIPT_VERSION = "build_programmatic_negative_candidates_v2_fixed"


def strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^\s*```(?:python|py)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def choose_prompt_col(df: pd.DataFrame) -> str:
    for col in ["prompt", "problem", "task_description", "text"]:
        if col in df.columns:
            return col
    raise ValueError("No prompt-like column found.")


def count_nonempty_lines(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


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


def analyze_validity(code: str, entry_point: Optional[str]) -> Dict[str, Any]:
    out = {
        "invalid_candidate_flag": 0,
        "candidate_validity_status": "ok",
        "redefines_target_flag": 0,
        "invalid_candidate_reason": None,
    }
    try:
        tree = ast.parse(code)
    except Exception:
        out["invalid_candidate_flag"] = 1
        out["candidate_validity_status"] = "syntax_error"
        out["invalid_candidate_reason"] = "syntax_error"
        return out

    if entry_point:
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == entry_point:
                out["invalid_candidate_flag"] = 1
                out["candidate_validity_status"] = "invalid_redefines_target"
                out["redefines_target_flag"] = 1
                out["invalid_candidate_reason"] = "redefines_target"
                return out
    return out


def extract_test_functions(tree: ast.Module) -> List[ast.FunctionDef]:
    return [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]


def remove_function_by_lineno(code: str, fn: ast.FunctionDef) -> str:
    lines = code.splitlines(keepends=True)
    start = fn.lineno - 1
    end = fn.end_lineno
    new_lines = lines[:start] + lines[end:]
    return "".join(new_lines).strip() + "\n"


def choose_target_fn(funcs: List[ast.FunctionDef], code: str) -> ast.FunctionDef:
    heur = re.compile(r"(boundary|edge|empty|none|zero|one|min|max|exception|error|raise)", re.I)
    for fn in funcs:
        src = ast.get_source_segment(code, fn) or ""
        if heur.search(fn.name) or heur.search(src):
            return fn
        if any(tok in src for tok in ["None", "[]", "{}", '""', "0", "-1", "1", "pytest.raises"]):
            return fn
    return funcs[-1]


def weaken_assert_line(line: str) -> str:
    m = re.match(r"^(\s*)assert\s+(.+)$", line)
    if not m:
        return line
    indent, expr = m.groups()
    lhs = expr
    if "==" in expr:
        lhs = expr.split("==", 1)[0].strip()
    elif "!=" in expr:
        lhs = expr.split("!=", 1)[0].strip()
    return indent + f"assert {lhs} is not None\n"


def weaken_all_asserts_in_fn(code: str, fn: ast.FunctionDef) -> str:
    lines = code.splitlines(keepends=True)
    for i in range(fn.lineno - 1, fn.end_lineno):
        if lines[i].lstrip().startswith("assert "):
            lines[i] = weaken_assert_line(lines[i])
    return "".join(lines).strip() + "\n"


def keep_first_test_and_weaken_rest(code: str, funcs: List[ast.FunctionDef]) -> str:
    lines = code.splitlines(keepends=True)
    skip_ranges = []
    for fn in funcs[1:]:
        skip_ranges.append((fn.lineno - 1, fn.end_lineno))

    out = []
    cur = 0
    for start, end in sorted(skip_ranges):
        out.extend(lines[cur:start])
        cur = end
    out.extend(lines[cur:])
    code2 = "".join(out).strip() + "\n"

    try:
        tree2 = ast.parse(code2)
        funcs2 = extract_test_functions(tree2)
        if funcs2:
            code2 = weaken_all_asserts_in_fn(code2, funcs2[0])
    except Exception:
        pass
    return code2


def sanitize_id_piece(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:120]


def build_records(
    anchor_csv: Path,
    formal_csv: Path,
    run_id: str,
    only_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    anchor = pd.read_csv(anchor_csv)
    formal = pd.read_csv(formal_csv)

    for df in [anchor, formal]:
        if "task_id" not in df.columns:
            raise ValueError("Missing required column: task_id")
        if "source" not in df.columns:
            raise ValueError("Missing required column: source")
        if "difficulty_bucket" not in df.columns:
            raise ValueError("Missing required column: difficulty_bucket")
        df["task_id"] = df["task_id"].astype(str)
        df["source"] = df["source"].astype(str)
        df["difficulty_bucket"] = df["difficulty_bucket"].astype(str)

    if "candidate_id" not in anchor.columns:
        raise ValueError("anchor_csv missing required column: candidate_id")
    if "candidate_test_code" not in anchor.columns:
        raise ValueError("anchor_csv missing required column: candidate_test_code")

    anchor["candidate_id"] = anchor["candidate_id"].astype(str)

    if only_source:
        anchor = anchor[anchor["source"] == only_source].copy()

    if anchor.empty:
        return []

    prompt_col = choose_prompt_col(formal)
    formal = formal.drop_duplicates(subset=["task_id"], keep="first")

    formal_keep_cols = [
        "task_id",
        prompt_col,
        "canonical_solution",
        "base_tests",
        "imports",
        "signature",
        "task_type",
        "eval_assets",
        "entry_point",
    ]
    formal_keep_cols = [c for c in formal_keep_cols if c in formal.columns]

    merged = anchor.merge(
        formal[formal_keep_cols],
        on="task_id",
        how="left",
        suffixes=("", "_formal"),
    )

    rows: List[Dict[str, Any]] = []
    seen_candidate_ids = set()

    for _, row in merged.iterrows():
        task_id = str(row["task_id"])
        source = str(row["source"])
        diff = str(row["difficulty_bucket"])
        anchor_cid = str(row["candidate_id"])
        entry_point = row.get("entry_point")
        code = strip_code_fences(str(row["candidate_test_code"]))

        try:
            tree = ast.parse(code)
        except Exception:
            continue

        funcs = extract_test_functions(tree)
        if not funcs:
            continue

        target_fn = choose_target_fn(funcs, code)

        variants: List[tuple[str, str]] = []

        if len(funcs) > 1:
            variants.append(("drop_targeted_test", remove_function_by_lineno(code, target_fn)))

        variants.append(("weaken_target_asserts", weaken_all_asserts_in_fn(code, target_fn)))

        exception_like = []
        name_re = re.compile(r"(exception|error|raise|invalid|none|empty)", re.I)
        for fn in funcs:
            src = ast.get_source_segment(code, fn) or ""
            if name_re.search(fn.name) or "pytest.raises" in src:
                exception_like.append(fn)

        for j, fn in enumerate(exception_like[:2], start=1):
            if len(funcs) > 1:
                variants.append((f"drop_exception_like_{j}", remove_function_by_lineno(code, fn)))

        if len(funcs) >= 2:
            variants.append(("keep_first_test_and_weaken_rest", keep_first_test_and_weaken_rest(code, funcs)))

        seen_code = set()
        prompt_value = row.get(f"{prompt_col}_formal", row.get(prompt_col))

        for idx, (variant_name, cand_code) in enumerate(variants, start=1):
            cand_code = strip_code_fences(cand_code)
            if not cand_code or cand_code in seen_code:
                continue
            seen_code.add(cand_code)

            parse_info = ast_parse_check(cand_code)
            validity_info = analyze_validity(cand_code, entry_point)

            safe_anchor = sanitize_id_piece(anchor_cid)
            cand_id = f"{task_id}__from_{safe_anchor}__progneg2__v{idx:02d}"

            if cand_id in seen_candidate_ids:
                # 极端情况下再补一个后缀，确保全局唯一
                dedup_n = 2
                cand_id_base = cand_id
                while cand_id in seen_candidate_ids:
                    cand_id = f"{cand_id_base}__dup{dedup_n}"
                    dedup_n += 1
            seen_candidate_ids.add(cand_id)

            rows.append(
                {
                    "run_id": run_id,
                    "script_version": SCRIPT_VERSION,
                    "candidate_id": cand_id,
                    "parent_candidate_id": anchor_cid,
                    "anchor_candidate_id": anchor_cid,
                    "task_id": task_id,
                    "source": source,
                    "difficulty_bucket": diff,
                    "test_source_type": "programmatic_constructed",
                    "candidate_test_origin_detail": f"from_positive_anchor_round2__{variant_name}",
                    "anchor_test_source_type": row.get("test_source_type"),
                    "anchor_candidate_test_origin_detail": row.get("candidate_test_origin_detail"),
                    "prompt": prompt_value,
                    "candidate_test_code": cand_code,
                    "raw_generation_text": cand_code,
                    "candidate_char_len": len(cand_code),
                    "candidate_nonempty_line_count": count_nonempty_lines(cand_code),
                    "generator_role": "programmatic_negative_construction",
                    "candidate_source": "programmatic_negative_from_positive_anchor",
                    **parse_info,
                    **validity_info,
                    "canonical_solution": row.get("canonical_solution"),
                    "base_tests": row.get("base_tests"),
                    "imports": row.get("imports"),
                    "signature": row.get("signature"),
                    "task_type": row.get("task_type"),
                    "eval_assets": row.get("eval_assets"),
                    "entry_point": row.get("entry_point"),
                }
            )

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor_csv", required=True)
    ap.add_argument("--formal_csv", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--only_source", default=None)
    args = ap.parse_args()

    project_root = Path(os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")).resolve()
    out_dir = project_root / "data" / "interim" / "generatedcandidates" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = build_records(
        Path(args.anchor_csv),
        Path(args.formal_csv),
        args.run_id,
        args.only_source,
    )

    out_jsonl = out_dir / "generatedcandidates.jsonl"
    compat_jsonl = out_dir / "smokecandidates.jsonl"
    manifest = out_dir / "manifest.json"

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(compat_jsonl, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_obj = {
        "run_id": args.run_id,
        "script_version": SCRIPT_VERSION,
        "anchor_csv": str(Path(args.anchor_csv).resolve()),
        "formal_csv": str(Path(args.formal_csv).resolve()),
        "only_source": args.only_source,
        "project_root": str(project_root),
        "output_jsonl": str(out_jsonl),
        "compat_jsonl": str(compat_jsonl),
        "num_candidates": len(rows),
        "notes": [
            "candidate_id encodes anchor_candidate_id to avoid collisions across same-task anchors",
            "parent_candidate_id / anchor_candidate_id are preserved for pair-level auditing",
            "this script constructs programmatic negatives from positive anchors",
        ],
    }

    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, ensure_ascii=False, indent=2)

    print("wrote:", out_jsonl)
    print("wrote:", compat_jsonl)
    print("wrote:", manifest)
    print("num_candidates =", len(rows))


if __name__ == "__main__":
    main()