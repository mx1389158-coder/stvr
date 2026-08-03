from __future__ import annotations

import argparse
import ast
import csv
import fnmatch
import inspect
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "build_external_project_tasks_v2_stvr_extended"
DEFAULT_CONFIG_REL = "stvr/configs/external/stvrexternalprojects.json"


@dataclass(frozen=True)
class Target:
    project: str
    project_src: str
    module: str
    rel_path: str
    function: str
    difficulty: str = "external_project"
    selection_origin: str = "manual"


SOURCE_CACHE: dict[Path, str] = {}
TREE_CACHE: dict[Path, ast.Module] = {}
GIT_HEAD_CACHE: dict[Path, str | None] = {}
SOURCE_LINES_CACHE: dict[int, list[str]] = {}


def read_source(path: Path) -> str:
    path = path.resolve()
    if path not in SOURCE_CACHE:
        SOURCE_CACHE[path] = path.read_text(encoding="utf-8")
    return SOURCE_CACHE[path]


def parse_source(path: Path) -> ast.Module:
    path = path.resolve()
    if path not in TREE_CACHE:
        TREE_CACHE[path] = ast.parse(read_source(path))
    return TREE_CACHE[path]


LEGACY_TARGETS = [
    Target("humanize", "humanize/src", "humanize.filesize", "humanize/filesize.py", "naturalsize"),
    Target("humanize", "humanize/src", "humanize.lists", "humanize/lists.py", "natural_list"),
    Target("humanize", "humanize/src", "humanize.number", "humanize/number.py", "ordinal"),
    Target("humanize", "humanize/src", "humanize.number", "humanize/number.py", "intcomma"),
    Target("humanize", "humanize/src", "humanize.number", "humanize/number.py", "intword"),
    Target("humanize", "humanize/src", "humanize.number", "humanize/number.py", "apnumber"),
    Target("humanize", "humanize/src", "humanize.number", "humanize/number.py", "fractional"),
    Target("humanize", "humanize/src", "humanize.number", "humanize/number.py", "scientific"),
    Target("humanize", "humanize/src", "humanize.number", "humanize/number.py", "clamp"),
    Target("humanize", "humanize/src", "humanize.number", "humanize/number.py", "metric"),
    Target("humanize", "humanize/src", "humanize.time", "humanize/time.py", "naturaldelta"),
    Target("humanize", "humanize/src", "humanize.time", "humanize/time.py", "precisedelta"),
    Target("packaging", "packaging/src", "packaging.utils", "packaging/utils.py", "canonicalize_name"),
    Target("packaging", "packaging/src", "packaging.utils", "packaging/utils.py", "is_normalized_name"),
    Target("packaging", "packaging/src", "packaging.utils", "packaging/utils.py", "canonicalize_version"),
    Target("packaging", "packaging/src", "packaging.utils", "packaging/utils.py", "parse_wheel_filename"),
    Target("packaging", "packaging/src", "packaging.utils", "packaging/utils.py", "parse_sdist_filename"),
    Target("packaging", "packaging/src", "packaging.tags", "packaging/tags.py", "parse_tag"),
    Target("packaging", "packaging/src", "packaging.tags", "packaging/tags.py", "interpreter_name"),
    Target("packaging", "packaging/src", "packaging.tags", "packaging/tags.py", "interpreter_version"),
    Target("packaging", "packaging/src", "packaging.markers", "packaging/markers.py", "default_environment"),
    Target("packaging", "packaging/src", "packaging.pylock", "packaging/pylock.py", "is_valid_pylock_path"),
]


class LaterLocalCallFinder(ast.NodeVisitor):
    def __init__(self, later_names: set[str]) -> None:
        self.later_names = later_names
        self.found: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self.later_names:
            self.found.add(node.func.id)
        self.generic_visit(node)


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_config_path(project_root: Path) -> Path:
    return project_root / DEFAULT_CONFIG_REL


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_git_head(path: Path) -> str | None:
    path = path.resolve()
    if path in GIT_HEAD_CACHE:
        return GIT_HEAD_CACHE[path]
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    value = proc.stdout.strip()
    GIT_HEAD_CACHE[path] = value or None
    return GIT_HEAD_CACHE[path]


def get_function_node(tree: ast.Module, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return node
    raise ValueError(f"Function not found: {function_name}")


def normalize_relative_imports(code: str, package: str) -> str:
    pattern = re.compile(r"^from \.(\w+(?:\.\w+)*) import ", flags=re.MULTILINE)
    code = pattern.sub(lambda m: f"from {package}.{m.group(1)} import ", code)
    code = re.sub(r"^from \. import ", f"from {package} import ", code, flags=re.MULTILINE)
    return code


def source_for_node(source: str, node: ast.AST) -> str:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", None)
    if lineno is None or end_lineno is None:
        raise ValueError("Node is missing source location")
    cache_key = id(source)
    if cache_key not in SOURCE_LINES_CACHE:
        SOURCE_LINES_CACHE[cache_key] = source.splitlines()
    lines = SOURCE_LINES_CACHE[cache_key]
    selected = list(lines[lineno - 1 : end_lineno])
    if not selected:
        raise ValueError("Could not recover source segment")
    col = int(getattr(node, "col_offset", 0) or 0)
    end_col = getattr(node, "end_col_offset", None)
    selected[0] = selected[0][col:]
    if end_col is not None and len(selected) == 1:
        selected[0] = selected[0][: max(0, int(end_col) - col)]
    elif end_col is not None:
        selected[-1] = selected[-1][: int(end_col)]
    return "\n".join(selected).rstrip()


def node_loc(source: str, node: ast.AST) -> int:
    return len([line for line in source_for_node(source, node).splitlines() if line.strip()])


def public_top_level_names(tree: ast.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and not node.name.startswith("_"):
            out[node.name] = getattr(node, "lineno", 10**9)
    return out


def has_later_local_calls(tree: ast.Module, fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    top_names = public_top_level_names(tree)
    later_names = {name for name, lineno in top_names.items() if lineno > getattr(fn_node, "lineno", 0)}
    finder = LaterLocalCallFinder(later_names)
    finder.visit(fn_node)
    return bool(finder.found)


def is_overload_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name) and decorator.id == "overload":
            return True
        if isinstance(decorator, ast.Attribute) and decorator.attr == "overload":
            return True
        if isinstance(decorator, ast.Call):
            func = decorator.func
            if isinstance(func, ast.Name) and func.id == "overload":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "overload":
                return True
    return False


def module_name_from_rel_path(rel_path: str) -> str:
    path = Path(rel_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_solution_source(
    source_path: Path,
    project_src_abs: Path,
    module_name: str,
    function_name: str,
) -> str:
    source = read_source(source_path)
    tree = parse_source(source_path)
    target = get_function_node(tree, function_name)

    future_lines: list[str] = []
    for node in tree.body:
        segment = source_for_node(source, node)
        if segment.startswith("from __future__ import "):
            future_lines.append(segment)

    pieces: list[str] = []
    pieces.extend(future_lines)
    pieces.append(
        "\n".join(
            [
                "import importlib as _utplm_importlib",
                "import sys as _utplm_sys",
                f"_utplm_project_src = {str(project_src_abs)!r}",
                "if _utplm_project_src not in _utplm_sys.path:",
                "    _utplm_sys.path.insert(0, _utplm_project_src)",
                f"_utplm_source_module = _utplm_importlib.import_module({module_name!r})",
                "globals().update(vars(_utplm_source_module))",
            ]
        )
    )
    pieces.append(source_for_node(source, target))
    return "\n\n".join(pieces).rstrip() + "\n"


def clean_docstring(doc: str | None, max_chars: int = 2200) -> str:
    if not doc:
        return ""
    doc = inspect.cleandoc(doc)
    doc = re.sub(r"\n{3,}", "\n\n", doc)
    return doc[:max_chars].strip()


def function_signature_from_source(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    segment = source_for_node(source, node)
    lines = segment.splitlines()
    header_lines: list[str] = []
    balance = 0
    for line in lines:
        header_lines.append(line.rstrip())
        balance += line.count("(") - line.count(")")
        if line.rstrip().endswith(":") and balance <= 0:
            break
    header = "\n".join(header_lines).strip()
    prefix = f"def {node.name}"
    if header.startswith("async def "):
        prefix = f"async def {node.name}"
    if not header.startswith(prefix):
        return "(...)"
    sig = header[len(prefix):].strip()
    if sig.endswith(":"):
        sig = sig[:-1]
    return sig


def build_prompt(module_name: str, function_name: str, signature: str, doc: str, source_excerpt: str) -> str:
    parts = [
        f"Open-source project target: {module_name}.{function_name}",
        "",
        f"Signature: {function_name}{signature}",
    ]
    if doc:
        parts.extend(["", "Docstring:", doc])
    parts.extend(
        [
            "",
            "Relevant implementation excerpt:",
            source_excerpt[:2600].strip(),
            "",
            "Write pytest-style tests for this target. Focus on externally visible behavior, normal cases, edge cases, and documented exceptions. Do not reimplement the target.",
        ]
    )
    return "\n".join(parts).strip()


def excluded(rel_path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pat) for pat in patterns)


def discover_project_targets(spec: dict[str, Any], external_root: Path) -> tuple[list[Target], dict[str, Any]]:
    project = str(spec["project"])
    project_src = str(spec["project_src"])
    project_src_abs = external_root / project_src
    if not project_src_abs.exists():
        raise FileNotFoundError(f"Missing project source for {project}: {project_src_abs}")

    max_functions = int(spec.get("max_functions", 16))
    min_function_loc = int(spec.get("min_function_loc", 2))
    max_function_loc = int(spec.get("max_function_loc", 90))
    include_globs = list(spec.get("include_globs") or ["**/*.py"])
    exclude_globs = list(spec.get("exclude_globs") or [])

    targets: list[Target] = []
    seen: set[tuple[str, str]] = set()
    preferred = list(spec.get("preferred_targets") or [])
    for item in preferred:
        rel_path = str(item["rel_path"])
        function = str(item["function"])
        module = str(item.get("module") or module_name_from_rel_path(rel_path))
        target = Target(project, project_src, module, rel_path, function, selection_origin="preferred")
        targets.append(target)
        seen.add((rel_path, function))

    candidates: list[tuple[str, str, str]] = []
    for pattern in include_globs:
        for path in sorted(project_src_abs.glob(pattern)):
            if not path.is_file() or path.suffix != ".py":
                continue
            rel_path = path.relative_to(project_src_abs).as_posix()
            if excluded(rel_path, exclude_globs):
                continue
            try:
                source = read_source(path)
                tree = parse_source(path)
            except Exception:
                continue
            module = module_name_from_rel_path(rel_path)
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef):
                    continue
                if node.name.startswith("_") or (rel_path, node.name) in seen:
                    continue
                if is_overload_function(node):
                    continue
                loc = node_loc(source, node)
                if loc < min_function_loc or loc > max_function_loc:
                    continue
                if has_later_local_calls(tree, node):
                    continue
                candidates.append((rel_path, module, node.name))
                seen.add((rel_path, node.name))

    for rel_path, module, function in sorted(candidates):
        if len(targets) >= max_functions:
            break
        targets.append(Target(project, project_src, module, rel_path, function, selection_origin="discovered"))
        seen.add((rel_path, function))

    report = {
        "project": project,
        "project_src": project_src,
        "max_functions": max_functions,
        "selected_functions": len(targets),
        "preferred_functions": sum(1 for t in targets if t.selection_origin == "preferred"),
        "discovered_functions": sum(1 for t in targets if t.selection_origin == "discovered"),
    }
    return targets, report


def load_targets(args: argparse.Namespace, project_root: Path, external_root: Path) -> tuple[list[Target], dict[str, Any], list[dict[str, Any]]]:
    if args.target_set == "legacy":
        return LEGACY_TARGETS, {"target_set": "legacy", "total_min_tasks": 0}, []

    config_path = Path(args.project_config).resolve() if args.project_config else default_config_path(project_root)
    config = load_json(config_path)
    project_reports: list[dict[str, Any]] = []
    targets: list[Target] = []
    missing: list[dict[str, Any]] = []

    for spec in config.get("projects", []):
        project_src_abs = external_root / str(spec["project_src"])
        if not project_src_abs.exists():
            missing.append(spec)
            continue
        project_targets, report = discover_project_targets(spec, external_root)
        targets.extend(project_targets)
        project_reports.append(report)

    if missing and not args.allow_missing_projects:
        lines = ["Missing external project sources. Prepare them first:"]
        for spec in missing:
            ref = str(spec.get("ref") or "").strip()
            suffix = f" && git checkout {ref}" if ref else ""
            lines.append(f"  git clone {spec.get('repo_url')} {external_root / spec['project']}{suffix}")
        raise FileNotFoundError("\n".join(lines))

    max_total = config.get("total_max_tasks")
    if max_total:
        targets = targets[: int(max_total)]

    config["project_config_path"] = str(config_path)
    return targets, config, project_reports


def task_id_for(target: Target, function_counts: dict[tuple[str, str], int]) -> str:
    base = f"external_{target.project}_{target.function}"
    if function_counts[(target.project, target.function)] <= 1:
        return re.sub(r"[^A-Za-z0-9_]+", "_", base)
    module_tail = target.module.split(".")[-1]
    return re.sub(r"[^A-Za-z0-9_]+", "_", f"external_{target.project}_{module_tail}_{target.function}")


def build_row(target: Target, external_root: Path, function_counts: dict[tuple[str, str], int]) -> dict[str, Any]:
    project_src_abs = external_root / target.project_src
    source_path = project_src_abs / target.rel_path
    module_source = read_source(source_path)
    tree = parse_source(source_path)
    fn_node = get_function_node(tree, target.function)
    signature = function_signature_from_source(module_source, fn_node)
    doc = clean_docstring(ast.get_docstring(fn_node))
    solution = build_solution_source(source_path, project_src_abs, target.module, target.function)
    source_excerpt = source_for_node(module_source, fn_node)
    task_id = task_id_for(target, function_counts)
    project_root = external_root / target.project
    project_commit = get_git_head(project_root)

    return {
        "task_id": task_id,
        "source": f"external_{target.project}",
        "difficulty_bucket": target.difficulty,
        "entry_point": target.function,
        "signature": f"def {target.function}{signature}:",
        "task_type": "external_project_function",
        "prompt": build_prompt(target.module, target.function, signature, doc, source_excerpt),
        "canonical_solution": solution,
        "imports": "",
        "base_tests": "[]",
        "eval_assets": json.dumps(
            {
                "external_project": target.project,
                "external_project_commit": project_commit,
                "module": target.module,
                "source_file": str(source_path),
                "target_function": target.function,
                "mutation_scope": "target_function_smoke",
                "selection_origin": target.selection_origin,
            },
            ensure_ascii=False,
        ),
        "external_project": target.project,
        "external_project_commit": project_commit,
        "external_module": target.module,
        "external_source_file": str(source_path),
        "external_project_src": str(project_src_abs),
        "selection_origin": target.selection_origin,
        "script_version": SCRIPT_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(find_repo_root()))
    parser.add_argument("--external_root", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--target_set", choices=["legacy", "stvr_extended"], default="stvr_extended")
    parser.add_argument("--project_config", default=None)
    parser.add_argument("--allow_missing_projects", action="store_true")
    parser.add_argument("--min_tasks", type=int, default=None)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else project_root / "data" / "raw" / "external_projects"
    out_csv = Path(args.out_csv).resolve() if args.out_csv else project_root / "data" / "processed" / "externalprojectvalidationtasks.csv"
    manifest_path = Path(args.manifest).resolve() if args.manifest else out_csv.with_suffix(".manifest.json")

    targets, config, project_reports = load_targets(args, project_root, external_root)
    min_tasks = args.min_tasks
    if min_tasks is None:
        min_tasks = int(config.get("total_min_tasks", 0) or 0)
    if len(targets) < min_tasks:
        raise RuntimeError(f"Selected {len(targets)} external tasks, below required minimum {min_tasks}")

    function_counts: dict[tuple[str, str], int] = {}
    for target in targets:
        key = (target.project, target.function)
        function_counts[key] = function_counts.get(key, 0) + 1

    rows = [build_row(target, external_root, function_counts) for target in targets]
    if not rows:
        raise RuntimeError("No external validation tasks were selected")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "target_set": args.target_set,
        "project_root": str(project_root),
        "external_root": str(external_root),
        "out_csv": str(out_csv),
        "n_tasks": len(rows),
        "n_projects": len({row["external_project"] for row in rows}),
        "projects": sorted({row["external_project"] for row in rows}),
        "project_reports": project_reports,
        "selection_policy": config.get("selection_policy"),
        "project_config_path": config.get("project_config_path"),
        "targets": [
            {
                "task_id": row["task_id"],
                "project": row["external_project"],
                "commit": row.get("external_project_commit"),
                "module": row["external_module"],
                "entry_point": row["entry_point"],
                "selection_origin": row["selection_origin"],
            }
            for row in rows
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
