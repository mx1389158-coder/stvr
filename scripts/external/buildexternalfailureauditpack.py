from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "build_external_failure_audit_pack_v1"

DEFAULT_RUNS = [
    ("base", "0", "external89bases11"),
    ("LL", "11", "external89lls11"),
    ("HL", "11", "external89hls11"),
    ("HH", "11", "external89hhs11"),
]

MANUAL_COLUMNS = [
    "manual_validity",
    "manual_failure_type",
    "auto_category_agree",
    "reveals_stricter_behavior",
    "manual_note",
    "auditor_id",
]

PREFERRED_FAILURE_CATEGORIES = [
    "execution_assertion_mismatch",
    "execution_call_contract_error",
    "test_runtime_failure",
    "assertion_weak_or_missing",
    "exception_path_gap",
    "mutation_survival_major",
    "mutation_unavailable",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_assets(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def stable_float(*parts: str) -> float:
    key = "||".join(parts).encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()[:12]
    return int(digest, 16) / float(16**12)


def short_text(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if len(text) <= limit:
        return text
    return text[: limit - 120].rstrip() + "\n...[truncated]...\n" + text[-100:].lstrip()


def read_tail(path_value: Any, limit: int = 1600) -> str:
    if not path_value:
        return ""
    path = Path(str(path_value))
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return short_text(text[-limit:], limit)


def source_excerpt_from_prompt(prompt: str, limit: int = 2200) -> str:
    marker = "Relevant implementation excerpt:"
    if marker not in prompt:
        return short_text(prompt, limit)
    excerpt = prompt.split(marker, 1)[1]
    tail_marker = "\n\nWrite pytest-style tests"
    if tail_marker in excerpt:
        excerpt = excerpt.split(tail_marker, 1)[0]
    return short_text(excerpt.strip(), limit)


def run_specs(values: list[str]) -> list[tuple[str, str, str]]:
    if not values:
        return DEFAULT_RUNS
    specs: list[tuple[str, str, str]] = []
    for value in values:
        parts = value.split(":", 2)
        if len(parts) != 3:
            raise ValueError(f"Run spec must be group:seed:run_id, got {value!r}")
        specs.append((parts[0], parts[1], parts[2]))
    return specs


def enrich_failure_rows(root: Path, specs: list[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, seed, run_id in specs:
        generated = {
            row.get("candidate_id"): row
            for row in load_jsonl(root / "data" / "interim" / "generatedcandidates" / run_id / "generatedcandidates.jsonl")
        }
        execution = {
            row.get("candidate_id"): row
            for row in load_jsonl(root / "data" / "interim" / "executionresults" / run_id / "executionresults.jsonl")
        }
        mutation = {
            row.get("candidate_id"): row
            for row in load_jsonl(root / "data" / "interim" / "mutationresults" / run_id / "mutationresults.jsonl")
        }
        static = {
            row.get("candidate_id"): row
            for row in load_jsonl(root / "data" / "interim" / "staticfeatures" / run_id / "staticfeatures.jsonl")
        }
        failure_logs = load_jsonl(root / "data" / "interim" / "failuretaxonomylogs" / run_id / "failuretaxonomylogs.jsonl")

        for failure in failure_logs:
            candidate_id = failure.get("candidate_id")
            gen = generated.get(candidate_id, {})
            exe = execution.get(candidate_id, {})
            mut = mutation.get(candidate_id, {})
            stat = static.get(candidate_id, {})
            assets = parse_assets(gen.get("eval_assets"))
            project = assets.get("external_project") or gen.get("external_project") or str(failure.get("source", "")).removeprefix("external_")
            row = {
                "group": group,
                "seed": seed,
                "run_id": run_id,
                "candidate_id": candidate_id,
                "task_id": failure.get("task_id") or gen.get("task_id"),
                "external_project": project,
                "external_module": assets.get("module") or gen.get("external_module"),
                "entry_point": gen.get("entry_point"),
                "signature": gen.get("signature"),
                "selection_origin": assets.get("selection_origin"),
                "auto_stage": failure.get("stage"),
                "auto_failure_category": failure.get("failure_category"),
                "auto_failure_subtype": failure.get("failure_subtype"),
                "auto_rule_id": failure.get("rule_id"),
                "auto_rule_strength": failure.get("rule_strength"),
                "auto_severity": failure.get("severity"),
                "auto_evidence_json": json.dumps(failure.get("evidence", {}), ensure_ascii=False, sort_keys=True),
                "execution_status": exe.get("execution_status"),
                "execution_pass": exe.get("execution_pass"),
                "exception_type": exe.get("exception_type"),
                "traceback_snippet": short_text(exe.get("traceback_snippet"), 1600),
                "stdout_tail": read_tail(exe.get("stdout_path")),
                "stderr_tail": read_tail(exe.get("stderr_path")),
                "num_tests_collected": exe.get("num_tests_collected"),
                "num_tests_passed": exe.get("num_tests_passed"),
                "num_tests_failed": exe.get("num_tests_failed"),
                "num_tests_errors": exe.get("num_tests_errors"),
                "mutation_status": mut.get("mutation_status"),
                "mutation_score": mut.get("mutation_score"),
                "mutants_total": mut.get("mutants_total"),
                "mutants_killed": mut.get("mutants_killed"),
                "mutants_survived": mut.get("mutants_survived"),
                "num_asserts": stat.get("num_asserts"),
                "assert_density": stat.get("assert_density"),
                "test_style": stat.get("test_style"),
                "candidate_test_code": short_text(gen.get("candidate_test_code"), 4500),
                "target_source_excerpt": source_excerpt_from_prompt(str(gen.get("prompt") or "")),
                "prompt_excerpt": short_text(gen.get("prompt"), 1800),
            }
            rows.append(row)
    return rows


def select_stratified(rows: list[dict[str, Any]], sample_size: int, seed: str) -> list[dict[str, Any]]:
    preferred = set(PREFERRED_FAILURE_CATEGORIES)
    candidates = [row for row in rows if row.get("auto_failure_category") in preferred]
    if len(candidates) < sample_size:
        candidates = list(rows)

    selected: list[dict[str, Any]] = []
    used_candidates: set[str] = set()
    group_counts: Counter[str] = Counter()
    project_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    stage_counts: Counter[str] = Counter()

    target_group = max(1, sample_size // max(1, len({str(r.get("group")) for r in candidates})))
    target_project = max(1, sample_size // max(1, len({str(r.get("external_project")) for r in candidates})))
    target_category = max(1, sample_size // max(1, len({str(r.get("auto_failure_category")) for r in candidates})))

    remaining = list(candidates)
    while remaining and len(selected) < sample_size:
        best_index = 0
        best_score = float("-inf")
        for index, row in enumerate(remaining):
            group = str(row.get("group") or "")
            project = str(row.get("external_project") or "")
            category = str(row.get("auto_failure_category") or "")
            stage = str(row.get("auto_stage") or "")
            candidate_id = str(row.get("candidate_id") or "")

            score = 0.0
            if candidate_id not in used_candidates:
                score += 120
            if group_counts[group] < target_group:
                score += 55
            if project_counts[project] < target_project:
                score += 35
            if category_counts[category] < target_category:
                score += 45
            if category_counts[category] == 0:
                score += 35
            if stage_counts[stage] == 0:
                score += 15
            score -= group_counts[group] * 4
            score -= project_counts[project] * 2
            score -= category_counts[category] * 3
            score += stable_float(seed, candidate_id, category, str(index))

            if score > best_score:
                best_score = score
                best_index = index

        row = remaining.pop(best_index)
        selected.append(row)
        used_candidates.add(str(row.get("candidate_id") or ""))
        group_counts[str(row.get("group") or "")] += 1
        project_counts[str(row.get("external_project") or "")] += 1
        category_counts[str(row.get("auto_failure_category") or "")] += 1
        stage_counts[str(row.get("auto_stage") or "")] += 1

    for index, row in enumerate(selected, start=1):
        row["audit_id"] = f"EXT89-AUDIT-{index:03d}"
        for col in MANUAL_COLUMNS:
            row.setdefault(col, "")
    return selected


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError("No rows selected for audit pack")
    preferred_order = [
        "audit_id",
        *MANUAL_COLUMNS,
        "group",
        "seed",
        "run_id",
        "external_project",
        "external_module",
        "task_id",
        "entry_point",
        "signature",
        "selection_origin",
        "candidate_id",
        "auto_stage",
        "auto_failure_category",
        "auto_failure_subtype",
        "auto_rule_id",
        "auto_rule_strength",
        "auto_severity",
        "auto_evidence_json",
        "execution_status",
        "execution_pass",
        "exception_type",
        "traceback_snippet",
        "stdout_tail",
        "stderr_tail",
        "num_tests_collected",
        "num_tests_passed",
        "num_tests_failed",
        "num_tests_errors",
        "mutation_status",
        "mutation_score",
        "mutants_total",
        "mutants_killed",
        "mutants_survived",
        "num_asserts",
        "assert_density",
        "test_style",
        "candidate_test_code",
        "target_source_excerpt",
        "prompt_excerpt",
    ]
    extra_cols = sorted({key for row in rows for key in row} - set(preferred_order))
    fieldnames = preferred_order + extra_cols
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default="/root/autodl-tmp/utplm")
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--sample_seed", default="20260625")
    parser.add_argument("--run", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out_csv = Path(args.out_csv).resolve() if args.out_csv else root / "outputs" / "audits" / "external89failureauditpack.csv"
    manifest_path = Path(args.manifest).resolve() if args.manifest else out_csv.with_suffix(".manifest.json")
    specs = run_specs(args.run)

    all_rows = enrich_failure_rows(root, specs)
    selected = select_stratified(all_rows, args.sample_size, args.sample_seed)
    write_csv(selected, out_csv)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "project_root": str(root),
        "out_csv": str(out_csv),
        "sample_size": len(selected),
        "sample_seed": args.sample_seed,
        "runs": [":".join(spec) for spec in specs],
        "n_failure_logs_available": len(all_rows),
        "selected_group_counts": dict(Counter(row["group"] for row in selected)),
        "selected_project_counts": dict(Counter(row["external_project"] for row in selected)),
        "selected_category_counts": dict(Counter(row["auto_failure_category"] for row in selected)),
        "selected_stage_counts": dict(Counter(row["auto_stage"] for row in selected)),
        "manual_columns": MANUAL_COLUMNS,
        "manual_validity_values": ["valid", "invalid", "unclear"],
        "manual_failure_type_values": [
            "oracle mismatch",
            "contract ambiguity",
            "runtime-env issue",
            "invalid test",
            "weak assertion",
            "strict-but-plausible",
            "other",
        ],
        "auto_category_agree_values": ["yes", "partial", "no"],
        "reveals_stricter_behavior_values": ["yes", "no", "unclear"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
