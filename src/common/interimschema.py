from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict

RUN_ID = os.environ.get("UTPLM_RUN_ID", "smoke_v1")

PROJECT_ROOT = Path(
    os.environ.get("UTPLM_PROJECT_ROOT", "/root/autodl-tmp/utplm")
).resolve()

INTERIM_ROOT = PROJECT_ROOT / "data" / "interim"
TMP_EVAL_ROOT = PROJECT_ROOT / "tmp" / "eval_runs" / RUN_ID

COMMON_KEYS = (
    "run_id",
    "candidate_id",
    "task_id",
    "source",
    "difficulty_bucket",
    "test_source_type",
    "script_version",
    "timestamp",
)

EXECUTION_STATUS = frozenset({
    "passed",
    "failed",
    "timeout",
    "infra_error",
    "syntax_error",
})

COVERAGE_STATUS = frozenset({
    "ok",
    "skipped_execution_failed",
    "tool_error",
})

MUTATION_STATUS = frozenset({
    "ok",
    "skipped_execution_failed",
    "timeout",
    "tool_error",
})

FAILURE_CATEGORIES = frozenset({
    "syntax_error",
    "import_or_env_error",
    "execution_timeout",
    "assertion_weak_or_missing",
    "boundary_case_gap",
    "exception_path_gap",
    "branch_distinguishing_gap",
    "mutation_survival_major",
    "other",
})

VALID_STAGES = frozenset({
    "generated_candidates",
    "execution_results",
    "coverage_results",
    "static_features",
    "mutation_results",
    "failure_taxonomy_logs",
})

STAGE_FILENAMES = {
    "generated_candidates": "generatedcandidates.jsonl",
    "execution_results": "executionresults.jsonl",
    "coverage_results": "coverageresults.jsonl",
    "static_features": "staticfeatures.jsonl",
    "mutation_results": "mutationresults.jsonl",
    "failure_taxonomy_logs": "failuretaxonomylogs.jsonl",
}

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def base_record(script_version: str) -> Dict[str, str]:
    return {
        "run_id": RUN_ID,
        "script_version": script_version,
        "timestamp": utc_now_iso(),
    }

def stage_dir(stage: str) -> Path:
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage: {stage}. Valid stages: {sorted(VALID_STAGES)}"
        )
    path = INTERIM_ROOT / stage.replace("_", "") / RUN_ID
    path.mkdir(parents=True, exist_ok=True)
    return path

def stage_file(stage: str) -> Path:
    if stage not in STAGE_FILENAMES:
        raise ValueError(
            f"Unknown stage: {stage}. Valid stages: {sorted(STAGE_FILENAMES)}"
        )

    p = stage_dir(stage) / STAGE_FILENAMES[stage]

    # 向后兼容旧 smoke 结果
    if stage == "generated_candidates":
        legacy = stage_dir(stage) / "smokecandidates.jsonl"
        if not p.exists() and legacy.exists():
            return legacy

    return p

def stage_manifest_file(stage: str) -> Path:
    return stage_dir(stage) / "manifest.json"