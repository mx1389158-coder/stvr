from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(".").resolve()
PROJECT_ROOT = Path(os.environ["UTPLM_PROJECT_ROOT"]).resolve()

# 可选环境变量：
#   a1_ANNOTATION_INPUT_CSV   指定输入 CSV
#   a1_ANNOTATION_OUTPUT_TAG  指定输出 tag（如 spotcheck_a1_pool_v1）
INPUT_OVERRIDE = os.environ.get("a1_ANNOTATION_INPUT_CSV", "").strip()
OUTPUT_TAG_OVERRIDE = os.environ.get("a1_ANNOTATION_OUTPUT_TAG", "").strip()

candidate_inputs = []
if INPUT_OVERRIDE:
    candidate_inputs.append(Path(INPUT_OVERRIDE).resolve())

candidate_inputs.extend(
    [
        PROJECT_ROOT / "outputs" / "summaries" / "a1" / "a1sampledcandidatesv1.csv",
        REPO_ROOT / "outputs" / "summaries" / "a1" / "a1sampledcandidatesv1.csv",
    ]
)

IN_CSV = None
for p in candidate_inputs:
    if p.exists():
        IN_CSV = p
        break

if IN_CSV is None:
    print("Tried input paths:")
    for p in candidate_inputs:
        print(" -", p)
    raise FileNotFoundError("Annotation input CSV not found.")

OUT_DIR = PROJECT_ROOT / "outputs" / "summaries" / "a1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN_CSV)

required_base_cols = ["sample_id", "candidate_id", "candidate_test_code"]
missing = [c for c in required_base_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns for annotation pack: {missing}")

prompt_source_col = None
for c in ["prompt", "problem", "task_description", "text"]:
    if c in df.columns:
        prompt_source_col = c
        break

if prompt_source_col is None:
    raise ValueError("No prompt-like column found in annotation input.")

df = df.copy()
df["sample_id"] = df["sample_id"].astype(str).str.strip()
df["candidate_id"] = df["candidate_id"].astype(str).str.strip()
df["candidate_test_code"] = df["candidate_test_code"].fillna("").astype(str)
df[prompt_source_col] = df[prompt_source_col].fillna("").astype(str)

if (df["sample_id"] == "").any():
    raise ValueError("Found empty sample_id in annotation input.")

if df["sample_id"].duplicated().any():
    dup_counts = df["sample_id"].value_counts()
    dup_counts = dup_counts[dup_counts > 1].head(20).to_dict()
    raise ValueError(f"sample_id is not unique. Top duplicates: {dup_counts}")

if (df["candidate_test_code"].str.strip() == "").any():
    bad = df.loc[df["candidate_test_code"].str.strip() == "", ["sample_id", "candidate_id"]]
    raise ValueError(
        "Found empty candidate_test_code in annotation input. "
        f"Examples: {bad.head(10).to_dict(orient='records')}"
    )

if (df[prompt_source_col].str.strip() == "").any():
    bad = df.loc[df[prompt_source_col].str.strip() == "", ["sample_id", "candidate_id"]]
    raise ValueError(
        "Found empty prompt/task text in annotation input. "
        f"Examples: {bad.head(10).to_dict(orient='records')}"
    )

run_ids = df["run_id"].dropna().astype(str).str.strip().unique().tolist() if "run_id" in df.columns else []
if len(run_ids) != 1:
    raise ValueError(f"Expected exactly one run_id in annotation input, got: {run_ids}")
run_id = run_ids[0]

safe_run = run_id.replace("/", "_").replace(" ", "_")
suffix = OUTPUT_TAG_OVERRIDE if OUTPUT_TAG_OVERRIDE else safe_run

OUT_R1 = OUT_DIR / f"a1annotationpackrater1{suffix}.csv"
OUT_R2 = OUT_DIR / f"a1annotationpackrater2{suffix}.csv"
OUT_KEY = OUT_DIR / f"a1annotationkeyinternal{suffix}.csv"
OUT_MANIFEST = OUT_DIR / f"a1annotationpackmanifest{suffix}.json"

SCRIPT_VERSION = "build_a1_annotation_pack_v6"
RUBRIC_VERSION = "a1_rubric_final_vfinal"

df = df.sort_values(["sample_id"], kind="stable").reset_index(drop=True)

blind = pd.DataFrame(
    {
        "sample_id": df["sample_id"],
        "task_prompt": df[prompt_source_col],
        "candidate_test_code": df["candidate_test_code"],
    }
)

blind["assertion_effectiveness"] = ""
blind["boundary_condition_checking"] = ""
blind["exception_path_applicability"] = ""
blind["exception_path_handling"] = ""
blind["branch_distinguishing_ability"] = ""
blind["fault_revealing_potential"] = ""
blind["teaching_value"] = ""
blind["primary_weakness_tags"] = ""
blind["overall_total_0_10"] = ""
blind["rater_note"] = ""

blind.sample(frac=1.0, random_state=11).reset_index(drop=True).to_csv(OUT_R1, index=False, encoding="utf-8")
blind.sample(frac=1.0, random_state=22).reset_index(drop=True).to_csv(OUT_R2, index=False, encoding="utf-8")
df.to_csv(OUT_KEY, index=False, encoding="utf-8")

manifest = {
    "script_version": SCRIPT_VERSION,
    "rubric_version": RUBRIC_VERSION,
    "project_root": str(PROJECT_ROOT),
    "repo_root": str(REPO_ROOT),
    "input_csv": str(IN_CSV),
    "run_id": run_id,
    "output_tag": suffix,
    "output_rater1_csv": str(OUT_R1),
    "output_rater2_csv": str(OUT_R2),
    "output_internal_key_csv": str(OUT_KEY),
    "num_rows": int(len(blind)),
    "prompt_source_col": prompt_source_col,
    "blind_columns": list(blind.columns),
    "notes": {
        "branch_distinguishing_ability": "canonical field name used for downstream analysis scripts",
        "overall_total_0_10": "supportive only; official a1 score should later be computed from main dimensions",
        "exception_path_applicability": "used to distinguish N/A from missing / unfilled",
        "teaching_value": "auxiliary only; not part of official a1 main score",
        "rater1_random_seed": 11,
        "rater2_random_seed": 22,
    },
}
with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print("[ok] wrote annotation packs")
print("run_id  =", run_id)
print("tag     =", suffix)
print("input   =", IN_CSV)
print("rows    =", len(blind))
print("rater1  =", OUT_R1)
print("rater2  =", OUT_R2)
print("key     =", OUT_KEY)
print("manifest=", OUT_MANIFEST)
