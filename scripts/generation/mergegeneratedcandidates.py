from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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

def remap_rows(
    rows: List[Dict[str, Any]],
    *,
    prefix: str,
    run_id: str,
    generator_model: str,
    generator_size: str,
    generator_role: str,
    candidate_source: str,
) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        old_cid = str(row["candidate_id"])
        new_row = dict(row)
        new_row["parent_candidate_id"] = old_cid
        new_row["candidate_id"] = f"{prefix}{old_cid}"
        new_row["run_id"] = run_id
        new_row["generator_model"] = generator_model
        new_row["generator_size"] = generator_size
        new_row["generator_role"] = generator_role
        new_row["candidate_source"] = candidate_source
        out.append(new_row)
    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_7b", required=True)
    parser.add_argument("--run_14b", required=True)
    parser.add_argument("--run_pool", required=True)
    parser.add_argument("--project_root", required=True)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    p7 = project_root / "data" / "interim" / "generatedcandidates" / args.run_7b / "generatedcandidates.jsonl"
    if not p7.exists():
        p7_legacy = project_root / "data" / "interim" / "generatedcandidates" / args.run_7b / "smokecandidates.jsonl"
        p7 = p7_legacy

    p14 = project_root / "data" / "interim" / "generatedcandidates" / args.run_14b / "generatedcandidates.jsonl"
    if not p14.exists():
        p14_legacy = project_root / "data" / "interim" / "generatedcandidates" / args.run_14b / "smokecandidates.jsonl"
        p14 = p14_legacy

    if not p7.exists():
        raise FileNotFoundError(f"7B candidate file not found: {p7}")
    if not p14.exists():
        raise FileNotFoundError(f"14B candidate file not found: {p14}")

    rows7 = remap_rows(
        load_jsonl(p7),
        prefix="s7_",
        run_id=args.run_pool,
        generator_model="Qwen2.5-7B-Instruct",
        generator_size="7B",
        generator_role="student_primary_candidate_generation",
        candidate_source="upstream_formal_shared_pool",
    )

    rows14 = remap_rows(
        load_jsonl(p14),
        prefix="t14_",
        run_id=args.run_pool,
        generator_model="Qwen2.5-14B-Instruct",
        generator_size="14B",
        generator_role="teacher_only_data_synthesis",
        candidate_source="upstream_formal_shared_pool",
    )

    merged = rows7 + rows14

    # 唯一性检查
    cids = [r["candidate_id"] for r in merged]
    if len(cids) != len(set(cids)):
        raise ValueError("duplicate candidate_id detected after merge")

    out_dir = project_root / "data" / "interim" / "generatedcandidates" / args.run_pool
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "generatedcandidates.jsonl"
    compat_jsonl = out_dir / "smokecandidates.jsonl"
    manifest = out_dir / "manifest.json"

    write_jsonl(merged, out_jsonl)
    write_jsonl(merged, compat_jsonl)

    manifest_obj = {
        "run_id": args.run_pool,
        "generator_sources": [args.run_7b, args.run_14b],
        "output_jsonl": str(out_jsonl),
        "compat_jsonl": str(compat_jsonl),
        "num_rows": len(merged),
        "source_breakdown": {
            "7b_rows": len(rows7),
            "14b_rows": len(rows14),
        },
        "notes": [
            "candidate_id is remapped with source prefix to ensure global uniqueness",
            "7B remains the only main experiment model",
            "14B is teacher-only data synthesis",
        ],
    }

    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_obj, f, ensure_ascii=False, indent=2)

    print("[ok] wrote", out_jsonl)
    print("[ok] wrote", compat_jsonl)
    print("[ok] wrote", manifest)
    print("merged rows =", len(merged))

if __name__ == "__main__":
    main()