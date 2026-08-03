from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_REL = "stvr/configs/external/stvrexternalprojects.json"


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(cmd: list[str], cwd: Path | None = None, dry_run: bool = False) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def git_head(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", default=str(find_repo_root()))
    parser.add_argument("--external_root", default=None)
    parser.add_argument("--project_config", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fetch_existing", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    external_root = Path(args.external_root).resolve() if args.external_root else project_root / "data" / "raw" / "external_projects"
    config_path = Path(args.project_config).resolve() if args.project_config else project_root / DEFAULT_CONFIG_REL
    config = load_json(config_path)
    external_root.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, Any]] = []
    for spec in config.get("projects", []):
        project = str(spec["project"])
        repo_url = str(spec.get("repo_url") or "")
        ref = str(spec.get("ref") or "").strip()
        dest = external_root / project
        if not repo_url:
            raise ValueError(f"Missing repo_url for project {project}")

        if dest.exists():
            if args.fetch_existing:
                run(["git", "-C", str(dest), "fetch", "--all", "--tags"], dry_run=args.dry_run)
            action = "exists"
        else:
            run(["git", "clone", repo_url, str(dest)], dry_run=args.dry_run)
            action = "cloned"

        if ref:
            run(["git", "-C", str(dest), "checkout", ref], dry_run=args.dry_run)
            action = f"{action}+checkout"

        report.append(
            {
                "project": project,
                "path": str(dest),
                "repo_url": repo_url,
                "requested_ref": ref or None,
                "resolved_commit": None if args.dry_run else git_head(dest),
                "action": action,
            }
        )

    print(json.dumps({"external_root": str(external_root), "projects": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
