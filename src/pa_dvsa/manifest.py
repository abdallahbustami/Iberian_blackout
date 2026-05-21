"""Run manifest generation for reproducible screening and validation runs."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Iterable

from . import __version__
from .common import sha256_file, write_json
from .environment import DEPENDENCIES


def _git_manifest(project_root: Path) -> dict[str, object]:
    git = shutil.which("git")
    if not git:
        return {"git_available": False, "is_repository": False}

    def run(args: Iterable[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [git, *args],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

    inside = run(["rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {"git_available": True, "is_repository": False}

    commit = run(["rev-parse", "HEAD"])
    branch = run(["branch", "--show-current"])
    status = run(["status", "--short"])
    return {
        "git_available": True,
        "is_repository": True,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_short": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for spec in DEPENDENCIES:
        try:
            versions[spec.name] = metadata.version(spec.name)
        except metadata.PackageNotFoundError:
            versions[spec.name] = None
    return versions


def _file_entry(project_root: Path, path: str | Path) -> dict[str, object]:
    file_path = Path(path)
    resolved = file_path if file_path.is_absolute() else project_root / file_path
    entry: dict[str, object] = {
        "path": str(file_path),
        "exists": resolved.exists(),
    }
    if resolved.exists() and resolved.is_file():
        stat = resolved.stat()
        entry.update({"bytes": stat.st_size, "sha256": sha256_file(resolved)})
    return entry


def collect_manifest(
    project_root: str | Path,
    run_name: str,
    *,
    command: list[str] | None = None,
    config_path: str | Path | None = None,
    input_paths: list[str | Path] | None = None,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Collect a reproducibility manifest for one run."""

    root = Path(project_root).resolve()
    inputs = input_paths or [
        "LaTeX/root.tex",
        "README.md",
        "pyproject.toml",
    ]
    payload = {
        "schema_version": "1.0",
        "tool_version": __version__,
        "run_name": run_name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "command": command or sys.argv,
        "random_seed": random_seed,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": _package_versions(),
        "git": _git_manifest(root),
        "config": _file_entry(root, config_path) if config_path else None,
        "inputs": [_file_entry(root, path) for path in inputs],
    }
    return payload


def write_manifest(
    project_root: str | Path,
    out_path: str | Path,
    run_name: str,
    *,
    command: list[str] | None = None,
    config_path: str | Path | None = None,
    input_paths: list[str | Path] | None = None,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Collect and write a manifest JSON file."""

    payload = collect_manifest(
        project_root,
        run_name,
        command=command,
        config_path=config_path,
        input_paths=input_paths,
        random_seed=random_seed,
    )
    write_json(out_path, payload)
    return payload
