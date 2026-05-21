"""Environment checks for the protection-aware DVSA implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib
from importlib import metadata
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Iterable

from . import __version__
from .common import sha256_file, write_json


@dataclass(frozen=True)
class DependencySpec:
    name: str
    import_name: str
    required: bool
    min_version: str | None = None
    purpose: str = ""


@dataclass(frozen=True)
class DependencyResult:
    name: str
    import_name: str
    required: bool
    min_version: str | None
    installed: bool
    version: str | None
    status: str
    purpose: str
    message: str


DEPENDENCIES: tuple[DependencySpec, ...] = (
    DependencySpec("numpy", "numpy", True, "1.24", "array operations and dense numerics"),
    DependencySpec("scipy", "scipy", True, "1.10", "sparse linear algebra and optimization"),
    DependencySpec("pandas", "pandas", True, "1.5", "result tables and validation data"),
    DependencySpec("matplotlib", "matplotlib", True, "3.6", "paper figures"),
    DependencySpec("andes", "andes", True, "1.6", "power-system DAE, PFlow, and TDS engine"),
)


REQUIRED_INPUTS: tuple[tuple[str, str], ...] = (
    ("LaTeX/root.tex", "current Applied Energy paper source"),
    ("docs-andes-app-en-stable.pdf", "ANDES manual"),
    ("Final Report on the Grid Incident in Spain and Portugal on 28 April 2025.pdf", "ENTSO-E final report"),
)


def _parse_version(version: str | None) -> tuple[int, ...]:
    if not version:
        return ()
    parts = re.findall(r"\d+", version)
    return tuple(int(part) for part in parts[:3])


def _meets_min_version(version: str | None, minimum: str | None) -> bool:
    if minimum is None:
        return True
    return _parse_version(version) >= _parse_version(minimum)


def check_dependency(spec: DependencySpec) -> DependencyResult:
    """Import one dependency and check its version when possible."""

    try:
        module = importlib.import_module(spec.import_name)
    except Exception as exc:  # pragma: no cover - exact import failures vary by host.
        status = "missing_required" if spec.required else "missing_optional"
        return DependencyResult(
            name=spec.name,
            import_name=spec.import_name,
            required=spec.required,
            min_version=spec.min_version,
            installed=False,
            version=None,
            status=status,
            purpose=spec.purpose,
            message=f"import failed: {exc}",
        )

    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = metadata.version(spec.name)
        except metadata.PackageNotFoundError:
            version = "unknown"

    if not _meets_min_version(version, spec.min_version):
        status = "version_too_old"
        message = f"installed {version}, expected >= {spec.min_version}"
    else:
        status = "ok"
        message = f"installed {version}"
    return DependencyResult(
        name=spec.name,
        import_name=spec.import_name,
        required=spec.required,
        min_version=spec.min_version,
        installed=True,
        version=version,
        status=status,
        purpose=spec.purpose,
        message=message,
    )


def _git_info(project_root: Path) -> dict[str, object]:
    git = shutil.which("git")
    if not git:
        return {"git_available": False, "is_repository": False}

    def run_git(args: Iterable[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [git, *args],
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    inside = run_git(["rev-parse", "--is-inside-work-tree"])
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return {"git_available": True, "is_repository": False, "message": inside.stderr.strip()}

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])
    return {
        "git_available": True,
        "is_repository": True,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "status_short": status.stdout.splitlines() if status.returncode == 0 else [],
    }


def _input_file_status(project_root: Path) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    for rel_path, purpose in REQUIRED_INPUTS:
        path = project_root / rel_path
        exists = path.exists()
        status: dict[str, object] = {
            "path": rel_path,
            "purpose": purpose,
            "exists": exists,
            "status": "ok" if exists else "missing",
        }
        if exists and path.is_file():
            stat = path.stat()
            status.update(
                {
                    "bytes": stat.st_size,
                    "sha256": sha256_file(path),
                }
            )
        statuses.append(status)
    return statuses


def _ensure_local_matplotlib_config(project_root: Path) -> str | None:
    """Set a writable Matplotlib config path when the user has not set one."""

    if os.environ.get("MPLCONFIGDIR"):
        return os.environ["MPLCONFIGDIR"]
    mpl_config = project_root / "results" / ".matplotlib"
    mpl_config.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config)
    return str(mpl_config)


def check_environment(project_root: str | Path) -> dict[str, object]:
    """Return a complete environment report for PA-DVSA workflows."""

    root = Path(project_root).resolve()
    matplotlib_config_dir = _ensure_local_matplotlib_config(root)
    python_ok = sys.version_info >= (3, 10)
    dependency_results = [check_dependency(spec) for spec in DEPENDENCIES]
    input_status = _input_file_status(root)
    required_deps_ok = all(result.status == "ok" for result in dependency_results if result.required)
    required_inputs_ok = all(item["status"] == "ok" for item in input_status)
    ready = bool(python_ok and required_deps_ok and required_inputs_ok)

    return {
        "schema_version": "1.0",
        "tool_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "ready_for_andes_screening": ready,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:3]),
            "minimum_required": "3.10",
            "status": "ok" if python_ok else "version_too_old",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "matplotlib": {
            "config_dir": matplotlib_config_dir,
        },
        "dependencies": [asdict(result) for result in dependency_results],
        "inputs": input_status,
        "git": _git_info(root),
        "readiness": {
            "python_ok": python_ok,
            "required_dependencies_ok": required_deps_ok,
            "required_inputs_ok": required_inputs_ok,
            "blocking_issues": [
                result.message
                for result in dependency_results
                if result.required and result.status != "ok"
            ]
            + [
                f"missing input: {item['path']}"
                for item in input_status
                if item["status"] != "ok"
            ],
        },
    }


def environment_to_markdown(report: dict[str, object]) -> str:
    """Render a Markdown environment report."""

    lines = [
        "# Environment Check",
        "",
        f"Project root: `{report['project_root']}`",
        f"Ready for ANDES screening: `{report['ready_for_andes_screening']}`",
        "",
        "## Python",
        "",
        f"- Executable: `{report['python']['executable']}`",
        f"- Version: `{report['python']['version'].split()[0]}`",
        f"- Status: `{report['python']['status']}`",
        "",
        "## Dependencies",
        "",
        "| Package | Required | Minimum | Installed | Status | Purpose |",
        "|---|---:|---:|---:|---|---|",
    ]
    for dep in report["dependencies"]:
        lines.append(
            f"| `{dep['name']}` | `{dep['required']}` | `{dep['min_version']}` | "
            f"`{dep['version']}` | `{dep['status']}` | {dep['purpose']} |"
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            "| Path | Status | Bytes | Purpose |",
            "|---|---|---:|---|",
        ]
    )
    for item in report["inputs"]:
        lines.append(
            f"| `{item['path']}` | `{item['status']}` | `{item.get('bytes', '')}` | "
            f"{item['purpose']} |"
        )
    lines.extend(["", "## Blocking Issues", ""])
    issues = report["readiness"]["blocking_issues"]
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_environment_report(project_root: str | Path, out_dir: str | Path) -> dict[str, object]:
    """Write JSON/Markdown environment reports."""

    out_path = Path(out_dir)
    payload = check_environment(project_root)
    write_json(out_path / "environment.json", payload)
    (out_path / "environment.md").write_text(environment_to_markdown(payload), encoding="utf-8")
    return payload
