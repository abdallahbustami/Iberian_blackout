"""Command line entry point for project traceability artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .environment import write_environment_report
from .manifest import write_manifest
from .traceability import write_traceability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate traceability, environment, and manifest artifacts."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository/project root. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--root-tex",
        default="LaTeX/root.tex",
        help="Path to the current paper source, relative to project root unless absolute.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/phase0",
        help="Output directory for traceability artifacts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero if the full ANDES screening environment is not ready.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(raw_args)
    project_root = Path(args.project_root).resolve()
    root_tex = Path(args.root_tex)
    if not root_tex.is_absolute():
        root_tex = project_root / root_tex
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    traceability = write_traceability(root_tex, out_dir)
    environment = write_environment_report(project_root, out_dir)
    write_manifest(
        project_root,
        out_dir / "manifest.json",
        "phase0",
        command=[sys.executable, "-m", "pa_dvsa.phase0", *raw_args],
        input_paths=[
            "LaTeX/root.tex",
            "docs-andes-app-en-stable.pdf",
            "Final Report on the Grid Incident in Spain and Portugal on 28 April 2025.pdf",
        ],
    )

    print(f"Wrote traceability artifacts to {out_dir}")
    print(
        "Traceability items: "
        f"{traceability['summary']['total_items']} "
        f"({traceability['summary']['required_outputs']} required outputs)"
    )
    print(f"Ready for ANDES screening: {environment['ready_for_andes_screening']}")
    if environment["readiness"]["blocking_issues"]:
        print("Blocking issues:")
        for issue in environment["readiness"]["blocking_issues"]:
            print(f"  - {issue}")
    if args.strict and not environment["ready_for_andes_screening"]:
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
