#!/usr/bin/env python3
"""Generate paper case-study figures and tables from regenerated artifacts."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "paper_case_studies" / "mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib as mpl

mpl.use("Agg")

from pa_dvsa.paper_case_studies.derived import build_derived_dataset
from pa_dvsa.paper_case_studies.figures import FIGURE_BUILDERS, FIGURE_STEMS
from pa_dvsa.paper_case_studies.loaders import (
    MissingArtifactError,
    load_artifacts,
    load_replication_bundle,
    relpath,
    sha256_file,
)
from pa_dvsa.paper_case_studies.style import configure_style, latex_available
from pa_dvsa.paper_case_studies.tables import TABLE_BUILDERS, TABLE_STEMS


LOG = logging.getLogger("paper_case_studies")
ALL_STEMS = FIGURE_STEMS + TABLE_STEMS
TARGET_FIGURE_FILES = {f"{stem}.pdf" for stem in FIGURE_STEMS}
TARGET_TABLE_FILES = {f"{stem}.{suffix}" for stem in TABLE_STEMS for suffix in ("tex", "csv")}


def _parse_formats(value: str) -> tuple[str, ...]:
    formats = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    allowed = {"pdf", "png", "svg"}
    bad = set(formats) - allowed
    if bad:
        raise argparse.ArgumentTypeError(f"Unsupported format(s): {', '.join(sorted(bad))}")
    return formats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("results/case_studies"))
    parser.add_argument(
        "--replication-root",
        type=Path,
        default=Path("results/activsg2000_iberian_replication"),
    )
    parser.add_argument("--out-fig", type=Path, default=Path("LaTeX/figures/generated"))
    parser.add_argument("--out-table", type=Path, default=Path("LaTeX/tables/generated"))
    parser.add_argument("--work", type=Path, default=Path("results/paper_case_studies"))
    parser.add_argument("--only", default="all", choices=("all",) + ALL_STEMS)
    parser.add_argument(
        "--formats",
        type=_parse_formats,
        default=("pdf",),
        help="Comma-separated figure formats. Defaults to pdf.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Retained for interface clarity; explicitly unsupported rows are labeled, not plotted as evaluated.",
    )
    parser.add_argument("--skip-derive", action="store_true")
    parser.add_argument(
        "--outline-text",
        dest="outline_text",
        action="store_true",
        default=True,
        help="Convert PDF text to vector paths to avoid Illustrator font prompts. Enabled by default.",
    )
    parser.add_argument(
        "--no-outline-text",
        dest="outline_text",
        action="store_false",
        help="Keep live text/fonts in generated PDFs.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _write_manifest(
    *,
    args: argparse.Namespace,
    input_hashes: dict[str, str],
    warnings: list[str],
    generated: list[dict[str, str]],
    used_external_latex: bool,
    archived: list[dict[str, str]] | None = None,
) -> None:
    manifest_path = args.work / "manifest.json"
    existing: dict[str, object] = {}
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    existing.update(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "command_options": {
                "data_root": relpath(args.data_root),
                "replication_root": relpath(args.replication_root),
                "out_fig": relpath(args.out_fig),
                "out_table": relpath(args.out_table),
                "only": args.only,
                "formats": list(args.formats),
                "strict": bool(args.strict),
                "skip_derive": bool(args.skip_derive),
                "outline_text": bool(args.outline_text),
            },
            "latex": {
                "requested": True,
                "external_latex_available": latex_available(),
                "matplotlib_text_usetex": used_external_latex,
                "note": (
                    "External LaTeX was used for Matplotlib text."
                    if used_external_latex
                    else "No latex binary was available; Matplotlib used its serif/mathtext fallback."
                ),
            },
            "inputs": input_hashes,
            "warnings": warnings,
            "generated_outputs": generated,
            "archived_outputs": archived or [],
        }
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _archive_stale_outputs(
    out_dir: Path,
    work: Path,
    *,
    target_files: set[str],
    suffixes: set[str],
    archive_name: str,
) -> list[dict[str, str]]:
    """Move stale generated files out of a LaTeX output directory.

    A full paper refresh should leave only the eight target PDF stems in the
    generated figure directory. Existing previews and stale stems are retained
    under the work directory for auditability instead of deleted.
    """

    archived: list[dict[str, str]] = []
    if not out_dir.exists():
        return archived
    candidates = [
        path
        for path in out_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    ]
    stale = [path for path in candidates if path.name not in target_files]
    if not stale:
        return archived
    archive_root = work / archive_name / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root.mkdir(parents=True, exist_ok=True)
    for path in stale:
        dst = archive_root / path.name
        shutil.move(str(path), str(dst))
        archived.append({"from": relpath(path), "to": relpath(dst), "sha256": sha256_file(dst)})
    return archived


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO), format="%(levelname)s: %(message)s")
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    used_tex = configure_style(prefer_usetex=True)
    if not used_tex:
        LOG.warning("No external latex binary found; using Matplotlib fallback text rendering.")
    archived = []
    if args.only == "all":
        archived.extend(
            _archive_stale_outputs(
                args.out_fig,
                args.work,
                target_files=TARGET_FIGURE_FILES,
                suffixes={".pdf", ".png", ".svg"},
                archive_name="archived_figures",
            )
        )
        archived.extend(
            _archive_stale_outputs(
                args.out_table,
                args.work,
                target_files=TARGET_TABLE_FILES,
                suffixes={".tex", ".csv"},
                archive_name="archived_tables",
            )
        )
    if archived:
        LOG.info("Archived %d stale figure file(s)", len(archived))
    if args.skip_derive:
        needs_replication = args.only in {"all", "fig01_replication_activsg2000"}
        bundle = load_replication_bundle(
            replication_root=args.replication_root,
            strict=args.strict and needs_replication,
        )
        from pa_dvsa.paper_case_studies.derived import _paths

        paths = _paths(args.work)
    else:
        bundle = load_artifacts(
            data_root=args.data_root,
            replication_root=args.replication_root,
            strict=args.strict,
        )
        paths = build_derived_dataset(
            bundle,
            work=args.work,
            strict=args.strict,
            allow_unsupported=True,
        )

    generated: list[dict[str, str]] = []
    stems = ALL_STEMS if args.only == "all" else (args.only,)
    for stem in stems:
        LOG.info("Generating %s", stem)
        if stem in FIGURE_BUILDERS:
            results = FIGURE_BUILDERS[stem](
                paths,
                bundle,
                out_fig=args.out_fig,
                formats=args.formats,
                outline_text=args.outline_text,
            )
            for result in results:
                generated.append({"artifact_id": stem, "kind": "figure", "path": relpath(result.path), "sha256": result.sha256})
        elif stem in TABLE_BUILDERS:
            results = TABLE_BUILDERS[stem](paths, bundle, out_table=args.out_table)
            for result in results:
                generated.append({"artifact_id": stem, "kind": "table", "path": relpath(result.path), "sha256": result.sha256})
        else:  # pragma: no cover - argparse guards this
            raise MissingArtifactError(f"Unknown artifact stem: {stem}")

    # Include derived data hashes in the top-level generated manifest.
    for path in (args.work / "dataset").glob("*"):
        if path.is_file():
            generated.append(
                {
                    "artifact_id": path.stem,
                    "kind": "derived_data",
                    "path": relpath(path),
                    "sha256": sha256_file(path),
                }
            )
    _write_manifest(
        args=args,
        input_hashes=bundle.input_hashes,
        warnings=bundle.warnings,
        generated=generated,
        used_external_latex=used_tex,
        archived=archived,
    )
    LOG.info("Wrote manifest to %s", relpath(args.work / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
