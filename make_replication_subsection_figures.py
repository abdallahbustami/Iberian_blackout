#!/usr/bin/env python3
"""Generate additional replication-subsection figures."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from pa_dvsa.replication.academic_replica import VARIANT_DESCRIPTIONS, run_academic_replica
from pa_dvsa.replication.subsection_figures import VARIANT_ORDER, make_all


def _prepend_local_tinytex() -> None:
    tinytex = Path("tools/TinyTeX/bin/universal-darwin")
    if tinytex.exists():
        os.environ["PATH"] = f"{tinytex.resolve()}{os.pathsep}{os.environ.get('PATH', '')}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replication-root", type=Path, default=Path("results/activsg2000_iberian_replication"))
    parser.add_argument("--variants-root", type=Path, default=Path("results/replication_subsection/variants"))
    parser.add_argument("--case-dataset-root", type=Path, default=Path("results/paper_case_studies/dataset"))
    parser.add_argument("--out-fig", type=Path, default=Path("LaTeX/figures/generated"))
    parser.add_argument("--work", type=Path, default=Path("results/replication_subsection"))
    parser.add_argument("--only", default="all", choices=("all", "fig09_replication_operator_conditions", "fig10_replication_mvar_absorption"))
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--tf", type=float, default=30.0)
    parser.add_argument("--tstep", type=float, default=0.01)
    parser.add_argument("--no-ensure-variants", action="store_true", help="Do not run missing replication variants.")
    parser.add_argument("--no-outline-text", action="store_true", help="Do not convert PDF text to paths.")
    return parser.parse_args()


def _variant_complete(path: Path) -> bool:
    required = ("summary.json", "events.json", "collector_traces.csv", "system_traces.csv")
    return all((path / name).exists() for name in required)


def ensure_variant_results(args: argparse.Namespace) -> None:
    args.variants_root.mkdir(parents=True, exist_ok=True)
    for variant in VARIANT_ORDER:
        if variant == "baseline":
            continue
        if variant not in VARIANT_DESCRIPTIONS:
            raise ValueError(f"unknown replication variant configured for figures: {variant}")
        out_dir = args.variants_root / variant
        if _variant_complete(out_dir):
            continue
        print(f"Generating missing replication variant: {variant}")
        run_academic_replica(
            variant=variant,
            out_dir=out_dir,
            tf_s=float(args.tf),
            tstep_s=float(args.tstep),
            plot=False,
            latex_fig_dir=None,
            formats=("pdf",),
            verbose_andes=False,
        )


def main() -> int:
    args = parse_args()
    _prepend_local_tinytex()
    if not args.no_ensure_variants:
        ensure_variant_results(args)
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    results, manifest = make_all(
        replication_root=args.replication_root,
        variants_root=args.variants_root,
        out_dir=args.out_fig,
        work_dir=args.work,
        case_dataset_root=args.case_dataset_root,
        formats=formats,
        outline_text=not args.no_outline_text,
        only=args.only,
    )
    manifest_path = args.work / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for result in results:
        print(result.path)
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
