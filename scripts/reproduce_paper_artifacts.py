#!/usr/bin/env python3
"""Regenerate all supported paper artifacts from source scripts.

The public repository does not need committed ``results/`` artifacts. This
orchestrator recreates the case-study outputs, the ACTIVSg2000 academic replica,
the normalized paper dataset, paper figures/tables, replication-subsection
figures, quantitative tables, and the finite-window proxy figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


DRY_RUN = False


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    if DRY_RUN:
        return
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _remove(path: Path) -> None:
    if path.exists():
        print(f"Removing {path.relative_to(ROOT)}", flush=True)
        if DRY_RUN:
            return
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove supported generated artifact directories before running.",
    )
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Skip nonlinear TDS in the rich paper sweep. This is faster but not paper-complete.",
    )
    parser.add_argument(
        "--skip-replication-variants",
        action="store_true",
        help="Do not regenerate additional ACTIVSg2000 ablation variants for replication-subsection figures.",
    )
    parser.add_argument("--fig-dir", type=Path, default=Path("LaTeX/figures/generated"))
    parser.add_argument("--table-dir", type=Path, default=Path("LaTeX/tables/generated"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the reproduction commands without executing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global DRY_RUN
    DRY_RUN = bool(args.dry_run)
    py = sys.executable
    fig_dir = args.fig_dir
    table_dir = args.table_dir

    if args.clean:
        for rel in (
            "results/case_studies",
            "results/activsg2000_iberian_replication",
            "results/paper_case_studies",
            "results/replication_subsection",
            "results/figures/finite_window_proxy",
        ):
            _remove(ROOT / rel)
        _remove(ROOT / fig_dir)
        _remove(ROOT / table_dir)

    # Legacy per-benchmark outputs are still used by the quantitative LP table.
    _run([py, "scripts/run_kundur_case_study.py"])
    for study in ("ieee39", "npcc_full", "gbnetwork"):
        _run([py, "scripts/run_load_shedding_case_study.py", study])

    # Large-system academic mechanism replica used by Figure 1 and the
    # replication-specific event library.
    _run(
        [
            py,
            "-m",
            "pa_dvsa.replication.academic_replica",
            "--plot",
            "--formats",
            "pdf",
            "--out",
            "results/activsg2000_iberian_replication",
            "--latex-fig-dir",
            str(fig_dir),
        ]
    )

    # Rich multi-scenario screen/TDS sweep used by Figures 2--8 and tables.
    sweep_cmd = [
        py,
        "scripts/run_paper_case_sweep.py",
        "--out",
        "results/paper_case_studies/dataset",
    ]
    if args.screen_only:
        sweep_cmd.append("--screen-only")
    _run(sweep_cmd)

    # Render the main eight paper figures/tables directly from the regenerated
    # normalized dataset. ``--skip-derive`` avoids requiring committed
    # per-case result directories for Figures 2--8.
    _run(
        [
            py,
            "run_case_studies.py",
            "--skip-derive",
            "--strict",
            "--out-fig",
            str(fig_dir),
            "--out-table",
            str(table_dir),
        ]
    )

    # Replication-subsection figures and ablation variants.
    repl_cmd = [
        py,
        "make_replication_subsection_figures.py",
        "--out-fig",
        str(fig_dir),
    ]
    if args.skip_replication_variants:
        repl_cmd.append("--no-ensure-variants")
    _run(repl_cmd)

    _run([py, "scripts/make_quantitative_case_elements.py"])
    _run([py, "scripts/generate_finite_window_proxy_figure.py"])
    print("\nReproduction complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
