#!/usr/bin/env python3
"""Figures 5 to 13 of the paper, plotted from the stored results into results/revised/figures/.

    python scripts/make_paper_figures.py
    python scripts/make_paper_figures.py --only validation,sequence
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from pa_dvsa.revised_figures import classic as K
from pa_dvsa.revised_figures.data import RESULTS, Results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=RESULTS / "figures")
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(K.FIGURES))
    a = ap.parse_args(argv)
    res = Results()
    K.configure()
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    for name, fn in K.FIGURES.items():
        if not only or name in only:
            print(f"{name:16s} {', '.join(p.name for p in fn(res, a.out))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
