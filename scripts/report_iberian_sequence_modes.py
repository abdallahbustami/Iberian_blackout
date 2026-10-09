"""Inter-area mode of the stressed state for several tie factors k (results/revised/iberian_sequence/modes.json).

    python scripts/report_iberian_sequence_modes.py 10 12 15
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pa_dvsa.replication import activsg2000_engine as E
from pa_dvsa.replication.iberian_sequence import sequence_config
from pa_dvsa.replication.iberian_state import dominant_interarea_mode, interarea_modes
from run_iberian_sequence import FINAL_BASE

OUT = ROOT / "results" / "revised" / "iberian_sequence" / "modes.json"


def mode_of(k: float) -> dict:
    kw = {**FINAL_BASE, "tie_scale": float(k)}
    kw.pop("voltage_control", None)
    cfg, _ = sequence_config(**kw)
    dis = E.detect_nonequilibrium_exciters(cfg)
    cfg = replace(cfg, tap_override=E.solve_uniform_tap(cfg, disabled_exciters=dis), pre_cascade_enabled=False)
    rs = E.build_replica_system(cfg, quiet=True, disabled_exciters=dis)
    modes = interarea_modes(rs)
    dom = dominant_interarea_mode(modes)
    right = sorted(modes, key=lambda m: -m["sigma"])[:3]
    return {"k": float(k), "dominant": dom, "rightmost": right}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("k", nargs="*", type=float, default=[10.0, 12.0, 15.0], help="tie factors (default 10 12 15)")
    ks = p.parse_args(argv).k
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for k in ks:
        r = mode_of(k)
        out[f"{k:g}"] = r
        d = r["dominant"]
        print(f"k={k:g}: {d['f_hz']:.3f} Hz, damping {100 * d['zeta']:.1f} %", flush=True)
        OUT.write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
