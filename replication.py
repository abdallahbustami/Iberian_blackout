#!/usr/bin/env python3
"""Launcher for the supported Iberian blackout replication studies.

Preferred module entry points:

* ``python -m pa_dvsa.replication.academic_replica`` for the paper-facing
  ACTIVSg2000 mechanism replica.
* ``python -m pa_dvsa.replication.activsg2000_engine`` for the lower-level
  ACTIVSg2000 engine.
* ``python -m pa_dvsa.replication.calibration`` for calibration sweeps.

This root launcher defaults to the paper-facing academic replica. Use an
explicit subcommand to run another path:

``python replication.py engine --plot``
``python replication.py calibration --use-existing-baseline``
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from pa_dvsa.replication.academic_replica import main as academic_main
from pa_dvsa.replication.activsg2000_engine import main as engine_main
from pa_dvsa.replication.calibration import main as calibration_main


CommandMain = Callable[[Sequence[str] | None], int]

COMMANDS: dict[str, CommandMain] = {
    "academic": academic_main,
    "paper": academic_main,
    "engine": engine_main,
    "calibration": calibration_main,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    if args and args[0] in COMMANDS:
        command = args.pop(0)
        return COMMANDS[command](args)
    return academic_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
