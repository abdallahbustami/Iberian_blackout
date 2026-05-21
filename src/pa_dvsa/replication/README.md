# Replication Package

This package contains the supported Iberian-style replication studies used by
the paper.  The root-level `replication.py` file is a convenience launcher.

## Modules

- `activsg2000_engine.py`: ANDES-backed ACTIVSg2000 physical mechanism engine.
  It contains the collector relays, operator actions, HVDC surrogate, AC tie
  relays, post-cascade protection, island accounting, artifact writing, and
  plotting helpers.
- `academic_replica.py`: paper-facing mechanism scenario built on the engine. This is
  the default academic mechanism replica with report anchors and named variants.
- `calibration.py`: deterministic calibration sweep for the ACTIVSg2000
  mechanism replica.

## Common Commands

```bash
PYTHONPATH=src python -m pa_dvsa.replication.academic_replica --plot
PYTHONPATH=src python -m pa_dvsa.replication.calibration --use-existing-baseline
PYTHONPATH=src python replication.py --plot
PYTHONPATH=src python replication.py engine --plot
```

The module form is preferred for new work. The root `replication.py` launcher
is kept for convenience and dispatches to the named package modules.
