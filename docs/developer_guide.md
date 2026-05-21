# Developer Guide

## Package Boundaries

- `andes_adapter.py` is the only layer that should depend directly on ANDES
  internals.
- `data_model.py` defines simulator-independent objects used throughout the
  workflow.
- `finite_window.py`, `resolvent_proxy.py`, `robust_screening.py`,
  `cascade_certificate.py`, and `mitigation.py` contain the numerical screen.
- `nonlinear_validation.py` contains simulator-agnostic replay logic plus the
  ANDES TDS callback adapter.
- `replication/` contains the supported ACTIVSg2000 academic mechanism replica.
- `paper_case_studies/` contains paper figure/table rendering from stored
  artifacts.

## Style

Use comments to explain physical assumptions, safety rules, and non-obvious
numerical choices. Avoid comments that restate Python syntax or describe
iteration history.

Public docs should use these terms:

- academic mechanism replica,
- protected-side voltage,
- finite-window response,
- robust screen,
- data-limited verdict,
- nonlinear TDS validation.

Avoid public references to archived exploratory paths except inside
`archive/legacy_experiments/README.md`.

## Tests

Run the full suite with:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

For fast checks after figure/table edits:

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_finite_window.py \
  tests/test_resolvent_proxy.py \
  tests/test_mitigation.py
```

For paper-output checks:

```bash
PYTHONPATH=src .venv/bin/python scripts/reproduce_paper_artifacts.py --clean
PYTHONPATH=src .venv/bin/python run_case_studies.py --skip-derive --only fig02_validation_matrix --strict
PYTHONPATH=src .venv/bin/python scripts/make_quantitative_case_elements.py
PYTHONPATH=src .venv/bin/python scripts/generate_finite_window_proxy_figure.py
```

## Artifact Hygiene

Before preparing a release, remove:

- `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.DS_Store`;
- `.venv/`;
- `tools/TinyTeX/`;
- generated `results/` artifacts unless intentionally packaging a reproduction
  bundle outside Git;
- preview images and duplicate PNG/SVG outputs when the paper consumes PDFs.

The supported Git repository should be reproducible from scripts rather than
from committed generated results.
