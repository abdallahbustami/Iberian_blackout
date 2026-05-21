# PA-DVSA: Protection-Aware Dynamic Voltage Security Assessment

PA-DVSA is a research codebase for screening overvoltage protection cascades
before committing operator actions to full nonlinear time-domain simulation. It
combines ANDES dynamic models, protected-side voltage reconstruction, finite
relay-window response maps, robust uncertainty envelopes, cascade fixed-point
prediction, mitigation LPs, and nonlinear TDS validation.

The repository has two public paths:

- an installable Python package, `pa_dvsa`, for the screening method and ANDES
  integration;
- a paper-reproduction path containing scripts that regenerate the case-study
  datasets, figures, and tables locally.

## What This Repository Contains

- Core PA-DVSA algorithms for protected-voltage margins, event-to-margin
  erosion matrices, finite-window relay assessment, robust screening, cascade
  fixed points, and mitigation LPs.
- An ANDES adapter for benchmark loading, DAE/Jacobian access, and nonlinear TDS
  protection replay.
- A modified ACTIVSg2000 academic mechanism replica of the Iberian-style
  overvoltage cascade.
- Reproducible paper case-study scripts for Kundur, IEEE 39, NPCC, GBnetwork,
  and the ACTIVSg2000 replica.

## What This Repository Does Not Claim

- The ACTIVSg2000 case is not a forensic dynamic equivalent of Spain and
  Portugal.
- The project is not an official ENTSO-E reproduction.
- The sparse screen is not a replacement for RMS/TDS validation.
- The replica is a modified academic benchmark used to study a mechanism:
  high-voltage operating points, loss of reactive absorption, collector-side
  overvoltage protection, AC/HVDC separation logic, and island viability.

## Method Overview

For each operating mode, PA-DVSA evaluates protected-side relay voltages and the
available margin to pickup. Candidate events are converted into signed
disturbance channels and mapped to normalized finite-window margin erosion
values,

\[
K_{ij}^{\mathrm{pk}}
=
\frac{\max_{0\le t\le T_i}[\Delta z_i(t)]_+}{h_i},
\]

where \(h_i\) is the protected-side margin to pickup. The resulting \(K\) matrix
is used to rank events, compute additive cascade fixed points, and decide which
scenarios should be certified, flagged, sent to nonlinear validation, or
mitigated.

The mitigation LP uses time-resolved lower bounds on control authority, not
nameplate MVAr alone. Slow controls that do not act inside the relay window
therefore contribute little to the certificate.

## Modeling Assumptions

The main case studies use the following assumptions.

- **Protected-side voltage matters.** Relays may monitor collector-side or
  transformer-side voltages that are not equal to the upstream transmission bus
  voltage.
- **Tap reconstruction is conservative.** If collector telemetry is unavailable,
  protected voltage is reconstructed through a fixed tap and error envelope.
- **Reactive absorption loss is the key positive feedback.** A trip can remove
  active generation and inductive absorption; the latter can raise nearby
  voltages and trigger additional protection.
- **Fixed-power-factor RES behavior couples MW schedules to MVAr absorption.**
  A fixed-PF ramp can remove reactive absorption when the system is already near
  protected-voltage limits.
- **Relay timing is finite-window.** Pickup, dwell, and reset/hysteresis are
  evaluated over the relay-relevant time window.
- **Missing protected-side data is data-limited, not safe.** If the uncertainty
  envelope cannot certify a positive protected-side margin, the screen reports a
  data-limited verdict.

See `docs/modeling_assumptions.md` for more detail.

## Installation

Use Python 3.10 or newer. The paper runs here used Python 3.13.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

ANDES is required for dynamic simulations and is installed through
`pyproject.toml`. LaTeX and Ghostscript are optional; they improve paper figure
text rendering and outlined PDF export.

## Quick Start

Run the test suite:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Generate traceability and environment artifacts:

```bash
PYTHONPATH=src .venv/bin/python -m pa_dvsa.phase0
```

Run the supported ACTIVSg2000 academic mechanism replica:

```bash
PYTHONPATH=src .venv/bin/python -m pa_dvsa.replication.academic_replica --plot
```

The convenience launcher is equivalent:

```bash
PYTHONPATH=src .venv/bin/python replication.py --plot
```

## Reproducing Paper Artifacts

Generated outputs are intentionally not committed. To rebuild the supported
paper artifacts from a clean checkout, run:

```bash
PYTHONPATH=src .venv/bin/python scripts/reproduce_paper_artifacts.py --clean
```

This runs the benchmark case studies, the ACTIVSg2000 academic replica, the
multi-scenario paper sweep, the main figure/table renderer, the replication
subsection figures, the quantitative tables, and the finite-window proxy
figure.

To inspect the commands without launching TDS jobs:

```bash
PYTHONPATH=src .venv/bin/python scripts/reproduce_paper_artifacts.py --dry-run --clean
```

The individual commands used by the orchestrator are:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_kundur_case_study.py

PYTHONPATH=src .venv/bin/python scripts/run_load_shedding_case_study.py ieee39

PYTHONPATH=src .venv/bin/python scripts/run_load_shedding_case_study.py npcc_full

PYTHONPATH=src .venv/bin/python scripts/run_load_shedding_case_study.py gbnetwork

PYTHONPATH=src .venv/bin/python -m pa_dvsa.replication.academic_replica --plot

PYTHONPATH=src .venv/bin/python scripts/run_paper_case_sweep.py \
  --out results/paper_case_studies/dataset

PYTHONPATH=src .venv/bin/python run_case_studies.py --skip-derive --strict

PYTHONPATH=src .venv/bin/python scripts/make_quantitative_case_elements.py

PYTHONPATH=src .venv/bin/python scripts/generate_finite_window_proxy_figure.py
```

Expected outputs:

- paper figures in the configured figure output directory;
- generated tables in the configured table output directory;
- normalized case-study datasets in `results/paper_case_studies/dataset/`;
- quantitative table inputs in
  `results/paper_case_studies/quantitative_elements/`;
- ACTIVSg2000 replica artifacts in
  `results/activsg2000_iberian_replication/`;
- finite-window proxy traces in `results/figures/finite_window_proxy/`.

The original manuscript source is not included in this public tree. Some
reproduction scripts keep the default output paths
`LaTeX/figures/generated/` and `LaTeX/tables/generated/` for compatibility with
the paper workflow; those directories are created on demand. Use `--out-fig`
and `--out-table` if you want generated artifacts elsewhere.

## Repository Map

- `src/pa_dvsa/`: maintained package code.
- `src/pa_dvsa/replication/`: supported ACTIVSg2000 academic replica.
- `src/pa_dvsa/paper_case_studies/`: paper figure/table generation.
- `scripts/`: reproducibility and case-study scripts.
- `tests/`: unit and integration tests.
- `data/activsg2000_stable/`: modified ACTIVSg2000 RAW/DYR input used by
  the supported academic mechanism replica.
- `data/`: small benchmark modifications needed by the supported runs.
- `results/`: local generated artifacts recreated by the reproduction scripts
  and ignored by Git.
- `docs/`: modeling, reproducibility, and developer notes.

## Supported Public Commands

```bash
python -m pa_dvsa.phase0
python -m pa_dvsa.replication.academic_replica --plot
python replication.py --plot
python replication.py engine --plot
python replication.py calibration --use-existing-baseline
```

Legacy exploratory commands are archived under `archive/legacy_experiments/`
and are not part of the supported public workflow.

## Citation And License

Add a `CITATION.cff` and an explicit license before public release. Until a
license is added, treat the repository as all-rights-reserved research code.
