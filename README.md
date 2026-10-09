# Protection-aware dynamic voltage security assessment

Code and results of the paper *The Iberian Blackout: A Black Swan or a Gray Rhino? A Protection-Aware Dynamic
Voltage Security Assessment* (A. Alalem Albustami and A. F. Taha).

The method screens candidate events (trips, ramps, topology actions, shunt switching) for overvoltage cascades.
For each event it computes the voltage at every overvoltage relay over the relay time window from a linear model
of the network, its controls and its relays, predicts which relays trip after a first trip, certifies the relays
that cannot pick up, identifies the relays whose data are insufficient for a conclusion, and computes the
least-cost fast reactive power actions that keep the relay voltages below pickup. The results are validated
against nonlinear time-domain simulation (TDS) in ANDES on four benchmark systems and on a 2000-bus system.

## Installation

Python 3.10 or later (tested with 3.11).

```bash
git clone https://github.com/abdallahbustami/Iberian_blackout.git
cd Iberian_blackout
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` pins the versions used for the paper. ANDES must be 1.9.3, since the results depend on its
models. `kvxopt` provides the sparse LU factorization used for the 2000-bus system. The figures use LaTeX fonts
when a LaTeX installation is found (for example [TinyTeX](https://yihui.org/tinytex/)) and matplotlib's fonts
otherwise.

## Quick start

```bash
pytest -m "not slow"                  # tests, about 2 min
python scripts/make_paper_figures.py  # Figs. 5-13 from the stored results, about 1 min
```

The figures are written to `results/revised/figures/`. The slow tests (`pytest -m slow`) run longer comparisons
with TDS (about 20 min).

## Reproducing the paper

The commands run from the repository root and write to `results/revised/`, replacing the stored files. Runtimes
are approximate, for a 12-core laptop.

| Paper | Command | Runtime |
|---|---|---|
| Sections 7.4-7.9, Table 4, Tables A.1, A.2 and A.4, Figs. 7-13 | `python scripts/run_revised_studies.py --jobs 6` | 20-30 min |
| Section 7.4, results with a 2 s window | `python scripts/run_revised_studies.py --horizon 2 --jobs 6` | 30-40 min |
| Section 7.2, base state of the 2000-bus system | `python -m pa_dvsa.replication.academic_replica --jobs 6` | 8 min |
| Section 7.2, base-state variants | `python -m pa_dvsa.replication.commitment_study --jobs 5` | 25 min |
| Table A.3, components of the 2000-bus system | `python -m pa_dvsa.replication.inventory` | 1 min |
| Section 7.2, stressed state, Fig. 5 | `python scripts/run_iberian_sequence.py --jobs 3` | 20-25 min |
| Table 3 and the facts of the main run | `python scripts/report_iberian_sequence.py` | 1 min |
| Reactive balance of the stressed state | `python scripts/report_iberian_sequence_balance.py` | 2 min |
| Inter-area modes for k = 10, 12 and 15 | `python scripts/report_iberian_sequence_modes.py` | 5 min |
| Section 7.3, screening of the base state, Fig. 6 | see below | 1 h |
| Section 7.3, screening of the stressed state | `python scripts/run_sequence_screen.py --jobs 6` | 75 min |
| Figs. 5-13 | `python scripts/make_paper_figures.py` | 1 min |

The screening of the base state builds four variants (case-data or reduced commitment, initial margin 0.02 or
0.01 pu), screens them, validates them with 50 ms relays in all four and with 1 s relays in the two 0.02 pu
variants, and measures the single-core runtimes:

```bash
python scripts/run_replica_screen.py --stages build,screen --jobs 2
python scripts/run_replica_screen.py --stages tds --relays fast --jobs 4
python scripts/run_replica_screen.py --stages tds --margins 0.02 --relays ov1 --jobs 4
python scripts/run_replica_screen.py --stages report
OPENBLAS_NUM_THREADS=1 python scripts/run_replica_screen.py --stages screen,report --jobs 1 --blas-threads 1
OPENBLAS_NUM_THREADS=1 python scripts/time_replica_screen_vs_tds.py --slugs stock_m0p02,C1p5_m0p02
OPENBLAS_NUM_THREADS=1 python scripts/verify_replica_krylov_engine.py --slug stock_m0p02
```

## Repository

```
src/pa_dvsa/              the method
  event_response.py         relay voltage response of each event and control (exact jump, linear model)
  linearization.py          linear model of a mode
  finite_window.py          window responses and the upper bound on their maximum
  cascade_certificate.py    threshold cascade map and trip check
  robust_screening.py       robust screening over the uncertainty set and the classes of the plants
  mitigation.py             mitigation LP
  protected_outputs.py      relay voltages and margins (measured, from the tap, or from measured P and Q)
  krylov_response.py        sparse and Krylov linear algebra for large systems
  tds_runner.py             nonlinear TDS with definite-time overvoltage relays
  benchmarks.py             the four benchmark systems, their events and controls
  revised_studies/          benchmark studies (screening, validation, mitigation, observability, uncertainty)
  replication/              the 2000-bus system (base state, stressed state, screening)
  revised_figures/          paper figures
scripts/                  entry points listed above
tests/                    tests (pytest)
data/activsg2000_stable/  ACTIVSg2000 case (RAW and DYR files)
results/revised/          stored results
```

## Stored results

`results/revised/` holds the result files behind every number, table and figure of the paper (60 MB). The traces
of individual simulations and the caches are not stored, and the scripts regenerate them.

| Folder | Contents |
|---|---|
| `kundur/`, `ieee39/`, `npcc/`, `gbnetwork/` | case design, screening (`screen_k.csv`), predicted cascades (`cascade_predictions.csv`), validation (`tds_seeds.csv`, `tds_validation.csv`), mitigation, observability, uncertainty, ranking, runtimes and figure data of each benchmark; `horizon_2s/` for the 2 s window |
| `summary/` | pooled tables over the four benchmarks (`headline_validation.csv` is Table 4) |
| `activsg2000/` | base state (`baseline/`, variants, `commitment_sweep/`, `condition_buildup.json`, `inventory.json`), its screening (`screen/`) and the screening of the stressed state (`screen_sequence/`) |
| `iberian_sequence/` | stressed-state runs (`final_*/`), Table 3 (`final_table.json`), reactive balance (`initial_balance.json`) and inter-area modes (`modes.json`) |

`scripts/report_iberian_sequence.py` needs the traces of all 16 stressed-state runs, so it runs only after
`scripts/run_iberian_sequence.py`.

## License

The code is released under the MIT License (see `LICENSE`). The ACTIVSg2000 case in `data/activsg2000_stable/`
is the synthetic grid of the Texas A&M University Electric Grid Test Case Repository (A. B. Birchfield et al.,
"Grid structural characteristics as validation criteria for synthetic networks," IEEE Transactions on Power
Systems, 2017) and is distributed with its original terms. The benchmark systems are built from the cases that
ship with ANDES.
