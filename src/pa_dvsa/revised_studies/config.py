"""Common settings of the benchmark studies (window, grid, epsilon, simulation protocol, uncertainty levels)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results" / "revised"

SYSTEMS = ("kundur", "ieee39", "npcc", "gbnetwork")
STUDIES = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11", "S12")

T_WINDOW_S = 1.0
DT_S = 0.005
EPSILON = 0.05
ENVELOPE_TOL_PU = 1e-4

T_SEED_S = 0.5
TDS_T_END_S = T_SEED_S + 1.5
TDS_TSTEP_S = 0.002
TDS_TSTEP_CHECK_S = 0.001
NOOP_DV_PU = 1e-4

BASE_MVA = 100.0
EPS_MEAS_PU = 0.002
TAP_BAND_REL = 0.025
TRANSMISSION_ONLY_TAP = (0.90, 1.10)
TRANSMISSION_ONLY_ERR_PU = (-0.02, 0.02)
EPS_PQ_REL = 0.02
GSU_Z_REL = 0.10
GSU_Z_REL_SENSITIVITY = (0.05, 0.20)
UNCERTAINTY_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
UNC_RELAY_THRESHOLD_PU = 0.02
UNC_TAP_REL = 0.025
UNC_RECONSTRUCTION_PU = 0.01
UNC_Q_ABSORPTION = 0.30

LP_RESTRICTIONS = {
    "all": None,
    "manual_reactor": ("manual_reactor",),
    "avr_vref": ("avr_vref",),
    "plant_q_command": ("plant_q_command",),
    "statcom": ("statcom",),
    "fast_reactor": ("fast_reactor",),
    "all_but_fast_reactor": ("statcom", "plant_q_command", "avr_vref", "manual_reactor"),
}
LP_ALPHA_TOL = 1e-7

ABLATION_ROWS = (
    "baseline",
    "no_ov_protection",
    "delayed_protection",
    "preserved_absorption",
    "voltage_mode_res",
    "fast_automatic_shunt",
    "missing_collector_telemetry",
)
SANITY_SEED = 20260924


def set_horizon(horizon_s: float) -> None:
    global T_WINDOW_S, TDS_T_END_S
    T_WINDOW_S = float(horizon_s)
    TDS_T_END_S = T_SEED_S + 1.5 * float(horizon_s)
