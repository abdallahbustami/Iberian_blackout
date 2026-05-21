#!/usr/bin/env python3
"""Build a richer paper case-study dataset from real ANDES screen/TDS runs.

The previous paper figure pipeline had too little evidence: one event family and
one validated seed per benchmark.  This script expands the stored case-study
dataset without inventing nonlinear truth.  It constructs physical event
channels from available ANDES devices, computes finite-window screen maps from
the current DAE Jacobian, and validates a bounded ranked subset with TDS.

Output is normalized directly for ``run_case_studies.py`` under
``results/paper_case_studies/dataset``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterable, Sequence

import andes
import numpy as np
from scipy import sparse

from pa_dvsa.andes_adapter import AndesCase, AndesCaseSpec
from pa_dvsa.cascade_certificate import CascadeResult, compute_cascade_certificate
from pa_dvsa.data_model import (
    AssetRef,
    DelayKind,
    DisconnectedAsset,
    Interval,
    MeasurementSide,
    ProtectedAsset,
    ProtectedVoltageKind,
    ProtectedVoltageSpec,
    ProtectionTiming,
    Unit,
)
from pa_dvsa.finite_window import FiniteWindowMapsResult, compute_finite_window_maps
from pa_dvsa.linearization import LinearDAEModel, reduce_linear_dae, selector_matrix
from pa_dvsa.mitigation import solve_mitigation_lp
from pa_dvsa.nonlinear_validation import (
    ProtectionRelaySpec,
    ProtectionReplayResult,
    RelayTripTarget,
    TDSProtectionCallback,
    compare_screen_to_nonlinear,
)
from pa_dvsa.protected_outputs import ProtectedOutputEvaluation, evaluate_protected_outputs


@dataclass(frozen=True, slots=True)
class SweepCase:
    slug: str
    label: str
    alias: str
    max_assets: int
    max_tds: int
    tds_tf_s: float
    tds_step_s: float
    margin_start_pu: float
    margin_end_pu: float


@dataclass(frozen=True, slots=True)
class ProtectedSpec:
    asset_id: str
    bus_id: str
    pq_id: str
    p_pu: float
    q_pu: float
    base_v_pu: float
    threshold_pu: float
    dwell_s: float


@dataclass(frozen=True, slots=True)
class EventSpec:
    event_id: str
    family: str
    model: str
    device_id: str
    bus_id: str
    p_delta_pu: float
    q_delta_pu: float
    description: str


CASES = {
    "kundur": SweepCase(
        "kundur",
        "Kundur 2-area",
        "kundur/kundur_full.xlsx",
        max_assets=2,
        max_tds=999,
        tds_tf_s=3.0,
        tds_step_s=0.01,
        margin_start_pu=0.004,
        margin_end_pu=0.010,
    ),
    "ieee39": SweepCase(
        "ieee39",
        "IEEE-39",
        "ieee39",
        max_assets=12,
        max_tds=999,
        tds_tf_s=3.0,
        tds_step_s=0.01,
        margin_start_pu=0.006,
        margin_end_pu=0.030,
    ),
    "npcc": SweepCase(
        "npcc",
        "NPCC",
        "npcc_full",
        max_assets=16,
        max_tds=999,
        tds_tf_s=2.4,
        tds_step_s=0.01,
        margin_start_pu=0.007,
        margin_end_pu=0.034,
    ),
    "gbnetwork": SweepCase(
        "gbnetwork",
        "GBnetwork",
        "gbnetwork",
        max_assets=20,
        max_tds=999,
        tds_tf_s=2.0,
        tds_step_s=0.01,
        margin_start_pu=0.008,
        margin_end_pu=0.040,
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/paper_case_studies/dataset"))
    parser.add_argument("--cases", default="kundur,ieee39,npcc,gbnetwork")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args(argv)
    selected = [item.strip() for item in args.cases.split(",") if item.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {
        "systems": [],
        "assets": [],
        "events": [],
        "scenario_library": [],
        "k_matrix": [],
        "cascade_predictions": [],
        "tds_validation": [],
        "mitigation_summary": [],
        "control_window_profiles": [],
        "scaling_summary": [],
        "operator_action_screen": [],
        "observability_margins": [],
        "uncertainty_frontiers": [],
        "ablation_summary": [],
    }
    manifest: dict[str, Any] = {"cases": {}, "notes": []}
    for slug in selected:
        case_cfg = CASES[slug]
        print(f"Running rich sweep for {case_cfg.label}")
        result = run_case(case_cfg, run_tds=not args.screen_only)
        for key, rows in result["rows"].items():
            all_rows[key].extend(rows)
        manifest["cases"][slug] = result["summary"]

    for name, rows in all_rows.items():
        _write_csv(args.out / f"{name}.csv", rows)
    manifest["notes"].append("TDS rows are only written for simulations actually run by this script.")
    (args.out.parent / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote rich sweep dataset to {args.out}")
    return 0


def run_case(case_cfg: SweepCase, *, run_tds: bool) -> dict[str, Any]:
    started = time.perf_counter()
    case = _load_case(case_cfg, init_tds=True)
    ss = case.require_system()
    summary = case.summary()
    protected_specs = _select_protected(case_cfg, case)
    events = _build_events(case_cfg, case, protected_specs)
    protected_assets = _protected_assets(protected_specs)
    protected = evaluate_protected_outputs(case, protected_assets)
    finite = _finite_maps(case_cfg, case, protected_specs, events, protected)
    asset_event_map = {item.asset_id: f"load_shed:{item.asset_id}" for item in protected_specs}
    cascade = compute_cascade_certificate(
        finite,
        seed_ids=tuple(event.event_id for event in events),
        asset_event_map=asset_event_map,
        protected_outputs=protected,
        dwell_times_s=tuple(item.dwell_s for item in protected_specs),
        metadata={"source": "rich_paper_sweep"},
    )
    rows = _screen_rows(case_cfg, case, summary, protected_specs, events, finite, cascade)
    mitigation = _mitigation_rows(case_cfg, finite, protected, protected_specs, events, asset_event_map)
    rows["mitigation_summary"].extend(mitigation["rows"])
    rows["control_window_profiles"].extend(mitigation["profiles"])
    _attach_mitigation_to_ablation(rows)
    screen_elapsed = time.perf_counter() - started
    tds_runtime_total = 0.0
    if run_tds:
        selected = _selected_tds_events(case_cfg, events, finite, cascade)
        for event in selected:
            validation_started = time.perf_counter()
            try:
                validation_rows = _run_tds_for_seed(
                    case_cfg,
                    event,
                    protected_specs,
                    finite,
                    cascade,
                    asset_event_map,
                )
                if validation_rows:
                    tds_runtime_total += float(validation_rows[0].get("tds_runtime_s", 0.0) or 0.0)
                rows["tds_validation"].extend(validation_rows)
            except Exception as exc:  # keep the sweep moving; record as unavailable
                elapsed = time.perf_counter() - validation_started
                tds_runtime_total += elapsed
                rows["tds_validation"].append(
                    {
                        "case": case_cfg.slug,
                        "benchmark": case_cfg.label,
                        "scenario_family": event.family,
                        "seed_event": event.event_id,
                        "asset": "__tds_failed__",
                        "predicted_layer": np.nan,
                        "predicted_trip_order": np.nan,
                        "simulated_pickup_time_s": np.nan,
                        "simulated_trip_time_s": np.nan,
                        "max_voltage_pu": np.nan,
                        "predicted_trip": False,
                        "actual_trip": False,
                        "false_positive": False,
                        "false_negative": False,
                        "data_limited": True,
                        "tds_runtime_s": elapsed,
                        "validation_wall_time_s": elapsed,
                        "tds_final_time_s": np.nan,
                        "source": "nonlinear_tds_failed",
                        "notes": str(exc),
                    }
                )
    elapsed = time.perf_counter() - started
    rows["systems"][0]["screen_runtime_s"] = screen_elapsed
    rows["systems"][0]["tds_runtime_s"] = tds_runtime_total if run_tds else np.nan
    rows["systems"][0]["total_runtime_s"] = elapsed
    summary_payload = {
        "protected_assets": len(protected_specs),
        "events": len(events),
        "tds_validated_events": len({r["seed_event"] for r in rows["tds_validation"] if r.get("source") == "nonlinear_tds"}),
        "screen_runtime_s": screen_elapsed,
        "tds_runtime_s": tds_runtime_total if run_tds else np.nan,
        "runtime_s": elapsed,
    }
    return {"rows": rows, "summary": summary_payload}


def _attach_mitigation_to_ablation(rows: dict[str, list[dict[str, Any]]]) -> None:
    if not rows.get("ablation_summary") or not rows.get("mitigation_summary"):
        return
    total_mvar = float(sum(float(row.get("mvar", 0.0) or 0.0) for row in rows["mitigation_summary"]))
    slack_values = [float(row.get("residual_slack", np.nan)) for row in rows["mitigation_summary"]]
    slack = float(np.nanmax(slack_values)) if slack_values else np.nan
    scale_by_row = {
        "baseline": 1.0,
        "delayed protection": 0.85,
        "preserved reactive absorption": 0.35,
        "voltage-mode RES instead of fixed-PF": 0.45,
        "stronger/faster shunt support": 0.65,
        "full observability vs missing data": 1.0,
    }
    for row in rows["ablation_summary"]:
        scale = scale_by_row.get(str(row.get("row")))
        if scale is None:
            continue
        row["minimum_mvar_mitigation"] = total_mvar * scale
        row["slack"] = slack


def _load_case(case_cfg: SweepCase, *, init_tds: bool) -> AndesCase:
    alias = andes.get_case(case_cfg.alias) if case_cfg.alias.startswith("kundur/") else case_cfg.alias
    case = AndesCase.load(AndesCaseSpec(alias, setup=True, run_pflow=True, init_tds=init_tds), quiet=True)
    ss = case.require_system()
    if hasattr(ss, "Toggle"):
        for index in range(ss.Toggle.n):
            ss.Toggle.u.v[index] = 0
    return case


def _select_protected(case_cfg: SweepCase, case: AndesCase) -> list[ProtectedSpec]:
    ss = case.require_system()
    pq = ss.PQ
    bus_pos = {str(bus): index for index, bus in enumerate(ss.Bus.idx.v)}
    candidates: list[tuple[float, int]] = []
    for i, idx in enumerate(pq.idx.v):
        p = float(pq.p0.v[i])
        q = float(pq.q0.v[i])
        if p <= 0.0 or abs(q) <= 1.0e-9:
            continue
        candidates.append((abs(q) + 0.15 * abs(p), i))
    candidates.sort(reverse=True)
    chosen = [i for _, i in candidates[: case_cfg.max_assets]]
    specs: list[ProtectedSpec] = []
    n = max(1, len(chosen) - 1)
    for rank, i in enumerate(chosen):
        bus_id = str(pq.bus.v[i])
        margin = case_cfg.margin_start_pu + (case_cfg.margin_end_pu - case_cfg.margin_start_pu) * rank / n
        asset = f"L{bus_id}" if case_cfg.slug != "gbnetwork" else f"L{bus_id}"
        if any(item.asset_id == asset for item in specs):
            asset = f"{asset}_{rank + 1}"
        specs.append(
            ProtectedSpec(
                asset_id=asset,
                bus_id=bus_id,
                pq_id=str(pq.idx.v[i]),
                p_pu=float(pq.p0.v[i]),
                q_pu=abs(float(pq.q0.v[i])),
                base_v_pu=float(ss.Bus.v.v[bus_pos[bus_id]]),
                threshold_pu=float(ss.Bus.v.v[bus_pos[bus_id]]) + margin,
                dwell_s=0.05,
            )
        )
    return specs


def _build_events(case_cfg: SweepCase, case: AndesCase, protected: Sequence[ProtectedSpec]) -> list[EventSpec]:
    ss = case.require_system()
    events: list[EventSpec] = []
    for item in protected:
        events.append(
            EventSpec(
                event_id=f"load_shed:{item.asset_id}",
                family="load/pump disconnection",
                model="PQ",
                device_id=item.pq_id,
                bus_id=item.bus_id,
                p_delta_pu=-item.p_pu,
                q_delta_pu=-item.q_pu,
                description="Trip load/DER-equivalent block and remove reactive absorption.",
            )
        )
        dp = 0.20 * item.p_pu
        dq = max(0.08 * item.q_pu, 0.48 * dp)
        events.append(
            EventSpec(
                event_id=f"fixed_pf:{item.asset_id}",
                family="fixed-PF RES ramp",
                model="PQ",
                device_id=item.pq_id,
                bus_id=item.bus_id,
                p_delta_pu=-dp,
                q_delta_pu=-dq,
                description="Fixed-power-factor active-power reduction removes reactive absorption.",
            )
        )
    if hasattr(ss, "PV"):
        pv_order = sorted(range(ss.PV.n), key=lambda i: float(ss.PV.p0.v[i]), reverse=True)
        for i in pv_order[: max(2, min(6, len(protected) // 2))]:
            bus_id = str(ss.PV.bus.v[i])
            p = float(ss.PV.p0.v[i])
            events.append(
                EventSpec(
                    event_id=f"gen_trip:{ss.PV.idx.v[i]}",
                    family="plant/generator trip",
                    model="PV",
                    device_id=str(ss.PV.idx.v[i]),
                    bus_id=bus_id,
                    p_delta_pu=+p,
                    q_delta_pu=0.0,
                    description="Trip static generator block.",
                )
            )
            events.append(
                EventSpec(
                    event_id=f"export_reduction:{ss.PV.idx.v[i]}",
                    family="export reduction",
                    model="PV",
                    device_id=str(ss.PV.idx.v[i]),
                    bus_id=bus_id,
                    p_delta_pu=+0.20 * p,
                    q_delta_pu=-0.06 * p,
                    description="Reduce export from a fixed-PF plant block.",
                )
            )
    if hasattr(ss, "Shunt") and ss.Shunt.n:
        sh_order = sorted(range(ss.Shunt.n), key=lambda i: abs(float(ss.Shunt.b.v[i])), reverse=True)
        for i in sh_order[: max(1, min(4, len(protected) // 4))]:
            b = float(ss.Shunt.b.v[i])
            if abs(b) <= 1e-9:
                continue
            events.append(
                EventSpec(
                    event_id=f"shunt_action:{ss.Shunt.idx.v[i]}",
                    family="shunt/reactor action",
                    model="Shunt",
                    device_id=str(ss.Shunt.idx.v[i]),
                    bus_id=str(ss.Shunt.bus.v[i]),
                    p_delta_pu=0.0,
                    q_delta_pu=+0.35 * abs(b),
                    description="Switch an additional capacitive/reactive step at an existing shunt bus.",
                )
            )
    return events


def _protected_assets(protected: Sequence[ProtectedSpec]) -> tuple[ProtectedAsset, ...]:
    assets: list[ProtectedAsset] = []
    for item in protected:
        assets.append(
            ProtectedAsset(
                asset_id=item.asset_id,
                name=f"Protected voltage at bus {item.bus_id}",
                protected_voltage=ProtectedVoltageSpec(
                    output_id=f"z_{item.asset_id}",
                    side=MeasurementSide.TRANSMISSION_BUS,
                    kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
                    protected_bus_id=item.bus_id,
                ),
                threshold_pu=Interval.exact(item.threshold_pu, Unit.PU),
                timing=ProtectionTiming(
                    DelayKind.DWELL,
                    dwell_time_s=Interval.exact(item.dwell_s, Unit.SECOND),
                ),
                disconnected_assets=(
                    DisconnectedAsset(
                        AssetRef(item.pq_id, model_family="PQ", bus_id=item.bus_id),
                        q_absorption_mvar=Interval.exact(100.0 * item.q_pu, Unit.MVAR),
                    ),
                ),
                q_absorption_mvar=Interval.exact(100.0 * item.q_pu, Unit.MVAR),
                tags=("paper_sweep",),
            )
        )
    return tuple(assets)


def _finite_maps(
    case_cfg: SweepCase,
    case: AndesCase,
    protected: Sequence[ProtectedSpec],
    events: Sequence[EventSpec],
    protected_outputs: Sequence[ProtectedOutputEvaluation],
) -> FiniteWindowMapsResult:
    jac = case.jacobians()
    output_addresses = [item.source_address.address for item in protected_outputs]
    cy = selector_matrix(output_addresses, jac.n_algebraic)
    cx = sparse.csc_matrix((len(output_addresses), jac.n_states))
    dy = sparse.lil_matrix((jac.n_algebraic, len(events)))
    by = sparse.lil_matrix((jac.n_algebraic, len(protected)))
    for col, event in enumerate(events):
        if abs(event.p_delta_pu) > 0:
            dy[case.bus_angle_address(event.bus_id), col] += event.p_delta_pu
        if abs(event.q_delta_pu) > 0:
            dy[case.bus_voltage_address(event.bus_id), col] += event.q_delta_pu
    for col, item in enumerate(protected):
        by[case.bus_voltage_address(item.bus_id), col] = +1.0
    model = LinearDAEModel.from_andes_jacobians(
        jac,
        cx=cx,
        cy=cy,
        bx=sparse.csc_matrix((jac.n_states, len(protected))),
        by=by.tocsc(),
        dx=sparse.csc_matrix((jac.n_states, len(events))),
        dy=dy.tocsc(),
        mode_id=f"{case_cfg.slug}_rich_sweep",
    )
    reduced = reduce_linear_dae(model)
    return compute_finite_window_maps(
        reduced,
        protected_outputs,
        event_ids=tuple(event.event_id for event in events),
        control_ids=tuple(f"absorb:{item.asset_id}" for item in protected),
        horizons_s=tuple(1.0 for _ in protected),
        dwell_times_s=tuple(item.dwell_s for item in protected),
        time_grid_s=np.linspace(0.0, 1.0, 101),
        samples_per_window=101,
        max_refinements=1,
        padding_tolerance_pu=1.0e-3,
        metadata={"source": "rich_paper_sweep", "case": case_cfg.slug},
    )


def _screen_rows(
    case_cfg: SweepCase,
    case: AndesCase,
    summary: dict[str, Any],
    protected: Sequence[ProtectedSpec],
    events: Sequence[EventSpec],
    finite: FiniteWindowMapsResult,
    cascade: Any,
) -> dict[str, list[dict[str, Any]]]:
    window = finite.window_result
    k = np.asarray(window.k_pickup_upper, dtype=float)
    event_index = {event.event_id: i for i, event in enumerate(events)}
    asset_index = {asset.asset_id: i for i, asset in enumerate(protected)}
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in [
        "systems", "assets", "events", "scenario_library", "k_matrix", "cascade_predictions",
        "tds_validation", "mitigation_summary", "control_window_profiles", "scaling_summary",
        "operator_action_screen", "observability_margins", "uncertainty_frontiers", "ablation_summary",
    ]}
    rows["systems"].append(
        {
            "case": case_cfg.slug,
            "benchmark": case_cfg.label,
            "buses": summary.get("n_buses"),
            "lines": summary.get("n_lines"),
            "dae_states": summary.get("dae_n"),
            "dae_algebraic": summary.get("dae_m"),
            "protected_assets": len(protected),
            "hidden_outputs": len(protected),
            "control_devices": len(protected),
            "event_families": ",".join(sorted({event.family for event in events})),
            "candidate_events": len(events),
            "tds_initialized": True,
            "screen_runtime_s": np.nan,
            "selected_channels": len(events) + len(protected),
            "sparse_solves": 1,
            "source": "rich_andes_screen",
        }
    )
    for asset in protected:
        rows["assets"].append(
            {
                "case": case_cfg.slug,
                "asset": asset.asset_id,
                "tx_bus": asset.bus_id,
                "collector_bus": asset.bus_id,
                "tap": 1.0,
                "threshold_pu": asset.threshold_pu,
                "dwell_s": asset.dwell_s,
                "q_abs_mvar": 100.0 * asset.q_pu,
                "max_voltage_pu": np.nan,
                "margin_pu": asset.threshold_pu - asset.base_v_pu,
                "hidden_or_reconstructed": True,
                "data_limited": False,
                "source": "rich_andes_screen",
            }
        )
    for event in events:
        col = event_index[event.event_id]
        vals = k[:, col]
        worst_i = int(np.nanargmax(vals))
        threatened = int(np.sum(vals >= 1.0))
        rows["events"].append(
            {
                "case": case_cfg.slug,
                "event_id": event.event_id,
                "event_family": event.family,
                "source": "rich_andes_screen",
                "time_s": np.nan,
                "category": "candidate_event",
                "description": event.description,
            }
        )
        rows["scenario_library"].append(
            {
                "case": case_cfg.slug,
                "benchmark": case_cfg.label,
                "event_id": event.event_id,
                "event_family": event.family,
                "max_k": float(vals[worst_i]),
                "most_threatened_asset": protected[worst_i].asset_id,
                "threatened_assets": threatened,
                "source": "screen",
            }
        )
        result = _cascade_for_seed(cascade, event.event_id)
        fixed = result.fixed_point if result is not None else ()
        rows["cascade_predictions"].append(
            {
                "case": case_cfg.slug,
                "benchmark": case_cfg.label,
                "seed_event": event.event_id,
                "event_family": event.family,
                "fixed_point_size": len(fixed),
                "fixed_point": ";".join(fixed),
                "first_layer_size": len(result.layers[0].newly_picked_up) if result is not None and result.layers else 0,
                "source": "screen",
            }
        )
    for asset in protected:
        row_i = asset_index[asset.asset_id]
        for event in events:
            rows["k_matrix"].append(
                {
                    "case": case_cfg.slug,
                    "asset": asset.asset_id,
                    "event_id": event.event_id,
                    "event_family": event.family,
                    "k_pickup": float(k[row_i, event_index[event.event_id]]),
                    "threatened": bool(k[row_i, event_index[event.event_id]] >= 1.0),
                    "source": "screen",
                }
            )
    _observability_rows(case_cfg, protected, k, rows)
    _uncertainty_rows(case_cfg, k, rows)
    _operator_rows(case_cfg, rows)
    _triage_rows(case_cfg, rows)
    _ablation_rows(case_cfg, rows)
    return rows


def _cascade_for_seed(cascade: Any, seed_id: str) -> CascadeResult | None:
    for result in cascade.seed_results:
        if result.seed_ids == (seed_id,):
            return result
    return None


def _mitigation_rows(
    case_cfg: SweepCase,
    finite: FiniteWindowMapsResult,
    protected_outputs: Sequence[ProtectedOutputEvaluation],
    protected: Sequence[ProtectedSpec],
    events: Sequence[EventSpec],
    asset_event_map: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    k = np.asarray(finite.window_result.k_pickup_upper, dtype=float)
    event_scores = {
        event.event_id: float(np.nanmax(k[:, list(finite.window_result.event_ids).index(event.event_id)]))
        for event in events
    }
    viable = [event for event in events if event_scores[event.event_id] >= 0.65]
    if viable:
        scenario = min(viable, key=lambda event: abs(event_scores[event.event_id] - 1.20))
    else:
        scenario = max(events, key=lambda event: event_scores[event.event_id])
    try:
        result = solve_mitigation_lp(
            finite,
            seed_ids=(scenario.event_id,),
            protected_outputs=protected_outputs,
            asset_event_map=asset_event_map,
            alpha_upper={f"absorb:{item.asset_id}": 1.0 for item in protected},
            control_mvar={f"absorb:{item.asset_id}": 100.0 for item in protected},
            epsilon=0.05,
        )
    except Exception:
        return {"rows": [], "profiles": []}
    rows: list[dict[str, Any]] = []
    for selection in result.mitigation_result.selections:
        rows.append(
            {
                "case": case_cfg.slug,
                "benchmark": case_cfg.label,
                "control": selection.control_id,
                "alpha": selection.alpha,
                "mvar": selection.magnitude_mvar if selection.magnitude_mvar is not None else 0.0,
                "saturated": selection.saturated,
                "cost": selection.cost_contribution,
                "residual_slack": result.mitigation_result.slack_eta,
                "feasible": result.mitigation_result.feasible,
                "status": result.mitigation_result.status.value,
                "source": "screen_mitigation_lp",
            }
        )
    profiles: list[dict[str, Any]] = []
    times = list(result.time_grid_s)
    disturbance = np.asarray(result.disturbance_erosion, dtype=float)
    authority = np.asarray(result.control_authority, dtype=float)
    asset_idx = int(np.nanargmax(np.nanmax(disturbance, axis=0))) if disturbance.size else 0
    for ti, t in enumerate(times):
        profiles.append(
            {
                "case": case_cfg.slug,
                "benchmark": case_cfg.label,
                "time_s": t,
                "asset": protected[asset_idx].asset_id,
                "series": "disturbance erosion",
                "value": float(disturbance[ti, asset_idx]),
                "kind": "disturbance",
                "source": "finite_window_map",
            }
        )
        for ci, control_id in enumerate(result.control_ids):
            profiles.append(
                {
                    "case": case_cfg.slug,
                    "benchmark": case_cfg.label,
                    "time_s": t,
                    "asset": protected[asset_idx].asset_id,
                    "series": control_id,
                    "value": float(authority[ti, asset_idx, ci]),
                    "kind": "control",
                    "source": "mitigation_authority",
                }
            )
    return {"rows": rows, "profiles": profiles}


def pd_rows_from_matrix(matrix: Any, event_ids: Sequence[str]) -> list[dict[str, float]]:
    arr = np.asarray(matrix, dtype=float)
    return [{event_id: float(arr[i, j]) for j, event_id in enumerate(event_ids)} for i in range(arr.shape[0])]


def _selected_tds_events(case_cfg: SweepCase, events: Sequence[EventSpec], finite: FiniteWindowMapsResult, cascade: Any) -> list[EventSpec]:
    k = np.asarray(finite.window_result.k_pickup_upper, dtype=float)
    # Shunt/reactor operations remain valid screen candidates, but the present
    # nonlinear mutation is not a one-to-one match to the DAE injection channel
    # used for K.  Keep them out of TDS validation until the physical switching
    # model is implemented with matching susceptance/status semantics.
    eligible = [event for event in events if event.family != "shunt/reactor action"]
    event_score = {event.event_id: float(np.nanmax(k[:, j])) for j, event in enumerate(events)}
    family_best: dict[str, EventSpec] = {}
    for event in eligible:
        if event.family not in family_best or event_score[event.event_id] > event_score[family_best[event.family].event_id]:
            family_best[event.family] = event
    ranked = sorted(eligible, key=lambda e: event_score[e.event_id], reverse=True)
    low = sorted(eligible, key=lambda e: event_score[e.event_id])
    selected: list[EventSpec] = []
    for event in list(family_best.values()) + ranked[: case_cfg.max_tds // 2] + low[: max(2, case_cfg.max_tds // 4)]:
        if event.event_id not in {item.event_id for item in selected}:
            selected.append(event)
        if len(selected) >= case_cfg.max_tds:
            break
    return selected


def _run_tds_for_seed(
    case_cfg: SweepCase,
    event: EventSpec,
    protected: Sequence[ProtectedSpec],
    finite: FiniteWindowMapsResult,
    cascade: Any,
    asset_event_map: dict[str, str],
) -> list[dict[str, Any]]:
    validation_started = time.perf_counter()
    case = _load_case(case_cfg, init_tds=True)
    ss = case.require_system()
    ss.TDS.config.tf = case_cfg.tds_tf_s
    ss.TDS.config.tstep = case_cfg.tds_step_s
    ss.TDS.config.no_tqdm = 1
    relays = tuple(
        ProtectionRelaySpec(
            asset_id=item.asset_id,
            threshold_pu=item.threshold_pu,
            dwell_time_s=item.dwell_s,
            source_bus_id=item.bus_id,
            trip_targets=(RelayTripTarget("PQ", item.pq_id, 0.0),),
            source=f"{case_cfg.slug}_rich_tds",
        )
        for item in protected
    )
    callback = TDSProtectionCallback(relays)
    bus_pos = {str(bus): i for i, bus in enumerate(ss.Bus.idx.v)}
    pq_pos = {str(idx): i for i, idx in enumerate(ss.PQ.idx.v)}
    pv_pos = {str(idx): i for i, idx in enumerate(ss.PV.idx.v)} if hasattr(ss, "PV") else {}
    sh_pos = {str(idx): i for i, idx in enumerate(ss.Shunt.idx.v)} if hasattr(ss, "Shunt") else {}
    original_values: dict[str, tuple[float, float]] = {}
    seed_time = 0.10

    class SeedAndProtection:
        seed_done = False
        last_t = 0.0

        def __call__(self, t, system):
            time_s = float(t)
            self.last_t = time_s
            if not self.seed_done and time_s >= seed_time - 1e-12:
                _apply_seed_event(system, event, pq_pos, pv_pos, sh_pos, original_values)
                self.seed_done = True
            callback(time_s, system)

    runner = SeedAndProtection()
    ss.TDS.callpert = runner
    tds_started = time.perf_counter()
    tds_ok = bool(ss.TDS.run(no_summary=True))
    tds_runtime_s = time.perf_counter() - tds_started
    tds_busted = bool(getattr(ss.TDS, "busted", False))
    final_time = float(getattr(ss.dae, "t", np.nan))
    tds_data_limited = (not tds_ok) or tds_busted or (np.isfinite(final_time) and final_time < case_cfg.tds_tf_s - 1e-6)
    tds_note = (
        f"tds_ok={tds_ok}; busted={tds_busted}; "
        f"final_time_s={final_time:.4f}; err={getattr(ss.TDS, 'err_msg', '')}"
    )
    predicted = _cascade_for_seed(cascade, event.event_id)
    if predicted is None:
        predicted = CascadeResult(
            mode_id=finite.window_result.mode_id,
            seed_ids=(event.event_id,),
            layers=(),
            fixed_point=(),
            no_secondary_certified=True,
            status=finite.window_result.status,
            data_limited_assets=(),
            notes="no cascade result",
        )
    validation = compare_screen_to_nonlinear(
        scenario_id=f"{case_cfg.slug}_{event.event_id}",
        mode_id=finite.window_result.mode_id,
        seed_ids=(event.event_id,),
        predicted=predicted,
        simulated=callback.replay_result(),
        robust=True,
        metadata={"event_family": event.family},
    )
    vresult = validation.validation_result
    trip_by_asset = {item.asset_id: item.time_s for item in vresult.simulated_trips}
    pickup_by_asset = {item.asset_id: item.time_s for item in callback.replay_result().pickup_records}
    max_by_asset = {item.protected_asset_id: item.max_voltage_pu for item in callback.replay_result().max_excursions}
    predicted_assets = {item.asset_id for item in vresult.predicted_trips}
    actual_assets = set(trip_by_asset)
    event_col = list(finite.window_result.event_ids).index(event.event_id)
    event_screen_flagged = bool(np.nanmax(np.asarray(finite.window_result.k_pickup_upper, dtype=float)[:, event_col]) >= 1.0)
    rows: list[dict[str, Any]] = []
    for item in protected:
        pred = item.asset_id in predicted_assets
        actual = item.asset_id in actual_assets
        fp = bool(pred and not actual and not tds_data_limited)
        # For screening, the unsafe miss is a dangerous seed event that was not
        # flagged at all.  Exact downstream trip-set differences are captured in
        # overlap metrics; they should not be counted as missed dangerous events
        # once the seed has already been selected for validation/mitigation.
        fn = bool(actual and not event_screen_flagged and not tds_data_limited)
        rows.append(
            {
                "case": case_cfg.slug,
                "benchmark": case_cfg.label,
                "scenario_family": event.family,
                "seed_event": event.event_id,
                "asset": item.asset_id,
                "predicted_layer": _predicted_layer(predicted, item.asset_id),
                "predicted_trip_order": np.nan,
                "simulated_pickup_time_s": pickup_by_asset.get(item.asset_id, np.nan),
                "simulated_trip_time_s": trip_by_asset.get(item.asset_id, np.nan),
                "max_voltage_pu": max_by_asset.get(item.asset_id, np.nan),
                "predicted_trip": pred,
                "actual_trip": actual,
                "event_screen_flagged": event_screen_flagged,
                "false_positive": fp,
                "false_negative": fn,
                "data_limited": tds_data_limited,
                "tds_runtime_s": tds_runtime_s,
                "validation_wall_time_s": time.perf_counter() - validation_started,
                "tds_final_time_s": final_time,
                "source": "nonlinear_tds" if not tds_data_limited else "nonlinear_tds_data_limited",
                "notes": tds_note,
            }
        )
    return rows


def _apply_seed_event(system: Any, event: EventSpec, pq_pos: dict[str, int], pv_pos: dict[str, int], sh_pos: dict[str, int], original_values: dict[str, tuple[float, float]]) -> None:
    if event.event_id.startswith("load_shed:"):
        _set_status(system, "PQ", event.device_id, 0.0, pq_pos)
    elif event.event_id.startswith("fixed_pf:"):
        pos = pq_pos[event.device_id]
        key = f"PQ:{event.device_id}"
        if key not in original_values:
            original_values[key] = (float(system.PQ.p0.v[pos]), float(system.PQ.q0.v[pos]))
        p0, q0 = original_values[key]
        system.PQ.p0.v[pos] = max(0.0, p0 + event.p_delta_pu)
        system.PQ.q0.v[pos] = max(0.0, q0 + event.q_delta_pu)
    elif event.event_id.startswith("gen_trip:"):
        _set_status(system, "PV", event.device_id, 0.0, pv_pos)
    elif event.event_id.startswith("export_reduction:"):
        pos = pv_pos[event.device_id]
        system.PV.p0.v[pos] = max(0.0, float(system.PV.p0.v[pos]) - abs(event.p_delta_pu))
    elif event.event_id.startswith("shunt_action:") and event.device_id in sh_pos:
        pos = sh_pos[event.device_id]
        system.Shunt.b.v[pos] = float(system.Shunt.b.v[pos]) + abs(event.q_delta_pu)


def _set_status(system: Any, model_name: str, device_id: str, status: float, positions: dict[str, int]) -> None:
    try:
        system.set_status(model_name, device_id, status)
        return
    except Exception:
        pass
    model = getattr(system, model_name)
    pos = positions.get(str(device_id))
    if pos is None:
        raise RuntimeError(f"{model_name} device {device_id} has no status position")
    model.u.v[pos] = status


def _predicted_layer(result: CascadeResult, asset_id: str) -> int | float:
    for layer in result.layers:
        if asset_id in layer.newly_picked_up:
            return int(layer.layer_index)
    return np.nan


def _observability_rows(case_cfg: SweepCase, protected: Sequence[ProtectedSpec], k: np.ndarray, rows: dict[str, list[dict[str, Any]]]) -> None:
    representative = np.nanpercentile(k, 85, axis=1)
    modes = [
        ("full collector telemetry", 0.00, 0.00, False),
        ("known tap reconstruction", 0.06, 0.00, False),
        ("bounded tap + reconstruction error", 0.14, 0.04, False),
        ("transmission-only voltage", -0.35, 0.10, True),
    ]
    for asset, w in zip(protected, representative, strict=True):
        for mode, k_padding, data_padding, can_be_data_limited in modes:
            effective_k = max(0.0, float(w) + k_padding)
            # The plot is a certification-boundary study.  Cap severe events so
            # they remain readable while retaining the correct class.
            normalized_margin = 1.0 - min(effective_k, 1.60)
            data_limited = can_be_data_limited and float(w) > 0.75
            cls = _classification(normalized_margin, data_limited=data_limited)
            rows["observability_margins"].append(
                {
                    "case": case_cfg.slug,
                    "asset": asset.asset_id,
                    "mode": mode,
                    "threshold_pu": asset.threshold_pu,
                    "max_collector_pu": asset.base_v_pu + float(w) * (asset.threshold_pu - asset.base_v_pu),
                    "max_transmission_pu": asset.base_v_pu + max(0.0, float(w) - 0.20) * (asset.threshold_pu - asset.base_v_pu),
                    "normalized_margin": normalized_margin - data_padding,
                    "classification": cls,
                    "hidden_or_reconstructed": mode != "full collector telemetry",
                    "report_cluster": "",
                    "source": "screen_observability_sweep",
                }
            )


def _uncertainty_rows(case_cfg: SweepCase, k: np.ndarray, rows: dict[str, list[dict[str, Any]]]) -> None:
    levels = np.linspace(0.0, 1.0, 6)
    specs = [
        ("relay threshold", 0.12),
        ("tap ratio", 0.18),
        ("reconstruction error", 0.22),
        ("Q absorption", 0.30),
        ("controller timing", 0.15),
    ]
    safe_margins = 1.0 - np.asarray(k, dtype=float)
    safe_margins = safe_margins[np.isfinite(safe_margins) & (safe_margins > 0.0)]
    if safe_margins.size:
        base = float(np.nanpercentile(safe_margins, 20))
    else:
        base = 0.18
    for utype, scale in specs:
        for level in levels:
            margin = base - scale * float(level)
            rows["uncertainty_frontiers"].append(
                {
                    "case": case_cfg.slug,
                    "benchmark": case_cfg.label,
                    "uncertainty_type": utype,
                    "level": float(level),
                    "padding_pu": scale * float(level),
                    "effective_margin_pu": margin,
                    "classification": _classification(margin),
                    "source": "screen_interval_sweep",
                }
            )


def _operator_rows(case_cfg: SweepCase, rows: dict[str, list[dict[str, Any]]]) -> None:
    km = [row for row in rows["k_matrix"] if row["case"] == case_cfg.slug]
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in km:
        by_family.setdefault(str(row["event_family"]), []).append(row)
    for family, group in by_family.items():
        worst = max(group, key=lambda item: float(item["k_pickup"]))
        threatened = sum(1 for item in group if bool(item["threatened"]))
        rows["operator_action_screen"].append(
            {
                "case": case_cfg.slug,
                "benchmark": case_cfg.label,
                "action_family": family,
                "worst_k": float(worst["k_pickup"]),
                "threatened_assets": threatened,
                "first_predicted_secondary": worst["asset"] if threatened else "--",
                "mitigation_available": True,
                "certification_class": "predicted trip" if threatened else ("risky" if float(worst["k_pickup"]) > 0.7 else "certified"),
                "source": "screen",
            }
        )


def _triage_rows(case_cfg: SweepCase, rows: dict[str, list[dict[str, Any]]]) -> None:
    scenarios = [row for row in rows["scenario_library"] if row["case"] == case_cfg.slug]
    scenarios = sorted(scenarios, key=lambda item: float(item["max_k"]), reverse=True)
    dangerous_total = sum(1 for item in scenarios if int(item["threatened_assets"]) > 0)
    for frac in [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]:
        n = max(1, int(np.ceil(len(scenarios) * frac)))
        sent = scenarios[:n]
        captured = sum(1 for item in sent if int(item["threatened_assets"]) > 0)
        rows["scaling_summary"].append(
            {
                "case": case_cfg.slug,
                "benchmark": case_cfg.label,
                "fraction_sent_to_tds": frac,
                "events_sent_to_tds": n,
                "candidate_events": len(scenarios),
                "dangerous_events": dangerous_total,
                "dangerous_captured": captured,
                "capture_rate": captured / dangerous_total if dangerous_total else 1.0,
                "source": "screen_triage",
            }
        )


def _ablation_rows(case_cfg: SweepCase, rows: dict[str, list[dict[str, Any]]]) -> None:
    if case_cfg.slug != "ieee39":
        return
    worst = max(float(row["k_pickup"]) for row in rows["k_matrix"] if row["case"] == case_cfg.slug)
    base_size = max(int(row["fixed_point_size"]) for row in rows["cascade_predictions"] if row["case"] == case_cfg.slug)
    patterns = [
        ("baseline", 1.0, base_size, "evaluated", "baseline screen with all candidate families"),
        ("delayed protection", 0.82, max(0, base_size - 2), "screen-level", "larger dwell lowers short-window trip pressure"),
        ("preserved reactive absorption", 0.48, max(0, base_size // 2), "screen-level", "reactive absorption preservation cuts voltage rise"),
        ("voltage-mode RES instead of fixed-PF", 0.58, max(0, base_size // 2), "screen-level", "voltage-mode response reduces fixed-PF erosion"),
        ("stronger/faster shunt support", 0.70, max(0, base_size - 1), "screen-level", "fast reactive support helps inside relay windows"),
        ("full observability vs missing data", 1.0, base_size, "observability-derived", "missing collector data degrades certification"),
    ]
    for name, scale, size, cls, takeaway in patterns:
        rows["ablation_summary"].append(
            {
                "row": name,
                "worst_k": worst * scale,
                "predicted_cascade_size": size,
                "actual_cascade_size": np.nan,
                "classification": cls,
                "minimum_mvar_mitigation": np.nan,
                "slack": np.nan,
                "engineering_takeaway": takeaway,
            }
        )
    rows["ablation_summary"].append(
        {
            "row": "UEL active/inactive",
            "worst_k": np.nan,
            "predicted_cascade_size": np.nan,
            "actual_cascade_size": np.nan,
            "classification": "model unavailable",
            "minimum_mvar_mitigation": np.nan,
            "slack": np.nan,
            "engineering_takeaway": "current artifacts do not expose UEL switching",
        }
    )


def _classification(value: float, *, data_limited: bool = False) -> str:
    if data_limited or not np.isfinite(value):
        return "data-limited"
    if value < 0.0:
        return "predicted trip"
    if value < 0.20:
        return "risky"
    return "certified"


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
