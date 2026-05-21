#!/usr/bin/env python3
"""Run an end-to-end Kundur stress-test case study.

This is not an official Iberian replica.  It is a small two-area Kundur
integration test that uses real ANDES Jacobians and TDS simulation to exercise
the protection-aware workflow end to end:

1. load ``kundur_full.xlsx``;
2. protect the two load buses with tight overvoltage margins;
3. linearize/reduce the ANDES DAE;
4. build finite-window pickup/trip maps and a cascade certificate;
5. solve the mitigation LP;
6. validate with a nonlinear TDS load-shedding run;
7. generate CSV/JSON/table/figure artifacts for the paper.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import andes
import numpy as np
from scipy import sparse

from pa_dvsa.andes_adapter import AndesCase, AndesCaseSpec
from pa_dvsa.benchmark_studies import (
    BenchmarkStudyConfig,
    BenchmarkTarget,
    ReplicaCollectorConfig,
    ScalabilityMetrics,
    build_prediction_study,
)
from pa_dvsa.cascade_certificate import compute_cascade_certificate
from pa_dvsa.data_model import (
    AssessmentStatus,
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
from pa_dvsa.finite_window import compute_finite_window_maps
from pa_dvsa.linearization import LinearDAEModel, reduce_linear_dae, selector_matrix
from pa_dvsa.mitigation import solve_mitigation_lp
from pa_dvsa.nonlinear_validation import (
    ProtectionRelaySpec,
    ProtectionReplayResult,
    RelayTripTarget,
    TDSProtectionCallback,
    compare_screen_to_nonlinear,
)
from pa_dvsa.paper_outputs import PaperOutputPaths, generate_paper_outputs
from pa_dvsa.protected_outputs import evaluate_protected_outputs
from pa_dvsa.robust_screening import compute_robust_screen
from pa_dvsa.zero_delay import compute_zero_delay_screen


RESULT_DIR = Path("results/case_studies/kundur")
BUS_IDS = ("7", "8")
PQ_IDS = ("PQ_0", "PQ_1")
ASSET_IDS = ("load7", "load8")
EVENT_IDS = ("load7_shed", "load8_shed")
CONTROL_IDS = ("absorb7", "absorb8")
ASSET_EVENT_MAP = dict(zip(ASSET_IDS, EVENT_IDS))
MARGIN_PU = 0.004
DWELL_S = 0.05
HORIZON_S = 1.0
TDS_TF_S = 3.0
TDS_STEP_S = 0.01
SEED_TIME_S = 0.10


def main() -> int:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    case_path = andes.get_case("kundur/kundur_full.xlsx")

    screen_case = _load_case(case_path, init_tds=True)
    summary = screen_case.summary()
    protected_assets = _protected_assets(screen_case)
    protected = evaluate_protected_outputs(screen_case, protected_assets)
    reduced = _reduced_model(screen_case, protected)

    zero_delay = compute_zero_delay_screen(
        reduced,
        protected,
        event_ids=EVENT_IDS,
        mode_id="kundur_load_shedding",
        metadata={"case": "kundur_full", "study": "tight-load-bus-overvoltage"},
    )
    finite = compute_finite_window_maps(
        reduced,
        protected,
        event_ids=EVENT_IDS,
        control_ids=CONTROL_IDS,
        horizons_s=(HORIZON_S, HORIZON_S),
        dwell_times_s=(DWELL_S, DWELL_S),
        time_grid_s=np.linspace(0.0, HORIZON_S, 101),
        samples_per_window=101,
        max_refinements=1,
        padding_tolerance_pu=1.0e-3,
        metadata={"case": "kundur_full", "seed_time_s": SEED_TIME_S},
    )
    cascade = compute_cascade_certificate(
        finite,
        seed_ids=("load7_shed",),
        asset_event_map=ASSET_EVENT_MAP,
        protected_outputs=protected,
        dwell_times_s=(DWELL_S, DWELL_S),
    )
    robust = compute_robust_screen(
        finite,
        protected_outputs=protected,
        seed_event_ids=("load7_shed",),
        asset_event_map=ASSET_EVENT_MAP,
        epsilon=0.05,
    )
    mitigation = solve_mitigation_lp(
        finite,
        seed_ids=("load7_shed",),
        protected_outputs=protected,
        asset_event_map=ASSET_EVENT_MAP,
        alpha_upper={"absorb7": 1.0, "absorb8": 1.0},
        control_mvar={"absorb7": 100.0, "absorb8": 100.0},
        epsilon=0.05,
    )
    nonlinear, trace_rows, tds_notes = _run_tds_validation(case_path, protected)
    cascade_result = cascade.cascade_result
    if cascade_result is None:
        raise RuntimeError("cascade certificate returned no top result")
    validation = compare_screen_to_nonlinear(
        scenario_id="kundur_load7_seed",
        mode_id=finite.window_result.mode_id,
        seed_ids=("load7_shed",),
        predicted=cascade_result,
        simulated=nonlinear,
        robust=True,
        metadata={"tds_notes": tds_notes},
    )
    prediction = build_prediction_study(
        finite,
        study_id="kundur_load_shedding",
        cascade=cascade,
        nonlinear=validation,
        robust_validation=True,
        metadata={"case": "kundur_full", "threshold_margin_pu": MARGIN_PU},
    )
    runtime_s = time.perf_counter() - started
    metrics = (
        ScalabilityMetrics(
            study_id="kundur_load_shedding",
            case_id="kundur_full",
            runtime_s=runtime_s,
            sparse_solve_count=1,
            selected_channels=len(EVENT_IDS) + len(CONTROL_IDS),
            protected_asset_count=len(ASSET_IDS),
            event_count=len(EVENT_IDS),
            control_count=len(CONTROL_IDS),
            status=AssessmentStatus.CERTIFIED,
            metadata={"dae_states": summary["dae_n"], "dae_algebraic": summary["dae_m"]},
        ),
    )
    config = _study_config(screen_case, protected)
    paper_manifest = generate_paper_outputs(
        output_id="kundur_load_shedding",
        config=config,
        prediction=prediction,
        robust=robust,
        mitigation=mitigation,
        scalability=metrics,
        paths=PaperOutputPaths.from_project_root(
            ".",
            artifact_dir=RESULT_DIR / "paper_outputs",
            figure_dir="LaTeX/figures/generated",
            table_dir="LaTeX/tables/generated/kundur",
        ),
        generate_figures=True,
    )

    _write_trace(trace_rows)
    payload = {
        "case_path": case_path,
        "summary": summary,
        "study_assumptions": {
            "not_official_replica": True,
            "threshold_margin_pu": MARGIN_PU,
            "dwell_s": DWELL_S,
            "seed": "load7_shed at t=0.10 s",
            "disturbance_sign": "D_y row = - Q_abs_pu at the load-bus voltage equation",
            "control_sign": "B_y row = +1.0 for a 100-MVAr reactive absorption command",
        },
        "protected_outputs": [item.to_dict() for item in protected],
        "zero_delay": zero_delay.to_dict(),
        "finite_window": finite.to_dict(),
        "cascade": cascade.to_dict(),
        "robust": robust.to_dict(),
        "mitigation": mitigation.to_dict(),
        "nonlinear_validation": validation.to_dict(),
        "metrics": [item.to_dict() for item in metrics],
        "paper_manifest": paper_manifest.to_dict(),
    }
    (RESULT_DIR / "kundur_case_study.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary(payload)
    print(f"Wrote {RESULT_DIR / 'kundur_case_study.json'}")
    print(f"Wrote {RESULT_DIR / 'tds_trace.csv'}")
    print(f"Wrote {RESULT_DIR / 'summary.md'}")
    print("Top-line result:")
    print(f"  Cascade fixed point: {cascade_result.fixed_point}")
    print(f"  Nonlinear trips: {[item.asset_id for item in nonlinear.trip_records]}")
    print(f"  Mitigation MVAr: {mitigation.mitigation_result.required_mvar_total:.3f}")
    print(f"  Validation status: {validation.validation_result.status.value}")
    return 0


def _load_case(case_path: str, *, init_tds: bool) -> AndesCase:
    case = AndesCase.load(
        AndesCaseSpec(case_path, setup=True, run_pflow=True, init_tds=init_tds),
        quiet=True,
    )
    system = case.require_system()
    if hasattr(system, "Toggle"):
        for index in range(system.Toggle.n):
            system.Toggle.u.v[index] = 0
    return case


def _protected_assets(case: AndesCase) -> tuple[ProtectedAsset, ...]:
    system = case.require_system()
    bus_pos = {str(bus): index for index, bus in enumerate(system.Bus.idx.v)}
    pq_pos = {str(idx): index for index, idx in enumerate(system.PQ.idx.v)}
    assets: list[ProtectedAsset] = []
    for asset_id, bus_id, pq_id in zip(ASSET_IDS, BUS_IDS, PQ_IDS):
        base_v = float(system.Bus.v.v[bus_pos[bus_id]])
        q_abs = abs(float(system.PQ.q0.v[pq_pos[pq_id]])) * float(case.base_mva or 100.0)
        assets.append(
            ProtectedAsset(
                asset_id=asset_id,
                name=f"Kundur load bus {bus_id} tight overvoltage relay",
                protected_voltage=ProtectedVoltageSpec(
                    output_id=f"z_{asset_id}",
                    side=MeasurementSide.TRANSMISSION_BUS,
                    kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
                    protected_bus_id=bus_id,
                ),
                threshold_pu=Interval.exact(base_v + MARGIN_PU, Unit.PU),
                timing=ProtectionTiming(
                    DelayKind.DWELL,
                    dwell_time_s=Interval.exact(DWELL_S, Unit.SECOND),
                ),
                disconnected_assets=(
                    DisconnectedAsset(
                        AssetRef(pq_id, model_family="PQ", bus_id=bus_id),
                        q_absorption_mvar=Interval.exact(q_abs, Unit.MVAR),
                    ),
                ),
                q_absorption_mvar=Interval.exact(q_abs, Unit.MVAR),
                tags=("kundur", "stress_test", "load_shedding"),
            )
        )
    return tuple(assets)


def _reduced_model(case: AndesCase, protected) :
    system = case.require_system()
    jac = case.jacobians()
    output_addresses = [item.source_address.address for item in protected]
    cy = selector_matrix(output_addresses, jac.n_algebraic)
    cx = sparse.csc_matrix((len(output_addresses), jac.n_states))
    dy = sparse.lil_matrix((jac.n_algebraic, len(EVENT_IDS)))
    by = sparse.lil_matrix((jac.n_algebraic, len(CONTROL_IDS)))
    for column, bus_id in enumerate(BUS_IDS):
        pq_q_abs_pu = abs(float(system.PQ.q0.v[column]))
        address = case.bus_voltage_address(bus_id)
        dy[address, column] = -pq_q_abs_pu
        by[address, column] = +1.0
    model = LinearDAEModel.from_andes_jacobians(
        jac,
        cx=cx,
        cy=cy,
        bx=sparse.csc_matrix((jac.n_states, len(CONTROL_IDS))),
        by=by.tocsc(),
        dx=sparse.csc_matrix((jac.n_states, len(EVENT_IDS))),
        dy=dy.tocsc(),
        mode_id="kundur_load_shedding",
    )
    return reduce_linear_dae(model)


def _run_tds_validation(case_path: str, protected) -> tuple[ProtectionReplayResult, list[dict], str]:
    case = _load_case(case_path, init_tds=True)
    system = case.require_system()
    system.TDS.config.tf = TDS_TF_S
    system.TDS.config.tstep = TDS_STEP_S
    system.TDS.config.no_tqdm = 1
    bus_pos = {str(bus): index for index, bus in enumerate(system.Bus.idx.v)}
    relays = tuple(
        ProtectionRelaySpec(
            asset_id=item.asset_id,
            threshold_pu=item.threshold_pu.nominal,
            dwell_time_s=DWELL_S,
            source_bus_id=item.source_bus_id,
            trip_targets=(RelayTripTarget("PQ", pq_id, 0.0),),
            source="kundur_tds",
        )
        for item, pq_id in zip(protected, PQ_IDS)
    )
    callback = TDSProtectionCallback(relays)
    trace: list[dict] = []

    class SeedAndProtection:
        seed_done = False
        seed_time = None

        def __call__(self, t, ss):
            time_s = float(t)
            if not self.seed_done and time_s >= SEED_TIME_S - 1.0e-12:
                ss.set_status("PQ", "PQ_0", 0)
                self.seed_done = True
                self.seed_time = time_s
            callback(time_s, ss)
            trace.append(
                {
                    "time_s": time_s,
                    "v_load7_pu": float(ss.Bus.v.v[bus_pos["7"]]),
                    "v_load8_pu": float(ss.Bus.v.v[bus_pos["8"]]),
                    "pq0_status": float(ss.PQ.u.v[0]),
                    "pq1_status": float(ss.PQ.u.v[1]),
                }
            )

    seed = SeedAndProtection()
    system.TDS.callpert = seed
    ok = bool(system.TDS.run(no_summary=True))
    notes = f"tds_ok={ok}; busted={system.TDS.busted}; seed_time={seed.seed_time}"
    return callback.replay_result(), trace, notes


def _study_config(case: AndesCase, protected) -> BenchmarkStudyConfig:
    system = case.require_system()
    q_abs = [item.q_absorption_mvar.nominal for item in _protected_assets(case)]
    collectors = tuple(
        ReplicaCollectorConfig(
            asset_id=asset_id,
            transmission_bus=int(bus_id),
            collector_bus=int(bus_id),
            tap_ratio=1.0,
            base_kv=float(system.Bus.Vn.v[list(system.Bus.idx.v).index(int(bus_id))]),
            threshold_pu=item.threshold_pu.nominal,
            dwell_time_s=DWELL_S,
            q_absorption_mvar=float(q),
            source_name=pq_id,
            shunt_name="none",
        )
        for asset_id, bus_id, pq_id, item, q in zip(ASSET_IDS, BUS_IDS, PQ_IDS, protected, q_abs)
    )
    return BenchmarkStudyConfig(
        study_id="kundur_load_shedding",
        target=BenchmarkTarget(
            case_id="kundur_full",
            case_alias="kundur/kundur_full.xlsx",
            description="Kundur two-area load-bus overvoltage stress test",
            category="small_signal_case_study",
            setup=True,
            run_pflow=True,
            init_tds=True,
        ),
        collectors=collectors,
        scenario_parameters={
            "threshold_margin_pu": MARGIN_PU,
            "dwell_s": DWELL_S,
            "seed_time_s": SEED_TIME_S,
            "tds_tf_s": TDS_TF_S,
            "tds_step_s": TDS_STEP_S,
        },
        feature_flags={
            "protection": True,
            "tds_validation": True,
            "mitigation_lp": True,
            "official_iberian_replica": False,
        },
        seed_ids=("load7_shed",),
        notes="Small Kundur stress test for workflow verification; thresholds are intentionally tight.",
    )


def _write_trace(rows: list[dict]) -> None:
    path = RESULT_DIR / "tds_trace.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("time_s", "v_load7_pu", "v_load8_pu", "pq0_status", "pq1_status"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(payload: dict) -> None:
    cascade = payload["cascade"]["seed_results"][0]
    validation = payload["nonlinear_validation"]["validation_result"]
    mitigation = payload["mitigation"]["mitigation_result"]
    lines = [
        "# Kundur Case Study",
        "",
        "This is a workflow stress test on the bundled ANDES Kundur two-area case, not an official Iberian replica.",
        "",
        "## Main Results",
        "",
        f"- Protected assets: {', '.join(payload['finite_window']['window_result']['protected_asset_ids'])}",
        f"- Seed: `load7_shed` at {SEED_TIME_S:.2f} s in nonlinear TDS.",
        f"- Cascade fixed point: {cascade['fixed_point']}",
        f"- Nonlinear trips: {[item['asset_id'] for item in validation['simulated_trips']]}",
        f"- False positives: {validation['false_positive_trips']}",
        f"- False negatives: {validation['false_negative_trips']}",
        f"- Validation status: {validation['status']}",
        f"- Mitigation feasible: {mitigation['feasible']}",
        f"- Required fast reactive absorption: {mitigation['required_mvar_total']:.3f} MVAr",
        "",
        "## Important Caveats",
        "",
        "- Relay thresholds are intentionally tight: base voltage plus 0.004 pu.",
        "- Disturbance channels are load-shedding/loss-of-reactive-absorption channels at buses 7 and 8.",
        "- This case exposed that ANDES status changes should use `system.set_status`, not direct `u.v` assignment.",
        "- The reduced Kundur model is reported as marginal, so finite-window maps are more meaningful than asymptotic stability claims.",
        "",
    ]
    (RESULT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
