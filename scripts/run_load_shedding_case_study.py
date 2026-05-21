#!/usr/bin/env python3
"""Run configurable load-shedding overvoltage cascade studies.

This script is the larger-system counterpart to ``run_kundur_case_study.py``.
It keeps the same protection-aware workflow but moves case-specific choices
into explicit study definitions so IEEE-39 and NPCC use the same code path.
    The scenarios are stress tests, not official Iberian replicas: thresholds are
    intentionally tight and the event channels are full PQ load-shedding channels
    at selected load buses.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Sequence

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
    NonlinearValidationReport,
    ProtectionRelaySpec,
    ProtectionReplayResult,
    RelayTripTarget,
    TDSProtectionCallback,
    compare_screen_to_nonlinear,
)
from pa_dvsa.paper_outputs import PaperOutputPaths, generate_paper_outputs
from pa_dvsa.protected_outputs import ProtectedOutputEvaluation, evaluate_protected_outputs
from pa_dvsa.robust_screening import compute_robust_screen
from pa_dvsa.zero_delay import compute_zero_delay_screen


@dataclass(frozen=True, slots=True)
class LoadSheddingStudy:
    study_id: str
    case_alias: str
    case_id: str
    description: str
    bus_ids: tuple[str, ...]
    pq_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    margin_pu: float = 0.004
    dwell_s: float = 0.05
    horizon_s: float = 1.0
    seed_asset_id: str = ""
    seed_time_s: float = 0.10
    tds_tf_s: float = 3.0
    tds_step_s: float = 0.01

    def __post_init__(self) -> None:
        sizes = {len(self.bus_ids), len(self.pq_ids), len(self.asset_ids)}
        if len(sizes) != 1:
            raise ValueError("bus_ids, pq_ids, and asset_ids must have equal length")
        if self.seed_asset_id and self.seed_asset_id not in self.asset_ids:
            raise ValueError(f"seed_asset_id {self.seed_asset_id!r} is not a protected asset")
        if not self.seed_asset_id:
            object.__setattr__(self, "seed_asset_id", self.asset_ids[0])

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(f"{asset_id}_shed" for asset_id in self.asset_ids)

    @property
    def control_ids(self) -> tuple[str, ...]:
        return tuple(f"absorb_{asset_id}" for asset_id in self.asset_ids)

    @property
    def seed_event_id(self) -> str:
        return f"{self.seed_asset_id}_shed"

    @property
    def asset_event_map(self) -> dict[str, str]:
        return dict(zip(self.asset_ids, self.event_ids))


STUDIES: dict[str, LoadSheddingStudy] = {
    "ieee39": LoadSheddingStudy(
        study_id="ieee39_load_shedding",
        case_alias="ieee39",
        case_id="ieee39_full",
        description="IEEE-39 four-load overvoltage cascade stress test",
        bus_ids=("3", "4", "8", "15"),
        pq_ids=("PQ_1", "PQ_2", "PQ_4", "PQ_6"),
        asset_ids=("load3", "load4", "load8", "load15"),
        seed_asset_id="load3",
    ),
    "npcc_full": LoadSheddingStudy(
        study_id="npcc_full_load_shedding",
        case_alias="npcc_full",
        case_id="npcc_full",
        description="NPCC RAW/DYR four-load overvoltage cascade stress test",
        bus_ids=("78", "91", "131", "53"),
        pq_ids=("PQ_40", "PQ_48", "PQ_86", "PQ_27"),
        asset_ids=("load78", "load91", "load131", "load53"),
        seed_asset_id="load78",
        tds_tf_s=2.0,
    ),
    "gbnetwork": LoadSheddingStudy(
        study_id="gbnetwork_load_shedding",
        case_alias="gbnetwork",
        case_id="gbnetwork",
        description="GBnetwork four-load overvoltage cascade stress test",
        bus_ids=("971", "818", "906", "745"),
        pq_ids=("PQ_296", "PQ_189", "PQ_249", "PQ_123"),
        asset_ids=("load971", "load818", "load906", "load745"),
        seed_asset_id="load971",
        tds_tf_s=2.0,
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=sorted(STUDIES))
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="skip nonlinear TDS validation and only write screen artifacts",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="write CSV/JSON/tables but skip generated PDF figures",
    )
    args = parser.parse_args(argv)
    payload = run_study(
        STUDIES[args.study],
        run_tds=not args.screen_only,
        generate_figures=not args.no_figures,
    )
    cascade = payload["cascade"]["seed_results"][0]
    validation = payload.get("nonlinear_validation")
    print(f"Wrote {payload['result_json']}")
    if payload.get("trace_csv"):
        print(f"Wrote {payload['trace_csv']}")
    print(f"Wrote {payload['summary_md']}")
    print("Top-line result:")
    print(f"  Cascade fixed point: {cascade['fixed_point']}")
    if validation is not None:
        result = validation["validation_result"]
        print(f"  Nonlinear trips: {[item['asset_id'] for item in result['simulated_trips']]}")
        print(f"  Validation status: {result['status']}")
    else:
        print("  Nonlinear trips: not run")
    print(
        "  Mitigation MVAr: "
        f"{payload['mitigation']['mitigation_result']['required_mvar_total']:.3f}"
    )
    return 0


def run_study(
    study: LoadSheddingStudy,
    *,
    run_tds: bool,
    generate_figures: bool,
) -> dict:
    result_dir = Path("results/case_studies") / study.study_id
    result_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    screen_case = _load_case(study, init_tds=True)
    summary = screen_case.summary()
    protected_assets = _protected_assets(study, screen_case)
    protected = evaluate_protected_outputs(screen_case, protected_assets)
    reduced = _reduced_model(study, screen_case, protected)

    zero_delay = compute_zero_delay_screen(
        reduced,
        protected,
        event_ids=study.event_ids,
        mode_id=study.study_id,
        metadata={"case": study.case_id, "study": study.description},
    )
    finite = compute_finite_window_maps(
        reduced,
        protected,
        event_ids=study.event_ids,
        control_ids=study.control_ids,
        horizons_s=tuple(study.horizon_s for _ in study.asset_ids),
        dwell_times_s=tuple(study.dwell_s for _ in study.asset_ids),
        time_grid_s=np.linspace(0.0, study.horizon_s, 101),
        samples_per_window=101,
        max_refinements=1,
        padding_tolerance_pu=1.0e-3,
        metadata={"case": study.case_id, "seed_time_s": study.seed_time_s},
    )
    cascade = compute_cascade_certificate(
        finite,
        seed_ids=(study.seed_event_id,),
        asset_event_map=study.asset_event_map,
        protected_outputs=protected,
        dwell_times_s=tuple(study.dwell_s for _ in study.asset_ids),
    )
    robust = compute_robust_screen(
        finite,
        protected_outputs=protected,
        seed_event_ids=(study.seed_event_id,),
        asset_event_map=study.asset_event_map,
        epsilon=0.05,
    )
    mitigation = solve_mitigation_lp(
        finite,
        seed_ids=(study.seed_event_id,),
        protected_outputs=protected,
        asset_event_map=study.asset_event_map,
        alpha_upper={control_id: 1.0 for control_id in study.control_ids},
        control_mvar={control_id: 100.0 for control_id in study.control_ids},
        epsilon=0.05,
    )

    nonlinear: ProtectionReplayResult | None = None
    validation: NonlinearValidationReport | None = None
    trace_rows: list[dict] = []
    tds_notes = "not_run"
    cascade_result = cascade.cascade_result
    if cascade_result is None:
        raise RuntimeError("cascade certificate returned no top result")
    if run_tds:
        nonlinear, trace_rows, tds_notes = _run_tds_validation(study, protected)
        validation = compare_screen_to_nonlinear(
            scenario_id=f"{study.study_id}_{study.seed_asset_id}_seed",
            mode_id=finite.window_result.mode_id,
            seed_ids=(study.seed_event_id,),
            predicted=cascade_result,
            simulated=nonlinear,
            robust=True,
            metadata={"tds_notes": tds_notes},
        )

    prediction = build_prediction_study(
        finite,
        study_id=study.study_id,
        cascade=cascade,
        nonlinear=validation,
        robust_validation=True,
        metadata={"case": study.case_id, "threshold_margin_pu": study.margin_pu},
    )
    runtime_s = time.perf_counter() - started
    metrics = (
        ScalabilityMetrics(
            study_id=study.study_id,
            case_id=study.case_id,
            runtime_s=runtime_s,
            sparse_solve_count=1,
            selected_channels=len(study.event_ids) + len(study.control_ids),
            protected_asset_count=len(study.asset_ids),
            event_count=len(study.event_ids),
            control_count=len(study.control_ids),
            status=AssessmentStatus.EMPIRICAL if run_tds else AssessmentStatus.NOT_EVALUATED,
            metadata={"dae_states": summary["dae_n"], "dae_algebraic": summary["dae_m"]},
        ),
    )
    config = _study_config(study, screen_case, protected)
    paper_manifest = generate_paper_outputs(
        output_id=study.study_id,
        config=config,
        prediction=prediction,
        robust=robust,
        mitigation=mitigation,
        scalability=metrics,
        paths=PaperOutputPaths.from_project_root(
            ".",
            artifact_dir=result_dir / "paper_outputs",
            figure_dir="LaTeX/figures/generated",
            table_dir=f"LaTeX/tables/generated/{study.study_id}",
        ),
        generate_figures=generate_figures,
    )

    trace_csv = None
    if trace_rows:
        trace_csv = _write_trace(result_dir, study, trace_rows)
    payload = {
        "result_json": str(result_dir / f"{study.study_id}.json"),
        "trace_csv": None if trace_csv is None else str(trace_csv),
        "summary_md": str(result_dir / "summary.md"),
        "summary": summary,
        "study_assumptions": {
            "not_official_replica": True,
            "threshold_margin_pu": study.margin_pu,
            "dwell_s": study.dwell_s,
            "seed": f"{study.seed_event_id} at t={study.seed_time_s:.2f} s",
            "disturbance_sign": (
                "D_y rows = -P0_pu at the load-bus angle equation and "
                "-Q0_pu at the load-bus voltage equation"
            ),
            "control_sign": "B_y row = +1.0 for a 100-MVAr reactive absorption command",
            "tds_notes": tds_notes,
        },
        "protected_outputs": [item.to_dict() for item in protected],
        "zero_delay": zero_delay.to_dict(),
        "finite_window": finite.to_dict(),
        "cascade": cascade.to_dict(),
        "robust": robust.to_dict(),
        "mitigation": mitigation.to_dict(),
        "nonlinear_validation": None if validation is None else validation.to_dict(),
        "metrics": [item.to_dict() for item in metrics],
        "paper_manifest": paper_manifest.to_dict(),
    }
    result_json = result_dir / f"{study.study_id}.json"
    result_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(result_dir, study, payload)
    return payload


def _load_case(study: LoadSheddingStudy, *, init_tds: bool) -> AndesCase:
    case = AndesCase.load(
        AndesCaseSpec(study.case_alias, setup=True, run_pflow=True, init_tds=init_tds),
        quiet=True,
    )
    system = case.require_system()
    if hasattr(system, "Toggle"):
        for index in range(system.Toggle.n):
            system.Toggle.u.v[index] = 0
    return case


def _protected_assets(study: LoadSheddingStudy, case: AndesCase) -> tuple[ProtectedAsset, ...]:
    system = case.require_system()
    bus_pos = {str(bus): index for index, bus in enumerate(system.Bus.idx.v)}
    pq_pos = {str(idx): index for index, idx in enumerate(system.PQ.idx.v)}
    assets: list[ProtectedAsset] = []
    for asset_id, bus_id, pq_id in zip(study.asset_ids, study.bus_ids, study.pq_ids):
        base_v = float(system.Bus.v.v[bus_pos[bus_id]])
        q_abs = abs(float(system.PQ.q0.v[pq_pos[pq_id]])) * float(case.base_mva or 100.0)
        assets.append(
            ProtectedAsset(
                asset_id=asset_id,
                name=f"{study.case_id} load bus {bus_id} tight overvoltage relay",
                protected_voltage=ProtectedVoltageSpec(
                    output_id=f"z_{asset_id}",
                    side=MeasurementSide.TRANSMISSION_BUS,
                    kind=ProtectedVoltageKind.DIRECT_BUS_VOLTAGE,
                    protected_bus_id=bus_id,
                ),
                threshold_pu=Interval.exact(base_v + study.margin_pu, Unit.PU),
                timing=ProtectionTiming(
                    DelayKind.DWELL,
                    dwell_time_s=Interval.exact(study.dwell_s, Unit.SECOND),
                ),
                disconnected_assets=(
                    DisconnectedAsset(
                        AssetRef(pq_id, model_family="PQ", bus_id=bus_id),
                        q_absorption_mvar=Interval.exact(q_abs, Unit.MVAR),
                    ),
                ),
                q_absorption_mvar=Interval.exact(q_abs, Unit.MVAR),
                tags=(study.case_id, "stress_test", "load_shedding"),
            )
        )
    return tuple(assets)


def _reduced_model(
    study: LoadSheddingStudy,
    case: AndesCase,
    protected: Sequence[ProtectedOutputEvaluation],
):
    system = case.require_system()
    jac = case.jacobians()
    output_addresses = [item.source_address.address for item in protected]
    cy = selector_matrix(output_addresses, jac.n_algebraic)
    cx = sparse.csc_matrix((len(output_addresses), jac.n_states))
    dy = sparse.lil_matrix((jac.n_algebraic, len(study.event_ids)))
    by = sparse.lil_matrix((jac.n_algebraic, len(study.control_ids)))
    pq_pos = {str(idx): index for index, idx in enumerate(system.PQ.idx.v)}
    for column, (bus_id, pq_id) in enumerate(zip(study.bus_ids, study.pq_ids)):
        pq_p_pu = float(system.PQ.p0.v[pq_pos[pq_id]])
        pq_q_pu = float(system.PQ.q0.v[pq_pos[pq_id]])
        dy[case.bus_angle_address(bus_id), column] = -pq_p_pu
        dy[case.bus_voltage_address(bus_id), column] = -pq_q_pu
        by[case.bus_voltage_address(bus_id), column] = +1.0
    model = LinearDAEModel.from_andes_jacobians(
        jac,
        cx=cx,
        cy=cy,
        bx=sparse.csc_matrix((jac.n_states, len(study.control_ids))),
        by=by.tocsc(),
        dx=sparse.csc_matrix((jac.n_states, len(study.event_ids))),
        dy=dy.tocsc(),
        mode_id=study.study_id,
    )
    return reduce_linear_dae(model)


def _run_tds_validation(
    study: LoadSheddingStudy,
    protected: Sequence[ProtectedOutputEvaluation],
) -> tuple[ProtectionReplayResult, list[dict], str]:
    case = _load_case(study, init_tds=True)
    system = case.require_system()
    system.TDS.config.tf = study.tds_tf_s
    system.TDS.config.tstep = study.tds_step_s
    system.TDS.config.no_tqdm = 1
    bus_pos = {str(bus): index for index, bus in enumerate(system.Bus.idx.v)}
    pq_pos = {str(idx): index for index, idx in enumerate(system.PQ.idx.v)}
    seed_pq_id = study.pq_ids[study.asset_ids.index(study.seed_asset_id)]
    relays = tuple(
        ProtectionRelaySpec(
            asset_id=item.asset_id,
            threshold_pu=item.threshold_pu.nominal,
            dwell_time_s=study.dwell_s,
            source_bus_id=item.source_bus_id,
            trip_targets=(RelayTripTarget("PQ", pq_id, 0.0),),
            source=f"{study.study_id}_tds",
        )
        for item, pq_id in zip(protected, study.pq_ids)
    )
    callback = TDSProtectionCallback(relays)
    trace: list[dict] = []

    class SeedAndProtection:
        seed_done = False
        seed_time = None

        def __call__(self, t, ss):
            time_s = float(t)
            if not self.seed_done and time_s >= study.seed_time_s - 1.0e-12:
                ss.set_status("PQ", seed_pq_id, 0)
                self.seed_done = True
                self.seed_time = time_s
            callback(time_s, ss)
            row = {"time_s": time_s}
            for asset_id, bus_id in zip(study.asset_ids, study.bus_ids):
                row[f"v_{asset_id}_pu"] = float(ss.Bus.v.v[bus_pos[bus_id]])
            for asset_id, pq_id in zip(study.asset_ids, study.pq_ids):
                row[f"{asset_id}_pq_status"] = float(ss.PQ.u.v[pq_pos[pq_id]])
            trace.append(row)

    seed = SeedAndProtection()
    system.TDS.callpert = seed
    ok = bool(system.TDS.run(no_summary=True))
    notes = f"tds_ok={ok}; busted={system.TDS.busted}; seed_time={seed.seed_time}"
    return callback.replay_result(), trace, notes


def _study_config(
    study: LoadSheddingStudy,
    case: AndesCase,
    protected: Sequence[ProtectedOutputEvaluation],
) -> BenchmarkStudyConfig:
    system = case.require_system()
    bus_pos = {str(bus): index for index, bus in enumerate(system.Bus.idx.v)}
    q_abs_by_asset = {
        item.asset_id: float(item.q_absorption_mvar.nominal)
        for item in _protected_assets(study, case)
    }
    collectors = tuple(
        ReplicaCollectorConfig(
            asset_id=asset_id,
            transmission_bus=int(bus_id),
            collector_bus=int(bus_id),
            tap_ratio=1.0,
            base_kv=float(system.Bus.Vn.v[bus_pos[bus_id]]),
            threshold_pu=item.threshold_pu.nominal,
            dwell_time_s=study.dwell_s,
            q_absorption_mvar=q_abs_by_asset[asset_id],
            source_name=pq_id,
            shunt_name="none",
        )
        for asset_id, bus_id, pq_id, item in zip(
            study.asset_ids,
            study.bus_ids,
            study.pq_ids,
            protected,
        )
    )
    return BenchmarkStudyConfig(
        study_id=study.study_id,
        target=BenchmarkTarget(
            case_id=study.case_id,
            case_alias=study.case_alias,
            description=study.description,
            category="load_shedding_stress_test",
            setup=True,
            run_pflow=True,
            init_tds=True,
        ),
        collectors=collectors,
        scenario_parameters={
            "threshold_margin_pu": study.margin_pu,
            "dwell_s": study.dwell_s,
            "seed_time_s": study.seed_time_s,
            "tds_tf_s": study.tds_tf_s,
            "tds_step_s": study.tds_step_s,
        },
        feature_flags={
            "protection": True,
            "tds_validation": True,
            "mitigation_lp": True,
            "official_iberian_replica": False,
        },
        seed_ids=(study.seed_event_id,),
        notes="Stress test for workflow verification; thresholds are intentionally tight.",
    )


def _write_trace(
    result_dir: Path,
    study: LoadSheddingStudy,
    rows: Sequence[dict],
) -> Path:
    path = result_dir / "tds_trace.csv"
    fieldnames = (
        ["time_s"]
        + [f"v_{asset_id}_pu" for asset_id in study.asset_ids]
        + [f"{asset_id}_pq_status" for asset_id in study.asset_ids]
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_summary(result_dir: Path, study: LoadSheddingStudy, payload: dict) -> None:
    cascade = payload["cascade"]["seed_results"][0]
    validation = payload.get("nonlinear_validation")
    mitigation = payload["mitigation"]["mitigation_result"]
    lines = [
        f"# {study.case_id} Load-Shedding Case Study",
        "",
        "This is a workflow stress test on a bundled ANDES benchmark, not an official Iberian replica.",
        "",
        "## Main Results",
        "",
        f"- Protected assets: {', '.join(payload['finite_window']['window_result']['protected_asset_ids'])}",
        f"- Seed: `{study.seed_event_id}` at {study.seed_time_s:.2f} s in nonlinear TDS.",
        f"- Cascade fixed point: {cascade['fixed_point']}",
    ]
    if validation is None:
        lines.extend(
            [
                "- Nonlinear trips: not run.",
                "- Validation status: not_evaluated",
            ]
        )
    else:
        result = validation["validation_result"]
        lines.extend(
            [
                f"- Nonlinear trips: {[item['asset_id'] for item in result['simulated_trips']]}",
                f"- False positives: {result['false_positive_trips']}",
                f"- False negatives: {result['false_negative_trips']}",
                f"- Validation status: {result['status']}",
            ]
        )
    lines.extend(
        [
            f"- Mitigation feasible: {mitigation['feasible']}",
            f"- Required fast reactive absorption: {mitigation['required_mvar_total']:.3f} MVAr",
            "",
            "## Important Caveats",
            "",
            f"- Relay thresholds are intentionally tight: base voltage plus {study.margin_pu:.4f} pu.",
            "- Disturbance channels are full PQ load-shedding channels.",
            "- Control channels are idealized fast reactive absorption commands at the monitored buses.",
            "- The scenario is intended to test the method implementation, not to calibrate real protection settings.",
            "",
        ]
    )
    (result_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
