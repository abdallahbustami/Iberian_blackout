from __future__ import annotations

import json
from pathlib import Path

from pa_dvsa.benchmark_studies import (
    ScalabilityMetrics,
    build_causal_ablation_suite,
    build_ieee39_official_report_replica_config,
    build_prediction_study,
)
from pa_dvsa.data_model import (
    AssessmentStatus,
    ControlSelection,
    MitigationResult,
    WindowMapResult,
)
from pa_dvsa.nonlinear_validation import ProtectionRelaySpec, ProtectionTrace, replay_protection_traces
from pa_dvsa.paper_outputs import (
    PaperOutputPaths,
    build_table_specs,
    generate_paper_outputs,
    write_csv_artifact,
    write_json_artifact,
)
from pa_dvsa.robust_screening import compute_robust_screen


def test_generate_paper_outputs_writes_artifacts_tables_and_figures(tmp_path: Path):
    config = build_ieee39_official_report_replica_config()
    window = _window()
    replay = replay_protection_traces(
        [
            ProtectionTrace(
                ProtectionRelaySpec("B", 1.1, dwell_time_s=0.1, source_bus_id="2"),
                (0.0, 0.1, 0.2),
                (1.0, 1.12, 1.13),
            )
        ]
    )
    prediction = build_prediction_study(window, study_id="study", seed_ids=("A",), nonlinear=replay)
    robust = compute_robust_screen(window, margins_pu=(1.0, 1.0), seed_assets=("A",))
    mitigation = MitigationResult(
        mode_id="mode",
        seed_ids=("A",),
        feasible=True,
        objective_value=2.0,
        slack_eta=0.0,
        selections=(ControlSelection("STATCOM_1", alpha=0.2, magnitude_mvar=20.0),),
        status=AssessmentStatus.CERTIFIED,
    )
    metrics = (
        ScalabilityMetrics(
            study_id="study",
            case_id="ieee39",
            runtime_s=1.2,
            sparse_solve_count=3,
            selected_channels=4,
            protected_asset_count=2,
            event_count=2,
            control_count=1,
            memory_peak_mb=12.5,
            status=AssessmentStatus.CERTIFIED,
        ),
    )
    paths = PaperOutputPaths.from_project_root(tmp_path)

    manifest = generate_paper_outputs(
        output_id="paper",
        config=config,
        prediction=prediction,
        ablations=build_causal_ablation_suite(config)[:2],
        robust=robust,
        mitigation=mitigation,
        scalability=metrics,
        paths=paths,
        generate_figures=True,
    )

    artifact_ids = {item.artifact_id for item in manifest.artifacts}
    assert "paper_manifest" in artifact_ids
    assert "prediction_overlay" in artifact_ids
    assert "table_case_calibration_tex" in artifact_ids
    assert "figure_study_k_pickup" in artifact_ids
    assert (tmp_path / "LaTeX/figures/generated/study_k_pickup.pdf").exists()
    assert (tmp_path / "LaTeX/tables/generated/case_calibration.tex").exists()
    manifest_json = json.loads((tmp_path / "results/paper_outputs/paper_manifest.json").read_text())
    assert manifest_json["root_tex_updated"] is False


def test_json_and_csv_artifacts_are_stable(tmp_path: Path):
    paths = PaperOutputPaths.from_project_root(tmp_path)
    payload = {"b": [2, 1], "a": {"x": 1.0}}
    rows = [{"b": 2, "a": 1}, {"a": 3, "b": 4}]

    first_json = write_json_artifact(paths, "stable", payload)
    first_csv = write_csv_artifact(paths, "stable_rows", rows)
    first_json_text = Path(first_json.path).read_text()
    first_csv_text = Path(first_csv.path).read_text()

    second_json = write_json_artifact(paths, "stable", payload)
    second_csv = write_csv_artifact(paths, "stable_rows", rows)

    assert Path(second_json.path).read_text() == first_json_text
    assert Path(second_csv.path).read_text() == first_csv_text
    assert second_json.sha256 == first_json.sha256
    assert second_csv.sha256 == first_csv.sha256


def test_table_specs_include_all_required_paper_tables():
    specs = build_table_specs()
    ids = {item.table_id for item in specs}

    assert ids == {
        "case_calibration",
        "screen_vs_tds_validation",
        "ablations",
        "uncertainty",
        "mitigation",
        "large_system_runtime",
    }


def _window() -> WindowMapResult:
    return WindowMapResult(
        mode_id="mode",
        protected_asset_ids=("A", "B"),
        event_ids=("A", "B"),
        control_ids=(),
        horizons_s=(1.0, 1.0),
        k_pickup_upper=((0.0, 0.0), (1.2, 0.0)),
        k_trip_upper=((0.0, 0.0), (1.1, 0.0)),
        status=AssessmentStatus.CERTIFIED,
    )
