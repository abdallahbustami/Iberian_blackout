from __future__ import annotations

import json
from pathlib import Path

import pytest

from pa_dvsa.replication.academic_replica import REPORT_ANCHORS, build_academic_replica_config


def test_academic_replica_baseline_config_has_no_equivalent_blackout_or_plotted_drop():
    cfg = build_academic_replica_config()

    assert cfg.allow_equivalent_blackout is False
    assert cfg.blackout_actions == ()
    assert cfg.enable_post_cascade_relays is True
    assert cfg.hvdc_surrogate.require_france_ac_separation is True
    assert cfg.hvdc_surrogate.require_collector_trips >= 6
    assert cfg.export_reduction_mode == "collector_fixed_pf"
    assert len(cfg.collector_specs) == 16


def test_academic_replica_variants_change_only_named_mechanisms():
    baseline = build_academic_replica_config()
    voltage_mode = build_academic_replica_config(variant="voltage_mode_res")
    no_post = build_academic_replica_config(variant="no_post_cascade_relays")
    no_pre = build_academic_replica_config(variant="no_pre_voltage_actions")

    assert voltage_mode.export_reduction_ramp_s > baseline.tf_s
    assert sum(item.q_absorption_mvar for item in voltage_mode.collector_specs) < sum(
        item.q_absorption_mvar for item in baseline.collector_specs
    )
    assert no_post.enable_post_cascade_relays is False
    assert no_pre.meshing_times_s == ()
    assert no_pre.reactor_actions == ()


def test_report_anchors_include_required_iberian_events():
    codes = {anchor.code for anchor in REPORT_ANCHORS}

    assert {
        "E3_Granada",
        "E4a_Badajoz",
        "E4b_Badajoz",
        "E5_multi_site",
        "Morocco_AC",
        "France_AC",
        "HVDC",
        "blackout",
    } <= codes


def test_generated_academic_replica_artifacts_have_defensible_chronology():
    out = Path("results/activsg2000_iberian_replication")
    summary_path = out / "summary.json"
    chronology_path = out / "chronology_validation.json"
    manifest_path = out / "academic_replica_manifest.json"
    screen_library_path = out / "screen_event_library.csv"
    if not summary_path.exists() or not chronology_path.exists():
        pytest.skip("academic replica artifacts have not been generated.")

    summary = json.loads(summary_path.read_text())
    chronology = json.loads(chronology_path.read_text())
    manifest = json.loads(manifest_path.read_text())

    assert summary["automatic_blackout_actions"] == 0
    assert summary["equivalent_blackout_enabled"] is False
    assert summary["collector_trips"] >= 6
    assert summary["first_trip_time_s"] > max(
        time
        for category, time in chronology["first_event_times_s"].items()
        if category == "operator_action"
    )
    assert chronology["checks"]["all_operator_actions_before_first_trip"] is True
    assert chronology["checks"]["morocco_before_france_ac"] is True
    assert chronology["checks"]["france_ac_before_hvdc"] is True
    assert chronology["checks"]["no_equivalent_blackout"] is True
    assert manifest["physicality_guardrails"]["final_voltage_drop_added_to_tds"] is False
    assert screen_library_path.exists() and screen_library_path.stat().st_size > 0
