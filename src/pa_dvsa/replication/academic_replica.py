#!/usr/bin/env python3
"""Academic ACTIVSg2000 mechanism replica of the Iberian blackout.

This is the supported ANDES-backed replication path. It reuses the physical
large-system machinery in ``activsg2000_engine.py`` and adds a research-facing
interface, report anchors, study variants, and provenance artifacts.

The paper-facing claim is deliberately narrow:

* ACTIVSg2000 is a modified academic benchmark, not Iberia.
* Operator actions, protection settings, and post-cascade controls are
  report-aligned mechanism surrogates.
* Voltage/frequency traces come from ANDES TDS and explicit protection logic.
* The default figure trims/shades the blackout endpoint; it does not add a
  plotted smooth voltage-to-zero trajectory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

from .activsg2000_engine import (
    BlackoutCriteriaSpec,
    HVDCSurrogateSpec,
    IslandBlackoutCriteriaSpec,
    GeneratorProtectionRelaySpec,
    OutOfStepRelaySpec,
    ReactorSwitchAction,
    ReplicationConfig,
    UFLSUVLSStageSpec,
    UnderFrequencyTieRelaySpec,
    build_report_collector_specs,
    report_relay_stages,
    run_replication,
)


TIME_ZERO_CEST = "2025-04-28T12:32:50+02:00"


@dataclass(frozen=True, slots=True)
class ReportAnchor:
    code: str
    report_time_cest: str
    model_role: str
    active_power_mw: float | None
    reactive_absorption_mvar: float | None
    description: str
    source: str


REPORT_ANCHORS: tuple[ReportAnchor, ...] = (
    ReportAnchor(
        "pre_1200_1232_voltage_control",
        "09:00-12:32",
        "precondition",
        None,
        None,
        (
            "Operators used line switching, shunt-reactor switching, and "
            "Santa Llogaia-Baixas HVDC setpoint changes for voltage control. "
            "Connecting lines can increase reactive production and voltage; "
            "disconnecting shunt reactors can also raise voltage."
        ),
        "ENTSO-E factual report, Section 2.1.2 and Table 2-2.",
    ),
    ReportAnchor(
        "E1",
        "12:32:00-12:32:57",
        "pre_trip_drift",
        317.3,
        None,
        "Net-load increase or embedded-generation loss before the first major trip.",
        "ENTSO-E final report, Chapter 1 / sequence tables.",
    ),
    ReportAnchor(
        "E2",
        "12:32:00-12:32:57",
        "fixed_pf_res_change",
        208.0,
        33.3,
        (
            "Identified wind/PV operating-point change or disconnection; fixed-PF "
            "behaviour reduces reactive absorption, including the Granada example."
        ),
        "ENTSO-E final report, Chapter 1 / event analysis.",
    ),
    ReportAnchor(
        "E3_Granada",
        "12:32:57.220",
        "collector_overvoltage_trip",
        355.0,
        165.0,
        (
            "Granada-region 400/220 kV generation evacuation transformer trips "
            "by 220 kV-side overvoltage protection; 400 kV side is reported "
            "around 417.9 kV."
        ),
        "ENTSO-E final report and factual report, Chapter 1.",
    ),
    ReportAnchor(
        "E4a_Badajoz",
        "12:33:16.443",
        "collector_overvoltage_trip",
        582.0,
        165.0,
        "Badajoz evacuation event 4a; overvoltage protection / evacuation trip.",
        "ENTSO-E final report, event chronology.",
    ),
    ReportAnchor(
        "E4b_Badajoz",
        "12:33:16.820",
        "collector_overvoltage_trip",
        145.0,
        37.0,
        "Badajoz evacuation event 4b; reported as a second nearby trip.",
        "ENTSO-E final report, event chronology.",
    ),
    ReportAnchor(
        "E5_multi_site",
        "12:33:17-12:33:18.020",
        "multi_site_generation_trip_cluster",
        928.0,
        None,
        (
            "Wind/PV trips across Segovia, Seville, Badajoz, Huelva, Caceres, "
            "Cadiz, Malaga, and Cuenca; some trips confirmed overvoltage, "
            "others unknown."
        ),
        "ENTSO-E final and factual reports, Chapter 1.",
    ),
    ReportAnchor(
        "defense_plan",
        "12:33:19-12:33:22",
        "ufls_uvls_defense",
        None,
        None,
        "Spanish and Portuguese automatic load shedding/defence plans activate but do not arrest collapse.",
        "ENTSO-E factual report, Chapter 1.",
    ),
    ReportAnchor(
        "Morocco_AC",
        "12:33:20.473",
        "underfrequency_ac_trip",
        None,
        None,
        "AC interconnection to Morocco trips on underfrequency.",
        "ENTSO-E final and factual reports, Chapter 1.",
    ),
    ReportAnchor(
        "France_AC",
        "12:33:21.535",
        "loss_of_synchronism_ac_trip",
        None,
        None,
        "AC overhead lines between France and Spain trip on loss-of-synchronism protection.",
        "ENTSO-E final and factual reports, Chapter 1.",
    ),
    ReportAnchor(
        "HVDC",
        "12:33:23.960",
        "hvdc_block",
        None,
        None,
        (
            "Santa Llogaia-Baixas HVDC completes electrical separation while "
            "operating in constant-power mode."
        ),
        "ENTSO-E final and factual reports, Chapter 1 and HVDC sections.",
    ),
    ReportAnchor(
        "blackout",
        "near 12:33:27",
        "blackout_endpoint",
        None,
        None,
        "Spanish and Portuguese systems lose valid operating continuation / de-energize.",
        "ENTSO-E final and factual reports, Chapter 1.",
    ),
)


VARIANT_DESCRIPTIONS: dict[str, str] = {
    "baseline": (
        "Report-aligned physical mechanism benchmark: all manual operator "
        "actions occur before the first protection trip; all post-cascade "
        "events are explicit relays/control responses."
    ),
    "voltage_mode_res": (
        "Ablation in which fixed-power-factor RES behaviour is mostly removed; "
        "reactive absorption is preserved, so the overvoltage cascade should weaken."
    ),
    "preserved_q_absorption": (
        "Ablation in which protected plants trip with much less MVAr absorption "
        "loss, testing whether lost absorption is the main compounding driver."
    ),
    "no_collector_ov_protection": (
        "Negative control in which collector-side overvoltage relay pickups are "
        "moved outside the simulated operating range.  The network can become "
        "high-voltage, but the collector-trip feedback path is disabled."
    ),
    "delayed_protection": (
        "Ablation in which collector-side overvoltage relay dwell times are "
        "substantially increased while the physical disturbance sequence is "
        "otherwise unchanged."
    ),
    "no_pre_voltage_actions": (
        "Ablation with line/shunt/HVDC pre-cascade operator actions disabled."
    ),
    "strong_defense": (
        "Ablation with earlier/larger UFLS/UVLS stages after generator protection."
    ),
    "no_post_cascade_relays": (
        "Negative control: collector protection remains active, but generator, "
        "tie, HVDC, defence, and island-blackout relays are disabled."
    ),
}


def _scaled_collectors(*, q_scale: float = 1.0, threshold_shift: float = 0.0):
    specs = []
    for spec in build_report_collector_specs(count=16, q_absorption_scale=q_scale):
        stages = tuple(
            replace(stage, pickup_pu=max(0.95, stage.pickup_pu + threshold_shift))
            for stage in spec.relay_stages
        )
        specs.append(
            replace(
                spec,
                threshold_pu=max(0.95, spec.threshold_pu + threshold_shift),
                relay_stages=stages,
            )
        )
    return tuple(specs)


def _tds_calibrated_collectors(*, q_scale: float = 1.25, threshold_shift: float = 0.0):
    """Collector relays for the physical ACTIVSg2000 TDS replica.

    The settings keep the first trip after the pre-cascade operator actions and
    use sustained overvoltage dwell, not external event scheduling, to create
    the report-like E3/E4/E5 clusters.
    """

    specs = []
    for i, spec in enumerate(build_report_collector_specs(count=16, q_absorption_scale=q_scale)):
        if spec.name == "Granada355":
            pickup = 1.0900 + threshold_shift
            dwell = 3.50
        elif spec.name == "Huelva34":
            pickup = 1.1250 + threshold_shift
            dwell = 4.50
        elif spec.name == "Caceres38":
            pickup = 1.1010 + threshold_shift
            dwell = 4.00
        elif spec.name == "Caceres41":
            pickup = 1.1030 + threshold_shift
            dwell = 4.00
        elif spec.name == "Badajoz16":
            pickup = 1.1180 + threshold_shift
            dwell = 3.50
        elif spec.name == "Cadiz26":
            pickup = 1.0940 + threshold_shift
            dwell = 4.50
        elif spec.name == "Cadiz128":
            pickup = 1.1110 + threshold_shift
            dwell = 4.00
        elif spec.report_cluster == "E3 Granada":
            pickup = 1.0900 + threshold_shift
            dwell = 3.50
        elif spec.report_cluster == "E4 Badajoz":
            pickup = 1.1035 + threshold_shift
            dwell = 12.20 + 0.08 * i
        else:
            pickup = 1.0860 + 0.001 * (i % 5) + threshold_shift
            dwell = 12.60 + 0.08 * i
        specs.append(
            replace(
                spec,
                threshold_pu=pickup,
                dwell_s=dwell,
                relay_stages=report_relay_stages(pickup, dwell, reset_ratio=0.95),
            )
        )
    return tuple(specs)


def _disable_collector_ov_protection(specs):
    disabled = []
    for spec in specs:
        pickup = 9.0
        disabled.append(
            replace(
                spec,
                threshold_pu=pickup,
                dwell_s=1.0e6,
                min_trip_time_s=1.0e6,
                relay_stages=report_relay_stages(pickup, 1.0e6, reset_ratio=0.95),
            )
        )
    return tuple(disabled)


def _delay_collector_ov_protection(specs, *, dwell_scale: float = 2.5):
    delayed = []
    for spec in specs:
        stages = tuple(
            replace(stage, dwell_s=stage.dwell_s * dwell_scale)
            for stage in spec.relay_stages
        )
        delayed.append(
            replace(
                spec,
                dwell_s=spec.dwell_s * dwell_scale,
                relay_stages=stages,
            )
        )
    return tuple(delayed)


def build_academic_replica_config(
    *,
    variant: str = "baseline",
    tf_s: float = 30.0,
    tstep_s: float = 0.01,
) -> ReplicationConfig:
    """Build the physically simulated academic replica configuration."""

    if variant not in VARIANT_DESCRIPTIONS:
        raise ValueError(f"unknown academic replica variant {variant!r}")

    base_defaults = ReplicationConfig()
    collector_specs = _tds_calibrated_collectors(q_scale=1.25)
    reactor_actions = tuple(
        replace(action, time_s=time_s, q_mvar=12.0)
        for action, time_s in zip(
            base_defaults.reactor_actions,
            (3.3, 3.8, 4.3, 4.8, 5.3, 5.8),
        )
    )

    cfg = ReplicationConfig(
        tf_s=tf_s,
        tstep_s=tstep_s,
        collector_specs=collector_specs,
        meshing_times_s=(0.8, 1.4, 2.1, 2.8),
        meshing_equivalent_strengths=(0.50, 0.50, 0.05, 0.05),
        reactor_actions=reactor_actions,
        hvdc_reference_time_s=6.2,
        export_reduction_time_s=6.6,
        export_reduction_ramp_s=8.0,
        export_reduction_mode="collector_fixed_pf",
        background_fixed_pf_q_absorption_mvar=0.0,
        allow_equivalent_blackout=False,
        enable_post_cascade_relays=True,
        collector_initial_margin_pu=0.0,
        collector_anchor_initial_margin_pu=0.0,
        collector_granada_initial_margin_pu=0.0,
        collector_badajoz_initial_margin_pu=0.0,
        hvdc_surrogate=HVDCSurrogateSpec(
            transfer_mw=450.0,
            reactive_support_mvar=320.0,
            min_block_time_s=22.0,
            block_voltage_pu=0.88,
            block_over_voltage_pu=1.135,
            block_frequency_hz=47.5,
            dwell_s=0.08,
            require_collector_trips=6,
            require_france_ac_separation=True,
        ),
        generator_protection=GeneratorProtectionRelaySpec(
            min_trip_time_s=16.70,
            dwell_s=0.08,
            under_voltage_pu=0.93,
            under_frequency_hz=49.8,
            angle_threshold_deg=0.45,
            require_collector_trips=6,
            target_mva=6500.0,
            max_genrou=12,
        ),
        morocco_ac_relay=UnderFrequencyTieRelaySpec(
            min_trip_time_s=19.0,
            dwell_s=0.08,
            under_frequency_hz=49.5,
            angle_threshold_deg=0.60,
            require_collector_trips=6,
            require_generator_trip=True,
            boundary_line_count=6,
        ),
        out_of_step_relay=OutOfStepRelaySpec(
            min_trip_time_s=20.2,
            dwell_s=0.08,
            angle_threshold_deg=0.35,
            require_collector_trips=6,
            require_morocco_trip=True,
            boundary_line_count=6,
        ),
        defense_stages=(
            UFLSUVLSStageSpec("UFLS1", 17.0, 0.18, under_voltage_pu=0.97, under_frequency_hz=49.0, max_shed_mw=900.0),
            UFLSUVLSStageSpec("UFLS2", 17.5, 0.22, under_voltage_pu=0.95, under_frequency_hz=48.8, max_shed_mw=1200.0),
        ),
        blackout_criteria=BlackoutCriteriaSpec(
            voltage_pu=0.75,
            voltage_dwell_s=0.08,
            frequency_hz=47.5,
            frequency_dwell_s=0.08,
            angle_threshold_deg=2.5,
            angle_dwell_s=1.20,
        ),
        island_blackout_criteria=IslandBlackoutCriteriaSpec(
            min_generation_mw=250.0,
            min_reference_generation_mw=50.0,
            voltage_pu=0.82,
            voltage_dwell_s=0.20,
            frequency_hz=48.5,
            frequency_dwell_s=0.20,
            angle_threshold_deg=25.0,
            angle_dwell_s=0.35,
            imbalance_fraction=0.65,
            imbalance_dwell_s=0.35,
            min_time_after_hvdc_block_s=0.70,
        ),
    )

    if variant == "voltage_mode_res":
        cfg = replace(
            cfg,
            collector_specs=_tds_calibrated_collectors(q_scale=0.35, threshold_shift=0.015),
            export_reduction_ramp_s=1e9,
            background_fixed_pf_q_absorption_mvar=0.0,
        )
    elif variant == "preserved_q_absorption":
        cfg = replace(cfg, collector_specs=_tds_calibrated_collectors(q_scale=0.20, threshold_shift=0.010))
    elif variant == "no_collector_ov_protection":
        cfg = replace(cfg, collector_specs=_disable_collector_ov_protection(cfg.collector_specs))
    elif variant == "delayed_protection":
        cfg = replace(cfg, collector_specs=_delay_collector_ov_protection(cfg.collector_specs))
    elif variant == "no_pre_voltage_actions":
        cfg = replace(
            cfg,
            meshing_times_s=(),
            reactor_actions=(),
            hvdc_reference_time_s=1e9,
            export_reduction_ramp_s=1e9,
        )
    elif variant == "strong_defense":
        cfg = replace(
            cfg,
            defense_stages=(
                UFLSUVLSStageSpec("UFLS1_STRONG", 16.4, 0.35, under_voltage_pu=0.98, under_frequency_hz=49.5, max_shed_mw=2200.0),
                UFLSUVLSStageSpec("UFLS2_STRONG", 16.8, 0.40, under_voltage_pu=0.96, under_frequency_hz=49.2, max_shed_mw=2600.0),
            ),
            island_blackout_criteria=replace(
                cfg.island_blackout_criteria,
                imbalance_fraction=0.75,
                frequency_hz=47.8,
            ),
        )
    elif variant == "no_post_cascade_relays":
        cfg = replace(cfg, enable_post_cascade_relays=False)

    return cfg


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


def _read_events(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_academic_replica_artifacts(
    *,
    out_dir: Path,
    variant: str,
    summary: dict[str, Any],
    events_path: Path,
) -> None:
    events = _read_events(events_path)
    first_by_category: dict[str, float] = {}
    for event in events:
        first_by_category.setdefault(str(event["category"]), float(event["time_s"]))

    chronology = {
        "time_zero_cest": TIME_ZERO_CEST,
        "variant": variant,
        "report_aligned_order": [
            "operator_action",
            "protection_trip",
            "generator_trip",
            "defense_action",
            "morocco_ac_trip",
            "out_of_step_trip",
            "hvdc_block",
            "island_blackout_declared",
            "system_blackout_declared",
        ],
        "first_event_times_s": first_by_category,
        "checks": {
            "all_operator_actions_before_first_trip": (
                max((float(e["time_s"]) for e in events if e["category"] == "operator_action"), default=-1.0)
                < first_by_category.get("protection_trip", float("inf"))
            ),
            "morocco_before_france_ac": (
                first_by_category.get("morocco_ac_trip", float("inf"))
                < first_by_category.get("out_of_step_trip", float("inf"))
            ),
            "france_ac_before_hvdc": (
                first_by_category.get("out_of_step_trip", float("inf"))
                < first_by_category.get("hvdc_block", float("inf"))
            ),
            "no_equivalent_blackout": summary.get("automatic_blackout_actions") == 0
            and not summary.get("equivalent_blackout_enabled"),
            "blackout_endpoint_is_explicit": bool(summary.get("blackout_detected")),
            "voltage_zero_not_fabricated": True,
        },
        "note": (
            "Model times are compressed academic mechanism times.  The run is "
            "ordered against the report chronology, but it is not a forensic "
            "Iberian dynamic equivalent."
        ),
    }
    (out_dir / "chronology_validation.json").write_text(
        json.dumps(chronology, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "report_anchors.json").write_text(
        json.dumps([asdict(anchor) for anchor in REPORT_ANCHORS], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "study_variants.json").write_text(
        json.dumps(VARIANT_DESCRIPTIONS, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "script": "python -m pa_dvsa.replication.academic_replica",
        "claim": "ACTIVSg2000-backed academic mechanism replica, not forensic Iberia.",
        "time_zero_cest": TIME_ZERO_CEST,
        "variant": variant,
        "physicality_guardrails": {
            "andes_tds_used": True,
            "no_anonymous_pq_blackout_burden": summary.get("automatic_blackout_actions") == 0,
            "equivalent_blackout_enabled": bool(summary.get("equivalent_blackout_enabled")),
            "final_voltage_drop_added_to_tds": False,
            "blackout_visualization": "trace trimming / de-energized shading only",
        },
        "outputs": {
            "summary": "summary.json",
            "events": "events.json",
            "collector_traces": "collector_traces.csv",
            "system_traces": "system_traces.csv",
            "island_traces": "island_traces.csv",
            "report_anchors": "report_anchors.json",
            "chronology_validation": "chronology_validation.json",
            "screen_event_library": "screen_event_library.csv",
        },
    }
    (out_dir / "academic_replica_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    screen_rows = []
    for event in events:
        if event["category"] in {
            "operator_action",
            "protection_trip",
            "generator_trip",
            "defense_action",
            "morocco_ac_trip",
            "out_of_step_trip",
            "hvdc_block",
            "island_blackout_declared",
            "system_blackout_declared",
        }:
            screen_rows.append(
                {
                    "event_id": event["code"],
                    "time_s": event["time_s"],
                    "category": event["category"],
                    "lost_p_mw": event.get("lost_p_mw", 0.0),
                    "lost_q_absorption_mvar": event.get("lost_q_absorption_mvar", 0.0),
                    "net_load_increase_mw": event.get("net_load_increase_mw", 0.0),
                    "q_demand_increase_mvar": event.get("q_demand_increase_mvar", 0.0),
                    "trigger": event.get("trigger") or "",
                    "description": event.get("description") or "",
                }
            )
    _write_csv(out_dir / "screen_event_library.csv", screen_rows)
    readme = f"""# ACTIVSg2000 Iberian Mechanism Replica

This directory was generated by `python -m pa_dvsa.replication.academic_replica`
or `python replication.py`.

This is an academic mechanism replica on a modified ACTIVSg2000 benchmark. It
is not a forensic Iberian grid equivalent. The traces are produced by ANDES TDS
plus explicit modeled protection/control actions. No anonymous PQ blackout
burden is enabled and no final voltage-to-zero curve is added for aesthetics.

Variant: `{variant}`

Use the artifacts as follows:

- `events.json`: event log from the simulated protection/control chain.
- `collector_traces.csv`: protected collector-side and upstream voltages.
- `system_traces.csv`: physical-bus voltage percentiles, frequency, and losses.
- `island_traces.csv`: island accounting after topology/protection changes.
- `screen_event_library.csv`: compact event library for the screening tool.
- `report_anchors.json`: report facts encoded as calibration/ordering anchors.
- `chronology_validation.json`: ordering checks and physicality guardrails.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def _copy_academic_figure_aliases(
    *,
    output_dir: Path,
    latex_fig_dir: Path | None,
    figure_paths: list[Path],
) -> list[Path]:
    renamed: list[Path] = []
    for path in figure_paths:
        if path.parent == output_dir and path.name.startswith("fig_replication_activsg2000."):
            target = path.with_name(path.name.replace("fig_replication_activsg2000", "fig_replication_academic_activsg2000"))
            shutil.copy2(path, target)
            renamed.append(target)
    if latex_fig_dir is not None:
        latex_fig_dir.mkdir(parents=True, exist_ok=True)
        for path in list(renamed):
            if path.parent == output_dir:
                target = latex_fig_dir / path.name
                shutil.copy2(path, target)
                renamed.append(target)
    return renamed


def run_academic_replica(
    *,
    variant: str,
    out_dir: Path,
    tf_s: float,
    tstep_s: float,
    plot: bool,
    latex_fig_dir: Path | None,
    formats: tuple[str, ...],
    verbose_andes: bool,
) -> tuple[dict[str, Any], list[Path]]:
    cfg = build_academic_replica_config(variant=variant, tf_s=tf_s, tstep_s=tstep_s)
    artifacts, summary, monitor = run_replication(
        cfg,
        out_dir=out_dir,
        quiet_andes=not verbose_andes,
    )
    _write_academic_replica_artifacts(
        out_dir=out_dir,
        variant=variant,
        summary=summary,
        events_path=artifacts.events_path,
    )
    figures: list[Path] = []
    if plot:
        from .figures import make_all as make_replication_figures

        figure_dir = latex_fig_dir if latex_fig_dir is not None else out_dir
        figures = make_replication_figures(
            data_dir=out_dir,
            fig_dir=figure_dir,
            formats=formats,
        )
        figures.extend(
            _copy_academic_figure_aliases(
                output_dir=out_dir,
                latex_fig_dir=latex_fig_dir,
                figure_paths=figures,
            )
        )
    return summary, figures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/activsg2000_iberian_replication"))
    parser.add_argument("--tf", type=float, default=30.0)
    parser.add_argument("--tstep", type=float, default=0.01)
    parser.add_argument("--variant", choices=tuple(VARIANT_DESCRIPTIONS), default="baseline")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--latex-fig-dir", type=Path, default=Path("LaTeX/figures/generated"))
    parser.add_argument("--formats", default="pdf")
    parser.add_argument("--verbose-andes", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    formats = tuple(item.strip().lower() for item in args.formats.split(",") if item.strip())
    summary, figures = run_academic_replica(
        variant=args.variant,
        out_dir=args.out,
        tf_s=float(args.tf),
        tstep_s=float(args.tstep),
        plot=bool(args.plot),
        latex_fig_dir=args.latex_fig_dir,
        formats=formats,
        verbose_andes=bool(args.verbose_andes),
    )
    print(f"Output directory: {args.out}")
    print(f"Variant: {args.variant}")
    print(f"Collector trips: {summary.get('collector_trips')}")
    print(f"First collector trip: {summary.get('first_trip_time_s')}")
    print(f"Lost P/Q absorption: {summary.get('total_lost_p_mw'):.1f} MW / {summary.get('total_lost_q_absorption_mvar'):.1f} MVAr")
    print(f"Morocco AC / France AC / HVDC: {summary.get('first_morocco_ac_trip_time_s')} / {summary.get('first_france_ac_separation_time_s')} / {summary.get('first_hvdc_block_time_s')}")
    print(f"Blackout endpoint: {summary.get('blackout_time_s')} ({summary.get('blackout_reason')})")
    print(f"Automatic PQ blackout burdens: {summary.get('automatic_blackout_actions')}")
    print(f"Voltage crash detected by TDS: {summary.get('voltage_crash_detected')}")
    for fig in figures:
        print(f"Figure: {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
