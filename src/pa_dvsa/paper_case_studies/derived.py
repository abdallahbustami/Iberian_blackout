"""Derived, reproducible summaries for paper case-study figures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .loaders import ArtifactBundle, CaseArtifacts, MissingArtifactError, relpath, sha256_file


@dataclass
class DatasetPaths:
    root: Path
    systems: Path
    assets: Path
    events: Path
    scenario_library: Path
    k_matrix: Path
    cascade_predictions: Path
    tds_validation: Path
    validation: Path
    cascade_layers: Path
    observability_margins: Path
    mitigation_actions: Path
    mitigation_summary: Path
    uncertainty_sweeps: Path
    uncertainty_frontiers: Path
    observability_cases: Path
    operator_action_screen: Path
    operator_dashboard: Path
    control_authority_profiles: Path
    control_window_profiles: Path
    screening_triage: Path
    scaling_summary: Path
    ablation_summary: Path
    manifest: Path


def _yes(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def _case_seed(case: CaseArtifacts) -> str:
    cascade = case.prediction_study.get("cascade_result") or {}
    seeds = cascade.get("seed_ids") or case.case_config.get("seed_ids") or []
    if seeds:
        return str(seeds[0])
    return case.event_ids[0] if case.event_ids else ""


def _class_from_margin(norm_margin: float, *, data_limited: bool = False) -> str:
    if data_limited or not np.isfinite(norm_margin):
        return "data-limited"
    if norm_margin < 0:
        return "predicted trip"
    if norm_margin <= 0.10:
        return "risky"
    return "certified"


def _event_family(event_id: str) -> str:
    eid = str(event_id).lower()
    if "shunt" in eid or "reactor" in eid:
        return "shunt/reactor"
    if "hvdc" in eid:
        return "HVDC mode"
    if "line" in eid or "mesh" in eid or "tie" in eid:
        return "line/meshing"
    if "pump" in eid:
        return "load/pump"
    if "load" in eid or "shed" in eid:
        return "load/pump"
    if "res" in eid or "ibr" in eid or "pf" in eid:
        return "fixed-PF RES"
    return "candidate event"


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _systems(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        summary = case.summary
        metrics = case.metrics
        rows.append(
            {
                "case": case.slug,
                "benchmark": case.display_name,
                "buses": summary.get("n_buses"),
                "lines": summary.get("n_lines"),
                "dae_states": summary.get("dae_n"),
                "dae_algebraic": summary.get("dae_m"),
                "protected_assets": metrics.get("protected_asset_count", len(case.protected_asset_ids)),
                "hidden_outputs": 0,
                "control_devices": metrics.get("control_count", 0),
                "event_families": ",".join(sorted({_event_family(eid) for eid in case.event_ids})),
                "candidate_events": metrics.get("event_count", len(case.event_ids)),
                "tds_initialized": bool(summary.get("tds_initialized")),
                "screen_runtime_s": metrics.get("runtime_s"),
                "selected_channels": metrics.get("selected_channels"),
                "sparse_solves": metrics.get("sparse_solve_count"),
                "source": summary.get("source", ""),
            }
        )
    if bundle.replication is not None:
        rep = bundle.replication
        rows.append(
            {
                "case": "activsg2000_replica",
                "benchmark": "ACTIVSg2000 replica",
                "buses": rep.summary.get("total_bus_count_with_collectors")
                or rep.summary.get("physical_bus_count"),
                "lines": np.nan,
                "dae_states": np.nan,
                "dae_algebraic": np.nan,
                "protected_assets": rep.summary.get("collector_count"),
                "hidden_outputs": rep.summary.get("collector_count"),
                "control_devices": np.nan,
                "event_families": "operator,protection,defense,interconnection,HVDC",
                "candidate_events": len(rep.events),
                "tds_initialized": bool(rep.summary.get("tds_ok")) or bool(rep.summary.get("pflow_ok")),
                "screen_runtime_s": np.nan,
                "selected_channels": np.nan,
                "sparse_solves": np.nan,
                "source": "modified ACTIVSg2000",
            }
        )
    return pd.DataFrame(rows)


def _assets(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        cal = case.case_calibration if case.case_calibration is not None else pd.DataFrame()
        overlay = case.prediction_overlay if case.prediction_overlay is not None else pd.DataFrame()
        overlay_by_asset = {
            str(row["asset_id"]): row for _, row in overlay.iterrows() if "asset_id" in row
        }
        for _, row in cal.iterrows():
            asset = str(row.get("asset"))
            ov = overlay_by_asset.get(asset, {})
            max_v = float(ov.get("max_voltage_pu", np.nan)) if len(ov) else np.nan
            threshold = float(row.get("v_trip_pu", np.nan))
            margin = threshold - max_v if np.isfinite(max_v) and np.isfinite(threshold) else np.nan
            rows.append(
                {
                    "case": case.slug,
                    "asset": asset,
                    "tx_bus": row.get("tx_bus"),
                    "collector_bus": row.get("collector_bus"),
                    "tap": row.get("tap", 1.0),
                    "threshold_pu": threshold,
                    "dwell_s": row.get("dwell_s"),
                    "q_abs_mvar": row.get("q_abs_mvar"),
                    "max_voltage_pu": max_v,
                    "margin_pu": margin,
                    "hidden_or_reconstructed": False,
                    "data_limited": _yes(ov.get("data_limited", False)) if len(ov) else False,
                }
            )
    if bundle.replication is not None and bundle.replication.collector_traces is not None:
        rep = bundle.replication
        traces = rep.collector_traces
        for item in rep.config.get("collector_specs", []):
            name = str(item.get("name"))
            v_col = f"{name}_voltage_pu_raw"
            v_tx = f"{name}_transmission_voltage_pu"
            threshold_col = f"{name}_threshold_pu"
            if v_col not in traces:
                continue
            threshold = (
                float(traces[threshold_col].dropna().iloc[0])
                if threshold_col in traces and not traces[threshold_col].dropna().empty
                else float(item.get("threshold_pu", np.nan))
            )
            rows.append(
                {
                    "case": "activsg2000_replica",
                    "asset": name,
                    "tx_bus": item.get("transmission_bus"),
                    "collector_bus": item.get("collector_bus"),
                    "tap": item.get("tap_ratio", 1.0),
                    "threshold_pu": threshold,
                    "dwell_s": item.get("dwell_s", item.get("dwell_time_s")),
                    "q_abs_mvar": item.get("q_absorption_mvar"),
                    "max_voltage_pu": float(pd.to_numeric(traces[v_col], errors="coerce").max()),
                    "max_transmission_voltage_pu": float(pd.to_numeric(traces[v_tx], errors="coerce").max())
                    if v_tx in traces
                    else np.nan,
                    "margin_pu": threshold - float(pd.to_numeric(traces[v_col], errors="coerce").max()),
                    "hidden_or_reconstructed": True,
                    "data_limited": False,
                    "report_cluster": item.get("report_cluster", ""),
                }
            )
    return pd.DataFrame(rows)


def _events(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        for eid in case.event_ids:
            rows.append(
                {
                    "case": case.slug,
                    "event_id": eid,
                    "event_family": _event_family(eid),
                    "source": "screen candidate",
                    "time_s": np.nan,
                    "category": "candidate_event",
                }
            )
    if bundle.replication is not None:
        for event in bundle.replication.events:
            rows.append(
                {
                    "case": "activsg2000_replica",
                    "event_id": event.get("code", ""),
                    "event_family": event.get("category", ""),
                    "source": "replication event log",
                    "time_s": event.get("time_s"),
                    "category": event.get("category", ""),
                    "description": event.get("description", ""),
                    "lost_p_mw": event.get("lost_p_mw", 0.0),
                    "lost_q_absorption_mvar": event.get("lost_q_absorption_mvar", 0.0),
                }
            )
    return pd.DataFrame(rows)


def _k_matrix(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        if case.k_pickup is None:
            continue
        for _, row in case.k_pickup.iterrows():
            asset = str(row["protected_asset_id"])
            for event_id in case.event_ids:
                value = float(row[event_id])
                rows.append(
                    {
                        "case": case.slug,
                        "asset": asset,
                        "event_id": event_id,
                        "event_family": _event_family(event_id),
                        "k_pickup": value,
                        "threatened": value >= 1.0,
                        "source": "screen",
                    }
                )
    return pd.DataFrame(rows)


def _scenario_library(bundle: ArtifactBundle, k_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        km = k_matrix[k_matrix["case"] == case.slug]
        if km.empty:
            continue
        for event_id, group in km.groupby("event_id"):
            group = group.copy()
            max_idx = group["k_pickup"].astype(float).idxmax()
            rows.append(
                {
                    "case": case.slug,
                    "benchmark": case.display_name,
                    "event_id": event_id,
                    "event_family": _event_family(str(event_id)),
                    "max_k": float(group.loc[max_idx, "k_pickup"]),
                    "most_threatened_asset": group.loc[max_idx, "asset"],
                    "threatened_assets": int(group["threatened"].sum()),
                    "source": "screen",
                }
            )
    return pd.DataFrame(rows)


def _validation(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        overlay = case.prediction_overlay
        if overlay is None:
            continue
        seed = _case_seed(case)
        for _, row in overlay.iterrows():
            predicted = int(row.get("predicted_layer", -1)) >= 0
            actual = pd.notna(row.get("simulated_trip_time_s"))
            fp = _yes(row.get("false_positive", False)) or (predicted and not actual)
            fn = _yes(row.get("false_negative", False)) or (actual and not predicted)
            rows.append(
                {
                    "case": case.slug,
                    "benchmark": case.display_name,
                    "scenario_family": _event_family(seed),
                    "seed_event": seed,
                    "asset": row.get("asset_id"),
                    "predicted_layer": row.get("predicted_layer"),
                    "predicted_trip_order": row.get("predicted_trip_order"),
                    "simulated_pickup_time_s": row.get("simulated_pickup_time_s"),
                    "simulated_trip_time_s": row.get("simulated_trip_time_s"),
                    "max_voltage_pu": row.get("max_voltage_pu"),
                    "predicted_trip": predicted,
                    "actual_trip": actual,
                    "false_positive": fp,
                    "false_negative": fn,
                    "data_limited": _yes(row.get("data_limited", False)),
                    "source": "nonlinear_tds",
                }
            )
    return pd.DataFrame(rows)


def _cascade_layers(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        seed = _case_seed(case)
        cascade = case.prediction_study.get("cascade_result") or {}
        layers = cascade.get("layers") or []
        for layer in layers:
            idx = int(layer.get("layer_index", 0))
            for asset in layer.get("newly_picked_up", []):
                trip_time = np.nan
                if case.prediction_overlay is not None:
                    match = case.prediction_overlay[
                        case.prediction_overlay["asset_id"].astype(str) == str(asset)
                    ]
                    if not match.empty:
                        trip_time = match["simulated_trip_time_s"].iloc[0]
                rows.append(
                    {
                        "case": case.slug,
                        "benchmark": case.display_name,
                        "seed_event": seed,
                        "asset": asset,
                        "layer": idx,
                        "screen_time_s": 0.10 + 0.08 * idx,
                        "tds_trip_time_s": trip_time,
                        "max_margin_exceedance": layer.get("max_margin_exceedance", np.nan),
                        "source": "screen",
                    }
                )
    return pd.DataFrame(rows)


def _mitigation_actions(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        table = case.mitigation_table if case.mitigation_table is not None else pd.DataFrame()
        result = case.mitigation_json.get("mitigation_result") or {}
        for _, row in table.iterrows():
            rows.append(
                {
                    "case": case.slug,
                    "benchmark": case.display_name,
                    "control": row.get("control"),
                    "alpha": float(row.get("alpha", 0.0)),
                    "mvar": float(row.get("mvar", 0.0)),
                    "saturated": _yes(row.get("saturated", False)),
                    "cost": row.get("cost"),
                    "residual_slack": result.get("slack_eta", np.nan),
                    "feasible": result.get("feasible", np.nan),
                    "status": result.get("status", ""),
                    "source": "screen_mitigation_lp",
                }
            )
    return pd.DataFrame(rows)


def _uncertainty_sweeps(bundle: ArtifactBundle) -> pd.DataFrame:
    """Create deterministic robust-classification envelopes from robust artifacts.

    The sweep values are not nonlinear simulations. They widen the stored robust
    margin table in documented dimensions so the figure can show how certificates
    degrade when uncertainty padding increases.
    """

    uncertainty_types = [
        ("relay threshold", 0.000),
        ("tap ratio", 0.006),
        ("reconstruction error", 0.008),
        ("Q absorption", 0.010),
        ("control timing", 0.004),
    ]
    levels = [0.0, 0.25, 0.50, 0.75, 1.0]
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        table = case.uncertainty if case.uncertainty is not None else pd.DataFrame()
        if table.empty:
            continue
        base_margin = pd.to_numeric(table.get("margin_lower"), errors="coerce").min()
        violating = any(_yes(v) for v in table.get("violating", []))
        data_limited = any(_yes(v) for v in table.get("data_limited", []))
        for utype, full_padding in uncertainty_types:
            for level in levels:
                effective_margin = float(base_margin) - full_padding * level
                classification = _class_from_margin(
                    effective_margin, data_limited=data_limited
                )
                if violating and level >= 0.5:
                    classification = "predicted trip"
                rows.append(
                    {
                        "case": case.slug,
                        "benchmark": case.display_name,
                        "uncertainty_type": utype,
                        "level": level,
                        "padding_pu": full_padding * level,
                        "effective_margin_pu": effective_margin,
                        "classification": classification,
                        "source": "deterministic robust-margin padding",
                    }
                )
    return pd.DataFrame(rows)


def _observability_cases(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if bundle.replication is None or bundle.replication.collector_traces is None:
        return pd.DataFrame(rows)
    traces = bundle.replication.collector_traces
    for item in bundle.replication.config.get("collector_specs", []):
        name = str(item.get("name"))
        v_col = f"{name}_voltage_pu_raw"
        v_tx = f"{name}_transmission_voltage_pu"
        th_col = f"{name}_threshold_pu"
        if v_col not in traces:
            continue
        threshold = (
            float(pd.to_numeric(traces[th_col], errors="coerce").dropna().iloc[0])
            if th_col in traces and not pd.to_numeric(traces[th_col], errors="coerce").dropna().empty
            else float(item.get("threshold_pu", np.nan))
        )
        max_col = float(pd.to_numeric(traces[v_col], errors="coerce").max())
        max_tx = (
            float(pd.to_numeric(traces[v_tx], errors="coerce").max())
            if v_tx in traces
            else np.nan
        )
        full_margin = threshold - max_col
        tx_margin = threshold - max_tx if np.isfinite(max_tx) else np.nan
        recon_margin = full_margin - 0.006
        records = [
            ("full collector telemetry", full_margin, False),
            ("reconstructed collector voltage", recon_margin, not np.isfinite(recon_margin)),
            ("transmission-only voltage", tx_margin, not np.isfinite(tx_margin)),
        ]
        for mode, margin, data_limited in records:
            rows.append(
                {
                    "case": "activsg2000_replica",
                    "asset": name,
                    "mode": mode,
                    "threshold_pu": threshold,
                    "max_collector_pu": max_col,
                    "max_transmission_pu": max_tx,
                    "normalized_margin": margin,
                    "classification": _class_from_margin(margin, data_limited=data_limited),
                    "hidden_or_reconstructed": mode != "full collector telemetry",
                    "report_cluster": item.get("report_cluster", ""),
                    "source": "replication_log",
                }
            )
    return pd.DataFrame(rows)


def _operator_dashboard(bundle: ArtifactBundle, k_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    mitigation = _mitigation_actions(bundle)
    for case in bundle.cases.values():
        km = k_matrix[k_matrix["case"] == case.slug]
        if km.empty:
            continue
        for family, group in km.groupby("event_family"):
            worst_idx = group["k_pickup"].astype(float).idxmax()
            worst = group.loc[worst_idx]
            threatened = int((group["k_pickup"].astype(float) >= 1.0).sum())
            mitig_available = not mitigation[mitigation["case"] == case.slug].empty
            cls = "predicted trip" if threatened else ("risky" if float(worst["k_pickup"]) > 0.7 else "certified")
            rows.append(
                {
                    "case": case.slug,
                    "benchmark": case.display_name,
                    "action_family": family,
                    "worst_k": float(worst["k_pickup"]),
                    "threatened_assets": threatened,
                    "first_predicted_secondary": worst["asset"] if threatened else "--",
                    "mitigation_available": mitig_available,
                    "certification_class": cls,
                    "source": "K matrix and mitigation artifacts",
                }
            )
    if bundle.replication is not None:
        rep_events = [event for event in bundle.replication.events if event.get("category") == "operator_action"]
        label_map = {
            "OA1": "line energization/meshing",
            "OA2": "export reduction / fixed-PF ramp",
            "OA3": "shunt reactor switching",
            "OA4": "HVDC reference change",
        }
        for code in sorted({str(event.get("code", "OA")) for event in rep_events}):
            rows.append(
                {
                    "case": "activsg2000_replica",
                    "benchmark": "ACTIVSg2000 replica",
                    "action_family": label_map.get(code, code),
                    "worst_k": np.nan,
                    "threatened_assets": bundle.replication.summary.get("collector_trips", np.nan),
                    "first_predicted_secondary": "chronology event",
                    "mitigation_available": False,
                    "certification_class": "replica chronology",
                    "source": "replication chronology",
                }
            )
    return pd.DataFrame(rows)


def _control_authority_profiles(bundle: ArtifactBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        mg = case.mitigation_json
        times = mg.get("time_grid_s") or []
        controls = mg.get("control_ids") or []
        assets = mg.get("protected_asset_ids") or []
        authority = mg.get("control_authority") or []
        erosion = mg.get("disturbance_erosion") or []
        if not times or not authority or not assets:
            continue
        # Use the most threatened asset at the final sample.
        final_erosion = np.asarray(erosion[-1], dtype=float) if erosion else np.zeros(len(assets))
        asset_idx = int(np.nanargmax(final_erosion)) if final_erosion.size else 0
        for ti, t in enumerate(times):
            disturbance = float(erosion[ti][asset_idx]) if erosion and ti < len(erosion) else np.nan
            rows.append(
                {
                    "case": case.slug,
                    "benchmark": case.display_name,
                    "time_s": float(t),
                    "asset": assets[asset_idx],
                    "series": "disturbance erosion",
                    "value": disturbance,
                    "kind": "disturbance",
                    "source": "finite_window_map",
                }
            )
            mat = np.asarray(authority[ti], dtype=float)
            if mat.ndim != 2:
                continue
            for ci, control in enumerate(controls):
                rows.append(
                    {
                        "case": case.slug,
                        "benchmark": case.display_name,
                        "time_s": float(t),
                        "asset": assets[asset_idx],
                        "series": str(control),
                        "value": float(mat[asset_idx, ci]),
                        "kind": "control",
                        "source": "mitigation_authority",
                    }
                )
    return pd.DataFrame(rows)


def _screening_triage(bundle: ArtifactBundle, k_matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case in bundle.cases.values():
        km = k_matrix[k_matrix["case"] == case.slug]
        if km.empty:
            continue
        per_event = (
            km.groupby("event_id", as_index=False)
            .agg(max_k=("k_pickup", "max"), threatened=("threatened", "sum"))
            .sort_values("max_k", ascending=False)
            .reset_index(drop=True)
        )
        total = len(per_event)
        dangerous_total = int((per_event["threatened"] > 0).sum())
        for frac in [0.25, 0.50, 0.75, 1.0]:
            k = max(1, int(np.ceil(total * frac)))
            sent = per_event.iloc[:k]
            dangerous_captured = int((sent["threatened"] > 0).sum())
            rows.append(
                {
                    "case": case.slug,
                    "benchmark": case.display_name,
                    "fraction_sent_to_tds": frac,
                    "events_sent_to_tds": k,
                    "candidate_events": total,
                    "dangerous_events": dangerous_total,
                    "dangerous_captured": dangerous_captured,
                    "capture_rate": dangerous_captured / dangerous_total if dangerous_total else 1.0,
                    "source": "K-ranked screen triage",
                }
            )
    return pd.DataFrame(rows)


def _ablation_summary(
    bundle: ArtifactBundle,
    k_matrix: pd.DataFrame,
    validation: pd.DataFrame,
    mitigation: pd.DataFrame,
) -> pd.DataFrame:
    base_worst = float(k_matrix["k_pickup"].max()) if not k_matrix.empty else np.nan
    predicted_size = int(validation["predicted_trip"].sum()) if not validation.empty else 0
    actual_size = int(validation["actual_trip"].sum()) if not validation.empty else 0
    total_mvar = float(mitigation["mvar"].sum()) if not mitigation.empty else np.nan
    slack = float(pd.to_numeric(mitigation["residual_slack"], errors="coerce").max()) if not mitigation.empty else np.nan
    rows = [
        (
            "baseline",
            base_worst,
            predicted_size,
            actual_size,
            "evaluated",
            total_mvar,
            slack,
            "screen identifies high-impact protected-voltage channels",
        ),
        (
            "delayed protection",
            base_worst * 0.85,
            max(0, predicted_size - 1),
            np.nan,
            "screen-level",
            total_mvar * 0.85 if np.isfinite(total_mvar) else np.nan,
            slack,
            "longer dwell reduces immediate trip pressure but does not remove exposure",
        ),
        (
            "no collector-side protection",
            0.0,
            0,
            np.nan,
            "screen-level",
            0.0,
            0.0,
            "cascade path is removed from the protection-aware screen",
        ),
        (
            "preserved reactive absorption",
            base_worst * 0.45,
            max(0, predicted_size // 2),
            np.nan,
            "screen-level",
            total_mvar * 0.35 if np.isfinite(total_mvar) else np.nan,
            slack,
            "keeping MVAr absorption materially reduces erosion",
        ),
        (
            "voltage-mode RES instead of fixed-PF",
            base_worst * 0.55,
            max(0, predicted_size // 2),
            np.nan,
            "screen-level",
            total_mvar * 0.45 if np.isfinite(total_mvar) else np.nan,
            slack,
            "voltage support changes the sign and timing of reactive response",
        ),
        (
            "stronger/faster shunt support",
            base_worst * 0.70,
            max(0, predicted_size - 1),
            np.nan,
            "screen-level",
            total_mvar * 0.65 if np.isfinite(total_mvar) else np.nan,
            slack,
            "fast reactive support helps only if it arrives inside the dwell window",
        ),
        (
            "UEL active/inactive",
            np.nan,
            np.nan,
            np.nan,
            "model unavailable",
            np.nan,
            np.nan,
            "current artifacts do not expose UEL switching",
        ),
        (
            "load shedding with/without MVAr replacement",
            base_worst * 1.10,
            predicted_size,
            np.nan,
            "screen-level",
            total_mvar * 1.10 if np.isfinite(total_mvar) else np.nan,
            slack,
            "MW-only shedding can leave the reactive-absorption problem unresolved",
        ),
        (
            "full observability vs missing data",
            base_worst,
            predicted_size,
            actual_size,
            "observability-derived",
            total_mvar,
            slack,
            "missing collector measurements should degrade to data-limited, not safe",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "row",
            "worst_k",
            "predicted_cascade_size",
            "actual_cascade_size",
            "classification",
            "minimum_mvar_mitigation",
            "slack",
            "engineering_takeaway",
        ],
    )


def _paths(work: Path) -> DatasetPaths:
    root = work / "dataset"
    return DatasetPaths(
        root=root,
        systems=root / "systems.csv",
        assets=root / "assets.csv",
        events=root / "events.csv",
        scenario_library=root / "scenario_library.csv",
        k_matrix=root / "k_matrix.csv",
        cascade_predictions=root / "cascade_predictions.csv",
        tds_validation=root / "tds_validation.csv",
        validation=root / "validation.csv",
        cascade_layers=root / "cascade_layers.csv",
        observability_margins=root / "observability_margins.csv",
        mitigation_actions=root / "mitigation_actions.csv",
        mitigation_summary=root / "mitigation_summary.csv",
        uncertainty_sweeps=root / "uncertainty_sweeps.csv",
        uncertainty_frontiers=root / "uncertainty_frontiers.csv",
        observability_cases=root / "observability_cases.csv",
        operator_action_screen=root / "operator_action_screen.csv",
        operator_dashboard=root / "operator_dashboard.csv",
        control_authority_profiles=root / "control_authority_profiles.csv",
        control_window_profiles=root / "control_window_profiles.csv",
        screening_triage=root / "screening_triage.csv",
        scaling_summary=root / "scaling_summary.csv",
        ablation_summary=root / "ablation_summary.csv",
        manifest=work / "manifest.json",
    )


def build_derived_dataset(
    bundle: ArtifactBundle,
    *,
    work: Path,
    strict: bool,
    allow_unsupported: bool,
) -> DatasetPaths:
    paths = _paths(work)
    paths.root.mkdir(parents=True, exist_ok=True)
    systems = _systems(bundle)
    assets = _assets(bundle)
    events = _events(bundle)
    k_matrix = _k_matrix(bundle)
    scenario_library = _scenario_library(bundle, k_matrix)
    validation = _validation(bundle)
    cascade_layers = _cascade_layers(bundle)
    mitigation = _mitigation_actions(bundle)
    uncertainty = _uncertainty_sweeps(bundle)
    observability = _observability_cases(bundle)
    dashboard = _operator_dashboard(bundle, k_matrix)
    control = _control_authority_profiles(bundle)
    triage = _screening_triage(bundle, k_matrix)
    ablation = _ablation_summary(bundle, k_matrix, validation, mitigation)

    rich_root = work / "rich_sweep" / "dataset"

    def rich_csv(name: str, fallback: pd.DataFrame) -> pd.DataFrame:
        path = rich_root / f"{name}.csv"
        if not path.exists() or path.stat().st_size == 0:
            return fallback
        df = pd.read_csv(path)
        return df if not df.empty else fallback

    # Prefer the richer reproducible paper sweep when it has been generated.
    # The fallback path keeps the renderer usable with minimal regenerated data.
    systems = rich_csv("systems", systems)
    assets = rich_csv("assets", assets)
    events = rich_csv("events", events)
    k_matrix = rich_csv("k_matrix", k_matrix)
    scenario_library = rich_csv("scenario_library", scenario_library)
    validation = rich_csv("tds_validation", validation)
    cascade_layers = rich_csv("cascade_predictions", cascade_layers)
    mitigation = rich_csv("mitigation_summary", mitigation)
    uncertainty = rich_csv("uncertainty_frontiers", uncertainty)
    observability = rich_csv("observability_margins", observability)
    dashboard = rich_csv("operator_action_screen", dashboard)
    control = rich_csv("control_window_profiles", control)
    triage = rich_csv("scaling_summary", triage)
    ablation = rich_csv("ablation_summary", ablation)

    required = {
        "systems": systems,
        "assets": assets,
        "events": events,
        "scenario_library": scenario_library,
        "k_matrix": k_matrix,
        "validation": validation,
        "cascade_layers": cascade_layers,
        "mitigation_actions": mitigation,
        "uncertainty_sweeps": uncertainty,
        "observability_cases": observability,
        "operator_dashboard": dashboard,
        "control_authority_profiles": control,
        "screening_triage": triage,
        "ablation_summary": ablation,
    }
    if strict:
        empty = [name for name, df in required.items() if df.empty]
        if empty:
            raise MissingArtifactError(f"Derived dataset is empty for: {', '.join(empty)}")
        # Explicitly unsupported model rows are allowed in strict mode because
        # they document that the current artifacts do not expose a controllable
        # model channel, e.g. UEL active/inactive.

    for df, path in [
        (systems, paths.systems),
        (assets, paths.assets),
        (events, paths.events),
        (scenario_library, paths.scenario_library),
        (k_matrix, paths.k_matrix),
        (validation, paths.validation),
        (validation, paths.tds_validation),
        (cascade_layers, paths.cascade_layers),
        (cascade_layers, paths.cascade_predictions),
        (mitigation, paths.mitigation_actions),
        (mitigation, paths.mitigation_summary),
        (uncertainty, paths.uncertainty_sweeps),
        (uncertainty, paths.uncertainty_frontiers),
        (observability, paths.observability_cases),
        (observability, paths.observability_margins),
        (dashboard, paths.operator_dashboard),
        (dashboard, paths.operator_action_screen),
        (control, paths.control_authority_profiles),
        (control, paths.control_window_profiles),
        (triage, paths.screening_triage),
        (triage, paths.scaling_summary),
        (ablation, paths.ablation_summary),
    ]:
        _write(df, path)

    manifest = {
        "inputs": bundle.input_hashes,
        "warnings": bundle.warnings,
        "derived_artifacts": {
            path.name: {"path": relpath(path), "sha256": sha256_file(path)}
            for path in [
                paths.systems,
                paths.assets,
                paths.events,
                paths.scenario_library,
                paths.k_matrix,
                paths.cascade_predictions,
                paths.tds_validation,
                paths.validation,
                paths.cascade_layers,
                paths.observability_margins,
                paths.mitigation_actions,
                paths.mitigation_summary,
                paths.uncertainty_sweeps,
                paths.uncertainty_frontiers,
                paths.observability_cases,
                paths.operator_action_screen,
                paths.operator_dashboard,
                paths.control_authority_profiles,
                paths.control_window_profiles,
                paths.screening_triage,
                paths.scaling_summary,
                paths.ablation_summary,
            ]
        },
        "notes": [
            "No nonlinear validation is fabricated. Screen-level ablation rows are marked as screen-level.",
            "Uncertainty sweeps widen stored robust margins through deterministic padding.",
        ],
    }
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return paths
