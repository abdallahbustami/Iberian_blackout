"""Deterministic calibration sweep for the ACTIVSg2000 Iberian-style replica.

The sweep intentionally varies only transparent mechanism parameters: collector
count/tap/Q scale, affected-area radius, relay reset ratio, tie-trip severity,
and generator-protection target.  It never enables the diagnostic equivalent
PQ-blackout burden.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from itertools import product
from pathlib import Path
import shutil
from typing import Any

from .activsg2000_engine import (
    GeneratorProtectionRelaySpec,
    OutOfStepRelaySpec,
    ReplicationConfig,
    build_report_collector_specs,
    make_replication_plot,
    run_replication,
)


BASELINE = {
    "collector_count": 16,
    "area_radius": 5,
    "tap_scale": 0.95,
    "q_absorption_scale": 1.0,
    "tie_trip_count": 8,
    "generator_target_mva": 8000.0,
    "reset_ratio": 0.95,
}


def _candidate_grid(full: bool) -> list[dict[str, Any]]:
    dimensions = {
        "collector_count": [12, 16, 20],
        "area_radius": [5, 6, 7],
        "tap_scale": [0.94, 0.95, 0.96],
        "q_absorption_scale": [1.0, 1.25, 1.5],
        "tie_trip_count": [8, 16, 24],
        "generator_target_mva": [4000.0, 6000.0, 8000.0, 10000.0],
        "reset_ratio": [0.92, 0.95, 0.97],
    }
    keys = list(dimensions)
    candidates = [dict(BASELINE)]
    if full:
        candidates.extend(
            dict(zip(keys, values, strict=True))
            for values in product(*(dimensions[key] for key in keys))
        )
    else:
        for key, values in dimensions.items():
            for value in values:
                candidate = dict(BASELINE)
                candidate[key] = value
                candidates.append(candidate)
    seen: set[tuple[tuple[str, Any], ...]] = set()
    unique: list[dict[str, Any]] = []
    for candidate in candidates:
        marker = tuple(sorted(candidate.items()))
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(candidate)
    return unique


def make_config(candidate: dict[str, Any], *, tf_s: float, tstep_s: float) -> ReplicationConfig:
    collector_specs = build_report_collector_specs(
        count=int(candidate["collector_count"]),
        tap_scale=float(candidate["tap_scale"]),
        q_absorption_scale=float(candidate["q_absorption_scale"]),
        reset_ratio=float(candidate["reset_ratio"]),
    )
    base = ReplicationConfig(
        tf_s=tf_s,
        tstep_s=tstep_s,
        area_bfs_radius=int(candidate["area_radius"]),
        collector_specs=collector_specs,
    )
    return replace(
        base,
        out_of_step_relay=replace(
            base.out_of_step_relay,
            boundary_line_count=int(candidate["tie_trip_count"]),
        ),
        generator_protection=replace(
            base.generator_protection,
            target_mva=float(candidate["generator_target_mva"]),
        ),
    )


def evaluate_candidate(summary: dict[str, Any], events: list[dict[str, Any]]) -> tuple[bool, str, float]:
    reasons: list[str] = []
    category_times: dict[str, list[float]] = {}
    for event in events:
        category_times.setdefault(str(event["category"]), []).append(float(event["time_s"]))
    first_trip = min(category_times.get("protection_trip", [float("inf")]))
    last_operator = max(category_times.get("operator_action", [-float("inf")]))
    gen_time = min(category_times.get("generator_trip", [float("inf")]))
    collector_before_gen = sum(
        1
        for event in events
        if event["category"] == "protection_trip" and float(event["time_s"]) < gen_time
    )
    hvdc_time = min(category_times.get("hvdc_block", [float("inf")]))
    france_time = min(category_times.get("out_of_step_trip", [float("inf")]))
    morocco_time = min(category_times.get("morocco_ac_trip", [float("inf")]))
    defense_time = min(category_times.get("defense_action", [float("inf")]))

    if last_operator >= first_trip:
        reasons.append("operator_action_after_first_trip")
    if collector_before_gen < 6:
        reasons.append("fewer_than_6_collector_trips_before_generator_protection")
    if not (gen_time < defense_time < morocco_time < france_time < hvdc_time):
        reasons.append("post_cascade_order_mismatch")
    if int(summary.get("automatic_blackout_actions", 0)) != 0:
        reasons.append("anonymous_pq_blackout_burden_present")
    if not bool(summary.get("blackout_detected")):
        reasons.append("no_blackout_endpoint")
    if hvdc_time <= france_time:
        reasons.append("hvdc_not_after_ac_separation")

    rejected = bool(reasons)
    # Lower score is better.  Hard ordering dominates, then timing/MW/MVAr.
    score = 0.0 if not rejected else 1_000_000.0 + 10_000.0 * len(reasons)
    score += abs(float(summary.get("collector_trips", 0)) - 13.0) * 25.0
    score += abs(float(summary.get("total_lost_p_mw", 0.0)) - 7600.0) / 50.0
    score += abs(float(summary.get("total_lost_q_absorption_mvar", 0.0)) - 830.0) / 5.0
    score += abs(float(summary.get("max_physical_voltage_pu", 1.1)) - 1.13) * 100.0
    return rejected, ";".join(reasons), score


def run_calibration(args: argparse.Namespace) -> int:
    out_root: Path = args.out
    calibration_dir = out_root / "calibration"
    run_dir = calibration_dir / "runs"
    if args.clean and calibration_dir.exists():
        shutil.rmtree(calibration_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.use_existing_baseline:
        summary_path = out_root / "summary.json"
        events_path = out_root / "events.json"
        if not summary_path.exists() or not events_path.exists():
            raise FileNotFoundError(
                "--use-existing-baseline requires summary.json and events.json in --out"
            )
        summary = json.loads(summary_path.read_text())
        events = json.loads(events_path.read_text())
        rejected, reason, score = evaluate_candidate(summary, events)
        row = {
            "candidate": "existing_baseline",
            **BASELINE,
            "rejected": int(rejected),
            "rejection_reason": reason,
            "score": score,
            "collector_trips": summary.get("collector_trips"),
            "blackout_detected": summary.get("blackout_detected"),
            "tds_ok": summary.get("tds_ok"),
            "blackout_time_s": summary.get("blackout_time_s"),
            "first_trip_time_s": summary.get("first_trip_time_s"),
            "first_hvdc_block_time_s": summary.get("first_hvdc_block_time_s"),
            "first_france_ac_separation_time_s": summary.get(
                "first_france_ac_separation_time_s"
            ),
            "total_lost_p_mw": summary.get("total_lost_p_mw"),
            "total_lost_q_absorption_mvar": summary.get(
                "total_lost_q_absorption_mvar"
            ),
        }
        manifest_path = calibration_dir / "manifest.csv"
        with manifest_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        (calibration_dir / "best_config.json").write_text(
            json.dumps(BASELINE, indent=2) + "\n"
        )
        (calibration_dir / "best_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (calibration_dir / "best_run_path.txt").write_text(str(out_root) + "\n")
        (calibration_dir / "rejected_candidates.json").write_text(
            json.dumps([row] if rejected else [], indent=2) + "\n"
        )
        print(f"Wrote {manifest_path}")
        print(f"Best run: {out_root} score={score:.2f}")
        return 0

    rows: list[dict[str, Any]] = []
    best: tuple[float, dict[str, Any], dict[str, Any], Path] | None = None
    rejected_payload: list[dict[str, Any]] = []

    candidates = _candidate_grid(bool(args.full))
    if args.max_runs is not None:
        candidates = candidates[: int(args.max_runs)]

    for index, candidate in enumerate(candidates, start=1):
        name = (
            f"c{index:04d}_n{candidate['collector_count']}_r{candidate['area_radius']}"
            f"_tap{candidate['tap_scale']}_q{candidate['q_absorption_scale']}"
            f"_tie{candidate['tie_trip_count']}_g{int(candidate['generator_target_mva'])}"
            f"_reset{candidate['reset_ratio']}"
        ).replace(".", "p")
        cfg = make_config(candidate, tf_s=float(args.tf), tstep_s=float(args.tstep))
        out_dir = run_dir / name
        try:
            artifacts, summary, monitor = run_replication(
                cfg,
                out_dir=out_dir,
                quiet_andes=not args.verbose_andes,
            )
            events = json.loads(artifacts.events_path.read_text())
            rejected, reason, score = evaluate_candidate(summary, events)
            row = {
                "candidate": name,
                **candidate,
                "rejected": int(rejected),
                "rejection_reason": reason,
                "score": score,
                "collector_trips": summary.get("collector_trips"),
                "blackout_detected": summary.get("blackout_detected"),
                "tds_ok": summary.get("tds_ok"),
                "blackout_time_s": summary.get("blackout_time_s"),
                "first_trip_time_s": summary.get("first_trip_time_s"),
                "first_hvdc_block_time_s": summary.get("first_hvdc_block_time_s"),
                "first_france_ac_separation_time_s": summary.get(
                    "first_france_ac_separation_time_s"
                ),
                "total_lost_p_mw": summary.get("total_lost_p_mw"),
                "total_lost_q_absorption_mvar": summary.get(
                    "total_lost_q_absorption_mvar"
                ),
            }
            rows.append(row)
            if rejected:
                rejected_payload.append(row)
            if best is None or score < best[0]:
                best = (score, candidate, summary, out_dir)
                if args.plot_best:
                    make_replication_plot(
                        artifacts,
                        cfg,
                        monitor,
                        latex_fig_dir=Path("LaTeX/figures/generated"),
                    )
        except Exception as exc:  # noqa: BLE001 - calibration must log failed candidates.
            row = {
                "candidate": name,
                **candidate,
                "rejected": 1,
                "rejection_reason": f"run_failed:{type(exc).__name__}:{exc}",
                "score": float("inf"),
            }
            rows.append(row)
            rejected_payload.append(row)

    manifest_path = calibration_dir / "manifest.csv"
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with manifest_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if best is not None:
        _score, candidate, summary, candidate_dir = best
        (calibration_dir / "best_config.json").write_text(
            json.dumps(candidate, indent=2) + "\n"
        )
        (calibration_dir / "best_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        (calibration_dir / "best_run_path.txt").write_text(str(candidate_dir) + "\n")
    (calibration_dir / "rejected_candidates.json").write_text(
        json.dumps(rejected_payload, indent=2) + "\n"
    )
    print(f"Wrote {manifest_path}")
    if best is not None:
        print(f"Best run: {best[3]} score={best[0]:.2f}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/activsg2000_iberian_replication"))
    parser.add_argument("--tf", type=float, default=30.0)
    parser.add_argument("--tstep", type=float, default=0.01)
    parser.add_argument("--max-runs", type=int, default=12)
    parser.add_argument("--full", action="store_true", help="Run the full Cartesian sweep.")
    parser.add_argument("--plot-best", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--verbose-andes", action="store_true")
    parser.add_argument(
        "--use-existing-baseline",
        action="store_true",
        help="Create calibration manifest from an already generated baseline run.",
    )
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    if argv is not None:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--out", type=Path, default=Path("results/activsg2000_iberian_replication"))
        parser.add_argument("--tf", type=float, default=30.0)
        parser.add_argument("--tstep", type=float, default=0.01)
        parser.add_argument("--max-runs", type=int, default=12)
        parser.add_argument("--full", action="store_true", help="Run the full Cartesian sweep.")
        parser.add_argument("--plot-best", action="store_true")
        parser.add_argument("--clean", action="store_true")
        parser.add_argument("--verbose-andes", action="store_true")
        parser.add_argument(
            "--use-existing-baseline",
            action="store_true",
            help="Create calibration manifest from an already generated baseline run.",
        )
        return run_calibration(parser.parse_args(argv))
    return run_calibration(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
