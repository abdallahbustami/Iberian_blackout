#!/usr/bin/env python3
"""Generate the finite-window/proxy comparison figure used in ``root.tex``.

The response curves are computed from the IEEE-39 ANDES case-study reduced
model.  The plotting layer is intentionally compact because the figure is used
as a single-column method illustration.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pa_dvsa.data_model import ProfileKind, TimeProfile
from pa_dvsa.finite_window import profile_response, step_response_d
from pa_dvsa.paper_case_studies.style import (
    PALETTE,
    configure_style,
    save_figure,
    soften_axes,
)
from pa_dvsa.protected_outputs import evaluate_protected_outputs
from pa_dvsa.resolvent_proxy import alpha_star, ghat
from scripts.run_load_shedding_case_study import (
    STUDIES,
    _load_case,
    _protected_assets,
    _reduced_model,
)


FIGURE_STEM = "finite_window_proxy"
OUT_DIR = Path("LaTeX/figures/generated")
DATA_DIR = Path("results/figures/finite_window_proxy")
CSV_PATH = DATA_DIR / "finite_window_proxy_waveforms.csv"
JSON_PATH = DATA_DIR / "finite_window_proxy_metadata.json"
SEARCH_CSV_PATH = DATA_DIR / "finite_window_proxy_channel_search.csv"
CASE_LABEL = {
    "ieee39_full": "IEEE-39",
    "npcc_full": "NPCC",
    "gbnetwork": "GBnetwork",
}


def _unit_input(study, event_id: str) -> np.ndarray:
    unit = np.zeros((len(study.event_ids), 1), dtype=float)
    unit[study.event_ids.index(event_id), 0] = 1.0
    return unit


def _compute_channel(reduced, study, *, asset_id: str, event_id: str, window_s: float, ramp_s: float):
    times = np.linspace(0.0, window_s, 401)
    output_index = study.asset_ids.index(asset_id)
    unit = _unit_input(study, event_id)
    step = step_response_d(
        reduced,
        times,
        disturbance_matrix=unit,
        event_ids=(event_id,),
    )
    ramp = profile_response(
        reduced,
        times,
        input_matrix=unit,
        profiles=(TimeProfile(ProfileKind.RAMP, duration_s=ramp_s),),
        channel_ids=(event_id,),
        channel_type="disturbance",
    )
    proxy = float(ghat(reduced, window_s, channel_type="disturbance", input_matrix=unit)[output_index, 0])
    step_values = np.asarray(step.values_pu[:, output_index, 0], dtype=float)
    ramp_values = np.asarray(ramp.values_pu[:, output_index, 0], dtype=float)
    peak_idx = int(np.argmax(step_values))
    return {
        "times": times,
        "step": step_values,
        "ramp": ramp_values,
        "proxy": proxy,
        "bound": alpha_star() * max(proxy, 0.0),
        "peak_time": float(times[peak_idx]),
        "peak_value": float(step_values[peak_idx]),
        "endpoint_value": float(step_values[-1]),
        "monotone": bool(np.all(np.diff(step_values) >= -1.0e-7)),
    }


def _display_id(value: str) -> str:
    return value.replace("load", "L").replace("_shed", "")


def _find_nonmonotone_channel(*, window_s: float = 1.0, ramp_s: float = 0.15) -> tuple[dict, list[dict]]:
    """Search benchmark reduced models for a visibly nonmonotone channel.

    We reject tiny channels even if their peak/proxy ratio is numerically large.
    The score favors channels whose finite-window peak exceeds both the proxy
    and the endpoint by a visible amount.
    """

    candidates: list[dict] = []
    for study_key, study in STUDIES.items():
        case = _load_case(study, init_tds=True)
        protected = evaluate_protected_outputs(case, _protected_assets(study, case))
        reduced = _reduced_model(study, case, protected)
        for asset_id in study.asset_ids:
            for event_id in study.event_ids:
                channel = _compute_channel(
                    reduced,
                    study,
                    asset_id=asset_id,
                    event_id=event_id,
                    window_s=window_s,
                    ramp_s=ramp_s,
                )
                peak = float(channel["peak_value"])
                endpoint = float(channel["endpoint_value"])
                proxy = float(channel["proxy"])
                peak_time = float(channel["peak_time"])
                if peak <= 0.0025 or proxy <= 1.0e-6 or bool(channel["monotone"]):
                    continue
                if peak_time >= 0.95 * window_s:
                    continue
                peak_proxy = peak / proxy
                peak_endpoint = peak / max(endpoint, 1.0e-9)
                if peak_proxy <= 1.05 or peak_endpoint <= 1.02:
                    continue
                score = peak_proxy * peak_endpoint * min(peak / 0.005, 2.0)
                candidates.append(
                    {
                        "study_key": study_key,
                        "study_id": study.study_id,
                        "case_id": study.case_id,
                        "asset_id": asset_id,
                        "event_id": event_id,
                        "window_s": window_s,
                        "ramp_s": ramp_s,
                        "peak_value": peak,
                        "peak_time": peak_time,
                        "endpoint_value": endpoint,
                        "proxy": proxy,
                        "bound": float(channel["bound"]),
                        "peak_proxy_ratio": peak_proxy,
                        "peak_endpoint_ratio": peak_endpoint,
                        "score": score,
                        "channel": channel,
                    }
                )
    if not candidates:
        raise RuntimeError("No visible nonmonotone finite-window channel found")
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[0], candidates


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configure_style(prefer_usetex=True)
    monotone_study = STUDIES["ieee39"]
    monotone_case = _load_case(monotone_study, init_tds=True)
    monotone_protected = evaluate_protected_outputs(monotone_case, _protected_assets(monotone_study, monotone_case))
    monotone_reduced = _reduced_model(monotone_study, monotone_case, monotone_protected)
    nonmonotone_best, search_candidates = _find_nonmonotone_channel(window_s=1.0, ramp_s=0.15)

    # These channels are deterministic IEEE-39 channels used in the screening
    # case study.  The first is monotone over the relay window; the second has a
    # nonmonotone early peak, which is exactly the case where direct sampling is
    # required for disturbance certification.
    specs = [
        {
            "panel": "a",
            "title": "Monotone channel",
            "subtitle": "IEEE-39 protected output L3 under seed L3",
            "asset_id": "load3",
            "event_id": "load3_shed",
            "window_s": 0.50,
            "ramp_s": 0.15,
            "badge": "proxy usable after monotone certificate",
        },
        {
            "panel": "b",
            "title": "Non-monotone channel",
            "subtitle": (
                f"{CASE_LABEL.get(str(nonmonotone_best['case_id']), str(nonmonotone_best['case_id']))} protected output "
                f"{_display_id(nonmonotone_best['asset_id'])} under seed "
                f"{_display_id(nonmonotone_best['event_id'])}"
            ),
            "asset_id": nonmonotone_best["asset_id"],
            "event_id": nonmonotone_best["event_id"],
            "window_s": float(nonmonotone_best["window_s"]),
            "ramp_s": float(nonmonotone_best["ramp_s"]),
            "badge": "sampling required; proxy not certified",
        },
    ]
    channels = [
        {
            **specs[0],
            **_compute_channel(
                monotone_reduced,
                monotone_study,
                asset_id=specs[0]["asset_id"],
                event_id=specs[0]["event_id"],
                window_s=specs[0]["window_s"],
                ramp_s=specs[0]["ramp_s"],
            ),
        },
        {**specs[1], **nonmonotone_best["channel"]},
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(3.45, 4.35),
        sharex=False,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.42},
    )
    fig.subplots_adjust(left=0.18, right=0.96, top=0.80, bottom=0.12)

    records: list[dict[str, object]] = []
    metadata = {
        "source_case": "ieee39_and_search_selected_benchmark",
        "mode_id": monotone_reduced.diagnostics.mode_id,
        "nonmonotone_search": {
            "systems": list(STUDIES),
            "selected_case": nonmonotone_best["case_id"],
            "selected_asset_id": nonmonotone_best["asset_id"],
            "selected_event_id": nonmonotone_best["event_id"],
            "selected_score": nonmonotone_best["score"],
            "selected_peak_proxy_ratio": nonmonotone_best["peak_proxy_ratio"],
            "selected_peak_endpoint_ratio": nonmonotone_best["peak_endpoint_ratio"],
        },
        "formulae": {
            "finite_window_peak": "max_{0 <= t <= T} Delta z_i(t)",
            "proxy": "Ghat(T) = F_d + C_r(1/T I - A_r)^(-1) D_r",
            "alpha_star": alpha_star(),
        },
        "channels": [],
    }

    for ax, channel in zip(axes, channels):
        times = channel["times"]
        step_values = channel["step"]
        ramp_values = channel["ramp"]
        proxy = float(channel["proxy"])
        bound = float(channel["bound"])
        peak_time = float(channel["peak_time"])
        peak_value = float(channel["peak_value"])

        ax.axvspan(0, channel["window_s"], color=PALETTE["empty"], alpha=0.60, zorder=0)
        ax.plot(
            times,
            step_values,
            color=PALETTE["trip"],
            lw=1.85,
            label="step disturbance",
            zorder=3,
        )
        ax.plot(
            times,
            ramp_values,
            color=PALETTE["control"],
            lw=1.55,
            linestyle=(0, (4.5, 2.0)),
            label="150 ms ramp",
            zorder=3,
        )
        ax.axhline(
            proxy,
            color=PALETTE["charcoal"],
            lw=1.05,
            linestyle=(0, (1.2, 1.8)),
            label=r"resolvent proxy $\widehat G(1/T)$",
            zorder=2,
        )
        ax.axhline(
            bound,
            color=PALETTE["risky"],
            lw=1.15,
            linestyle=(0, (5.0, 2.0)),
            label=r"monotone bound $\alpha_\star\widehat G(1/T)$",
            zorder=2,
        )
        ax.scatter(
            [peak_time],
            [peak_value],
            s=42,
            facecolors=PALETTE["trip"],
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label=r"finite-window peak",
        )
        ax.annotate(
            r"$\max \Delta z_i(t)$",
            xy=(peak_time, peak_value),
            xytext=(0.58 * channel["window_s"], peak_value * 0.86),
            ha="left",
            va="center",
            fontsize=6.9,
            color=PALETTE["trip"],
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 0.75,
                "color": PALETTE["trip"],
                "shrinkA": 1.5,
                "shrinkB": 2.5,
            },
        )

        ax.text(
            0.00,
            1.10,
            rf"\textbf{{{channel['panel']}}}  {channel['title']}"
            if matplotlib.rcParams.get("text.usetex")
            else f"{channel['panel']}  {channel['title']}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.0,
            fontweight="bold",
            color=PALETTE["charcoal"],
        )
        ax.text(
            0.02,
            0.93,
            channel["subtitle"],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.9,
            color=PALETTE["charcoal"],
        )
        badge_color = PALETTE["safe"] if channel["monotone"] else PALETTE["data"]
        ax.text(
            0.98,
            0.08,
            channel["badge"],
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color=PALETTE["charcoal"],
            bbox={
                "boxstyle": "round,pad=0.22,rounding_size=0.10",
                "facecolor": PALETTE["light_teal"] if channel["monotone"] else PALETTE["light_purple"],
                "edgecolor": badge_color,
                "linewidth": 0.75,
            },
        )
        ax.text(
            0.98,
            0.88,
            rf"peak/proxy $={peak_value / proxy:.2f}$" if proxy > 0 else "proxy unavailable",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color=PALETTE["charcoal"],
        )

        ax.set_ylabel(r"$\Delta z_i$ (p.u.)")
        ax.set_xlim(0.0, channel["window_s"])
        y_values = np.concatenate([step_values, ramp_values, np.array([proxy, bound])])
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        pad = max(0.002, 0.12 * (y_max - y_min))
        ax.set_ylim(y_min - 0.25 * pad, y_max + 1.10 * pad)
        soften_axes(ax, grid=True)

        metadata["channels"].append(
            {
                "panel": channel["panel"],
                "asset_id": channel["asset_id"],
                "event_id": channel["event_id"],
                "window_s": channel["window_s"],
                "ramp_duration_s": channel["ramp_s"],
                "monotone": channel["monotone"],
                "finite_window_peak_pu": peak_value,
                "finite_window_peak_time_s": peak_time,
                "endpoint_value_pu": float(channel["endpoint_value"]),
                "proxy_pu": proxy,
                "alpha_star_proxy_pu": bound,
            }
        )
        for time_s, step_value, ramp_value in zip(times, step_values, ramp_values):
            records.append(
                {
                    "panel": channel["panel"],
                    "asset_id": channel["asset_id"],
                    "event_id": channel["event_id"],
                    "time_s": float(time_s),
                    "step_delta_z_pu": float(step_value),
                    "ramp_delta_z_pu": float(ramp_value),
                    "proxy_pu": proxy,
                    "alpha_star_proxy_pu": bound,
                    "monotone": channel["monotone"],
                }
            )

    axes[-1].set_xlabel("time in assessment window (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    # Keep one readable legend above the panels and remove duplicates.
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.985),
        ncol=2,
        frameon=False,
        fontsize=6.7,
        handlelength=2.4,
        columnspacing=0.9,
    )

    results = save_figure(fig, FIGURE_STEM, OUT_DIR, ("pdf",), outline_text=False)
    plt.close(fig)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with SEARCH_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "study_key",
            "study_id",
            "case_id",
            "asset_id",
            "event_id",
            "window_s",
            "peak_value",
            "peak_time",
            "endpoint_value",
            "proxy",
            "peak_proxy_ratio",
            "peak_endpoint_ratio",
            "score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in search_candidates:
            writer.writerow({key: candidate[key] for key in fieldnames})
    JSON_PATH.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    for result in results:
        print(f"Wrote {result.path}")
    print(f"Wrote {CSV_PATH}")
    print(f"Wrote {SEARCH_CSV_PATH}")
    print(f"Wrote {JSON_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
