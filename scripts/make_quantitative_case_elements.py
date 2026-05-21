"""Generate quantitative case-study tables and mitigation LP figure.

The outputs are derived from the normalized paper-case-study artifacts and the
per-benchmark mitigation LP JSON files. Missing nonlinear timing/runtime data is
reported as unavailable instead of inferred.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pa_dvsa.paper_case_studies.style import (
    FIGURE_SPECS,
    PALETTE,
    configure_style,
    save_figure,
    soften_axes,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "results" / "paper_case_studies" / "dataset"
CASE_STUDIES = ROOT / "results" / "case_studies"
REPLICA = ROOT / "results" / "activsg2000_iberian_replication"
VARIANTS = ROOT / "results" / "replication_subsection" / "replication_variant_summary.csv"
OUT_TABLE = ROOT / "LaTeX" / "tables" / "generated"
OUT_FIG = ROOT / "LaTeX" / "figures" / "generated"
WORK = ROOT / "results" / "paper_case_studies" / "quantitative_elements"

SYSTEM_ORDER = ("kundur", "ieee39", "npcc", "gbnetwork")
SYSTEM_LABEL = {
    "kundur": "Kundur",
    "ieee39": "IEEE 39",
    "npcc": "NPCC",
    "gbnetwork": "GBnetwork",
}
CASE_OUTPUT_DIR = {
    "kundur": CASE_STUDIES / "kundur" / "paper_outputs",
    "ieee39": CASE_STUDIES / "ieee39_load_shedding" / "paper_outputs",
    "npcc": CASE_STUDIES / "npcc_full_load_shedding" / "paper_outputs",
    "gbnetwork": CASE_STUDIES / "gbnetwork_load_shedding" / "paper_outputs",
}
FAMILY_ORDER = (
    "load/pump disconnection",
    "fixed-PF RES ramp",
    "plant/generator trip",
    "export reduction",
    "shunt/reactor action",
)
FAMILY_LABEL = {
    "load/pump disconnection": "Load/pump disconnection",
    "fixed-PF RES ramp": "Fixed-PF RES ramp",
    "plant/generator trip": "Plant/generator trip",
    "export reduction": "Export reduction",
    "shunt/reactor action": "Shunt/reactor action",
}


def _bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _tex(value: object) -> str:
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in repl.items():
        text = text.replace(src, dst)
    return text


def _fmt_num(value: object, decimals: int = 1) -> str:
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(f):
        return "--"
    return f"{f:.{decimals}f}"


def _fmt_int(value: object) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(f):
        return "--"
    return str(int(round(f)))


def _short_event(event_id: object) -> str:
    text = str(event_id)
    text = text.replace("load_shed:", "LS ")
    text = text.replace("fixed_pf:", "PF ")
    text = text.replace("gen_trip:", "GT ")
    text = text.replace("plant_trip:", "PT ")
    text = text.replace("_shed", "")
    text = text.replace("_", " ")
    return text


def _short_control(control: object) -> str:
    text = str(control)
    text = text.replace("absorb_load", "Absorb L")
    text = text.replace("absorb:", "Absorb ")
    text = text.replace("absorb", "Absorb ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_required_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    df = pd.read_csv(path)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df


def _seed_level_validation(tds: pd.DataFrame) -> pd.DataFrame:
    tds = tds.copy()
    for col in [
        "predicted_trip",
        "actual_trip",
        "data_limited",
        "false_positive",
        "false_negative",
        "event_screen_flagged",
    ]:
        if col in tds:
            tds[col] = _bool(tds[col])

    rows = []
    for (case, family, seed), group in tds.groupby(["case", "scenario_family", "seed_event"]):
        pred_set = set(group.loc[group["predicted_trip"], "asset"].astype(str))
        actual_set = set(group.loc[group["actual_trip"], "asset"].astype(str))
        data_limited = bool(group["data_limited"].any())
        any_pred = bool(pred_set)
        any_actual = bool(actual_set)
        union = pred_set | actual_set
        overlap = np.nan if data_limited or not union else len(pred_set & actual_set) / len(union)
        rows.append(
            {
                "case": case,
                "scenario_family": family,
                "seed_event": seed,
                "any_predicted": any_pred,
                "any_actual": any_actual,
                "data_limited": data_limited,
                "unsafe_fn": (not any_pred) and any_actual and not data_limited,
                "conservative_fp": any_pred and (not any_actual) and not data_limited,
                "trip_set_overlap": overlap,
            }
        )
    return pd.DataFrame(rows)


def build_validation_summary() -> pd.DataFrame:
    systems = _load_required_csv(DATASET / "systems.csv", ["case", "candidate_events", "screen_runtime_s"])
    library = _load_required_csv(
        DATASET / "scenario_library.csv",
        ["case", "event_id", "event_family", "max_k"],
    )
    tds = _load_required_csv(
        DATASET / "tds_validation.csv",
        ["case", "scenario_family", "seed_event", "asset", "predicted_trip", "actual_trip", "data_limited"],
    )
    seed = _seed_level_validation(tds)
    runtime_by_case = systems.set_index("case")["screen_runtime_s"].to_dict()
    system_tds_runtime_by_case = (
        systems.set_index("case")["tds_runtime_s"].to_dict()
        if "tds_runtime_s" in systems.columns
        else {}
    )

    rows = []
    for case in SYSTEM_ORDER:
        for family in FAMILY_ORDER:
            lib_f = library[(library["case"] == case) & (library["event_family"] == family)].copy()
            seed_f = seed[(seed["case"] == case) & (seed["scenario_family"] == family)].copy()
            if lib_f.empty and seed_f.empty:
                continue

            dangerous = int(seed_f["any_actual"].sum()) if not seed_f.empty else 0
            unsafe_fn = int(seed_f["unsafe_fn"].sum()) if not seed_f.empty else 0
            conservative_fp = int(seed_f["conservative_fp"].sum()) if not seed_f.empty else 0
            overlap_values = seed_f["trip_set_overlap"].dropna()
            overlap = np.nan if overlap_values.empty else 100.0 * float(overlap_values.mean())
            if seed_f.empty:
                tds_runtime = "--"
            elif "tds_runtime_s" in tds.columns:
                seed_runtime = (
                    tds[
                        (tds["case"] == case)
                        & (tds["scenario_family"] == family)
                    ]
                    .groupby("seed_event")["tds_runtime_s"]
                    .first()
                    .dropna()
                )
                tds_runtime = "--" if seed_runtime.empty else _fmt_num(float(seed_runtime.median()), 2)
            else:
                total_tds = system_tds_runtime_by_case.get(case)
                tds_runtime = "--" if total_tds is None or not math.isfinite(float(total_tds)) else "stored total"

            if dangerous:
                ranked = lib_f.sort_values("max_k", ascending=False)
                top_n = max(1, int(math.ceil(0.5 * len(ranked))))
                top_seeds = set(ranked.head(top_n)["event_id"].astype(str))
                dangerous_seeds = set(seed_f.loc[seed_f["any_actual"], "seed_event"].astype(str))
                captured = len(top_seeds & dangerous_seeds)
                topk = f"{captured}/{dangerous}"
            else:
                topk = "--"

            rows.append(
                {
                    "System / scenario family": f"{SYSTEM_LABEL[case]} / {FAMILY_LABEL[family]}",
                    "Events screened": int(len(lib_f)),
                    "TDS cases": int(seed_f["seed_event"].nunique()),
                    "Dangerous TDS cases": dangerous,
                    "Unsafe FN": unsafe_fn,
                    "Conservative FP": conservative_fp,
                    "Trip-set overlap": "--" if np.isnan(overlap) else f"{overlap:.0f}\\%",
                    "Top-k capture": topk,
                    "Screen runtime (s)": _fmt_num(runtime_by_case.get(case), 1),
                    "TDS runtime (s)": tds_runtime,
                    "source": "screen+nonlinear_tds",
                }
            )

    # The ACTIVSg2000 case is a mechanism-replica event log, not a screen-vs-TDS
    # validation sweep. Include it explicitly so it is not conflated with the
    # nonlinear validation rows above.
    event_count = "--"
    tds_cases = "--"
    dangerous = "--"
    if (REPLICA / "screen_event_library.csv").exists():
        ev = pd.read_csv(REPLICA / "screen_event_library.csv")
        event_count = str(len(ev))
    if (REPLICA / "summary.json").exists():
        summary = json.load((REPLICA / "summary.json").open())
        tds_cases = "1"
        dangerous = "1" if summary.get("blackout_detected") else "0"
    rows.append(
        {
            "System / scenario family": "ACTIVSg2000 / Mechanism replica log",
            "Events screened": event_count,
            "TDS cases": tds_cases,
            "Dangerous TDS cases": dangerous,
            "Unsafe FN": "--",
            "Conservative FP": "--",
            "Trip-set overlap": "--",
            "Top-k capture": "--",
            "Screen runtime (s)": "--",
            "TDS runtime (s)": "--",
            "source": "replication_log",
        }
    )
    return pd.DataFrame(rows)


def _event_time(events: list[dict], categories: set[str]) -> float | None:
    times = []
    for event in events:
        if str(event.get("category", "")) in categories:
            try:
                times.append(float(event.get("time_s")))
            except (TypeError, ValueError):
                continue
    return min(times) if times else None


def _variant_outcome_map() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}

    def store(
        variant: str,
        *,
        trips: int,
        blackout: bool,
        first_trip: float | None = None,
        blackout_time: float | None = None,
        lost_p_mw: float | None = None,
        lost_q_mvar: float | None = None,
    ) -> None:
        if trips == 0 and not blackout:
            result = "0 trips; no blackout"
        elif blackout:
            suffix = f" at {blackout_time:.1f} s" if blackout_time is not None and math.isfinite(blackout_time) else ""
            result = f"{trips} trips; blackout{suffix}"
        else:
            result = f"{trips} trips; no blackout"
        out[variant] = {
            "collector_trips": trips,
            "blackout_detected": blackout,
            "first_trip_time_s": first_trip,
            "blackout_time_s": blackout_time,
            "total_lost_p_mw": lost_p_mw,
            "total_lost_q_absorption_mvar": lost_q_mvar,
            "result": result,
        }

    if VARIANTS.exists():
        variants = pd.read_csv(VARIANTS)
        for _, row in variants.iterrows():
            trips = int(row.get("collector_trips", 0) or 0)
            blackout = bool(row.get("blackout_detected", False))
            store(
                str(row["variant"]),
                trips=trips,
                blackout=blackout,
                first_trip=float(row["first_trip_time_s"]) if pd.notna(row.get("first_trip_time_s")) else None,
                blackout_time=float(row["blackout_time_s"]) if pd.notna(row.get("blackout_time_s")) else None,
                lost_p_mw=float(row["total_lost_p_mw"]) if pd.notna(row.get("total_lost_p_mw")) else None,
                lost_q_mvar=float(row["total_lost_q_absorption_mvar"])
                if pd.notna(row.get("total_lost_q_absorption_mvar"))
                else None,
            )

    variant_root = ROOT / "results" / "replication_subsection" / "variants"
    for summary_path in variant_root.glob("*/summary.json"):
        variant = summary_path.parent.name
        try:
            summary = json.load(summary_path.open())
        except Exception:
            continue
        events = []
        events_path = summary_path.parent / "events.json"
        if events_path.exists():
            try:
                events = json.load(events_path.open())
            except Exception:
                events = []
        trips = int(summary.get("collector_trips", 0) or 0)
        if events:
            trips = sum(1 for event in events if str(event.get("category", "")) == "protection_trip")
        blackout = bool(summary.get("blackout_detected", False))
        if events:
            blackout = blackout or any(
                str(event.get("category", "")) in {"system_blackout_declared", "island_blackout_declared"}
                for event in events
            )
        first_trip = (
            summary.get("first_trip_time_s")
            or summary.get("first_trip")
            or _event_time(events, {"protection_trip"})
        )
        blackout_time = (
            summary.get("blackout_time_s")
            or summary.get("blackout_time")
            or _event_time(events, {"system_blackout_declared", "island_blackout_declared"})
        )
        store(
            variant,
            trips=trips,
            blackout=blackout,
            first_trip=float(first_trip) if first_trip is not None else None,
            blackout_time=float(blackout_time) if blackout_time is not None else None,
            lost_p_mw=summary.get("total_lost_p_mw"),
            lost_q_mvar=summary.get("total_lost_q_absorption_mvar"),
        )
    return out


def build_ablation_summary() -> pd.DataFrame:
    screen = _load_required_csv(
        DATASET / "ablation_summary.csv",
        ["row", "worst_k", "predicted_cascade_size", "classification", "minimum_mvar_mitigation"],
    )
    by_row = {str(row["row"]): row for _, row in screen.iterrows()}
    outcomes = _variant_outcome_map()

    def screen_value(row_name: str, col: str) -> str:
        row = by_row.get(row_name)
        if row is None:
            return "--"
        return _fmt_num(row.get(col), 1) if col == "worst_k" else _fmt_int(row.get(col))

    def outcome(variant: str, default: str = "--") -> str:
        return str(outcomes.get(variant, {}).get("result", default))

    rows = [
        {
            "Scenario": "Baseline",
            "Evidence": "Replica TDS + screen",
            "Worst max K": screen_value("baseline", "worst_k"),
            "Predicted fixed point": screen_value("baseline", "predicted_cascade_size"),
            "Outcome": outcome("baseline"),
            "Main interpretation": "Protection trips remove reactive absorption and propagate.",
        },
        {
            "Scenario": "No collector OV protection",
            "Evidence": "Replica TDS",
            "Worst max K": "--",
            "Predicted fixed point": "--",
            "Outcome": outcome("no_collector_ov_protection", "No matching artifact"),
            "Main interpretation": "Voltage can rise, but the relay feedback path is removed.",
        },
        {
            "Scenario": "Delayed protection",
            "Evidence": "Replica TDS + screen",
            "Worst max K": screen_value("delayed protection", "worst_k"),
            "Predicted fixed point": screen_value("delayed protection", "predicted_cascade_size"),
            "Outcome": outcome("delayed_protection", "Screen-level only"),
            "Main interpretation": "Extra dwell time lowers short-window trip pressure.",
        },
        {
            "Scenario": "Reactive absorption preserved after trip",
            "Evidence": "Replica TDS + screen",
            "Worst max K": screen_value("preserved reactive absorption", "worst_k"),
            "Predicted fixed point": screen_value("preserved reactive absorption", "predicted_cascade_size"),
            "Outcome": outcome("preserved_q_absorption"),
            "Main interpretation": "Removing MW alone is not the same mechanism.",
        },
        {
            "Scenario": "Voltage-mode RES instead of fixed PF",
            "Evidence": "Replica TDS + screen",
            "Worst max K": screen_value("voltage-mode RES instead of fixed-PF", "worst_k"),
            "Predicted fixed point": screen_value("voltage-mode RES instead of fixed-PF", "predicted_cascade_size"),
            "Outcome": outcome("voltage_mode_res"),
            "Main interpretation": "Fast local reactive response restores margin.",
        },
        {
            "Scenario": "Automatic shunt / fast reactor response",
            "Evidence": "Screen-level",
            "Worst max K": screen_value("stronger/faster shunt support", "worst_k"),
            "Predicted fixed point": screen_value("stronger/faster shunt support", "predicted_cascade_size"),
            "Outcome": "No matching replica run",
            "Main interpretation": "MVAr support helps only when available inside the relay window.",
        },
        {
            "Scenario": "UEL active at key units",
            "Evidence": "Unavailable",
            "Worst max K": "--",
            "Predicted fixed point": "--",
            "Outcome": "Model unavailable",
            "Main interpretation": "UEL behavior was not available as a validated ablation artifact.",
        },
        {
            "Scenario": "Missing collector telemetry",
            "Evidence": "Robust screen",
            "Worst max K": screen_value("full observability vs missing data", "worst_k"),
            "Predicted fixed point": screen_value("full observability vs missing data", "predicted_cascade_size"),
            "Outcome": "Data-limited screen",
            "Main interpretation": "Robust result becomes data limited without a protected-side envelope.",
        },
    ]
    return pd.DataFrame(rows)


@dataclass
class MitigationCase:
    case: str
    label: str
    seed: str
    binding_assets: str
    controls: str
    requested_mvar: float
    response_window: str
    slack_before: float
    eta: float
    feasible: bool
    selections: list[dict]
    binding_points: list[dict]


def _load_mitigation_case(case: str) -> MitigationCase:
    path = CASE_OUTPUT_DIR[case] / "mitigation.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing mitigation artifact: {path}")
    data = json.load(path.open())
    result = data["mitigation_result"]
    selections = result.get("selections", [])
    binding = result.get("binding_constraints", [])
    seed = ", ".join(_short_event(x) for x in result.get("seed_ids", [])) or "--"
    assets = sorted({str(x.get("protected_asset_id", "--")).replace("load", "L") for x in binding})
    times = [float(x.get("time_s", np.nan)) for x in binding if math.isfinite(float(x.get("time_s", np.nan)))]
    if times:
        response = f"{min(times):.2f}--{max(times):.2f} s" if min(times) != max(times) else f"{min(times):.2f} s"
    else:
        response = "--"

    controls = []
    for item in selections:
        label = _short_control(item.get("control_id", "--"))
        mvar = float(item.get("magnitude_mvar", 0.0))
        sat = "*" if bool(item.get("saturated", False)) else ""
        controls.append(f"{label} {mvar:.1f}{sat}")

    disturbance = np.array(data.get("disturbance_erosion", []), dtype=float)
    slack_before = max(float(np.nanmax(disturbance) - 0.95), 0.0) if disturbance.size else np.nan
    return MitigationCase(
        case=case,
        label=SYSTEM_LABEL[case],
        seed=seed,
        binding_assets=", ".join(assets) if assets else "--",
        controls="; ".join(controls) if controls else "--",
        requested_mvar=float(result.get("required_mvar_total", np.nan)),
        response_window=response,
        slack_before=slack_before,
        eta=float(result.get("slack_eta", np.nan)),
        feasible=bool(result.get("feasible", False)),
        selections=selections,
        binding_points=binding,
    )


def build_mitigation_summary() -> pd.DataFrame:
    cases = [_load_mitigation_case(case) for case in SYSTEM_ORDER]
    rows = []
    for c in cases:
        rows.append(
            {
                "Seed event": f"{c.label}: {c.seed}",
                "Binding protected asset": c.binding_assets,
                "Selected fast controls": c.controls,
                "Requested MVAr": _fmt_num(c.requested_mvar, 1),
                "Response / binding time": c.response_window,
                "Slack before -> after": f"{_fmt_num(c.slack_before, 1)} $\\to$ {_fmt_num(c.eta, 1)}",
                "Interpretation": "Certified by the LP; residual slack is zero."
                if c.feasible and c.eta <= 1e-8
                else "Positive slack remains; available controls are insufficient.",
            }
        )
    return pd.DataFrame(rows)


def _write_latex_table(
    df: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    columns: str,
    size: str = r"\scriptsize",
    note: str | None = None,
) -> None:
    def header_cell(value: object) -> str:
        text = str(value)
        if "\\" in text or "$" in text:
            return text
        return _tex(text)

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        size,
        rf"\begin{{tabularx}}{{\textwidth}}{{{columns}}}",
        r"\toprule",
        " & ".join(header_cell(c) for c in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, str) and ("$" in value or r"\%" in value or r"\to" in value):
                cells.append(value)
            else:
                cells.append(_tex(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabularx}"])
    if note:
        lines.append(rf"\vspace{{0.25em}}\par\footnotesize {note}")
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines) + "\n")


def write_tables() -> dict[str, Path]:
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    validation = build_validation_summary()
    validation_csv = WORK / "tab_validation_summary.csv"
    validation.to_csv(validation_csv, index=False)
    validation_tex = OUT_TABLE / "tab_validation_summary.tex"
    _write_latex_table(
        validation.drop(columns=["source"]),
        validation_tex,
        caption=(
            "Validation summary for the protection-aware screen. A false negative is unsafe if "
            "nonlinear TDS produces a protected pickup or trip that the robust screen certified as safe."
        ),
        label="tab:validation_summary",
        columns=(
            r">{\raggedright\arraybackslash}X"
            r"rrrrrrrcc"
        ),
        note=(
            r"Trip-set overlap is the mean Jaccard overlap between predicted and nonlinear trip sets "
            r"over non-data-limited seeds with at least one predicted or actual trip. Top-\(k\) capture "
            r"uses the top half of events ranked by \(K^{\rm pk}\). Runtime columns report one screen "
            r"pass for the full system and median nonlinear TDS wall-clock time per validated seed; "
            r"\(--\) means that family was screened but not validated with a matching nonlinear switching model."
        ),
    )

    ablation = build_ablation_summary()
    ablation_csv = WORK / "tab_ablation_summary.csv"
    ablation.to_csv(ablation_csv, index=False)
    ablation_tex = OUT_TABLE / "tab_ablation_summary.tex"
    ablation_for_tex = ablation.rename(
        columns={
            "Worst max K": r"Worst \(\max_i K_{ij}^{\rm pk}\)",
        }
    )
    _write_latex_table(
        ablation_for_tex,
        ablation_tex,
        caption=(
            "Causal ablations on the mechanism replica and the associated screen-level intervention model. "
            "The evidence column separates physical replica runs from screen-only sensitivity studies."
        ),
        label="tab:ablation_summary",
        columns=(
            r">{\raggedright\arraybackslash}p{0.19\textwidth}"
            r">{\raggedright\arraybackslash}p{0.12\textwidth}"
            r"cc"
            r">{\raggedright\arraybackslash}p{0.15\textwidth}"
            r">{\raggedright\arraybackslash}X"
        ),
        size=r"\scriptsize",
        note=(
            r"\(--\) denotes an unsupported or unavailable screen quantity, not a zero effect. "
            r"Replica TDS rows are physical ACTIVSg2000 mechanism-replica variants; screen-level rows "
            r"are finite-window sensitivity studies without a matching nonlinear variant."
        ),
    )

    mitigation = build_mitigation_summary()
    mitigation_csv = WORK / "tab_mitigation_summary.csv"
    mitigation.to_csv(mitigation_csv, index=False)
    mitigation_tex = OUT_TABLE / "tab_mitigation_summary.tex"
    _write_latex_table(
        mitigation,
        mitigation_tex,
        caption=(
            "Mitigation LP output for representative high-risk seed events. Positive slack means "
            "available controls are insufficient under the modeled uncertainty margin."
        ),
        label="tab:mitigation_summary",
        columns=(
            r">{\raggedright\arraybackslash}X"
            r">{\raggedright\arraybackslash}p{0.12\textwidth}"
            r">{\raggedright\arraybackslash}X"
            r"cc"
            r">{\raggedright\arraybackslash}p{0.13\textwidth}"
            r">{\raggedright\arraybackslash}X"
        ),
        note=(
            r"An asterisk on a control indicates that the LP saturated that control. "
            r"Slack is reported before mitigation and after the LP optimum."
        ),
    )
    return {
        "validation_csv": validation_csv,
        "validation_tex": validation_tex,
        "ablation_csv": ablation_csv,
        "ablation_tex": ablation_tex,
        "mitigation_csv": mitigation_csv,
        "mitigation_tex": mitigation_tex,
    }


def make_mitigation_figure() -> list:
    configure_style(prefer_usetex=True)
    cases = [_load_mitigation_case(case) for case in SYSTEM_ORDER]
    fig = plt.figure(figsize=(7.0, 3.65))
    ax1 = fig.add_axes([0.10, 0.18, 0.43, 0.70])
    ax2 = fig.add_axes([0.62, 0.18, 0.34, 0.70])

    y = np.arange(len(cases))
    colors = [PALETTE["control"], PALETTE["data"], PALETTE["risky"], PALETTE["safe"]]
    for idx, case in enumerate(cases):
        left = 0.0
        for j, sel in enumerate(case.selections):
            width = float(sel.get("magnitude_mvar", 0.0))
            hatch = "//" if bool(sel.get("saturated", False)) else None
            ax1.barh(
                idx,
                width,
                left=left,
                height=0.55,
                color=colors[j % len(colors)],
                edgecolor="white",
                linewidth=0.8,
                hatch=hatch,
            )
            if width >= 18:
                ax1.text(
                    left + width / 2,
                    idx,
                    _short_control(sel.get("control_id", "")),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="white",
                )
            left += width
        ax1.text(
            left + max(4.0, max(c.requested_mvar for c in cases) * 0.015),
            idx,
            f"{case.requested_mvar:.1f}",
            ha="left",
            va="center",
            fontsize=7.2,
            color=PALETTE["charcoal"],
        )

    ax1.set_yticks(y)
    ax1.set_yticklabels([case.label for case in cases])
    ax1.invert_yaxis()
    ax1.set_xlabel("requested fast absorption (MVAr)")
    ax1.set_title("LP control request", loc="left", fontsize=9.2, fontweight="bold", color=PALETTE["charcoal"])
    soften_axes(ax1, grid=True)

    for idx, case in enumerate(cases):
        points = case.binding_points or []
        valid_points = []
        for point in points:
            try:
                t = float(point.get("time_s", np.nan))
                slack = max(float(point.get("slack", 0.0)), 0.0)
            except (TypeError, ValueError):
                continue
            if math.isfinite(t):
                valid_points.append((t, slack))
        if not valid_points:
            continue
        xs = [t for t, _ in valid_points]
        slacks = [slack for _, slack in valid_points]
        color = PALETTE["trip"] if case.eta > 1e-8 else PALETTE["control"]
        ax2.plot(
            [min(xs), max(xs)],
            [idx, idx],
            color=color,
            linewidth=2.6 if case.eta > 1e-8 else 1.3,
            alpha=0.72,
            solid_capstyle="round",
            zorder=2,
        )
        if case.eta > 1e-8:
            peak_idx = int(np.argmax(slacks))
            ax2.scatter(
                [xs[peak_idx]],
                [idx],
                s=62,
                facecolors=PALETTE["trip"],
                edgecolors=PALETTE["trip"],
                linewidths=1.0,
                zorder=4,
            )
        else:
            sample_x = xs if len(xs) <= 4 else [min(xs), max(xs)]
            ax2.scatter(
                sample_x,
                [idx] * len(sample_x),
                s=38,
                facecolors="white",
                edgecolors=PALETTE["control"],
                linewidths=1.0,
                zorder=4,
            )
        ax2.text(
            1.03,
            idx,
            rf"$\eta^\star={case.eta:.1f}$",
            ha="left",
            va="center",
            fontsize=7.0,
            color=PALETTE["trip"] if case.eta > 1e-8 else PALETTE["safe"],
        )
    ax2.axvspan(0, 0.05, color=PALETTE["light_red"], alpha=0.45, zorder=0)
    ax2.text(
        0.065,
        0.88,
        "relay dwell",
        transform=ax2.transAxes,
        rotation=90,
        ha="center",
        va="top",
        fontsize=6.8,
        color=PALETTE["trip"],
    )
    ax2.set_xlim(0, 1.32)
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("binding time (s)")
    ax2.set_title("Binding relay-window constraints", loc="left", fontsize=9.2, fontweight="bold", color=PALETTE["charcoal"])
    soften_axes(ax2, grid=True)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=PALETTE["control"], markersize=6, label="feasible binding"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["trip"], markeredgecolor=PALETTE["trip"], markersize=6, label="positive slack"),
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=PALETTE["charcoal"], hatch="//", label="saturated control"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.57, 0.985), ncol=3, frameon=False, fontsize=7.0)

    return save_figure(fig, "fig10_mitigation_lp_outputs", OUT_FIG, ("pdf",), outline_text=False)


def main() -> None:
    table_paths = write_tables()
    fig_results = make_mitigation_figure()
    manifest = {
        "tables": {k: str(v.relative_to(ROOT)) for k, v in table_paths.items()},
        "figures": [{"path": str(r.path.relative_to(ROOT)), "sha256": r.sha256} for r in fig_results],
        "inputs": [
            str((DATASET / name).relative_to(ROOT))
            for name in [
                "systems.csv",
                "scenario_library.csv",
                "tds_validation.csv",
                "ablation_summary.csv",
            ]
        ]
        + [str((CASE_OUTPUT_DIR[case] / "mitigation.json").relative_to(ROOT)) for case in SYSTEM_ORDER],
    }
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Generated:")
    for path in table_paths.values():
        print(f"  {path.relative_to(ROOT)}")
    for result in fig_results:
        print(f"  {result.path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
