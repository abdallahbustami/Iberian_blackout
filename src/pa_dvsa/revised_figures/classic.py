"""Figures 5 to 13 of the paper, plotted from results/revised/ (driver: scripts/make_paper_figures.py)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
import numpy as np
import pandas as pd

from ..paper_case_studies.style import PALETTE, configure_style, panel_label, soften_axes
from .data import REPO_ROOT, RESULTS, Results, _bool, read_csv, read_json

PAPER_FIG_DIR = REPO_ROOT / "paper" / "figures"
SYSTEMS = ("kundur", "ieee39", "npcc", "gbnetwork")
SYSTEM_LABELS = {"kundur": "Kundur", "ieee39": "IEEE 39", "npcc": "NPCC", "gbnetwork": "GBnetwork"}
SYSTEM_COLORS = {"kundur": PALETTE["control"], "ieee39": PALETTE["safe"], "npcc": PALETTE["risky"],
                 "gbnetwork": PALETTE["trip"]}
SYSTEM_MARKERS = {"kundur": "o", "ieee39": "s", "npcc": "^", "gbnetwork": "D"}

GROUP = {
    "load_disconnection": "load disconnection",
    "fixed_pf_ramp": "fixed-PF ramp",
    "plant_trip": "plant / generator trip",
    "generator_trip": "plant / generator trip",
    "generation_redispatch": "export reduction",
    "shunt_reactor_disconnection": "shunt / line switching",
    "capacitor_switching": "shunt / line switching",
    "line_energization": "shunt / line switching",
}
GROUP_ORDER = ("load disconnection", "fixed-PF ramp", "plant / generator trip", "export reduction",
               "shunt / line switching")
GROUP_COLOR = {
    "load disconnection": PALETTE["trip"],
    "fixed-PF ramp": PALETTE["risky"],
    "plant / generator trip": PALETTE["data"],
    "export reduction": PALETTE["control"],
    "shunt / line switching": PALETTE["safe"],
}


def configure() -> None:
    configure_style(prefer_usetex=True)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in ("pdf", "png"):
        p = out_dir / f"{stem}.{fmt}"
        fig.savefig(p, dpi=300 if fmt == "png" else None)
        paths.append(p)
    plt.close(fig)
    if PAPER_FIG_DIR.is_dir():
        _copy_outlined(out_dir / f"{stem}.pdf", PAPER_FIG_DIR / f"{stem}.pdf")
    return paths


def _copy_outlined(src: Path, dst: Path) -> None:
    gs = shutil.which("gs")
    if gs is None:
        print(f"warning: Ghostscript not found, {dst.name} copied with fonts")
        shutil.copy2(src, dst)
        return
    subprocess.run([gs, "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pdfwrite", "-dNoOutputFonts",
                    "-dCompatibilityLevel=1.5", "-dAutoRotatePages=/None", "-o", str(dst), str(src)], check=True)


def _resize_rows(height_in: float, top: float, bottom: float, ratios: list[float], hspace: float,
                 scale: list[float]) -> tuple[float, dict]:
    n = len(ratios)
    avg = (top - bottom) * height_in / (n + hspace * (n - 1))
    rows = [avg * n * r / sum(ratios) * f for r, f in zip(ratios, scale)]
    gap = hspace * avg
    top_in, bottom_in = (1 - top) * height_in, bottom * height_in
    new_h = top_in + bottom_in + sum(rows) + gap * (n - 1)
    return new_h, {"top": 1 - top_in / new_h, "bottom": bottom_in / new_h, "height_ratios": rows,
                   "hspace": gap / (sum(rows) / n)}


def _num(text: str) -> str:
    m = re.search(r"(\d+)$", str(text))
    return str(int(m.group(1))) if m else str(text)


def event_name(event_id: str) -> str:
    fam, _, dev = str(event_id).partition(":")
    if fam == "load_disconnection":
        return f"load {_num(dev)}"
    if fam == "plant_trip":
        return f"plant {_num(dev)} trip"
    if fam == "fixed_pf_ramp":
        return f"plant {_num(dev)} ramp"
    if fam == "generator_trip":
        return "generator trip"
    if fam == "shunt_reactor_disconnection":
        return "reactor disconnection"
    if fam == "capacitor_switching":
        return "capacitor switching"
    if fam == "line_energization":
        return "line energization"
    if fam == "generation_redispatch":
        return "export reduction"
    return str(event_id)


def _seed_tables(res: Results) -> dict[str, pd.DataFrame]:
    out = {}
    for slug in SYSTEMS:
        t = res.sys(slug).cascade_table("tds_seeds.csv", res.cascade)
        for c in ("dangerous", "predicted_dangerous", "certified_no_secondary"):
            t[c] = _bool(t[c]).fillna(False)
        out[slug] = t
    return out


def fig_validation(res: Results, out_dir: Path) -> list[Path]:
    seeds = _seed_tables(res)
    counts = {}
    for slug, t in seeds.items():
        d, p, c = t["dangerous"], t["predicted_dangerous"], t["certified_no_secondary"]
        counts[slug] = {"tp": int((d & p).sum()), "fp": int((~d & p).sum()), "nc": int((~d & ~p & ~c).sum()),
                        "tn": int((~d & ~p & c).sum()), "fn": int((d & ~p).sum()), "total": int(len(t))}
    tot = {k: sum(c[k] for c in counts.values()) for k in ("tp", "fp", "nc", "tn", "fn", "total")}
    segments = (("tp", PALETTE["trip"], "white"), ("fp", PALETTE["risky"], PALETTE["charcoal"]),
                ("nc", PALETTE["background"], PALETTE["charcoal"]), ("tn", PALETTE["safe"], "white"))
    fig = plt.figure(figsize=(7.2, 1.75))
    panel_w = 0.215
    gap = (0.98 - 0.04 - 4 * panel_w) / 3.0
    fn_slot = 0.16

    for idx, slug in enumerate(SYSTEMS):
        x0 = 0.04 + idx * (panel_w + gap)
        title_ax = fig.add_axes([x0, 0.82, panel_w, 0.13])
        bar_ax = fig.add_axes([x0, 0.47, panel_w, 0.33])
        for ax in (title_ax, bar_ax):
            ax.set_axis_off()
        c = counts[slug]
        title_ax.text(0.0, 0.45, SYSTEM_LABELS[slug], ha="left", va="center", fontsize=9.6, fontweight="bold",
                      color=PALETTE["charcoal"])
        title_ax.text(1.0, 0.45, f"$n={c['total']}$", ha="right", va="center", fontsize=8.0,
                      color=PALETTE["charcoal"])
        bar_ax.add_patch(Rectangle((0.0, 0.1), 1.0, 0.8, transform=bar_ax.transAxes, facecolor="white",
                                   edgecolor=PALETTE["grid"], linewidth=0.7))
        denom = max(1, c["tp"] + c["fp"] + c["nc"] + c["tn"])
        cursor = 0.0
        small: list[list] = []
        for key, color, tcolor in segments:
            w = (1.0 - fn_slot) * c[key] / denom
            if w <= 0:
                continue
            bar_ax.add_patch(Rectangle((cursor, 0.1), w, 0.8, transform=bar_ax.transAxes, facecolor=color,
                                       edgecolor="white", linewidth=0.8))
            if w >= 0.08:
                bar_ax.text(cursor + w / 2, 0.5, f"{c[key]}", transform=bar_ax.transAxes, ha="center", va="center",
                            fontsize=11.0, fontweight="bold", color=tcolor)
            else:
                small.append([cursor + w / 2, c[key]])
            cursor += w
        for i in range(1, len(small)):
            small[i][0] = max(small[i][0], small[i - 1][0] + 0.075)
        for x, n in small:
            bar_ax.text(x, 0.98, f"{n}", transform=bar_ax.transAxes, ha="center", va="bottom", fontsize=7.2,
                        color=PALETTE["charcoal"])
        bar_ax.add_patch(Rectangle((1.0 - fn_slot, 0.1), fn_slot, 0.8, transform=bar_ax.transAxes,
                                   facecolor=PALETTE["light_teal"], edgecolor=PALETTE["safe"], linewidth=1.5,
                                   linestyle=(0, (3.5, 1.5))))
        bar_ax.text(1.0 - fn_slot / 2, 0.58, f"{c['fn']}", transform=bar_ax.transAxes, ha="center", va="center",
                    fontsize=12.0, fontweight="bold", color=PALETTE["safe"])
    leg = fig.add_axes([0.04, 0.27, 0.95, 0.12])
    leg.set_axis_off()
    entries = (("dangerous, flagged", PALETTE["trip"], False), ("safe, flagged", PALETTE["risky"], False),
               ("safe, not certified", PALETTE["background"], False), ("safe, certified", PALETTE["safe"], False),
               ("dangerous, not flagged (missed)", PALETTE["safe"], True))
    xs = (0.0, 0.17, 0.31, 0.49, 0.65)
    for (label, color, dashed), lx in zip(entries, xs):
        leg.add_patch(Rectangle((lx, 0.2), 0.022, 0.6, transform=leg.transAxes,
                                facecolor=PALETTE["light_teal"] if dashed else color,
                                edgecolor=PALETTE["safe"] if dashed else "white", linewidth=1.0 if dashed else 0.6,
                                linestyle=(0, (3.5, 1.5)) if dashed else "-"))
        leg.text(lx + 0.03, 0.5, label, transform=leg.transAxes, ha="left", va="center", fontsize=7.2,
                 color=PALETTE["safe"] if dashed else PALETTE["charcoal"])
    banner = fig.add_axes([0.04, 0.04, 0.95, 0.16])
    banner.set_axis_off()
    banner.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=banner.transAxes, facecolor="#EAEDEF",
                                    edgecolor="none", boxstyle="round,pad=0.0,rounding_size=0.06",
                                    mutation_aspect=4.0))
    banner.text(0.5, 0.5, f"{tot['total']} events $\\cdot$ {tot['tp']} dangerous, all flagged $\\cdot$ "
                f"{tot['fp']} safe events flagged $\\cdot$ {tot['tn']} certified, none tripped $\\cdot$ "
                f"{tot['fn']} missed", ha="center", va="center", fontsize=8.0, color=PALETTE["charcoal"])
    return _save(fig, out_dir, "fig_validation_classic")


def _node_positions(items: list[str], x: float, lo: float = 0.06, hi: float = 0.60) -> dict[str, tuple[float, float]]:
    if not items:
        return {}
    ys = [(lo + hi) / 2] if len(items) == 1 else np.linspace(hi, lo, len(items)).tolist()
    return {item: (x, float(y)) for item, y in zip(items, ys)}


def _trim(items: list[str], limit: int) -> list[str]:
    return items if len(items) <= limit else [*items[: limit - 1], f"+{len(items) - limit + 1}"]


PICKUP_ONLY = "#E5A496"
TDS_STAR = "#9E3B32"


def fig_cascade(res: Results, out_dir: Path) -> list[Path]:
    fig = plt.figure(figsize=(7.0, 3.95))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.78], hspace=0.30, wspace=0.16, left=0.075, right=0.985,
                          top=0.91, bottom=0.165)
    node_size = 240
    y_seed = 0.33
    kmax_all = []
    for slug in SYSTEMS:
        km = res.sys(slug).fig_data("fig3_event_kmax_sorted.csv", res.cascade)
        kmax_all.append(float(km["max_k_pk_nonself"].astype(float).max()))
    k_top = 10 ** np.ceil(np.log10(max(kmax_all) * 3.0))
    for col, slug in enumerate(SYSTEMS):
        s = res.sys(slug)
        nodes = s.fig_data("fig3_cascade_nodes.csv", res.cascade).copy()
        for c_ in ("is_self", "predicted_trip", "tds_trip_in_window"):
            nodes[c_] = _bool(nodes[c_]).fillna(False)
        seed = str(nodes["seed_id"].iloc[0])
        other = nodes[~nodes["is_self"]]
        pick = other[other["asset_class"] == "pickup_predicted"]
        ax = fig.add_subplot(gs[0, col])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        x_seed, x_first, x_second = 0.12, 0.50, 0.84
        radius = 0.055
        first = pick[pick["layer"] == 1].sort_values("k_pk_seed", ascending=False)
        later = pick[pick["layer"] >= 2].sort_values(["layer", "score"], ascending=[True, False])
        tds = set(other[other["tds_trip_in_window"]]["asset_id"])
        pred_trip = set(other[other["predicted_trip"]]["asset_id"])
        kseed = dict(zip(other["asset_id"], other["k_pk_seed"]))
        klass = dict(zip(other["asset_id"], other["asset_class"]))
        first_ids = _trim([str(a) for a in first["asset_id"]], 4)
        later_ids = _trim([str(a) for a in later["asset_id"]], 4)
        dashed = False
        if not first_ids:
            near = other.sort_values("k_pk_seed", ascending=False).head(2)
            first_ids = [str(a) for a in near["asset_id"]]
            dashed = True
        pos = {"__seed__": (x_seed, y_seed)}
        pos.update(_node_positions(first_ids, x_first))
        pos.update(_node_positions(later_ids, x_second))
        for tgt in first_ids:
            yt = pos[tgt][1]
            ax.add_patch(FancyArrowPatch((x_seed + radius, y_seed), (x_first - radius, yt), transform=ax.transAxes,
                                         connectionstyle="arc3,rad=0.10",
                                         color=PALETTE["charcoal"] if dashed else PALETTE["trip"],
                                         alpha=0.45 if dashed else 0.60, linewidth=0.9, arrowstyle="-|>",
                                         mutation_scale=7, linestyle=(0, (3, 2)) if dashed else "-"))
            if not tgt.startswith("+"):
                t = ax.text(0.30, y_seed + 0.56 * (yt - y_seed), f"{float(kseed[tgt]):.2f}", transform=ax.transAxes,
                            ha="center", va="center", fontsize=6.4, color=PALETTE["charcoal"])
                t.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
        if later_ids:
            edges = s.fig_data("fig3_cascade_edges.csv", res.cascade)
            e1 = edges[edges["source_layer"] == 1]
            shown_first = [a for a in first_ids if not a.startswith("+")]
            hidden_later = [str(a) for a in later["asset_id"] if str(a) not in later_ids]
            for tgt in later_ids:
                members = hidden_later if tgt.startswith("+") else [tgt]
                srcs = sorted(set(e1[e1["target_asset"].isin(members)]["source_asset"]) & set(shown_first))
                for src in srcs:
                    ax.add_patch(FancyArrowPatch((x_first + radius, pos[src][1]), (x_second - radius, pos[tgt][1]),
                                                 transform=ax.transAxes, connectionstyle="arc3,rad=0.05",
                                                 color=PALETTE["trip"], alpha=0.28, linewidth=0.7,
                                                 arrowstyle="-|>", mutation_scale=6))
        for name, (x, y) in pos.items():
            if name == "__seed__":
                face, label = PALETTE["charcoal"], ""
            elif name.startswith("+"):
                face, label = PALETTE["trip"], name
            elif dashed:
                face = PALETTE["light_amber"] if klass[name] == "risky" else PALETTE["light_teal"]
                label = _num(name)
            else:
                face, label = (PALETTE["trip"] if name in pred_trip else PICKUP_ONLY), _num(name)
            ax.scatter([x], [y], transform=ax.transAxes, marker="o", s=node_size, facecolor=face, edgecolor="white",
                       linewidth=1.1, zorder=3, clip_on=False)
            ax.text(x, y, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.8,
                    color=PALETTE["charcoal"] if dashed and name != "__seed__" else "white", zorder=4)
            if name in tds:
                ax.scatter(x + radius * 0.95, y + radius * 0.85, transform=ax.transAxes, marker="*", s=40,
                           facecolor=TDS_STAR, edgecolor="white", linewidth=0.4, zorder=5, clip_on=False)
        n_pred = int(other["predicted_trip"].sum())
        n_tds = int(other["tds_trip_in_window"].sum())
        ax.text(0.0, 1.0, SYSTEM_LABELS[slug], transform=ax.transAxes, ha="left", va="top", fontsize=9.2,
                fontweight="bold", color=PALETTE["charcoal"])
        sub = (f"{event_name(seed)}\nno pickup predicted" if dashed else
               f"{event_name(seed)}\n{n_pred} trips predicted, {n_tds} in TDS")
        ax.text(0.0, 0.875, sub, transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
                color=PALETTE["charcoal"], linespacing=1.15)

        axk = fig.add_subplot(gs[1, col])
        km = s.fig_data("fig3_event_kmax_sorted.csv", res.cascade).copy()
        k = km["max_k_pk_nonself"].astype(float).to_numpy()
        kk = np.clip(k, 1e-3, None)
        ranks = km["rank"].to_numpy()
        axk.step(ranks, kk, where="post", color=PALETTE["grid"], linewidth=1.6, zorder=1)
        axk.scatter(ranks, kk, c=[GROUP_COLOR[GROUP[f]] for f in km["family"]], s=13, edgecolor="none", zorder=3)
        axk.axhline(1.0, color=PALETTE["trip"], linewidth=1.0, linestyle=(0, (4, 2)), zorder=2)
        top = km.iloc[0]
        axk.annotate(f"{event_name(top['event_id'])}\n$K_j^{{\\max}}={float(top['max_k_pk_nonself']):.2f}$",
                     xy=(ranks[0], kk[0]), xycoords="data", xytext=(0.30, 0.97), textcoords="axes fraction",
                     ha="left", va="top", fontsize=6.4, color=PALETTE["charcoal"],
                     arrowprops={"arrowstyle": "->", "lw": 0.6, "color": PALETTE["charcoal"], "shrinkB": 2.5})
        axk.set_yscale("log")
        axk.set_ylim(1e-3, k_top)
        axk.set_yticks([1e-3, 1e-2, 1e-1, 1, 10][: int(np.log10(k_top)) + 4])
        axk.set_xlim(0.5, len(km) + 0.5)
        axk.set_xlabel("event rank")
        if col == 0:
            axk.set_ylabel(r"largest excursion $K_j^{\max}$")
        else:
            axk.set_yticklabels([])
        soften_axes(axk, grid=True, ygrid=True)
    node_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PALETTE["trip"], markeredgecolor="white",
               markersize=8, label="trip predicted"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PICKUP_ONLY, markeredgecolor="white",
               markersize=8, label="pickup, no trip predicted"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PALETTE["light_amber"], markeredgecolor="white",
               markersize=8, label="close to pickup"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=PALETTE["light_teal"], markeredgecolor="white",
               markersize=8, label="certified"),
        Line2D([0], [0], marker="*", linestyle="", markerfacecolor=TDS_STAR, markeredgecolor="white",
               markersize=8, label="tripped in TDS")]
    fig.legend(handles=node_handles, loc="upper center", bbox_to_anchor=(0.53, 1.0), ncol=5, frameon=False,
               fontsize=7.0, handletextpad=0.3, columnspacing=1.2)
    handles = [Line2D([0], [0], marker="o", linestyle="", markerfacecolor=GROUP_COLOR[g], markeredgecolor="none",
                      markersize=5, label=g) for g in GROUP_ORDER]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=5, frameon=False, fontsize=7.0,
               handletextpad=0.2, columnspacing=1.2)
    return _save(fig, out_dir, "fig_cascade_classic")


def fig_families(res: Results, out_dir: Path) -> list[Path]:
    rows = []
    for slug in SYSTEMS:
        df = res.sys(slug).fig_data("fig5_family_summary.csv", res.cascade)
        for _, r in df.iterrows():
            rows.append({"system": slug, "group": GROUP[r["family"]], "k": float(r["worst_k_pk_nonself"])})
    fam = pd.DataFrame(rows).groupby(["system", "group"], as_index=False)["k"].max()
    h = 0.75 * 2.45
    top_in, bottom_in = 0.36, 0.40
    fig = plt.figure(figsize=(3.45, h))
    ax = fig.add_axes([0.25, bottom_in / h, 0.72, 1 - (top_in + bottom_in) / h])
    lo, hi = 1e-3, 20
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(len(GROUP_ORDER) - 0.45, -0.75)
    ax.axvspan(lo, 1, color=PALETTE["light_teal"], alpha=0.25, zorder=0)
    ax.axvspan(1, hi, color=PALETTE["light_red"], alpha=0.30, zorder=0)
    ax.axvline(1, color=PALETTE["trip"], linewidth=1.1, linestyle=(0, (5, 2.5)), zorder=2)
    ax.text(1.4e-3, -0.58, "below pickup", ha="left", va="center", fontsize=7.0, fontweight="bold", color=PALETTE["safe"])
    ax.text(hi * 0.85, -0.58, "pickup", ha="right", va="center", fontsize=7.0, fontweight="bold",
            color=PALETTE["trip"])
    offs = {"kundur": -0.24, "ieee39": -0.08, "npcc": 0.08, "gbnetwork": 0.24}
    for yi, g in enumerate(GROUP_ORDER):
        ax.hlines(yi, lo, hi, color=PALETTE["grid"], linewidth=0.6, zorder=1)
        for slug in SYSTEMS:
            r = fam[(fam["system"] == slug) & (fam["group"] == g)]
            if r.empty:
                continue
            v = float(r["k"].iloc[0])
            ax.scatter(max(v, 1.2e-3), yi + offs[slug], marker=SYSTEM_MARKERS[slug], s=46,
                       facecolor=SYSTEM_COLORS[slug], edgecolor="white", linewidth=0.8, zorder=4)
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels([g.replace(" / ", " /\n") if "/" in g else g for g in GROUP_ORDER], fontsize=7.2)
    ax.set_xlabel(r"largest excursion $K_j^{\max}$ in the family", fontsize=8.0)
    ax.tick_params(axis="x", labelsize=7.2)
    handles = [Line2D([0], [0], marker=SYSTEM_MARKERS[s], markerfacecolor=SYSTEM_COLORS[s], markeredgecolor="white",
                      linestyle="", markersize=6.5, label=SYSTEM_LABELS[s]) for s in SYSTEMS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.25 + 0.72 * 0.42, 1.0), ncol=4, frameon=False,
               fontsize=6.8, handletextpad=0.2, columnspacing=0.8)
    soften_axes(ax, grid=False)
    ax.tick_params(axis="y", length=0)
    return _save(fig, out_dir, "fig_families_classic")


def fig_mitigation(res: Results, out_dir: Path, slug: str = "npcc") -> list[Path]:
    s = res.sys(slug)
    er = s.fig_data("fig6_erosion_screen.csv", res.cascade)
    tr = s.fig_data("fig6_tds_traces.csv", res.cascade)
    env = s.fig_data("fig6_control_envelopes.csv", res.cascade)
    mit = s.csv("mitigation.csv")
    seed = str(er["seed_id"].iloc[0])
    asset = str(er["asset_id"].iloc[0])
    m = mit[(mit["seed_id"] == seed) & (mit["restriction"] == "all")].iloc[0]
    t = er["t_s"].to_numpy()
    d = er["erosion_sampled_no_mitigation"].to_numpy()
    r = er["erosion_sampled_with_mitigation"].to_numpy()
    h, lay = _resize_rows(4.35, 0.95, 0.10, [1.0, 0.62], 0.34, [0.70, 1.0])
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.45, h), gridspec_kw={"height_ratios": lay["height_ratios"],
                                                                           "hspace": lay["hspace"]})
    ax_a.fill_between(t, 1.0, d, where=d >= 1.0, color=PALETTE["light_red"], alpha=0.55, linewidth=0, zorder=1)
    ax_a.fill_between(t, r, d, where=d >= r, color=PALETTE["light_blue"], alpha=0.42, linewidth=0, zorder=1)
    ax_a.plot(t, d, color=PALETTE["trip"], linewidth=1.9, label="event, method")
    ax_a.plot(t, r, color=PALETTE["control"], linewidth=1.9, label="with LP actions, method")
    trm = tr[_bool(tr["is_most_affected"]).fillna(False) & (tr["t_rel_s"] >= 0)]
    for run, color, lab in (("no_mitigation", PALETTE["trip"], "event, TDS"),
                            ("with_mitigation", PALETTE["control"], "with LP actions, TDS")):
        g = trm[trm["run"] == run]
        trip = g["tds_trip_rel_s"].iloc[0]
        g = g[(g["t_rel_s"] <= 1.0) & ((g["t_rel_s"] <= trip) if pd.notna(trip) else True)]
        ax_a.plot(g["t_rel_s"], g["erosion"], color=color, linewidth=1.0, linestyle=(0, (3, 1.6)), label=lab)
        if pd.notna(trip) and trip <= 1.0:
            ax_a.scatter([trip], [float(g["erosion"].iloc[-1])], marker="x", s=24, color=color, linewidth=1.0,
                         zorder=5)
    ax_a.axhline(1.0, color=PALETTE["charcoal"], linewidth=0.8, linestyle=(0, (1, 1.5)), zorder=2)
    ax_a.text(0.985, 1.03, "pickup", transform=ax_a.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.6,
              color=PALETTE["charcoal"])
    ymax = max(1.25, float(np.nanmax(d))) * 1.36
    ax_a.set_ylim(min(-0.15, float(np.nanmin(r)) - 0.05), ymax)
    ax_a.set_xlim(0, 1.0)
    ax_a.text(0.98, 0.43, f"NPCC, plant {_num(asset)}\nLP actions: {float(m['mvar_total_rated']):.0f} MVAr, "
              f"$\\eta^\\star={float(m['eta']):.2f}$", transform=ax_a.get_yaxis_transform(), ha="right", va="center",
              fontsize=7.0, color=PALETTE["charcoal"], linespacing=1.1)
    ax_a.set_ylabel("normalized voltage\n" r"rise $\Delta z_i/h_i$")
    ax_a.set_xlabel("time after the event (s)")
    ax_a.legend(loc="upper right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, fontsize=6.2, handlelength=1.8,
                columnspacing=0.8, handletextpad=0.4)
    soften_axes(ax_a, grid=True)
    panel_label(ax_a, "a")

    style = {"statcom": (PALETTE["control"], "STATCOM"), "plant_q_command": (PALETTE["data"], "plant $Q$ command"),
             "avr_vref": (PALETTE["safe"], "AVR setpoint"), "fast_reactor": (PALETTE["charcoal"], "fast reactor"),
             "manual_reactor": (PALETTE["risky"], "manual reactor")}
    for cls, (color, lab) in style.items():
        g = env[(env["control_class"] == cls) & (env["t_s"] <= 1.0 + 1e-9)]
        if g.empty:
            continue
        ax_b.plot(g["t_s"], g["envelope"], linewidth=1.6, color=color,
                  label=f"{lab} ({1000 * float(g['delay_s'].iloc[0]):.0f} ms)" if float(g['delay_s'].iloc[0]) < 1
                  else f"{lab} ({float(g['delay_s'].iloc[0]):.0f} s)")
    ax_b.set_xlim(0, 1.0)
    ax_b.set_ylim(-0.03, 1.05)
    ax_b.set_ylabel("fraction of\nfull action")
    ax_b.set_xlabel("time after the command (s)")
    ax_b.legend(loc="center right", bbox_to_anchor=(1.0, 0.42), frameon=False, fontsize=6.2, ncol=1,
                handlelength=1.6)
    soften_axes(ax_b, grid=True)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.18, right=0.97, bottom=lay["bottom"], top=lay["top"])
    return _save(fig, out_dir, "fig_mitigation_classic")


REGIMES = (("transmission_only", "transmission\nonly", PALETTE["background"]),
           ("bounded_tap", "tap\n$\\pm2.5\\%$", PALETTE["light_purple"]),
           ("known_tap", "known tap,\nno $P$, $Q$", PALETTE["light_blue"]),
           ("measured_pq", "known tap,\n$P$, $Q$", PALETTE["light_amber"]),
           ("telemetry", "collector\ntelemetry", PALETTE["light_teal"]))
CLASS_ORDER = (("data_limited", "insufficient data", PALETTE["data"]),
               ("pickup_predicted", "pickup predicted", PALETTE["trip"]),
               ("risky", "close to pickup", PALETTE["risky"]),
               ("certified", "certified", PALETTE["safe"]))


def fig_observability(res: Results, out_dir: Path) -> list[Path]:
    obs = pd.concat([res.sys(s).csv("observability.csv") for s in SYSTEMS], ignore_index=True)
    summ = pd.concat([res.sys(s).csv("observability_summary.csv") for s in SYSTEMS], ignore_index=True)
    obs = obs[~_bool(obs["sensitivity_only"]).fillna(False)]
    summ = summ[~_bool(summ["sensitivity_only"]).fillna(False)]
    h, lay = _resize_rows(4.35, 0.95, 0.10, [1.08, 0.86], 0.40, [0.75, 0.80])
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.45, h), gridspec_kw={"height_ratios": lay["height_ratios"],
                                                                           "hspace": lay["hspace"]})
    rng = np.random.default_rng(7)
    allv = []
    for j, (reg, _lab, color) in enumerate(REGIMES):
        v = obs[obs["regime"] == reg]["h_lower_pu"].to_numpy(dtype=float)
        allv.extend(v.tolist())
        q10, q25, q50, q75, q90 = np.nanpercentile(v, [10, 25, 50, 75, 90])
        ax_a.vlines(j, q10, q90, color=PALETTE["charcoal"], linewidth=1.0, alpha=0.22, zorder=1)
        ax_a.vlines(j, q25, q75, color=color, linewidth=9.0, alpha=0.9, zorder=2)
        ax_a.scatter(np.full(v.size, j) + rng.normal(0, 0.045, v.size), v, s=9, color=PALETTE["charcoal"],
                     alpha=0.38, edgecolor="none", zorder=3)
        ax_a.scatter([j], [q50], marker="D", s=30, color=PALETTE["charcoal"], edgecolor="white", linewidth=0.55,
                     zorder=4)
    ax_a.set_ylim(min(allv) - 0.01, max(allv) + 0.01)
    ax_a.axhspan(ax_a.get_ylim()[0], 0, color=PALETTE["light_red"], alpha=0.22, zorder=0)
    ax_a.axhline(0, color=PALETTE["trip"], lw=1.0, ls=(0, (4, 2)))
    ax_a.text(0.5, 0.002, "pickup", ha="center", va="bottom", fontsize=6.8, color=PALETTE["trip"])
    ax_a.set_xticks(range(len(REGIMES)), [lab for _r, lab, _c in REGIMES], fontsize=6.6)
    ax_a.set_ylabel("worst-case\n" r"margin $\underline h_i$ (pu)")
    soften_axes(ax_a, grid=True)
    panel_label(ax_a, "a")
    y = np.arange(len(REGIMES)) * 0.82
    for i, (reg, _lab, _c) in enumerate(REGIMES):
        sub = summ[summ["regime"] == reg]
        n = {c: int(sub[f"n_{c}"].sum()) for c, _l, _col in CLASS_ORDER}
        total = sum(n.values())
        left = 0.0
        for c, _l, color in CLASS_ORDER:
            w = n[c] / total
            if w <= 0:
                continue
            ax_b.barh(y[i], w, left=left, height=0.66, color=color, edgecolor="none", zorder=2)
            if w >= 0.14:
                ax_b.text(left + w / 2, y[i], f"{100 * w:.0f}\\%", ha="center", va="center", fontsize=7.0,
                          color="white" if c != "risky" else PALETTE["charcoal"], fontweight="bold")
            left += w
        ax_b.text(1.02, y[i], f"{int(sub['n_seeds_certified_no_secondary'].sum())}", ha="left", va="center",
                  fontsize=6.8, color=PALETTE["charcoal"], clip_on=False)
    ax_b.text(1.02, -0.55, "certified\nevents", ha="left", va="bottom", fontsize=6.0, color=PALETTE["charcoal"],
              clip_on=False, linespacing=0.9)
    ax_b.set_yticks(y, [lab for _r, lab, _c in REGIMES], fontsize=6.8)
    ax_b.invert_yaxis()
    ax_b.set_xlim(0, 1.0)
    ax_b.set_xticks([0, 0.5, 1.0], ["0", "50", "100"])
    ax_b.set_xlabel(r"share of protected plants (\%)")
    handles = [Patch(facecolor=color, edgecolor="white", label=lab) for _c, lab, color in CLASS_ORDER]
    ax_b.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, 1.24), ncol=2, frameon=False,
                fontsize=6.4)
    soften_axes(ax_b, grid=False)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.20, right=0.89, bottom=lay["bottom"], top=lay["top"])
    return _save(fig, out_dir, "fig_observability_classic")


def fig_uncertainty(res: Results, out_dir: Path) -> list[Path]:
    unc = pd.concat([res.sys(s).csv("uncertainty.csv") for s in SYSTEMS], ignore_index=True)
    pooled = unc.groupby(["class", "level"])["n_seeds_certified_no_secondary"].sum()
    n_seeds = int(unc[(unc["class"] == "relay_threshold") & (unc["level"] == 0)]["n_seeds"].sum())
    lines = (("relay_threshold", "relay threshold", PALETTE["trip"], "-", "o"),
             ("reconstruction_error", "reconstruction error", PALETTE["risky"], "-", "o"),
             ("tap_ratio_reconstructed", "tap ratio, $z_i$ from $P$, $Q$", PALETTE["data"], "-", "o"),
             ("q_absorption", "reactive absorption", PALETTE["control"], "-", "o"),
             ("tap_ratio", "tap ratio, telemetry", PALETTE["data"], (0, (1.0, 1.4)), "s"),
             ("controller_timing", "controller timing", PALETTE["safe"], (0, (4, 2)), "o"))
    h = 0.70 * 2.85
    top_in, bottom_in = 0.40, 0.43
    fig, ax = plt.subplots(1, 1, figsize=(3.45, h))
    for cls, lab, color, ls, mk in lines:
        g = pooled.loc[cls]
        ax.plot(g.index.to_numpy() * 100, g.to_numpy(), lw=1.65, ls=ls, marker=mk, ms=3.4, color=color, label=lab,
                zorder=3)
    ax.set_xlim(-2, 104)
    ax.set_ylim(0, 125)
    ax.set_xlabel(r"uncertainty level (\%)")
    ax.set_ylabel(f"certified events (of {n_seeds})")
    fig.legend(loc="upper center", bbox_to_anchor=(0.55, 1.0), frameon=False, fontsize=6.4, ncol=3,
               columnspacing=0.8, handlelength=1.7, handletextpad=0.4)
    soften_axes(ax, grid=True)
    fig.subplots_adjust(left=0.17, right=0.97, bottom=bottom_in / h, top=1 - top_in / h)
    return _save(fig, out_dir, "fig_uncertainty_classic")


def fig_ranking(res: Results, out_dir: Path, slug: str = "gbnetwork") -> list[Path]:
    ab = res.sys(slug).ablations().set_index("row")
    rows = (("baseline", "none"), ("no_ov_protection", "no overvoltage\nrelays"),
            ("delayed_protection", "relay delay\n0.2 s"), ("preserved_absorption", "absorption\npreserved"),
            ("voltage_mode_res", "volt-var\ncontrol"))
    kv = np.array([float(ab.loc[r, "worst_k_pk_nonself"]) for r, _l in rows])
    rc = np.array([len(str(ab.loc[r, "tds_would_trip_in_window"]).split(";"))
                   if pd.notna(ab.loc[r, "tds_would_trip_in_window"]) else 0 for r, _l in rows])
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.45, 4.35), gridspec_kw={"height_ratios": [1.05, 1.0],
                                                                              "hspace": 0.42})
    y = np.arange(len(rows))
    mk, mr = float(kv.max()), float(rc.max())
    sk = mr / mk
    ax_a.barh(y, -kv * sk, height=0.55, color=PALETTE["trip"], edgecolor="white", linewidth=0.8, zorder=2)
    ax_a.barh(y, rc, height=0.55, color=PALETTE["control"], edgecolor="white", linewidth=0.8, zorder=2)
    for yi, k, n in zip(y, kv, rc):
        ax_a.text(-k * sk - mr * 0.05, yi, f"{k:.2f}", ha="right", va="center", fontsize=7.2, color=PALETTE["trip"])
        ax_a.text(n + mr * 0.05, yi, f"{n}", ha="left", va="center", fontsize=7.2, color=PALETTE["control"])
    ax_a.axvline(0, color=PALETTE["charcoal"], linewidth=0.7)
    ax_a.set_yticks(y, [l for _r, l in rows], fontsize=7.0)
    ax_a.set_xlim(-mr * 1.45, mr * 1.45)
    ax_a.set_ylim(len(rows) - 0.4, -0.95)
    ax_a.set_xticks([])
    ax_a.spines["bottom"].set_visible(False)
    ax_a.text(-mr * 0.6, -0.78, r"largest $K^{\rm pk}$", ha="center", va="bottom", fontsize=7.6,
              color=PALETTE["trip"], fontweight="bold")
    ax_a.text(mr * 0.6, -0.78, "plants tripping in TDS", ha="center", va="bottom", fontsize=7.6,
              color=PALETTE["control"], fontweight="bold")
    ax_a.text(0, len(rows) - 0.45, f"{SYSTEM_LABELS[slug]}, {event_name(ab['designated_seed'].iloc[0])}",
              ha="center", va="top", fontsize=6.8, color=PALETTE["charcoal"], fontstyle="italic")
    soften_axes(ax_a, grid=False)
    ax_a.tick_params(axis="y", length=0)
    panel_label(ax_a, "a")

    ax_b.plot([0, 100], [0, 100], color=PALETTE["grid"], linewidth=1.0, linestyle=(0, (4, 2)), label="random order")
    colors = {"ieee39": PALETTE["safe"], "npcc": PALETTE["data"], "gbnetwork": PALETTE["trip"]}
    marks = {"ieee39": "s", "npcc": "^", "gbnetwork": "D"}
    grid = np.array([0, 10, 20, 30, 40, 50, 75, 100]) / 100.0
    for sl in ("ieee39", "npcc", "gbnetwork"):
        tri = res.sys(sl).csv("triage.csv")
        fs, fc = tri["fraction_sent"].to_numpy(), tri["fraction_captured"].to_numpy()
        yv = np.array([fc[fs <= x + 1e-12].max() for x in grid])
        ax_b.plot(fs * 100, fc * 100, drawstyle="steps-post", lw=1.6, color=colors[sl], zorder=2)
        ax_b.plot(grid * 100, yv * 100, marker=marks[sl], ms=4.2, lw=0, color=colors[sl], zorder=3,
                  label=SYSTEM_LABELS[sl])
    ax_b.set_xlabel(r"events simulated, ranked by $K_j^{\max}$ (\%)")
    ax_b.set_ylabel(r"dangerous events found (\%)")
    ax_b.set_xlim(0, 105)
    ax_b.set_ylim(0, 105)
    ax_b.legend(loc="lower right", frameon=False, fontsize=6.8, ncol=1)
    soften_axes(ax_b, grid=True)
    panel_label(ax_b, "b")
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.11, top=0.95)
    return _save(fig, out_dir, "fig_ranking_classic")


def fig_replica(res: Results, out_dir: Path) -> list[Path]:
    root = RESULTS / "activsg2000"
    steps = read_json(root / "condition_buildup.json")
    base = read_json(root / "baseline" / "summary.json")
    pickup = float(steps[0]["max_collector_v_pu"]) / float(steps[0]["max_ratio_to_ov1_pickup"])
    labels = ["initial", "line 1", "line 2", "export\nreduction", "output\nreduction", "TDS\nmaximum"]
    ratios = [float(s["max_ratio_to_ov1_pickup"]) for s in steps] + [float(base["max_collector_v_pu"]) / pickup]
    b_scale = 0.75
    h, lay = _resize_rows(4.95, 0.935, 0.11, [1.0, 0.95], 0.50, [0.75, b_scale])
    fig = plt.figure(figsize=(3.45, h))
    gs = fig.add_gridspec(2, 1, left=0.24, right=0.975, **lay)
    ax_w = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0])
    floor = min(ratios) - 0.004
    ytop = 1.004
    prev = ratios[0]
    for i, r in enumerate(ratios):
        kind = "initial" if i == 0 else ("final" if i == len(ratios) - 1 else "step")
        if kind in ("initial", "final"):
            bottom, h = floor, r - floor
            color = PALETTE["charcoal"] if kind == "initial" else PALETTE["trip"]
            edge = PALETTE["charcoal"]
        else:
            bottom, h = min(prev, r), abs(r - prev)
            color = PALETTE["risky"] if r >= prev else PALETTE["safe"]
            edge = "white"
        ax_w.bar(i, max(h, 2e-5), bottom=bottom, width=0.66, color=color, edgecolor=edge, linewidth=0.8,
                 alpha=0.92, zorder=3)
        txt = {"step": f"{(r - prev) * 100:+.2f} pp", "final": f"{r:.7f}"}.get(kind, f"{r:.4f}")
        ax_w.text(i, (max(r, prev) if kind == "step" else r) + 0.0006, txt, ha="center", va="bottom", fontsize=6.4, rotation=90 if kind == "step" else 0,
                  color=PALETTE["trip"] if kind == "final" else PALETTE["charcoal"])
        prev = r
    ax_w.plot(range(len(ratios)), ratios, color=PALETTE["charcoal"], lw=0.85, marker="o", ms=3.0, mfc="white",
              zorder=4)
    ax_w.axhline(1.0, color=PALETTE["trip"], lw=1.0, ls=(0, (5, 2.4)))
    ax_w.axhspan(1.0, ytop, color=PALETTE["light_red"], alpha=0.38, zorder=0)
    ax_w.set_xticks(range(len(labels)), labels, fontsize=6.8)
    ax_w.set_ylabel(r"largest $z_i/V_i^{\rm trip}$")
    ax_w.set_ylim(floor, ytop)
    ax_w.text(-0.4, 1.0003, "pickup", ha="left", va="bottom", fontsize=6.6, color=PALETTE["trip"])
    soften_axes(ax_w, grid=True)
    panel_label(ax_w, "a")

    ev = read_csv(root / "screen" / "fig_data" / "fig9_bottom_k_events.csv")
    ev = ev[(ev["relay"] == "fast") & ev["family"].isin(GROUP)].copy()
    ev["group"] = ev["family"].map(GROUP)
    ev["k"] = ev["max_k_pk_nonself"].astype(float).clip(lower=1e-3)
    groups = [g for g in GROUP_ORDER if g in set(ev["group"])]
    for yi, g in enumerate(groups):
        v = ev[ev["group"] == g]["k"].to_numpy()
        q10, q50, q90 = np.quantile(v, [0.1, 0.5, 0.9])
        ax_p.plot([v.min(), v.max()], [yi, yi], color=PALETTE["grid"], lw=1.0, zorder=1)
        ax_p.plot([q10, q90], [yi, yi], color=PALETTE["background"], lw=8.5, solid_capstyle="round", zorder=2)
        ax_p.scatter([q50], [yi], marker="D", s=32, facecolor=PALETTE["charcoal"], edgecolor="white", linewidth=0.7,
                     zorder=4)
        n1 = int((v >= 1.0).sum())
        ax_p.text(0.985, yi, f"{n1}/{len(v)}", transform=ax_p.get_yaxis_transform(), ha="right", va="center",
                  fontsize=6.4, color=PALETTE["trip"] if n1 else PALETTE["charcoal"])
    xmax = max(60.0, float(ev["k"].max()) * 2.0)
    ax_p.axvspan(1.0, xmax, color=PALETTE["light_red"], alpha=0.30, zorder=0)
    ax_p.axvline(1.0, color=PALETTE["trip"], lw=1.0, ls=(0, (5, 2.4)), zorder=3)
    ax_p.set_xscale("log")
    ax_p.set_xlim(1e-3, xmax)
    ax_p.set_ylim(len(groups) - 0.52, -0.52)
    ax_p.set_yticks(range(len(groups)), [g.replace(" / ", " /\n") if "/" in g else g for g in groups], fontsize=6.8)
    ax_p.set_xlabel(r"largest excursion $K_j^{\max}$ of the event")
    hl = [Line2D([0], [0], color=PALETTE["background"], lw=6.0, solid_capstyle="round", label=r"10--90\% range"),
          Line2D([0], [0], marker="D", color="none", markerfacecolor=PALETTE["charcoal"], markeredgecolor="white",
                 markersize=5.0, label="median"),
          Line2D([0], [0], color=PALETTE["grid"], lw=1.0, label="min--max")]
    ax_p.legend(handles=hl, loc="upper center", bbox_to_anchor=(0.45, 1 + 0.20 / b_scale), ncol=3, frameon=False,
                fontsize=6.2, handletextpad=0.3, columnspacing=0.6)
    soften_axes(ax_p, grid=True, ygrid=False)
    panel_label(ax_p, "b")
    return _save(fig, out_dir, "fig_replica_classic")


def fig_sequence(res: Results, out_dir: Path, label: str = "final_main", zoom_from: float = 14.6) -> list[Path]:
    d = RESULTS / "iberian_sequence" / label
    st = read_csv(d / "system_traces.csv")
    col = read_csv(d / "collector_traces.csv")
    ev = read_json(d / "events.json")
    design = read_json(d / "case_design.json")
    t, tc = st["time_s"].to_numpy(), col["time_s"].to_numpy()
    names = [b["name"] for b in design["blocks"]]
    trips = [e for e in ev if e["category"] == "plant_trip"]
    gen_trips = [e for e in ev if e["category"] == "generator_trip"]
    ufls = [e for e in ev if e["category"] == "ufls"]
    t_first = trips[0]["time_s"]
    t_sep = next(e["time_s"] for e in ev if e["category"] == "out_of_step_separation")
    t_end = float(t[-1])
    prot = {p["name"]: p for p in design["plant_protection"]}
    ov1, ov2 = prot["OV1"], prot["OV2"]
    thr = float(design["separation"]["threshold_deg"])
    red = design["pre_cascade_actions"][0]
    trig_color = {"plant_OV1": PALETTE["trip"], "plant_OV2": PALETTE["data"], "plant_OVI": PALETTE["charcoal"]}

    h, lay = _resize_rows(3.55, 0.875, 0.115, [1.55, 1.0], 0.16, [0.75, 1.0])
    fig = plt.figure(figsize=(7.0, h))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.85, 1.0], wspace=0.24, left=0.068, right=0.988, **lay)
    ax_v = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[1, 0], sharex=ax_v)
    ax_g = fig.add_subplot(gs[0, 1])
    ax_f = fig.add_subplot(gs[1, 1], sharex=ax_g)

    for n in names:
        on = col[f"{n}_online"].to_numpy() > 0.5
        ax_v.plot(tc, np.where(on, col[f"{n}_v_pu"].to_numpy(), np.nan), color="#B9B1A3", lw=0.55, zorder=1)
    hv = col[[f"{n}_hv_v_pu" for n in names]].median(axis=1).to_numpy()
    ax_v.plot(tc, hv, color=PALETTE["control"], lw=1.15, zorder=2)
    for e in trips:
        ax_v.scatter([e["time_s"]], [e["measured"]["v_pu"]], marker="X", s=24, lw=0.3, edgecolor="white",
                     color=trig_color[e["trigger"]], zorder=4)
    for p in (ov1, ov2):
        ax_v.axhline(float(p["pickup"]), color=PALETTE["trip"], lw=0.75, ls=(0, (5, 2.4)), zorder=1)
        ax_v.text(0.25, float(p["pickup"]) + 0.006, rf"{float(p['pickup']):.2f} pu, {float(p['delay_s']):g} s",
                  fontsize=7.2, color=PALETTE["trip"], va="bottom", ha="left")
    ax_v.text(0.5 * t_first, 0.985, rf"fixed-PF output reduction ({float(red['size']):.0f} MW over "
              rf"{float(red['duration_s']):.0f} s)", transform=ax_v.get_xaxis_transform(), ha="center", va="top",
              fontsize=7.2, color=PALETTE["charcoal"])
    for ax in (ax_v, ax_a):
        for x in (t_first, t_sep):
            ax.axvline(x, color=PALETTE["charcoal"], lw=0.7, ls=(0, (2, 1.5)), zorder=0)
    for x, txt in ((t_first, "first trip"), (t_sep, "separation")):
        ax_v.text(x - 0.12, 1.0, txt, transform=ax_v.get_xaxis_transform(), fontsize=7.2,
                  color=PALETTE["charcoal"], ha="right", va="bottom")
    ax_v.set_ylabel("voltage (pu)")
    ax_v.set_xlim(0.0, t_end + 0.1)
    ax_v.set_ylim(0.80, 1.36)
    ax_v.set_yticks([0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    plt.setp(ax_v.get_xticklabels(), visible=False)
    h = [Line2D([], [], color="#B9B1A3", lw=1.2, label="relay-side voltages"),
         Line2D([], [], color=PALETTE["control"], lw=1.2, label="transmission side, median"),
         Line2D([], [], ls="none", marker="X", ms=5.5, mec="white", mew=0.3, color=PALETTE["trip"],
                label=rf"{float(ov1['pickup']):.2f} pu, {float(ov1['delay_s']):g} s trip"),
         Line2D([], [], ls="none", marker="X", ms=5.5, mec="white", mew=0.3, color=PALETTE["data"],
                label=rf"{float(ov2['pickup']):.2f} pu, {float(ov2['delay_s']):g} s trip"),
         Line2D([], [], ls="none", marker="X", ms=5.5, mec="white", mew=0.3, color=PALETTE["charcoal"],
                label=r"Granada, 1.10 pu, 0 s")]
    fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=5, frameon=False, fontsize=7.2,
               handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
    soften_axes(ax_v, grid=True)
    panel_label(ax_v, "a")

    ang = np.where(t <= t_sep + 1e-9, st["coi_angle_change_deg"].to_numpy(), np.nan)
    ax_a.plot(t, ang, color=PALETTE["charcoal"], lw=0.95)
    ax_a.axhline(thr, color=PALETTE["trip"], lw=0.75, ls=(0, (5, 2.4)))
    ax_a.text(0.25, thr - 6, rf"separation at {thr:.0f}$^\circ$", fontsize=7.2, color=PALETTE["trip"], va="top")
    ax_a.set_ylabel(r"angle ($^\circ$)")
    ax_a.set_xlabel(r"time after 12:32:00 (s)")
    ax_a.set_ylim(-5, 140)
    ax_a.set_yticks([0, 60, 120])
    soften_axes(ax_a, grid=True)
    panel_label(ax_a, "b")

    m = t >= zoom_from - 0.2
    dg = st["cum_embedded_pv_lost_mw"].to_numpy() / 1e3
    plants = (st["cum_lost_p_mw"].to_numpy() - st["cum_generator_trip_mw"].to_numpy()) / 1e3 - dg
    ax_g.stackplot(t[m], plants[m], dg[m], colors=[PALETTE["risky"], PALETTE["light_amber"]], alpha=0.95, lw=0,
                   labels=["plants", "embedded PV"])
    ax_g.plot(t[m], st["cum_ufls_shed_mw"].to_numpy()[m] / 1e3, color=PALETTE["charcoal"], lw=1.0,
              ls=(0, (4, 1.6)), label="load shed (UFLS)")
    ax_g.axvline(t_sep, color=PALETTE["charcoal"], lw=0.7, ls=(0, (2, 1.5)), zorder=0)
    ax_g.set_ylabel("disconnected (GW)")
    ax_g.legend(loc="upper left", fontsize=6.8, frameon=False, handlelength=1.3, borderaxespad=0.1)
    plt.setp(ax_g.get_xticklabels(), visible=False)
    soften_axes(ax_g, grid=True)
    panel_label(ax_g, "c")

    f = st["affected_island_f_eq_hz"].to_numpy()
    ax_f.plot(t[m], f[m], color=PALETTE["control"], lw=1.1)
    for e in ufls:
        ax_f.scatter([e["time_s"]], [np.interp(e["time_s"], t, f)], marker="v", s=18, color=PALETTE["risky"],
                     edgecolor=PALETTE["charcoal"], lw=0.3, zorder=4)
    ybot = 46.0
    gt = np.array([e["time_s"] for e in gen_trips])
    ax_f.scatter(gt, np.full(len(gt), ybot + 0.18), marker="|", s=40, color=PALETTE["charcoal"], lw=0.8, zorder=4)
    ax_f.axvline(t_sep, color=PALETTE["charcoal"], lw=0.7, ls=(0, (2, 1.5)), zorder=0)
    ax_f.set_xlim(zoom_from, t_end + 0.1)
    ax_f.set_ylim(ybot, 50.3)
    ax_f.set_ylabel("frequency (Hz)")
    ax_f.set_xlabel(r"time after 12:32:00 (s)")
    hf = [Line2D([], [], ls="none", marker="v", ms=4.5, mfc=PALETTE["risky"], mec=PALETTE["charcoal"], mew=0.3,
                 label="UFLS stage"),
          Line2D([], [], ls="none", marker="|", ms=6, color=PALETTE["charcoal"], mew=0.8, label="unit trip")]
    ax_f.legend(handles=hf, loc="lower left", fontsize=6.8, frameon=False, handlelength=1.0, borderaxespad=0.1,
                bbox_to_anchor=(0.0, 0.12))
    soften_axes(ax_f, grid=True)
    panel_label(ax_f, "d")
    return _save(fig, out_dir, "fig_sequence_classic")


FIGURES = {
    "sequence": fig_sequence,
    "replica": fig_replica,
    "validation": fig_validation,
    "cascade": fig_cascade,
    "families": fig_families,
    "mitigation": fig_mitigation,
    "observability": fig_observability,
    "uncertainty": fig_uncertainty,
    "ranking": fig_ranking,
}
