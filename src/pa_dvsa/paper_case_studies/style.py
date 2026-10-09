"""Palette, fonts and axis style of the paper figures."""

from __future__ import annotations

import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTE = {
    "safe": "#6FAE9F",
    "risky": "#DDBB6A",
    "trip": "#C96F5A",
    "control": "#5E81AC",
    "data": "#8B7EAD",
    "background": "#D9D3C7",
    "charcoal": "#2F3A45",
    "soft_bg": "#FBFAF7",
    "grid": "#ECE7DE",
    "light_red": "#F0D7D1",
    "light_teal": "#DDEDE8",
    "light_purple": "#E5E0F0",
    "light_blue": "#DDE7F2",
    "light_amber": "#F2E6C4",
    "empty": "#F3F1EC",
}


def latex_available() -> bool:
    return bool(shutil.which("latex"))


def configure_style(*, prefer_usetex: bool = True) -> bool:
    use_tex = prefer_usetex and latex_available()
    mpl.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 8.4,
            "axes.labelsize": 9.4,
            "xtick.labelsize": 8.1,
            "ytick.labelsize": 8.1,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.75,
            "axes.edgecolor": PALETTE["charcoal"],
            "xtick.major.width": 0.72,
            "ytick.major.width": 0.72,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "path",
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    return use_tex


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        rf"\textbf{{{label}}}" if mpl.rcParams.get("text.usetex") else label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        color=PALETTE["charcoal"],
    )


def soften_axes(ax: plt.Axes, *, grid: bool = False, ygrid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["charcoal"])
    ax.spines["bottom"].set_color(PALETTE["charcoal"])
    if grid:
        axis = "y" if ygrid else "both"
        ax.grid(axis=axis, color=PALETTE["grid"], lw=0.8, alpha=0.65)


