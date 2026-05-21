"""Shared plotting style for paper case-study artifacts."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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

FAMILY_COLOR = {
    "load/pump disconnection": PALETTE["trip"],
    "fixed-PF RES ramp": PALETTE["risky"],
    "plant/generator trip": PALETTE["data"],
    "export reduction": PALETTE["control"],
    "shunt/reactor action": PALETTE["safe"],
}

ABLATION_SHORT = {
    "baseline": "baseline",
    "delayed protection": "delayed\nprotection",
    "preserved reactive absorption": "preserved\nQ absorption",
    "voltage-mode RES instead of fixed-PF": "voltage-mode\nIBR",
    "stronger/faster shunt support": "faster shunt\nsupport",
}


FIGURE_SPECS = {
    "column": (3.45, 4.35),
    "column_short": (3.45, 3.65),
    "wide": (7.00, 4.60),
    "wide_short": (7.00, 3.85),
    "wide_tall": (7.00, 4.85),
}


@dataclass
class SaveResult:
    path: Path
    sha256: str


def latex_available() -> bool:
    return bool(shutil.which("latex"))


def configure_style(*, prefer_usetex: bool = True) -> bool:
    """Apply a consistent journal-style Matplotlib configuration.

    Returns
    -------
    bool
        Whether Matplotlib was configured to use an external LaTeX binary.
    """

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


def tex_escape(value: object) -> str:
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


def badge(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    transform=None,
    fontsize: int = 13,
) -> None:
    ax.text(
        x,
        y,
        text,
        transform=transform or ax.transData,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="white",
        bbox={
            "boxstyle": "round,pad=0.18,rounding_size=0.08",
            "fc": color,
            "ec": "none",
            "alpha": 0.96,
        },
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(
    fig: plt.Figure,
    stem: str,
    out_dir: Path,
    formats: Iterable[str],
    *,
    outline_text: bool = False,
) -> list[SaveResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[SaveResult] = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        kwargs = {}
        if fmt == "png":
            kwargs["dpi"] = 600
        if fmt == "pdf" and outline_text:
            _save_pdf_with_outlined_text(fig, path)
        else:
            fig.savefig(path, **kwargs)
        written.append(SaveResult(path=path, sha256=sha256_file(path)))
        if fmt == "svg" and outline_text and shutil.which("inkscape"):
            sidecar = out_dir / f"{stem}_outlined.svg"
            subprocess.run(
                [
                    "inkscape",
                    str(path),
                    "--export-text-to-path",
                    "--export-filename",
                    str(sidecar),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if sidecar.exists():
                written.append(SaveResult(path=sidecar, sha256=sha256_file(sidecar)))
    return written


def svg_to_outlined_pdf(svg_path: Path, pdf_path: Path) -> None:
    """Convert an SVG whose text is already path-converted into a PDF."""

    try:
        import cairosvg
    except Exception:
        cairosvg = None
    if cairosvg is not None:
        try:
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
            return
        except Exception:
            # Some macOS environments have the Python package but not the
            # native Cairo dylib. Fall through to PyMuPDF, which ships wheels.
            pass

    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Outlined PDF export requires either cairosvg+cairo or PyMuPDF. "
            "Install PyMuPDF with `python -m pip install pymupdf`, or rerun "
            "with --no-outline-text."
        ) from exc
    doc = pymupdf.open(str(svg_path))
    try:
        pdf_path.write_bytes(doc.convert_to_pdf())
    finally:
        doc.close()


def _save_pdf_with_outlined_text(fig: plt.Figure, pdf_path: Path) -> None:
    """Write a PDF with text converted to vector paths.

    Illustrator can complain about missing TeX fonts when a regular Matplotlib
    PDF contains live font objects. Use Ghostscript on the native Matplotlib PDF
    so transparency and alpha blending remain intact.
    """

    tmp_pdf = pdf_path.with_name(f".{pdf_path.stem}.native_tmp.pdf")
    fig.savefig(tmp_pdf)
    try:
        outline_pdf_text(tmp_pdf, pdf_path)
    finally:
        tmp_pdf.unlink(missing_ok=True)


def outline_pdf_text(input_pdf: Path, output_pdf: Path) -> None:
    """Convert PDF fonts to outlines while preserving native PDF graphics."""

    gs = shutil.which("gs")
    if not gs:
        for candidate in ("/usr/local/bin/gs", "/opt/homebrew/bin/gs"):
            if Path(candidate).exists():
                gs = candidate
                break
    if not gs:
        raise RuntimeError(
            "Ghostscript is required for outlined PDF export. Install `gs`, or "
            "rerun with --no-outline-text."
        )
    tmp_out = output_pdf.with_name(f".{output_pdf.stem}.outlined_tmp.pdf")
    cmd = [
        gs,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-dCompatibilityLevel=1.7",
        "-dPDFSETTINGS=/prepress",
        "-dNoOutputFonts",
        "-sDEVICE=pdfwrite",
        f"-sOutputFile={tmp_out}",
        str(input_pdf),
    ]
    try:
        subprocess.run(cmd, check=True)
        tmp_out.replace(output_pdf)
    finally:
        tmp_out.unlink(missing_ok=True)
