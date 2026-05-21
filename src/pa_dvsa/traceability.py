"""Traceability extraction from the Applied Energy paper source.

Each code module, test, result, and figure should trace back to a concrete
equation, algorithm, statement, or planned case-study output in
``LaTeX/root.tex``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable

from . import __version__
from .common import sha256_file, write_json


SECTION_RE = re.compile(
    r"\\(?P<level>section|subsection|subsubsection)\{(?P<title>.*?)\}"
    r"(?:\\label\{(?P<label>[^}]+)\})?"
)
LABEL_RE = re.compile(r"\\label\{(?P<label>[^}]+)\}")
CAPTION_RE = re.compile(r"\\caption\{(?P<caption>.*?)\}")
BEGIN_RE = re.compile(r"\\begin\{(?P<env>[^}]+)\}(?:\[(?P<title>[^\]]+)\])?")
END_RE = re.compile(r"\\end\{(?P<env>[^}]+)\}")


@dataclass(frozen=True)
class TraceItem:
    """One paper artifact that implementation must respect."""

    kind: str
    label: str
    line: int
    section: str
    title: str
    implementation_status: str
    notes: str = ""


@dataclass(frozen=True)
class RequiredOutput:
    """One planned result or figure required by the paper."""

    output_id: str
    source_label: str
    paper_location: str
    description: str
    acceptance_check: str
    implementation_status: str = "planned"


def _strip_latex_comment(line: str) -> str:
    """Remove LaTeX comments while preserving escaped percent signs."""

    escaped = False
    for idx, char in enumerate(line):
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "%" and not escaped:
            return line[:idx]
        escaped = False
    return line


def _clean_latex_text(text: str) -> str:
    """Return compact text suitable for traceability summaries."""

    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", "", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _line_lookup(lines: Iterable[str]) -> list[str]:
    return [_strip_latex_comment(line).rstrip("\n") for line in lines]


def _current_section(section_stack: dict[str, str]) -> str:
    for key in ("subsubsection", "subsection", "section"):
        value = section_stack.get(key)
        if value:
            return value
    return "front matter"


def _kind_for_label(label: str, env: str) -> str:
    if label.startswith("eq:"):
        return "equation"
    if label.startswith("alg:"):
        return "algorithm"
    if label.startswith("fig:"):
        return "figure"
    if label.startswith("tab:"):
        return "table"
    if label.startswith("prop:"):
        return "proposition"
    if label.startswith("cor:"):
        return "corollary"
    if label.startswith("rem:"):
        return "remark"
    if label.startswith("sec:") or label.startswith("subsec:"):
        return "section"
    return env or "label"


def _required_outputs() -> list[RequiredOutput]:
    """Return planned paper outputs implied by the case-study sections."""

    return [
        RequiredOutput(
            output_id="finite_window_proxy_figure",
            source_label="fig:finite_window_proxy",
            paper_location="Hybrid DAE Model and Finite-Window Voltage Maps",
            description=(
                "Plot protected-voltage step/ramp waveforms, true finite-window "
                "maxima, and the resolvent proxy for at least one monotone channel "
                "and one non-monotone or oscillatory channel."
            ),
            acceptance_check=(
                "Generated data must include waveform samples, certified or "
                "empirical peak classification, proxy value, and channel "
                "classification."
            ),
        ),
        RequiredOutput(
            output_id="ieee39_mechanism_layout",
            source_label="fig:case_layout",
            paper_location="Case A",
            description=(
                "Document the benchmark replica layout: protected collector sides, "
                "tap ratios, relay thresholds, tripped assets, and removed MW/Mvar."
            ),
            acceptance_check=(
                "Every protected asset in the figure must have a matching config "
                "entry and result row."
            ),
        ),
        RequiredOutput(
            output_id="screen_vs_nonlinear_heatmap",
            source_label="fig:K_heatmap",
            paper_location="Case B",
            description=(
                "Heatmap of pickup or trip erosion matrices with nonlinear RMS "
                "trip-sequence overlays."
            ),
            acceptance_check=(
                "Compare predicted first pickup, first trip, final trip set, "
                "maximum protected-voltage excursion, and trip time."
            ),
        ),
        RequiredOutput(
            output_id="causal_ablations",
            source_label="subsec:case_ablations",
            paper_location="Case C",
            description=(
                "Run controlled ablations: no collector protection, delayed "
                "protection, fixed-PF versus voltage control, preserved reactive "
                "absorption, automatic shunt/STATCOM/HVDC support, UEL/limit modes, "
                "and load or pump shedding with/without MVAr replacement."
            ),
            acceptance_check=(
                "Each ablation must reuse the same base seed scenario and report "
                "cascade stop/go, first trip, final trip set, and peak voltage."
            ),
        ),
        RequiredOutput(
            output_id="uncertainty_large_scale_screening",
            source_label="subsec:case_large",
            paper_location="Case D",
            description=(
                "Monte Carlo or interval sweeps over relay thresholds, delays, "
                "taps, reactive absorption, fixed-PF values, ramp magnitudes, shunt "
                "status, controller time constants, and compliance envelopes; plus "
                "large-system sparse screening on available ANDES benchmarks."
            ),
            acceptance_check=(
                "Report robust no-secondary-trip status, data-limited assets, "
                "runtime, sparse solve counts, and highest-risk seeds."
            ),
        ),
        RequiredOutput(
            output_id="mitigation_lp_qp_results",
            source_label="eq:mitigation_qp",
            paper_location="Cascade Screening and Mitigation",
            description=(
                "Solve the time-grid mitigation LP/QP for high-risk seed sets and "
                "report selected controls, required fast MVAr, binding constraints, "
                "and residual slack."
            ),
            acceptance_check=(
                "Mitigation constraints must use time-resolved lower-bounded "
                "control coefficients, not scalar ranking scores."
            ),
        ),
    ]


def extract_traceability(root_tex: str | Path) -> dict[str, object]:
    """Extract traceability items from ``root.tex``."""

    root_path = Path(root_tex).resolve()
    raw_lines = root_path.read_text(encoding="utf-8").splitlines()
    lines = _line_lookup(raw_lines)

    section_stack: dict[str, str] = {}
    env_stack: list[tuple[str, str]] = []
    pending_captions: dict[str, str] = {}
    items: list[TraceItem] = []
    seen_labels: set[str] = set()

    for lineno, line in enumerate(lines, start=1):
        section_match = SECTION_RE.search(line)
        if section_match:
            level = section_match.group("level")
            title = _clean_latex_text(section_match.group("title"))
            section_stack[level] = title
            if level == "section":
                section_stack.pop("subsection", None)
                section_stack.pop("subsubsection", None)
            elif level == "subsection":
                section_stack.pop("subsubsection", None)
            label = section_match.group("label")
            if label and label not in seen_labels:
                items.append(
                    TraceItem(
                        kind="section",
                        label=label,
                        line=lineno,
                        section=title,
                        title=title,
                        implementation_status="reference",
                    )
                )
                seen_labels.add(label)

        begin_match = BEGIN_RE.search(line)
        if begin_match:
            env = begin_match.group("env")
            env_title = _clean_latex_text(begin_match.group("title") or "")
            env_stack.append((env, env_title))

        caption_match = CAPTION_RE.search(line)
        if caption_match and env_stack:
            pending_captions[env_stack[-1][0]] = _clean_latex_text(caption_match.group("caption"))

        current_env, current_env_title = env_stack[-1] if env_stack else ("", "")
        for label_match in LABEL_RE.finditer(line):
            label = label_match.group("label")
            if label in seen_labels:
                continue
            kind = _kind_for_label(label, current_env)
            title = pending_captions.get(current_env, current_env_title)
            if kind == "section":
                title = _current_section(section_stack)
            if not title:
                if kind == "equation":
                    title = label.replace("eq:", "").replace("_", " ")
                else:
                    title = label
            status = "paper_source"
            if kind == "section":
                status = "reference"
            if kind in {"equation", "algorithm", "figure", "table", "proposition", "corollary", "remark"}:
                status = "must_implement_or_validate"
            items.append(
                TraceItem(
                    kind=kind,
                    label=label,
                    line=lineno,
                    section=_current_section(section_stack),
                    title=title,
                    implementation_status=status,
                )
            )
            seen_labels.add(label)

        end_match = END_RE.search(line)
        if end_match and env_stack:
            end_env = end_match.group("env")
            while env_stack:
                popped_env, _ = env_stack.pop()
                pending_captions.pop(popped_env, None)
                if popped_env == end_env:
                    break

    required_outputs = _required_outputs()
    counts: dict[str, int] = {}
    for item in items:
        counts[item.kind] = counts.get(item.kind, 0) + 1

    return {
        "schema_version": "1.0",
        "tool_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "path": str(root_path),
            "sha256": sha256_file(root_path),
            "line_count": len(raw_lines),
        },
        "summary": {
            "counts_by_kind": counts,
            "total_items": len(items),
            "required_outputs": len(required_outputs),
        },
        "items": [asdict(item) for item in items],
        "required_outputs": [asdict(output) for output in required_outputs],
    }


def traceability_to_markdown(traceability: dict[str, object]) -> str:
    """Render a compact Markdown traceability report."""

    source = traceability["source"]
    summary = traceability["summary"]
    counts = summary["counts_by_kind"]
    lines = [
        "# Paper Traceability Matrix",
        "",
        f"Source: `{source['path']}`",
        f"SHA-256: `{source['sha256']}`",
        f"Line count: `{source['line_count']}`",
        "",
        "## Summary",
        "",
    ]
    for kind in sorted(counts):
        lines.append(f"- `{kind}`: {counts[kind]}")
    lines.extend(
        [
            f"- `required_outputs`: {summary['required_outputs']}",
            "",
            "## Required Outputs",
            "",
            "| Output | Source | Acceptance Check |",
            "|---|---|---|",
        ]
    )
    for output in traceability["required_outputs"]:
        lines.append(
            f"| `{output['output_id']}` | `{output['source_label']}` | "
            f"{output['acceptance_check']} |"
        )
    lines.extend(
        [
            "",
            "## Paper Artifacts",
            "",
            "| Kind | Label | Line | Section | Status |",
            "|---|---:|---:|---|---|",
        ]
    )
    for item in traceability["items"]:
        lines.append(
            f"| `{item['kind']}` | `{item['label']}` | {item['line']} | "
            f"{item['section']} | `{item['implementation_status']}` |"
        )
    return "\n".join(lines) + "\n"


def write_traceability(root_tex: str | Path, out_dir: str | Path) -> dict[str, object]:
    """Extract and write JSON/Markdown traceability artifacts."""

    out_path = Path(out_dir)
    payload = extract_traceability(root_tex)
    write_json(out_path / "traceability.json", payload)
    (out_path / "traceability.md").write_text(traceability_to_markdown(payload), encoding="utf-8")
    return payload
