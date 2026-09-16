"""Assembles the one artifact meant to survive the run: a single
self-contained HTML file with the report text, embedded plots (base64, no
external file refs), a metrics table, verification notes, and cost/latency."""
from __future__ import annotations

import base64
import html
from pathlib import Path

try:
    import markdown as _markdown
except ImportError:
    _markdown = None

from schema import AnalysisState


def _markdown_to_html(text: str) -> str:
    if _markdown is not None:
        return _markdown.markdown(text, extensions=["tables"])
    return f"<pre>{html.escape(text)}</pre>"


def _embed_plots(plot_paths: list[str]) -> str:
    if not plot_paths:
        return ""
    figures = []
    for p in plot_paths:
        path = Path(p)
        if not path.exists():
            continue
        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        figures.append(
            f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(path.stem)}">'
            f"<figcaption>{html.escape(path.name)}</figcaption></figure>"
        )
    return "<h2>Plots</h2><div class='plots'>" + "".join(figures) + "</div>"


def _metrics_table(state: AnalysisState) -> str:
    if not state.models_tried:
        return ""
    metric_keys: list[str] = []
    for m in state.models_tried:
        for k in m.metrics:
            if k not in metric_keys:
                metric_keys.append(k)
    header = "<tr><th>Model</th><th>Baseline?</th>" + "".join(f"<th>{html.escape(k)}</th>" for k in metric_keys) + "</tr>"
    rows = ""
    for m in state.models_tried:
        cells = "".join(f"<td>{m.metrics.get(k, '')}</td>" for k in metric_keys)
        rows += f"<tr><td>{html.escape(m.name)}</td><td>{'yes' if m.is_baseline else ''}</td>{cells}</tr>"
    return f"<h2>Models</h2><table>{header}{rows}</table>"


def _verification_section(state: AnalysisState) -> str:
    status = "PASS" if state.verification.ok else "FLAGGED"
    css_class = "ok" if state.verification.ok else "flagged"
    notes = "".join(f"<li>{html.escape(n)}</li>" for n in state.verification.notes)
    notes_html = f"<ul>{notes}</ul>" if notes else "<p>No discrepancies found.</p>"
    return f"<h2>Verification: <span class='{css_class}'>{status}</span></h2>{notes_html}"


def _cost_section(state: AnalysisState) -> str:
    c = state.cost
    rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v:.1f}s</td></tr>" for k, v in c.node_seconds.items())
    return (
        "<h2>Cost &amp; Latency</h2>"
        f"<p>Tokens: {c.total_tokens} (in: {c.input_tokens}, out: {c.output_tokens})<br>"
        f"Sandbox time: {c.sandbox_seconds:.1f}s<br>"
        f"Total wall time: {c.total_wall_seconds:.1f}s</p>"
        f"<table><tr><th>Node</th><th>Seconds</th></tr>{rows}</table>"
    )


PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Analysis Report — {run_id}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
h1, h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 0.3rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }}
figure {{ margin: 1rem 0; }}
figure img {{ max-width: 100%; border: 1px solid #ddd; }}
.ok {{ color: #0a7d2c; }}
.flagged {{ color: #b3261e; }}
code, pre {{ background: #f5f5f5; padding: 0.2rem 0.4rem; }}
</style></head>
<body>
<h1>Analysis Report</h1>
<p><strong>Run:</strong> {run_id} &middot; <strong>Question:</strong> {question}</p>
{report_body}
{plots}
{metrics}
{verification}
{cost}
</body></html>
"""


def assemble_report(state: AnalysisState) -> str:
    return PAGE_TEMPLATE.format(
        run_id=html.escape(state.run_id),
        question=html.escape(state.question),
        report_body=_markdown_to_html(state.report_markdown or "_No report text produced._"),
        plots=_embed_plots(state.plots),
        metrics=_metrics_table(state),
        verification=_verification_section(state),
        cost=_cost_section(state),
    )
