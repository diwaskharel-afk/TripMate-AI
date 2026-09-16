"""Streamlit demo: upload a CSV, ask a question, watch the pipeline stream
phase-by-phase, then download the self-contained HTML report.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import streamlit as st

from config import Settings
from graph import GraphContext, build_graph
from report import assemble_report
from sandbox import AnalysisSandbox
from schema import AnalysisState

st.set_page_config(page_title="Agentic Data Analyst", page_icon="📊", layout="centered")
st.title("📊 Agentic Data Analyst")
st.caption(
    "Upload a CSV, ask a question. The agent classifies the task (explore / predict / "
    "forecast), then runs an inspectable LangGraph pipeline in a live sandbox."
)

uploaded = st.file_uploader("Dataset (CSV)", type=["csv"])
question = st.text_area(
    "What do you want to know?",
    placeholder="e.g. Predict insurance charges for new customers from their attributes.",
)
model_name = st.selectbox("Model", ["openai:gpt-4.1-mini", "openai:gpt-4.1"], index=0)
run_limit = st.slider("Max model calls per pipeline node", 2, 12, 6)

NODE_LABELS = {
    "classify": "Classifying task",
    "ingest": "Ingesting & validating data",
    "eda": "Exploratory analysis",
    "feature_prep": "Feature preparation",
    "model_predict": "Training & evaluating models",
    "model_forecast": "Backtesting & forecasting",
    "report": "Writing report",
    "verify": "Verifying report against ground truth",
}


async def _stream_pipeline(settings: Settings, status_boxes: dict) -> AnalysisState:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.run_output_dir(run_id)
    state = AnalysisState(run_id=run_id, dataset_path=str(settings.dataset_path), question=settings.question)

    with AnalysisSandbox(settings.dataset_path, out_dir / "plots") as sandbox:
        ctx = GraphContext(sandbox, settings.model_name, settings.run_limit_per_node)
        app = build_graph(ctx)
        async for event in app.astream(state, stream_mode="values"):
            state = event if isinstance(event, AnalysisState) else AnalysisState.model_validate(event)
            for node_name, box in status_boxes.items():
                if node_name in state.cost.node_seconds:
                    box.update(state=" done", label=f"✅ {NODE_LABELS[node_name]}")
        state.cost.sandbox_seconds = sandbox.sandbox_seconds

    (out_dir / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    report_html = assemble_report(state)
    (out_dir / "report.html").write_text(report_html, encoding="utf-8")
    return state


if st.button("Run analysis", type="primary", disabled=uploaded is None or not question.strip()):
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = Path(tmp) / (uploaded.name or "data.csv")
        dataset_path.write_bytes(uploaded.getvalue())

        settings = Settings()
        settings.dataset_path = dataset_path
        settings.question = question
        settings.model_name = model_name
        settings.run_limit_per_node = run_limit

        status_boxes = {
            name: st.status(f"⏳ {label}", expanded=False)
            for name, label in NODE_LABELS.items()
        }

        try:
            state = asyncio.run(_stream_pipeline(settings, status_boxes))
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

        st.success(f"Done. Task classified as: **{state.task_type}**")
        report_html = assemble_report(state)
        st.download_button(
            "Download report.html",
            data=report_html,
            file_name=f"report-{state.run_id}.html",
            mime="text/html",
        )
        st.components.v1.html(report_html, height=800, scrolling=True)
