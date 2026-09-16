"""The analysis pipeline as an explicit LangGraph state machine.

classify -> ingest -> eda -> [eda-only: report] | [predict|forecast: feature_prep
-> model_predict|model_forecast] -> verify -> report -> END

Each node runs one phase-scoped agent turn (agent.run_phase) against the
shared sandbox, then merges whatever the agent wrote into /home/user/state.json
back into the typed AnalysisState. This is the "inspectable state machine"
instead of one long ReAct loop deciding everything itself.
"""
from __future__ import annotations

import json
import logging

from langgraph.graph import END, StateGraph

from agent import load_phase_prompt, run_phase
from cost import record_usage, track_node
from sandbox import AnalysisSandbox
from schema import AnalysisState

logger = logging.getLogger(__name__)


def _merge_state_json(state: AnalysisState, raw_json: str | None) -> None:
    """Merge whatever the sandbox-side state.json contains into the typed
    AnalysisState, field by field, ignoring unknown/malformed keys instead of
    failing the whole node on a partial write."""
    if not raw_json:
        return
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("state.json from sandbox was not valid JSON; skipping merge")
        return

    for field_name in AnalysisState.model_fields:
        if field_name not in payload or field_name in ("run_id", "dataset_path", "question"):
            continue
        try:
            merged = AnalysisState.model_validate({**state.model_dump(), field_name: payload[field_name]})
            setattr(state, field_name, getattr(merged, field_name))
        except Exception as exc:
            logger.warning("Skipping invalid field %r from sandbox state.json: %s", field_name, exc)


class GraphContext:
    """Bundles the run-scoped dependencies (sandbox, model name, run limit)
    that every node closure needs but that aren't part of the typed state."""

    def __init__(self, sandbox: AnalysisSandbox, model_name: str, run_limit_per_node: int):
        self.sandbox = sandbox
        self.model_name = model_name
        self.run_limit_per_node = run_limit_per_node

    async def run_node(self, state: AnalysisState, node_name: str, prompt_name: str, **fmt) -> None:
        tools = [self.sandbox.make_run_python_tool()]
        prompt = load_phase_prompt(prompt_name, **fmt)
        user_message = (
            f"Current state.json (may be partial):\n{state.model_dump_json(indent=2)}"
        )
        with track_node(state.cost, node_name):
            text, usage = await run_phase(
                self.model_name, tools, self.run_limit_per_node, prompt, user_message
            )
        record_usage(state.cost, usage)
        raw_state_json = self.sandbox.pull_state_json()
        _merge_state_json(state, raw_state_json)
        logger.info("[%s] done: %s", node_name, text[:200])


def build_graph(ctx: GraphContext):
    graph = StateGraph(AnalysisState)

    async def classify_node(state: AnalysisState) -> AnalysisState:
        await ctx.run_node(state, "classify", "classify", question=state.question)
        return state

    async def ingest_node(state: AnalysisState) -> AnalysisState:
        await ctx.run_node(state, "ingest", "ingest")
        return state

    async def eda_node(state: AnalysisState) -> AnalysisState:
        await ctx.run_node(state, "eda", "eda")
        return state

    async def feature_prep_node(state: AnalysisState) -> AnalysisState:
        await ctx.run_node(state, "feature_prep", "feature_prep")
        return state

    async def model_predict_node(state: AnalysisState) -> AnalysisState:
        await ctx.run_node(state, "model_predict", "model_predict")
        return state

    async def model_forecast_node(state: AnalysisState) -> AnalysisState:
        await ctx.run_node(state, "model_forecast", "model_forecast")
        return state

    async def verify_node(state: AnalysisState) -> AnalysisState:
        from verifier import verify_report

        state.verification = await verify_report(ctx.model_name, state)
        return state

    async def report_node(state: AnalysisState) -> AnalysisState:
        await ctx.run_node(state, "report", "report")
        return state

    graph.add_node("classify", classify_node)
    graph.add_node("ingest", ingest_node)
    graph.add_node("eda", eda_node)
    graph.add_node("feature_prep", feature_prep_node)
    graph.add_node("model_predict", model_predict_node)
    graph.add_node("model_forecast", model_forecast_node)
    graph.add_node("verify", verify_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "ingest")
    graph.add_edge("ingest", "eda")

    def route_after_eda(state: AnalysisState) -> str:
        if state.task_type == "predict":
            return "feature_prep_predict"
        if state.task_type == "forecast":
            return "feature_prep_forecast"
        return "report"

    graph.add_conditional_edges(
        "eda",
        route_after_eda,
        {
            "feature_prep_predict": "feature_prep",
            "feature_prep_forecast": "feature_prep",
            "report": "report",
        },
    )

    def route_after_feature_prep(state: AnalysisState) -> str:
        return "model_forecast" if state.task_type == "forecast" else "model_predict"

    graph.add_conditional_edges(
        "feature_prep",
        route_after_feature_prep,
        {"model_predict": "model_predict", "model_forecast": "model_forecast"},
    )

    graph.add_edge("model_predict", "report")
    graph.add_edge("model_forecast", "report")
    # verify runs LAST: it checks the just-written report_markdown against the
    # ground-truth metrics/models_tried already in state, then annotates
    # state.verification. report.py's HTML assembly (outside the graph) uses that.
    graph.add_edge("report", "verify")
    graph.add_edge("verify", END)

    return graph.compile()
