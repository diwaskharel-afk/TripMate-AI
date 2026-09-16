"""Thin CLI entrypoint. Usage:

    python main.py --dataset datasets/insurance.csv \\
        --question "Run a linear regression and estimate insurance charges in 2029"

All the actual work lives in graph.py/sandbox.py/report.py; this just wires
config -> sandbox -> graph -> report and writes outputs/<run_id>/.
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from config import Settings
from graph import GraphContext, build_graph
from report import assemble_report
from sandbox import AnalysisSandbox
from schema import AnalysisState


async def run_pipeline(settings: Settings) -> AnalysisState:
    """Run one full analysis and return the final AnalysisState. This is the
    function both the eval harness and the Streamlit app call directly."""
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.run_output_dir(run_id)
    plots_dir = out_dir / "plots"

    state = AnalysisState(
        run_id=run_id,
        dataset_path=str(settings.dataset_path),
        question=settings.question,
    )

    with AnalysisSandbox(settings.dataset_path, plots_dir) as sandbox:
        ctx = GraphContext(sandbox, settings.model_name, settings.run_limit_per_node)
        app = build_graph(ctx)
        state = await app.ainvoke(state)
        if not isinstance(state, AnalysisState):
            state = AnalysisState.model_validate(state)
        state.cost.sandbox_seconds = sandbox.sandbox_seconds

    (out_dir / "state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    (out_dir / "report.html").write_text(assemble_report(state), encoding="utf-8")
    return state


def parse_args(argv: list[str] | None = None) -> Settings:
    parser = argparse.ArgumentParser(description="Agentic data-analysis pipeline")
    parser.add_argument("--dataset", type=Path, help="Path to the input CSV")
    parser.add_argument("--question", type=str, help="What to analyze/predict/forecast")
    parser.add_argument("--model", type=str, help="LangChain model string, e.g. openai:gpt-4.1-mini")
    parser.add_argument("--run-limit", type=int, help="Max model calls per pipeline node")
    parser.add_argument("--output-dir", type=Path, help="Where to write outputs/<run_id>/")
    args = parser.parse_args(argv)

    settings = Settings()
    if args.dataset:
        settings.dataset_path = args.dataset
    if args.question:
        settings.question = args.question
    if args.model:
        settings.model_name = args.model
    if args.run_limit:
        settings.run_limit_per_node = args.run_limit
    if args.output_dir:
        settings.output_dir = args.output_dir
    return settings


def main() -> None:
    settings = parse_args()
    state = asyncio.run(run_pipeline(settings))
    out_dir = settings.output_dir / state.run_id
    print(f"Done. task_type={state.task_type} report={out_dir / 'report.html'}")


if __name__ == "__main__":
    main()
