"""Verifier pass: one extra no-tools LLM call that cross-checks the written
report's numeric claims against the ground-truth AnalysisState fields, so
hallucinated numbers get flagged before a human sees the report.
"""
from __future__ import annotations

import json

from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

from schema import AnalysisState, VerificationResult


class _VerifierOutput(BaseModel):
    ok: bool = Field(description="True if every numeric claim in the report matches ground truth")
    notes: list[str] = Field(default_factory=list, description="Discrepancies found, empty if none")


VERIFIER_PROMPT = """You are a fact-checker. Below is a data-analysis report and the \
ground-truth structured state it should have been derived from. Check every \
number the report cites (metrics, percentages, forecast values) against the \
ground truth. Flag anything that doesn't match or that isn't traceable to the \
ground truth at all. Don't flag stylistic issues, only factual/numeric ones.

GROUND TRUTH STATE:
{ground_truth}

REPORT:
{report}
"""


async def verify_report(model_name: str, state: AnalysisState) -> VerificationResult:
    if not state.report_markdown.strip():
        return VerificationResult(ok=False, notes=["No report_markdown was produced to verify."])

    ground_truth = {
        "models_tried": [m.model_dump() for m in state.models_tried],
        "forecast": [f.model_dump() for f in state.forecast],
        "eda_findings": state.eda_findings,
        "task_type": state.task_type,
    }

    model = init_chat_model(model_name)
    structured_model = model.with_structured_output(_VerifierOutput)
    prompt = VERIFIER_PROMPT.format(
        ground_truth=json.dumps(ground_truth, indent=2),
        report=state.report_markdown,
    )
    try:
        result: _VerifierOutput = await structured_model.ainvoke(prompt)
        return VerificationResult(ok=result.ok, notes=result.notes)
    except Exception as exc:
        return VerificationResult(ok=False, notes=[f"Verifier call failed: {exc}"])
