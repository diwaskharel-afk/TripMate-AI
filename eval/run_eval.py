"""Eval harness: runs the full pipeline against 3 benchmark datasets (one per
branch) and asserts task classification, schema validity, and that the best
model beats the naive baseline. This is what turns "I built an agent" into
"I built and evaluated an agent" — run with:

    python -m eval.run_eval

Requires OPENAI_API_KEY and E2B_API_KEY in the environment; each case spends
real API/sandbox credit, so this is not run automatically by CI unless those
secrets are configured.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Settings
from main import run_pipeline
from schema import AnalysisState

DATASETS_DIR = Path(__file__).parent / "datasets"
RESULTS_PATH = Path(__file__).parent / "results.json"

# Lower-is-better metric keys we know how to compare against baseline.
LOWER_IS_BETTER = {"rmse", "mae", "mse"}
HIGHER_IS_BETTER = {"accuracy", "f1", "r2", "precision", "recall", "roc_auc"}


@dataclass
class EvalCase:
    name: str
    dataset: Path
    question: str
    expected_task_type: str


@dataclass
class CaseResult:
    name: str
    passed: bool
    reasons: list[str] = field(default_factory=list)
    task_type: str | None = None
    run_id: str | None = None


CASES = [
    EvalCase(
        name="eda_only",
        dataset=DATASETS_DIR / "eda_survey.csv",
        question="Explore this employee survey dataset and summarize what drives satisfaction.",
        expected_task_type="eda",
    ),
    EvalCase(
        name="supervised_regression",
        dataset=DATASETS_DIR / "insurance_like.csv",
        question="Predict insurance charges for new customers from their attributes.",
        expected_task_type="predict",
    ),
    EvalCase(
        name="forecasting",
        dataset=DATASETS_DIR / "daily_sales.csv",
        question="Forecast daily sales for the next 30 days based on historical trend and marketing spend.",
        expected_task_type="forecast",
    ),
]


def _beats_baseline(state: AnalysisState) -> tuple[bool, str]:
    baseline = state.baseline_model()
    best = state.best_model()
    if baseline is None or best is None:
        return False, "missing baseline or candidate model in models_tried"
    shared_keys = set(baseline.metrics) & set(best.metrics)
    for key in shared_keys:
        if key in LOWER_IS_BETTER and best.metrics[key] < baseline.metrics[key]:
            return True, f"{key}: {best.metrics[key]} < baseline {baseline.metrics[key]}"
        if key in HIGHER_IS_BETTER and best.metrics[key] > baseline.metrics[key]:
            return True, f"{key}: {best.metrics[key]} > baseline {baseline.metrics[key]}"
    return False, f"best model did not beat baseline on any shared metric ({shared_keys})"


async def run_case(case: EvalCase) -> CaseResult:
    settings = Settings()
    settings.dataset_path = case.dataset
    settings.question = case.question

    result = CaseResult(name=case.name, passed=True)
    try:
        state = await run_pipeline(settings)
    except Exception as exc:
        return CaseResult(name=case.name, passed=False, reasons=[f"pipeline raised: {exc}"])

    result.task_type = state.task_type
    result.run_id = state.run_id

    if state.task_type != case.expected_task_type:
        result.passed = False
        result.reasons.append(f"expected task_type={case.expected_task_type}, got {state.task_type}")

    try:
        AnalysisState.model_validate(json.loads(state.model_dump_json()))
    except Exception as exc:
        result.passed = False
        result.reasons.append(f"state.json failed schema validation: {exc}")

    if case.expected_task_type in ("predict", "forecast"):
        ok, detail = _beats_baseline(state)
        if not ok:
            result.passed = False
        result.reasons.append(f"baseline check: {detail}")

    return result


async def main() -> int:
    started = time.time()
    results = [await run_case(case) for case in CASES]
    duration = time.time() - started

    print("\n=== Eval results ===")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name} (task_type={r.task_type}, run_id={r.run_id})")
        for reason in r.reasons:
            print(f"    - {reason}")

    n_passed = sum(r.passed for r in results)
    print(f"\n{n_passed}/{len(results)} passed in {duration:.1f}s")

    RESULTS_PATH.write_text(
        json.dumps(
            {
                "passed": n_passed,
                "total": len(results),
                "duration_seconds": round(duration, 1),
                "cases": [
                    {
                        "name": r.name,
                        "passed": r.passed,
                        "task_type": r.task_type,
                        "run_id": r.run_id,
                        "reasons": r.reasons,
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return 0 if n_passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
