"""Pydantic data contracts shared across the pipeline.

`AnalysisState` is the single source of truth passed through the LangGraph
graph and dumped to disk as `state.json` after every run. Anything the final
report claims must trace back to a field here.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TaskType = Literal["eda", "predict", "forecast"]


class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_count: int = 0
    missing_pct: float = 0.0


class ModelResult(BaseModel):
    name: str
    metrics: dict[str, float] = Field(default_factory=dict)
    is_baseline: bool = False


class ForecastPoint(BaseModel):
    period: str
    point_estimate: float
    lower: float | None = None
    upper: float | None = None


class CostReport(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    node_seconds: dict[str, float] = Field(default_factory=dict)
    sandbox_seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def total_wall_seconds(self) -> float:
        return sum(self.node_seconds.values())


class VerificationResult(BaseModel):
    ok: bool = True
    notes: list[str] = Field(default_factory=list)


class AnalysisState(BaseModel):
    """The full pipeline state. Passed node-to-node in graph.py and persisted
    as outputs/<run_id>/state.json."""

    run_id: str = ""
    dataset_path: str = ""
    question: str = ""

    shape: tuple[int, int] | None = None
    columns: list[ColumnInfo] = Field(default_factory=list)

    task_type: TaskType | None = None
    task_type_reasoning: str = ""
    target_column: str | None = None

    assumptions: list[str] = Field(default_factory=list)
    eda_findings: list[str] = Field(default_factory=list)
    feature_prep_notes: list[str] = Field(default_factory=list)

    models_tried: list[ModelResult] = Field(default_factory=list)
    forecast: list[ForecastPoint] = Field(default_factory=list)

    plots: list[str] = Field(default_factory=list)

    report_markdown: str = ""
    verification: VerificationResult = Field(default_factory=VerificationResult)
    cost: CostReport = Field(default_factory=CostReport)

    def best_model(self) -> ModelResult | None:
        candidates = [m for m in self.models_tried if not m.is_baseline]
        return candidates[-1] if candidates else None

    def baseline_model(self) -> ModelResult | None:
        for m in self.models_tried:
            if m.is_baseline:
                return m
        return None
