"""Runtime configuration. Everything that used to be hardcoded in
preprocessing_agent.py lives here, sourced from env vars with CLI overrides
applied by main.py."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).parent
load_dotenv(REPO_ROOT / ".env")


@dataclass
class Settings:
    dataset_path: Path = field(default_factory=lambda: Path(
        os.environ.get("DATASET_PATH", "datasets/insurance.csv")
    ))
    question: str = field(default_factory=lambda: os.environ.get(
        "ANALYSIS_QUESTION",
        "Explore this dataset and summarize the key findings.",
    ))
    model_name: str = field(default_factory=lambda: os.environ.get(
        "MODEL_NAME", "openai:gpt-4.1-mini"
    ))
    run_limit_per_node: int = field(default_factory=lambda: int(
        os.environ.get("RUN_LIMIT_PER_NODE", "6")
    ))
    output_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("OUTPUT_DIR", "outputs")
    ))
    openai_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))
    e2b_api_key: str | None = field(default_factory=lambda: os.environ.get("E2B_API_KEY"))

    def run_output_dir(self, run_id: str) -> Path:
        d = self.output_dir / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d
