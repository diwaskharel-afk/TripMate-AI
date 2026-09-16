"""Per-run cost/latency instrumentation. Trivial but easy to forget."""
from __future__ import annotations

import time
from contextlib import contextmanager

from schema import CostReport


@contextmanager
def track_node(cost: CostReport, node_name: str):
    start = time.monotonic()
    try:
        yield
    finally:
        cost.node_seconds[node_name] = cost.node_seconds.get(node_name, 0.0) + (
            time.monotonic() - start
        )


def record_usage(cost: CostReport, usage: dict) -> None:
    cost.input_tokens += usage.get("input_tokens", 0) or 0
    cost.output_tokens += usage.get("output_tokens", 0) or 0
