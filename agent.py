"""Builds the small, phase-scoped ReAct agent each LangGraph node runs.

Each node gets its own `create_agent` instance with its own tiny message
history (just base rules + phase prompt + current state summary), instead of
one giant transcript growing across the whole pipeline. That's the
context-engineering win: every call's payload is bounded by the node's job,
not by everything that happened before it.
"""
from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

PROMPTS_DIR = Path(__file__).parent / "prompts"
BASE_RULES = (PROMPTS_DIR / "base_rules.md").read_text(encoding="utf-8")


def load_phase_prompt(name: str, **format_kwargs: str) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    if format_kwargs:
        text = text.format(**format_kwargs)
    return f"{BASE_RULES}\n\n---\n\n{text}"


def build_phase_agent(model_name: str, tools: list, run_limit: int):
    """A fresh agent per node call — cheap, and keeps middleware state (the
    call-limit counter) scoped to just this node's turn."""
    return create_agent(
        model=model_name,
        tools=tools,
        middleware=[ModelCallLimitMiddleware(run_limit=run_limit)],
    )


async def run_phase(
    model_name: str,
    tools: list,
    run_limit: int,
    system_prompt: str,
    user_message: str,
):
    """Run one phase-scoped agent turn and return (final_text, usage_metadata)."""
    agent = build_phase_agent(model_name, tools, run_limit)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]
    result = await agent.ainvoke({"messages": messages})
    final_message = result["messages"][-1]
    usage = getattr(final_message, "usage_metadata", None) or {}
    text = getattr(final_message, "content", "") or ""
    return text, usage
