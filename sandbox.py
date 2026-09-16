"""E2B sandbox wrapper. Owns the one thing that used to be scattered through
preprocessing_agent.py: the live sandbox lifecycle, the run_python tool, and
pulling artifacts (plots, state.json) back to the local filesystem."""
from __future__ import annotations

import time
from pathlib import Path

from e2b_code_interpreter import Sandbox
from langchain.tools import tool


def _truncate(text: str, max_chars: int, keep: str = "head") -> str:
    """Truncate text, keeping either the head or tail, with a note about what was cut."""
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    note = f"\n... [truncated: {omitted} characters omitted] ..."
    if keep == "head":
        return text[:max_chars] + note
    return note + "\n" + text[-max_chars:]


class AnalysisSandbox:
    """Thin wrapper around an e2b Sandbox scoped to one analysis run."""

    REMOTE_DATA_PATH = "/home/user/my-file"
    REMOTE_STATE_PATH = "/home/user/state.json"
    MAX_STDOUT_CHARS = 2000
    MAX_STDERR_CHARS = 1500

    def __init__(self, dataset_path: Path, plots_dir: Path):
        self.dataset_path = dataset_path
        self.plots_dir = plots_dir
        self._downloaded_plots: set[str] = set()
        self._sandbox_seconds = 0.0
        self._sbx: Sandbox | None = None

    def __enter__(self):
        start = time.monotonic()
        self._sbx = Sandbox.create(metadata={"name": "data_analysis_agent_run"})
        with open(self.dataset_path, "rb") as f:
            self._sbx.files.write(self.REMOTE_DATA_PATH, f)
        self._create_time = start
        return self

    def __exit__(self, *exc) -> None:
        if self._sbx is not None:
            self._sbx.kill()
        self._sandbox_seconds = time.monotonic() - self._create_time

    @property
    def sandbox_seconds(self) -> float:
        return self._sandbox_seconds

    def download_new_plots(self) -> list[str]:
        """Pull any plot_*.png files the agent has written that we haven't seen yet."""
        assert self._sbx is not None
        saved = []
        for entry in self._sbx.files.list("/home/user"):
            name = entry.name
            if not (name.startswith("plot_") and name.endswith(".png")):
                continue
            if name in self._downloaded_plots:
                continue
            content = self._sbx.files.read(f"/home/user/{name}", format="bytes")
            self.plots_dir.mkdir(parents=True, exist_ok=True)
            local_path = self.plots_dir / name
            local_path.write_bytes(content)
            self._downloaded_plots.add(name)
            saved.append(str(local_path))
        return saved

    def pull_file(self, remote_path: str) -> bytes | None:
        """Pull an arbitrary file back from the sandbox (e.g. state.json)."""
        assert self._sbx is not None
        try:
            return self._sbx.files.read(remote_path, format="bytes")
        except Exception:
            return None

    def pull_state_json(self) -> str | None:
        content = self.pull_file(self.REMOTE_STATE_PATH)
        return content.decode("utf-8") if content is not None else None

    def make_run_python_tool(self):
        """Build a fresh run_python tool bound to this sandbox instance.

        A fresh tool object is built per graph node so LangChain's tool-call
        history in each node's own agent loop stays scoped to that node
        (nodes don't share message history), while all of them execute
        against the same persistent sandbox kernel.
        """
        sbx = self

        @tool
        def run_python(code: str) -> str:
            """Execute Python code in the live sandbox and return stdout/stderr."""
            assert sbx._sbx is not None
            execution = sbx._sbx.run_code(code)

            if execution.error:
                raw_err = execution.error.traceback or f"{execution.error.name}: {execution.error.value}"
                err_out = _truncate(raw_err, sbx.MAX_STDERR_CHARS, keep="tail")
                return f"ERROR: {err_out}"

            output = "\n".join(execution.logs.stdout) or "(no output)"
            output = _truncate(output, sbx.MAX_STDOUT_CHARS, keep="head")

            saved = sbx.download_new_plots()
            if saved:
                output += "\n(saved plots: " + ", ".join(saved) + ")"
            return output

        return run_python
