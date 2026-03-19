"""Runner — multi-turn agent execution loop.

Drives an agent through a task by building observations,
routing commands to mock backends, and collecting results.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cli_bench.agents.base import BenchAgent
from cli_bench.mock_backends.base import BaseMockBackend
from cli_bench.models.observation import Observation
from cli_bench.models.task import BenchTask
from cli_bench.models.tool_adapter import ToolAdapter

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of running an agent on a single task."""

    task_id: str
    turns: int
    finished: bool
    final_state: dict[str, Any]
    action_log: list[dict[str, Any]]
    elapsed_ms: int
    agent_result: str | None = None


class Runner:
    """Multi-turn agent execution loop.

    Routes agent commands to the appropriate mock backend by matching
    command[0] (the tool binary name) against the backends dict keys.
    """

    def __init__(
        self,
        agent: BenchAgent,
        backends: dict[str, BaseMockBackend],
        tool_adapters_dir: Path | None = None,
    ) -> None:
        self._agent = agent
        self._backends = backends
        self._tool_adapters = self._load_adapters(tool_adapters_dir) if tool_adapters_dir else {}

    @staticmethod
    def _load_adapters(adapters_dir: Path) -> dict[str, ToolAdapter]:
        """Load all tool adapter YAML files from the given directory.

        Returns a dict keyed by the adapter's binary name (e.g. ``gh``, ``slack``).
        """
        adapters: dict[str, ToolAdapter] = {}
        if not adapters_dir.is_dir():
            logger.warning("Tool adapters directory does not exist: %s", adapters_dir)
            return adapters
        for yaml_path in sorted(adapters_dir.glob("*.yaml")):
            try:
                adapter = ToolAdapter.from_yaml(yaml_path)
                adapters[adapter.binary] = adapter
            except Exception:
                logger.warning("Failed to load tool adapter: %s", yaml_path, exc_info=True)
        return adapters

    async def run_task(
        self,
        task: BenchTask,
        memory: dict | None = None,
    ) -> RunResult:
        """Execute agent against task. Returns RunResult with final state and action log."""
        start_ms = _now_ms()
        action_log: list[dict[str, Any]] = []
        stdout = ""
        stderr = ""

        # Inject tool adapters into backends so --help interception works
        for tool_name, backend in self._backends.items():
            adapter = self._tool_adapters.get(tool_name)
            if adapter is not None:
                backend.set_tool_adapter(adapter)

        tool_prompts = self._build_tool_prompts(task.tools_provided, task.doc_visibility)

        timeout_ms = task.timeout_seconds * 1000

        for turn in range(task.max_turns):
            # Timeout enforcement: check elapsed time before each turn
            if (_now_ms() - start_ms) >= timeout_ms:
                elapsed = _now_ms() - start_ms
                return RunResult(
                    task_id=task.id,
                    turns=turn,
                    finished=False,
                    final_state=self._snapshot_all(),
                    action_log=action_log,
                    elapsed_ms=elapsed,
                )

            observation = Observation(
                task=task.description,
                tools=tool_prompts,
                stdout=stdout,
                stderr=stderr,
                turn=turn,
                memory=memory,
            )

            action = await self._agent.act(observation)

            if action.is_finish:
                elapsed = _now_ms() - start_ms
                return RunResult(
                    task_id=task.id,
                    turns=turn + 1,
                    finished=True,
                    final_state=self._snapshot_all(),
                    action_log=action_log,
                    elapsed_ms=elapsed,
                    agent_result=action.result,
                )

            if action.is_command and action.cmd:
                binary = action.cmd[0]
                backend = self._backends.get(binary)
                if backend is not None:
                    mock_result = backend.execute(action.cmd)
                    stdout = mock_result.stdout
                    stderr = mock_result.stderr
                else:
                    stdout = ""
                    stderr = f"unknown tool: {binary}"

                action_log.append({
                    "command": action.cmd,
                    "stdout": stdout,
                    "stderr": stderr,
                })

        elapsed = _now_ms() - start_ms
        return RunResult(
            task_id=task.id,
            turns=task.max_turns,
            finished=False,
            final_state=self._snapshot_all(),
            action_log=action_log,
            elapsed_ms=elapsed,
        )

    def _build_tool_prompts(
        self,
        tools: list[str],
        doc_visibility: str = "full",
    ) -> list[dict[str, Any]]:
        """Build tool prompt dicts, filtered by *doc_visibility*.

        - ``"full"``: current behaviour — full docs, commands, examples.
        - ``"description_only"``: name + description + discovery hint.
        - ``"name_only"``: just binary name + discovery hint.
        """
        prompts: list[dict[str, Any]] = []
        for tool in tools:
            adapter = self._tool_adapters.get(tool)
            if adapter is None:
                prompts.append({"name": tool})
                continue

            if doc_visibility == "name_only":
                prompts.append({
                    "name": adapter.binary,
                    "hint": f"Use '{adapter.binary} --help' to discover commands.",
                })
            elif doc_visibility == "description_only":
                prompts.append({
                    "name": adapter.binary,
                    "description": adapter.description,
                    "hint": f"Use '{adapter.binary} --help' to discover commands.",
                })
            else:
                # full — unchanged
                prompts.append({
                    "name": adapter.binary,
                    "description": adapter.description,
                    "commands": [
                        {
                            "name": cmd.name,
                            "description": cmd.description,
                            "args": [
                                {
                                    "name": arg.name,
                                    "type": arg.type,
                                    "required": arg.required,
                                    "description": arg.description,
                                    **({"default": arg.default} if arg.default is not None else {}),
                                    **({"values": arg.values} if arg.values else {}),
                                }
                                for arg in cmd.args
                            ],
                            "output_format": cmd.output_format,
                            "side_effects": cmd.side_effects,
                            **({"example": cmd.example} if cmd.example else {}),
                        }
                        for cmd in adapter.commands
                    ],
                    "full_documentation": adapter.to_prompt(),
                })
        return prompts

    def _snapshot_all(self) -> dict[str, Any]:
        """Snapshot state from all backends."""
        return {
            name: backend.get_state_snapshot()
            for name, backend in self._backends.items()
        }


def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.monotonic() * 1000)
