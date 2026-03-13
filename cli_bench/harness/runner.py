"""Runner — multi-turn agent execution loop.

Drives an agent through a task by building observations,
routing commands to mock backends, and collecting results.
"""

import time
from dataclasses import dataclass, field
from typing import Any

from cli_bench.agents.base import BenchAgent
from cli_bench.mock_backends.base import BaseMockBackend
from cli_bench.models.observation import Observation
from cli_bench.models.task import BenchTask


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
    ) -> None:
        self._agent = agent
        self._backends = backends

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

        tool_prompts = [{"name": tool} for tool in task.tools_provided]

        for turn in range(task.max_turns):
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

    def _snapshot_all(self) -> dict[str, Any]:
        """Snapshot state from all backends."""
        return {
            name: backend.get_state_snapshot()
            for name, backend in self._backends.items()
        }


def _now_ms() -> int:
    """Current time in milliseconds."""
    return int(time.monotonic() * 1000)
