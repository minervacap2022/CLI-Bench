"""Observation and Action types for the benchmark agent loop.

Dataclasses (not Pydantic) for lightweight runtime types.
"""

from dataclasses import dataclass, field


@dataclass
class Observation:
    """What the agent sees each turn: task, tools, execution output, memory."""

    task: str
    tools: list[dict]
    stdout: str = ""
    stderr: str = ""
    turn: int = 0
    memory: dict | None = None

    @property
    def is_first_turn(self) -> bool:
        return self.turn == 0


@dataclass
class Action:
    """What the agent does: run a command or finish with a result."""

    cmd: list[str] | None = None
    result: str | None = None

    @classmethod
    def command(cls, cmd: list[str]) -> "Action":
        """Create an action that runs a command."""
        return cls(cmd=cmd)

    @classmethod
    def finish(cls, result: str) -> "Action":
        """Create an action that finishes the task with a result."""
        return cls(result=result)

    @property
    def is_command(self) -> bool:
        return self.cmd is not None

    @property
    def is_finish(self) -> bool:
        return self.result is not None
