"""DummyAgent and ScriptedAgent — reference implementations for testing.

DummyAgent: baseline that immediately finishes (score = 0).
ScriptedAgent: returns pre-configured actions in sequence.
"""

from cli_bench.agents.base import BenchAgent
from cli_bench.models.observation import Action, Observation


class DummyAgent(BenchAgent):
    """Baseline agent that immediately finishes. Score = 0."""

    async def act(self, observation: Observation) -> Action:
        return Action.finish("No action taken")

    def reset(self) -> None:
        pass


class ScriptedAgent(BenchAgent):
    """Returns pre-configured actions in sequence. For testing only."""

    def __init__(self, actions: list[Action]) -> None:
        self._actions = list(actions)
        self._index = 0

    async def act(self, observation: Observation) -> Action:
        if self._index >= len(self._actions):
            return Action.finish("Script exhausted")
        action = self._actions[self._index]
        self._index += 1
        return action

    def reset(self) -> None:
        self._index = 0
