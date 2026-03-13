"""BenchAgent abstract base class for benchmark agent loop.

All agents must implement act() and reset().
"""

from abc import ABC, abstractmethod

from cli_bench.models.observation import Action, Observation


class BenchAgent(ABC):
    """Abstract base for benchmark agents.

    Agents observe task state and produce actions (commands or finish).
    """

    @abstractmethod
    async def act(self, observation: Observation) -> Action:
        """Given an observation, return the next action."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset agent state between runs."""
        ...
