"""Scoring weights and task score computation.

Dataclasses for lightweight runtime types.
"""

from dataclasses import dataclass


@dataclass
class ScoringWeights:
    """Weight configuration for score calculation."""

    outcome: float = 0.6
    efficiency: float = 0.2
    recovery: float = 0.2
    memory_utilization: float = 0.0
    preference_adherence: float = 0.0
    tone_appropriateness: float = 0.0


@dataclass
class TaskScore:
    """Computed score for a single task execution."""

    outcome: float
    efficiency: float
    recovery: float
    total: float
    memory_utilization: float = 0.0
    preference_adherence: float = 0.0
    tone_appropriateness: float = 0.0

    @classmethod
    def calculate(
        cls,
        outcome: float,
        efficiency: float,
        recovery: float,
        weights: ScoringWeights,
        memory_utilization: float = 0.0,
        preference_adherence: float = 0.0,
        tone_appropriateness: float = 0.0,
    ) -> "TaskScore":
        """Compute weighted total score from individual dimensions."""
        total = (
            outcome * weights.outcome
            + efficiency * weights.efficiency
            + recovery * weights.recovery
            + memory_utilization * weights.memory_utilization
            + preference_adherence * weights.preference_adherence
            + tone_appropriateness * weights.tone_appropriateness
        )
        return cls(
            outcome=outcome,
            efficiency=efficiency,
            recovery=recovery,
            total=total,
            memory_utilization=memory_utilization,
            preference_adherence=preference_adherence,
            tone_appropriateness=tone_appropriateness,
        )

    @staticmethod
    def pass_k(scores: list["TaskScore"], threshold: float = 0.5) -> float:
        """Return 1.0 if ALL scores have outcome >= threshold, else 0.0."""
        if all(s.outcome >= threshold for s in scores):
            return 1.0
        return 0.0
