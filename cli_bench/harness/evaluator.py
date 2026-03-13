"""Evaluator — scoring engine for benchmark task runs.

Scores agent performance on outcome (state diff), efficiency (command count),
and recovery (error handling) dimensions.
"""

from cli_bench.harness.runner import RunResult
from cli_bench.mock_backends.base import BaseMockBackend
from cli_bench.models.scoring import ScoringWeights, TaskScore
from cli_bench.models.task import BenchTask


class Evaluator:
    """Score a run result against task expectations."""

    def evaluate(
        self,
        task: BenchTask,
        result: RunResult,
        backends: dict[str, BaseMockBackend],
    ) -> TaskScore:
        """Score a run result against task expectations.

        Dimensions:
        - outcome: diff each backend's state against task.expected_state
        - efficiency: min(1.0, optimal_commands / max(1, actual_commands))
        - recovery: error handling analysis from action_log
        """
        outcome = self._score_outcome(task, backends)
        efficiency = self._score_efficiency(task, result)
        recovery = self._score_recovery(result)

        weights = ScoringWeights(
            outcome=task.scoring.outcome,
            efficiency=task.scoring.efficiency,
            recovery=task.scoring.recovery,
            memory_utilization=task.scoring.memory_utilization,
            preference_adherence=task.scoring.preference_adherence,
            tone_appropriateness=task.scoring.tone_appropriateness,
        )

        return TaskScore.calculate(
            outcome=outcome,
            efficiency=efficiency,
            recovery=recovery,
            weights=weights,
        )

    def _score_outcome(
        self,
        task: BenchTask,
        backends: dict[str, BaseMockBackend],
    ) -> float:
        """Diff each backend's state against task.expected_state. Average the scores."""
        expected = task.expected_state
        if not expected:
            return 1.0

        scores: list[float] = []
        for service_name, expected_service_state in expected.items():
            backend = backends.get(service_name)
            if backend is None:
                scores.append(0.0)
                continue
            diff = backend.diff(expected_service_state)
            scores.append(diff.score)

        if not scores:
            return 1.0
        return sum(scores) / len(scores)

    def _score_efficiency(self, task: BenchTask, result: RunResult) -> float:
        """min(1.0, optimal_commands / max(1, actual_commands))."""
        actual = max(1, len(result.action_log))
        return min(1.0, task.optimal_commands / actual)

    def _score_recovery(self, result: RunResult) -> float:
        """Analyze action_log for error recovery.

        - 1.0 if errors encountered AND agent recovered (subsequent successful commands)
        - 0.5 if no errors encountered (neutral)
        - 0.0 if errors encountered AND agent didn't recover
        """
        action_log = result.action_log
        if not action_log:
            return 0.5

        error_indices: list[int] = []
        for i, entry in enumerate(action_log):
            if entry.get("stderr"):
                error_indices.append(i)

        if not error_indices:
            return 0.5

        # Check if there's a successful command after the last error
        last_error_idx = error_indices[-1]
        for i in range(last_error_idx + 1, len(action_log)):
            if not action_log[i].get("stderr"):
                return 1.0

        return 0.0
