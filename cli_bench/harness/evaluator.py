"""Evaluator — scoring engine for benchmark task runs.

Scores agent performance on outcome (state diff), efficiency (command count),
recovery (error handling), and exploration (tool discovery) dimensions.
"""

from cli_bench.harness.runner import RunResult
from cli_bench.mock_backends.base import BaseMockBackend
from cli_bench.models.scoring import ScoringWeights, TaskScore
from cli_bench.models.task import BenchTask

_HELP_FLAGS = {"--help", "-h"}


def _is_help_command(entry: dict) -> bool:
    """Return True if an action-log entry is a ``--help`` / ``-h`` invocation."""
    cmd = entry.get("command", [])
    return bool(set(cmd) & _HELP_FLAGS)


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
        - exploration: tool discovery via --help (non-trivial only when
          ``doc_visibility`` is not ``"full"``)
        """
        outcome = self._score_outcome(task, backends)
        efficiency = self._score_efficiency(task, result)
        recovery = self._score_recovery(result)
        exploration = self._score_exploration(task, result)

        weights = ScoringWeights(
            outcome=task.scoring.outcome,
            efficiency=task.scoring.efficiency,
            recovery=task.scoring.recovery,
            exploration=task.scoring.exploration,
            memory_utilization=task.scoring.memory_utilization,
            preference_adherence=task.scoring.preference_adherence,
            tone_appropriateness=task.scoring.tone_appropriateness,
        )

        return TaskScore.calculate(
            outcome=outcome,
            efficiency=efficiency,
            recovery=recovery,
            exploration=exploration,
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
        """``min(1.0, optimal / actual)`` — help calls excluded in exploration mode."""
        if task.doc_visibility != "full":
            # Don't penalise for --help calls when agent has to discover tools
            non_help = [e for e in result.action_log if not _is_help_command(e)]
            actual = max(1, len(non_help))
        else:
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

    def _score_exploration(self, task: BenchTask, result: RunResult) -> float:
        """Score tool-discovery behaviour via ``--help``.

        Returns ``0.5`` (neutral) when ``doc_visibility == "full"`` so that
        exploration has no effect on tasks that provide full documentation.

        Components (when *not* full):
        - **Proactive discovery (0.4):** agent called ``--help`` before its
          first real (non-help) command.
        - **Exploration efficiency (0.3):** ``min(1, optimal / actual)`` for
          help calls.
        - **Reactive recovery (0.3):** after an error the agent consulted
          ``--help`` before the next real command.
        """
        if task.doc_visibility == "full":
            return 0.5

        action_log = result.action_log
        if not action_log:
            return 0.0

        help_indices: list[int] = []
        non_help_indices: list[int] = []
        for i, entry in enumerate(action_log):
            if _is_help_command(entry):
                help_indices.append(i)
            else:
                non_help_indices.append(i)

        # No help calls at all → zero exploration score
        if not help_indices:
            return 0.0

        # --- Proactive discovery (0.4) ---
        first_non_help = non_help_indices[0] if non_help_indices else len(action_log)
        proactive = 1.0 if help_indices[0] < first_non_help else 0.0

        # --- Exploration efficiency (0.3) ---
        optimal = task.optimal_help_calls if task.optimal_help_calls is not None else 2
        efficiency = min(1.0, optimal / max(1, len(help_indices)))

        # --- Reactive recovery (0.3) ---
        error_indices = [
            i for i, e in enumerate(action_log) if e.get("stderr") and not _is_help_command(e)
        ]
        if not error_indices:
            reactive = 0.5  # neutral — no errors to recover from
        else:
            recovered = False
            for err_idx in error_indices:
                # Check if a --help call appears after error and before next non-help
                for h_idx in help_indices:
                    if h_idx > err_idx:
                        recovered = True
                        break
                if recovered:
                    break
            reactive = 1.0 if recovered else 0.0

        return proactive * 0.4 + efficiency * 0.3 + reactive * 0.3
