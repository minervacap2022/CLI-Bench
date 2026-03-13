"""Tests for the Evaluator scoring engine."""

import pytest

from cli_bench.harness.evaluator import Evaluator
from cli_bench.harness.runner import RunResult
from cli_bench.mock_backends.github import GitHubMockBackend
from cli_bench.models.scoring import TaskScore
from cli_bench.models.task import BenchTask


def _make_task(
    expected_state: dict,
    optimal_commands: int = 1,
    scoring_outcome: float = 0.6,
    scoring_efficiency: float = 0.2,
    scoring_recovery: float = 0.2,
) -> BenchTask:
    """Helper to create a BenchTask with customizable fields."""
    return BenchTask(
        id="eval-test",
        title="Eval test",
        difficulty="easy",
        category="test",
        description="Test evaluation",
        tools_provided=["gh"],
        initial_state={"github": {"repos": {}}},
        expected_state=expected_state,
        max_turns=10,
        optimal_commands=optimal_commands,
        scoring={
            "outcome": scoring_outcome,
            "efficiency": scoring_efficiency,
            "recovery": scoring_recovery,
        },
    )


def _make_run_result(
    action_log: list[dict] | None = None,
    finished: bool = True,
    turns: int = 1,
) -> RunResult:
    """Helper to create a RunResult with customizable fields."""
    return RunResult(
        task_id="eval-test",
        turns=turns,
        finished=finished,
        final_state={},
        action_log=action_log or [],
        elapsed_ms=100,
        agent_result="Done",
    )


class TestPerfectOutcome:
    def test_all_expected_state_matches(self) -> None:
        """When backend state matches expected, outcome = 1.0."""
        the_issue = {"number": 1, "title": "Bug", "state": "open", "assignee": None, "labels": [], "body": ""}
        expected_state = {
            "github": {
                "repos": {
                    "acme/app": {
                        "issues": [the_issue],
                        "pulls": [],
                        "commits": [],
                    }
                }
            }
        }
        task = _make_task(expected_state=expected_state)

        # Set up backend with exactly matching state
        backend = GitHubMockBackend({
            "repos": {
                "acme/app": {
                    "issues": [the_issue],
                    "pulls": [],
                    "commits": [],
                }
            }
        })

        result = _make_run_result()
        evaluator = Evaluator()
        score = evaluator.evaluate(task, result, {"github": backend})

        assert isinstance(score, TaskScore)
        assert score.outcome == 1.0


class TestPartialOutcome:
    def test_some_state_matches(self) -> None:
        """When backend state partially matches expected, 0 < outcome < 1."""
        bug_issue = {"number": 1, "title": "Bug", "state": "open"}
        feature_issue = {"number": 2, "title": "Feature", "state": "open"}
        expected_state = {
            "github": {
                "repos": {
                    "acme/app": {
                        "issues": [bug_issue, feature_issue],
                    }
                }
            }
        }
        task = _make_task(expected_state=expected_state)

        # Backend has only one of two expected issues
        backend = GitHubMockBackend({
            "repos": {
                "acme/app": {
                    "issues": [bug_issue],
                    "pulls": [],
                    "commits": [],
                }
            }
        })

        result = _make_run_result()
        evaluator = Evaluator()
        score = evaluator.evaluate(task, result, {"github": backend})

        assert 0.0 < score.outcome < 1.0


class TestEfficiencyOptimal:
    def test_actual_equals_optimal(self) -> None:
        """When actual commands == optimal_commands, efficiency = 1.0."""
        task = _make_task(
            expected_state={"github": {"repos": {}}},
            optimal_commands=2,
        )

        action_log = [
            {"command": ["gh", "issue", "list"], "stdout": "[]", "stderr": ""},
            {"command": ["gh", "issue", "create"], "stdout": "{}", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)

        backend = GitHubMockBackend({"repos": {}})
        evaluator = Evaluator()
        score = evaluator.evaluate(task, result, {"github": backend})

        assert score.efficiency == 1.0


class TestEfficiencyDouble:
    def test_actual_is_double_optimal(self) -> None:
        """When actual = 2x optimal, efficiency = 0.5."""
        task = _make_task(
            expected_state={"github": {"repos": {}}},
            optimal_commands=2,
        )

        action_log = [
            {"command": ["gh", "issue", "list"], "stdout": "[]", "stderr": ""},
            {"command": ["gh", "issue", "list"], "stdout": "[]", "stderr": ""},
            {"command": ["gh", "issue", "create"], "stdout": "{}", "stderr": ""},
            {"command": ["gh", "issue", "create"], "stdout": "{}", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)

        backend = GitHubMockBackend({"repos": {}})
        evaluator = Evaluator()
        score = evaluator.evaluate(task, result, {"github": backend})

        assert score.efficiency == pytest.approx(0.5)


class TestRecoveryWithErrors:
    def test_error_then_success(self) -> None:
        """Error followed by successful command yields recovery = 1.0."""
        task = _make_task(expected_state={"github": {"repos": {}}})

        action_log = [
            {"command": ["gh", "issue", "list"], "stdout": "", "stderr": "error: not found"},
            {"command": ["gh", "issue", "list", "--repo", "acme/app"], "stdout": "[]", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)

        backend = GitHubMockBackend({"repos": {}})
        evaluator = Evaluator()
        score = evaluator.evaluate(task, result, {"github": backend})

        assert score.recovery == 1.0


class TestRecoveryNoErrors:
    def test_no_errors_encountered(self) -> None:
        """No errors encountered gives recovery = 0.5 (neutral)."""
        task = _make_task(expected_state={"github": {"repos": {}}})

        action_log = [
            {"command": ["gh", "issue", "list"], "stdout": "[]", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)

        backend = GitHubMockBackend({"repos": {}})
        evaluator = Evaluator()
        score = evaluator.evaluate(task, result, {"github": backend})

        assert score.recovery == 0.5


class TestRecoveryFailedToRecover:
    def test_error_without_recovery(self) -> None:
        """Errors encountered but agent didn't recover yields recovery = 0.0."""
        task = _make_task(expected_state={"github": {"repos": {}}})

        action_log = [
            {"command": ["gh", "issue", "list"], "stdout": "", "stderr": "error: not found"},
            {"command": ["gh", "issue", "list"], "stdout": "", "stderr": "error: still broken"},
        ]
        result = _make_run_result(action_log=action_log)

        backend = GitHubMockBackend({"repos": {}})
        evaluator = Evaluator()
        score = evaluator.evaluate(task, result, {"github": backend})

        assert score.recovery == 0.0


class TestFullScore:
    def test_complete_scoring_pipeline(self) -> None:
        """Full scoring pipeline produces weighted total."""
        task = _make_task(
            expected_state={"github": {"repos": {}}},
            optimal_commands=1,
            scoring_outcome=0.6,
            scoring_efficiency=0.2,
            scoring_recovery=0.2,
        )

        action_log = [
            {"command": ["gh", "issue", "list"], "stdout": "[]", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)

        backend = GitHubMockBackend({"repos": {}})
        evaluator = Evaluator()
        score = evaluator.evaluate(task, result, {"github": backend})

        assert isinstance(score, TaskScore)
        # outcome=1.0, efficiency=1.0 (1 cmd, optimal=1), recovery=0.5 (no errors)
        # total = 1.0*0.6 + 1.0*0.2 + 0.5*0.2 = 0.6 + 0.2 + 0.1 = 0.9
        assert score.total == pytest.approx(0.9)
