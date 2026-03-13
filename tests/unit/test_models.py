"""Tests for Observation, Action, BenchTask, TaskScore models."""

from pathlib import Path

import pytest
import yaml

from cli_bench.models.observation import Action, Observation
from cli_bench.models.scoring import ScoringWeights, TaskScore
from cli_bench.models.task import BenchTask, ScoringConfig, StateAssertion


class TestObservation:
    def test_initial_observation(self) -> None:
        """Observation at turn=0 has is_first_turn=True."""
        obs = Observation(
            task="Create a GitHub issue",
            tools=[{"name": "gh", "binary": "gh"}],
        )
        assert obs.turn == 0
        assert obs.is_first_turn is True
        assert obs.stdout == ""
        assert obs.stderr == ""
        assert obs.memory is None

    def test_observation_with_output(self) -> None:
        """Observation at turn=3 has is_first_turn=False and preserves stdout/stderr."""
        obs = Observation(
            task="List issues",
            tools=[],
            stdout='{"issues": []}',
            stderr="warning: rate limit",
            turn=3,
            memory={"last_repo": "minervacap2022/Klik"},
        )
        assert obs.turn == 3
        assert obs.is_first_turn is False
        assert obs.stdout == '{"issues": []}'
        assert obs.stderr == "warning: rate limit"
        assert obs.memory == {"last_repo": "minervacap2022/Klik"}


class TestAction:
    def test_action_command(self) -> None:
        """Action.command() creates an action with is_command=True, is_finish=False."""
        action = Action.command(["gh", "issue", "list", "--repo", "test/repo"])
        assert action.is_command is True
        assert action.is_finish is False
        assert action.cmd == ["gh", "issue", "list", "--repo", "test/repo"]
        assert action.result is None

    def test_action_finish(self) -> None:
        """Action.finish() creates an action with is_finish=True, is_command=False."""
        action = Action.finish("Created issue #42")
        assert action.is_finish is True
        assert action.is_command is False
        assert action.result == "Created issue #42"
        assert action.cmd is None


class TestStateAssertion:
    def test_basic_assertion(self) -> None:
        assertion = StateAssertion(field="issues.count", value=1)
        assert assertion.field == "issues.count"
        assert assertion.value == 1
        assert assertion.contains is None
        assert assertion.not_value is None

    def test_contains_assertion(self) -> None:
        assertion = StateAssertion(
            field="issues.labels",
            contains=["bug", "urgent"],
        )
        assert assertion.contains == ["bug", "urgent"]


class TestBenchTask:
    def test_bench_task_from_yaml(self, tmp_path: Path) -> None:
        """BenchTask.from_yaml loads all fields correctly from a YAML file."""
        task_data = {
            "id": "cli-gh-001",
            "title": "Create a GitHub issue",
            "difficulty": "easy",
            "category": "github",
            "description": "Create a new issue in the test/repo repository with title 'Bug report'",
            "tools_provided": ["gh"],
            "initial_state": {"issues": []},
            "expected_state": {"issues": [{"title": "Bug report"}]},
            "scoring": {
                "outcome": 0.7,
                "efficiency": 0.2,
                "recovery": 0.1,
            },
            "max_turns": 5,
            "optimal_commands": 1,
            "timeout_seconds": 120,
        }
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml.dump(task_data))

        task = BenchTask.from_yaml(yaml_path)

        assert task.id == "cli-gh-001"
        assert task.title == "Create a GitHub issue"
        assert task.difficulty == "easy"
        assert task.category == "github"
        assert task.description == "Create a new issue in the test/repo repository with title 'Bug report'"
        assert task.tools_provided == ["gh"]
        assert task.initial_state == {"issues": []}
        assert task.expected_state == {"issues": [{"title": "Bug report"}]}
        assert task.scoring.outcome == 0.7
        assert task.scoring.efficiency == 0.2
        assert task.scoring.recovery == 0.1
        assert task.max_turns == 5
        assert task.optimal_commands == 1
        assert task.timeout_seconds == 120
        assert task.persona is None
        assert task.memory_required is None

    def test_bench_task_with_persona_and_memory(self, tmp_path: Path) -> None:
        """BenchTask loads optional persona and memory_required fields."""
        task_data = {
            "id": "klik-mem-001",
            "title": "Schedule meeting with preferences",
            "difficulty": "hard",
            "category": "calendar",
            "description": "Schedule a meeting considering user preferences",
            "tools_provided": ["gcal", "slack"],
            "initial_state": {},
            "expected_state": {"meetings": [{"title": "Standup"}]},
            "max_turns": 10,
            "persona": "alex-pm",
            "memory_required": ["preferred_meeting_time", "team_members"],
        }
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml.dump(task_data))

        task = BenchTask.from_yaml(yaml_path)
        assert task.persona == "alex-pm"
        assert task.memory_required == ["preferred_meeting_time", "team_members"]

    def test_bench_task_defaults(self, tmp_path: Path) -> None:
        """BenchTask uses correct defaults for optional fields."""
        task_data = {
            "id": "cli-test-001",
            "title": "Test task",
            "difficulty": "medium",
            "category": "test",
            "description": "A test task",
            "tools_provided": ["echo"],
            "initial_state": {},
            "expected_state": {"done": True},
            "max_turns": 3,
        }
        yaml_path = tmp_path / "task.yaml"
        yaml_path.write_text(yaml.dump(task_data))

        task = BenchTask.from_yaml(yaml_path)
        assert task.optimal_commands == 1
        assert task.timeout_seconds == 300
        assert task.scoring.outcome == 0.6
        assert task.scoring.efficiency == 0.2
        assert task.scoring.recovery == 0.2


class TestTaskScore:
    def test_task_score_calculation(self) -> None:
        """TaskScore.calculate produces correct weighted total."""
        weights = ScoringWeights(outcome=0.6, efficiency=0.2, recovery=0.2)
        score = TaskScore.calculate(
            outcome=1.0,
            efficiency=0.5,
            recovery=0.8,
            weights=weights,
        )
        # total = 1.0*0.6 + 0.5*0.2 + 0.8*0.2 = 0.6 + 0.1 + 0.16 = 0.86
        assert score.outcome == 1.0
        assert score.efficiency == 0.5
        assert score.recovery == 0.8
        assert score.total == pytest.approx(0.86)

    def test_task_score_with_klik_dimensions(self) -> None:
        """TaskScore.calculate handles KLIK-specific dimensions."""
        weights = ScoringWeights(
            outcome=0.4,
            efficiency=0.1,
            recovery=0.1,
            memory_utilization=0.2,
            preference_adherence=0.1,
            tone_appropriateness=0.1,
        )
        score = TaskScore.calculate(
            outcome=1.0,
            efficiency=0.5,
            recovery=0.8,
            weights=weights,
            memory_utilization=0.9,
            preference_adherence=0.7,
            tone_appropriateness=1.0,
        )
        # total = 1.0*0.4 + 0.5*0.1 + 0.8*0.1 + 0.9*0.2 + 0.7*0.1 + 1.0*0.1
        # = 0.4 + 0.05 + 0.08 + 0.18 + 0.07 + 0.1 = 0.88
        assert score.total == pytest.approx(0.88)
        assert score.memory_utilization == 0.9
        assert score.preference_adherence == 0.7
        assert score.tone_appropriateness == 1.0

    def test_pass_k_all_pass(self) -> None:
        """pass_k returns 1.0 when ALL scores have outcome >= threshold."""
        weights = ScoringWeights()
        scores = [
            TaskScore.calculate(outcome=0.8, efficiency=0.5, recovery=0.5, weights=weights),
            TaskScore.calculate(outcome=0.6, efficiency=0.3, recovery=0.4, weights=weights),
            TaskScore.calculate(outcome=1.0, efficiency=1.0, recovery=1.0, weights=weights),
            TaskScore.calculate(outcome=0.5, efficiency=0.2, recovery=0.1, weights=weights),
            TaskScore.calculate(outcome=0.9, efficiency=0.7, recovery=0.6, weights=weights),
        ]
        assert TaskScore.pass_k(scores, threshold=0.5) == 1.0

    def test_pass_k_some_fail(self) -> None:
        """pass_k returns 0.0 when any score has outcome < threshold."""
        weights = ScoringWeights()
        scores = [
            TaskScore.calculate(outcome=0.8, efficiency=0.5, recovery=0.5, weights=weights),
            TaskScore.calculate(outcome=0.3, efficiency=0.3, recovery=0.4, weights=weights),
            TaskScore.calculate(outcome=1.0, efficiency=1.0, recovery=1.0, weights=weights),
        ]
        assert TaskScore.pass_k(scores, threshold=0.5) == 0.0
