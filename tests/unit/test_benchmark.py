"""Tests for BenchmarkRunner full pipeline."""

from pathlib import Path

import pytest
import yaml

from cli_bench.agents.dummy import DummyAgent
from cli_bench.harness.benchmark import BenchmarkReport, BenchmarkRunner, TaskResult


def _write_task_yaml(path: Path, task_id: str, difficulty: str, category: str) -> None:
    """Write a minimal task YAML file."""
    data = {
        "id": task_id,
        "title": f"Task {task_id}",
        "difficulty": difficulty,
        "category": category,
        "description": f"Description for {task_id}",
        "tools_provided": ["gh"],
        "initial_state": {
            "github": {
                "repos": {
                    "acme/app": {
                        "issues": [],
                        "pulls": [],
                        "commits": [],
                    }
                }
            }
        },
        "expected_state": {
            "github": {
                "repos": {
                    "acme/app": {
                        "issues": [{"title": "New issue"}],
                    }
                }
            }
        },
        "max_turns": 5,
        "optimal_commands": 1,
        "scoring": {
            "outcome": 0.6,
            "efficiency": 0.2,
            "recovery": 0.2,
        },
    }
    path.write_text(yaml.dump(data))


class TestBenchmarkWithDummyAgent:
    @pytest.mark.asyncio
    async def test_two_tasks_produces_report(self, tmp_path: Path) -> None:
        """Running 2 tasks with DummyAgent produces a valid BenchmarkReport."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        _write_task_yaml(tasks_dir / "task1.yaml", "cli-gh-001", "easy", "github")
        _write_task_yaml(tasks_dir / "task2.yaml", "cli-gh-002", "medium", "github")

        agent = DummyAgent()
        runner = BenchmarkRunner(tasks_dir=tasks_dir, agent=agent, k=2)

        report = await runner.run_all()

        assert isinstance(report, BenchmarkReport)
        assert len(report.results) == 2
        assert report.total_time_ms >= 0

        # Each task should have k=2 scores
        for task_result in report.results:
            assert isinstance(task_result, TaskResult)
            assert len(task_result.scores) == 2
            assert isinstance(task_result.mean_score, float)
            assert isinstance(task_result.pass_k, float)

        # DummyAgent does nothing, so outcome should be low (state won't match)
        # Overall score should be a valid float
        assert isinstance(report.overall_score, float)
        assert isinstance(report.overall_pass_k, float)

        # by_difficulty and by_category should have entries
        assert "easy" in report.by_difficulty
        assert "medium" in report.by_difficulty
        assert "github" in report.by_category

    @pytest.mark.asyncio
    async def test_run_single_task(self, tmp_path: Path) -> None:
        """Running a single task by ID produces a TaskResult."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        _write_task_yaml(tasks_dir / "task1.yaml", "cli-gh-001", "easy", "github")

        agent = DummyAgent()
        runner = BenchmarkRunner(tasks_dir=tasks_dir, agent=agent, k=3)

        result = await runner.run_single("cli-gh-001")

        assert isinstance(result, TaskResult)
        assert result.task_id == "cli-gh-001"
        assert len(result.scores) == 3
