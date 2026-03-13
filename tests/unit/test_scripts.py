"""Tests for CLI entry point scripts and HuggingFace upload.

Tests argument parsing, agent creation, result saving, and HF upload logic.
No real API calls or file system operations beyond tmp_path.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml


class TestRunBenchmarkScript:
    """Tests for run_benchmark.py script functions."""

    def test_create_agent_dummy(self) -> None:
        """_create_agent('dummy') returns a DummyAgent."""
        from scripts.run_benchmark import _create_agent

        agent = _create_agent("dummy", None)
        from cli_bench.agents.dummy import DummyAgent

        assert isinstance(agent, DummyAgent)

    def test_create_agent_openai(self) -> None:
        """_create_agent('openai') returns an OpenAIAgent with given model."""
        from scripts.run_benchmark import _create_agent

        agent = _create_agent("openai", "gpt-4o-mini")
        from cli_bench.agents.openai_agent import OpenAIAgent

        assert isinstance(agent, OpenAIAgent)
        assert agent.model == "gpt-4o-mini"

    def test_create_agent_openai_default_model(self) -> None:
        """_create_agent('openai', None) uses default model gpt-4o."""
        from scripts.run_benchmark import _create_agent

        agent = _create_agent("openai", None)
        from cli_bench.agents.openai_agent import OpenAIAgent

        assert isinstance(agent, OpenAIAgent)
        assert agent.model == "gpt-4o"

    def test_create_agent_anthropic(self) -> None:
        """_create_agent('anthropic') returns an AnthropicAgent with given model."""
        from scripts.run_benchmark import _create_agent

        agent = _create_agent("anthropic", "claude-sonnet-4-20250514")
        from cli_bench.agents.anthropic_agent import AnthropicAgent

        assert isinstance(agent, AnthropicAgent)
        assert agent.model == "claude-sonnet-4-20250514"

    def test_create_agent_anthropic_default_model(self) -> None:
        """_create_agent('anthropic', None) uses default model."""
        from scripts.run_benchmark import _create_agent

        agent = _create_agent("anthropic", None)
        from cli_bench.agents.anthropic_agent import AnthropicAgent

        assert isinstance(agent, AnthropicAgent)
        assert agent.model == "claude-sonnet-4-20250514"

    def test_save_report_creates_json(self, tmp_path: Path) -> None:
        """_save_report writes report.json to the output directory."""
        from scripts.run_benchmark import _save_report

        # Create a mock report with the expected attributes
        report = MagicMock()
        report.overall_score = 0.75
        report.overall_pass_k = 0.6
        report.by_difficulty = {"easy": 0.9, "hard": 0.5}
        report.results = []
        report.__dict__ = {
            "overall_score": 0.75,
            "overall_pass_k": 0.6,
            "by_difficulty": {"easy": 0.9, "hard": 0.5},
            "results": [],
            "by_category": {},
            "total_cost_usd": 0.0,
            "total_time_ms": 100,
        }

        output_dir = tmp_path / "output"
        _save_report(output_dir, report)

        report_path = output_dir / "report.json"
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["overall_score"] == 0.75

    def test_save_result_creates_task_json(self, tmp_path: Path) -> None:
        """_save_result writes {task_id}.json to the output directory."""
        from scripts.run_benchmark import _save_result

        result = MagicMock()
        result.task_id = "cli-gh-001"
        result.mean_score = 0.8
        result.pass_k = 1.0

        output_dir = tmp_path / "output"
        _save_result(output_dir, result)

        result_path = output_dir / "cli-gh-001.json"
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["task_id"] == "cli-gh-001"
        assert data["mean_score"] == 0.8


class TestGenerateReportScript:
    """Tests for generate_report.py script functions."""

    def test_generate_markdown_report(self, tmp_path: Path) -> None:
        """generate_markdown produces valid markdown from report data."""
        from scripts.generate_report import generate_markdown

        report_data = {
            "overall_score": 0.72,
            "overall_pass_k": 0.6,
            "by_difficulty": {"easy": 0.9, "medium": 0.7, "hard": 0.5},
            "by_category": {"github": 0.8, "slack": 0.6},
            "total_cost_usd": 1.23,
            "total_time_ms": 45000,
        }
        md = generate_markdown(report_data)
        assert "0.72" in md
        assert "easy" in md
        assert "github" in md

    def test_generate_markdown_empty_report(self) -> None:
        """generate_markdown handles empty report data."""
        from scripts.generate_report import generate_markdown

        report_data = {
            "overall_score": 0.0,
            "overall_pass_k": 0.0,
            "by_difficulty": {},
            "by_category": {},
            "total_cost_usd": 0.0,
            "total_time_ms": 0,
        }
        md = generate_markdown(report_data)
        assert "0.0" in md


class TestUploadToHfScript:
    """Tests for upload_to_hf.py script functions."""

    def test_collect_task_files_finds_yaml(self, tmp_path: Path) -> None:
        """collect_task_files returns all .yaml files in tasks dir."""
        from scripts.upload_to_hf import collect_task_files

        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "task1.yaml").write_text("id: t1")
        (tasks_dir / "task2.yaml").write_text("id: t2")
        (tasks_dir / "readme.md").write_text("ignore me")

        files = collect_task_files(tasks_dir)
        assert len(files) == 2
        assert all(f.suffix == ".yaml" for f in files)

    def test_collect_task_files_empty_dir(self, tmp_path: Path) -> None:
        """collect_task_files returns empty list for empty directory."""
        from scripts.upload_to_hf import collect_task_files

        tasks_dir = tmp_path / "empty"
        tasks_dir.mkdir()

        files = collect_task_files(tasks_dir)
        assert files == []

    def test_build_repo_id(self) -> None:
        """build_repo_id formats org/benchmark correctly."""
        from scripts.upload_to_hf import build_repo_id

        assert build_repo_id("minervacap2022", "CLI_Bench") == "minervacap2022/CLI-Bench"
