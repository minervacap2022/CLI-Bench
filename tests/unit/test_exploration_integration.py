"""Integration tests for the tool exploration mode.

End-to-end tests that run ScriptedAgent through the full pipeline
(BenchmarkRunner or Runner+Evaluator) with name_only/description_only tasks,
verifying that:
- Minimal docs are provided to the agent
- --help commands return useful text and are recorded
- Exploration scoring works within the full pipeline
- Existing full-mode tasks are unaffected
"""

from pathlib import Path

import pytest
import yaml

from cli_bench.agents.dummy import DummyAgent, ScriptedAgent
from cli_bench.harness.benchmark import BenchmarkRunner
from cli_bench.harness.evaluator import Evaluator
from cli_bench.harness.runner import Runner
from cli_bench.mock_backends.fictional import FictionalMockBackend
from cli_bench.mock_backends.github import GitHubMockBackend
from cli_bench.models.observation import Action, Observation
from cli_bench.models.task import BenchTask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_ADAPTERS_DIR = Path("cli_bench/tool_adapters")


def _write_exploration_task(path: Path) -> dict:
    """Write a name_only exploration task YAML and return the data dict."""
    data = {
        "id": "integ-explore-001",
        "title": "(kforge) exploration integration test",
        "difficulty": "medium",
        "category": "custom_cli_exploration",
        "description": "List all artifacts in namespace team-backend using kforge.",
        "tools_provided": ["kforge"],
        "doc_visibility": "name_only",
        "optimal_help_calls": 2,
        "initial_state": {
            "kforge": {
                "artifacts": [
                    {"id": "art-001", "namespace": "team-backend", "type": "docker", "name": "api-server"},
                    {"id": "art-002", "namespace": "ml-models", "type": "docker", "name": "inference"},
                ],
                "pipelines": [],
                "registries": [],
            }
        },
        "expected_state": {
            "kforge": {
                "command_history": [
                    {"pattern": "kforge artifact list.*--namespace team-backend"},
                ],
                "output_contains": ["art-001"],
            }
        },
        "max_turns": 10,
        "optimal_commands": 1,
        "scoring": {
            "outcome": 0.5,
            "efficiency": 0.15,
            "recovery": 0.15,
            "exploration": 0.2,
        },
    }
    path.write_text(yaml.dump(data))
    return data


# =========================================================================
# Integration: Runner + Evaluator with ScriptedAgent
# =========================================================================

class TestExplorationEndToEnd:
    """End-to-end: ScriptedAgent discovers kforge via --help, then lists artifacts."""

    @pytest.mark.asyncio
    async def test_scripted_agent_explores_and_succeeds(self) -> None:
        """Agent uses --help to discover commands, then executes correctly."""
        task_data = {
            "id": "integ-explore-001",
            "title": "(kforge) exploration integration test",
            "difficulty": "medium",
            "category": "custom_cli_exploration",
            "description": "List all artifacts in namespace team-backend using kforge.",
            "tools_provided": ["kforge"],
            "doc_visibility": "name_only",
            "optimal_help_calls": 2,
            "initial_state": {
                "kforge": {
                    "artifacts": [
                        {"id": "art-001", "namespace": "team-backend", "type": "docker", "name": "api-server"},
                        {"id": "art-002", "namespace": "ml-models", "type": "docker", "name": "inference"},
                    ],
                    "pipelines": [],
                    "registries": [],
                }
            },
            "expected_state": {
                "kforge": {
                    "command_history": [
                        {"pattern": "kforge artifact list.*--namespace team-backend"},
                    ],
                    "output_contains": ["art-001"],
                }
            },
            "max_turns": 10,
            "optimal_commands": 1,
            "scoring": {
                "outcome": 0.5,
                "efficiency": 0.15,
                "recovery": 0.15,
                "exploration": 0.2,
            },
        }
        task = BenchTask(**task_data)

        # Script: --help (root) → --help (artifact) → real command → finish
        agent = ScriptedAgent([
            Action.command(["kforge", "--help"]),
            Action.command(["kforge", "artifact", "--help"]),
            Action.command(["kforge", "artifact", "list", "--namespace", "team-backend"]),
            Action.finish("Listed artifacts"),
        ])

        initial = task.initial_state["kforge"]
        backend = FictionalMockBackend(initial, tool_name="kforge")
        backends = {"kforge": backend}

        runner = Runner(
            agent=agent,
            backends=backends,
            tool_adapters_dir=TOOL_ADAPTERS_DIR,
        )
        run_result = await runner.run_task(task)

        # Verify agent finished successfully
        assert run_result.finished is True
        assert run_result.turns == 4

        # Verify help commands returned useful content
        assert len(run_result.action_log) == 3  # 2 help + 1 real command
        # Root help should contain subcommands
        assert "AVAILABLE COMMANDS:" in run_result.action_log[0]["stdout"]
        assert "artifact" in run_result.action_log[0]["stdout"]
        # Subcommand help should list actions
        assert "list" in run_result.action_log[1]["stdout"]
        # Real command should return data
        assert "art-001" in run_result.action_log[2]["stdout"]

        # Verify command history recorded everything
        history = backend.get_command_history()
        assert len(history) == 3
        assert "kforge --help" in history[0]
        assert "kforge artifact --help" in history[1]
        assert "kforge artifact list" in history[2]

        # Score via evaluator
        evaluator = Evaluator()
        score = evaluator.evaluate(task, run_result, backends)

        # Outcome: command_history pattern should match
        assert score.outcome > 0.0
        # Efficiency: only 1 non-help command, optimal=1 → 1.0
        assert score.efficiency == 1.0
        # Exploration: proactive (help before real cmd) + good efficiency
        assert score.exploration > 0.7
        # Total should reflect all dimensions
        assert score.total > 0.0

    @pytest.mark.asyncio
    async def test_scripted_agent_no_help_low_exploration(self) -> None:
        """Agent that skips --help gets 0.0 exploration score."""
        task = BenchTask(
            id="integ-explore-002",
            title="No help test",
            difficulty="medium",
            category="custom_cli_exploration",
            description="List artifacts without using help.",
            tools_provided=["kforge"],
            doc_visibility="name_only",
            optimal_help_calls=2,
            initial_state={
                "kforge": {
                    "artifacts": [
                        {"id": "art-001", "namespace": "team-backend", "type": "docker", "name": "api-server"},
                    ],
                    "pipelines": [],
                    "registries": [],
                }
            },
            expected_state={
                "kforge": {
                    "command_history": [
                        {"pattern": "kforge artifact list"},
                    ],
                }
            },
            max_turns=5,
            optimal_commands=1,
            scoring={
                "outcome": 0.5,
                "efficiency": 0.15,
                "recovery": 0.15,
                "exploration": 0.2,
            },
        )

        agent = ScriptedAgent([
            Action.command(["kforge", "artifact", "list", "--namespace", "team-backend"]),
            Action.finish("Done"),
        ])

        backend = FictionalMockBackend(
            task.initial_state["kforge"], tool_name="kforge"
        )
        backends = {"kforge": backend}

        runner = Runner(agent=agent, backends=backends, tool_adapters_dir=TOOL_ADAPTERS_DIR)
        run_result = await runner.run_task(task)

        evaluator = Evaluator()
        score = evaluator.evaluate(task, run_result, backends)

        assert score.exploration == 0.0
        # But outcome and efficiency should still be good
        assert score.outcome > 0.0
        assert score.efficiency == 1.0


# =========================================================================
# Integration: BenchmarkRunner pipeline with exploration task
# =========================================================================

class TestBenchmarkExplorationPipeline:
    @pytest.mark.asyncio
    async def test_exploration_task_through_benchmark(self, tmp_path: Path) -> None:
        """name_only task runs through BenchmarkRunner pipeline without errors."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _write_exploration_task(tasks_dir / "task.yaml")

        agent = DummyAgent()
        runner = BenchmarkRunner(
            tasks_dir=tasks_dir,
            agent=agent,
            k=1,
            tool_adapters_dir=TOOL_ADAPTERS_DIR,
        )
        report = await runner.run_all()

        assert len(report.results) == 1
        result = report.results[0]
        assert result.task_id == "integ-explore-001"
        assert len(result.scores) == 1
        # DummyAgent finishes immediately → exploration=0, outcome low
        score = result.scores[0]
        assert score.exploration == 0.0  # no help calls, no actions


# =========================================================================
# Integration: Verify help works on real GitHub backend
# =========================================================================

class TestGitHubHelpIntegration:
    @pytest.mark.asyncio
    async def test_gh_help_with_github_backend(self) -> None:
        """GitHub backend returns help text when adapter is injected."""
        task = BenchTask(
            id="integ-gh-help",
            title="GH help test",
            difficulty="easy",
            category="project_mgmt_exploration",
            description="List open issues using gh, discovering commands via --help.",
            tools_provided=["gh"],
            doc_visibility="description_only",
            optimal_help_calls=2,
            initial_state={
                "gh": {
                    "repos": {
                        "acme/app": {
                            "issues": [
                                {"number": 1, "title": "Bug", "state": "open",
                                 "assignee": None, "labels": [], "body": ""},
                            ],
                            "pulls": [],
                            "commits": [],
                        }
                    }
                }
            },
            expected_state={
                "gh": {
                    "command_history": [
                        {"pattern": "gh issue list.*--repo acme/app.*--state open"},
                    ],
                }
            },
            max_turns=8,
            optimal_commands=1,
            scoring={
                "outcome": 0.5,
                "efficiency": 0.15,
                "recovery": 0.15,
                "exploration": 0.2,
            },
        )

        agent = ScriptedAgent([
            Action.command(["gh", "--help"]),
            Action.command(["gh", "issue", "--help"]),
            Action.command(["gh", "issue", "list", "--help"]),
            Action.command(["gh", "issue", "list", "--repo", "acme/app", "--state", "open"]),
            Action.finish("Listed issues"),
        ])

        backend = GitHubMockBackend(task.initial_state["gh"])
        backends = {"gh": backend}

        runner = Runner(agent=agent, backends=backends, tool_adapters_dir=TOOL_ADAPTERS_DIR)
        run_result = await runner.run_task(task)

        assert run_result.finished is True

        # Root help
        assert "AVAILABLE COMMANDS:" in run_result.action_log[0]["stdout"]
        assert "issue" in run_result.action_log[0]["stdout"]
        assert "pr" in run_result.action_log[0]["stdout"]

        # Subcommand help
        assert "list" in run_result.action_log[1]["stdout"]
        assert "create" in run_result.action_log[1]["stdout"]

        # Action help
        assert "--repo" in run_result.action_log[2]["stdout"]
        assert "--state" in run_result.action_log[2]["stdout"]

        # Real command returned data
        assert "Bug" in run_result.action_log[3]["stdout"]

        # Score
        evaluator = Evaluator()
        score = evaluator.evaluate(task, run_result, backends)
        assert score.outcome > 0.0
        assert score.efficiency == 1.0  # help excluded, 1 real cmd
        assert score.exploration > 0.7  # proactive help


# =========================================================================
# Regression: full-mode task is completely unaffected
# =========================================================================

class TestFullModeRegression:
    @pytest.mark.asyncio
    async def test_full_mode_task_unchanged(self, tmp_path: Path) -> None:
        """A full-mode task through the benchmark pipeline behaves identically."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        data = {
            "id": "regr-full-001",
            "title": "Full mode regression",
            "difficulty": "easy",
            "category": "project_mgmt",
            "description": "List issues in acme/app.",
            "tools_provided": ["gh"],
            "initial_state": {
                "gh": {
                    "repos": {
                        "acme/app": {
                            "issues": [
                                {"number": 1, "title": "Bug", "state": "open",
                                 "assignee": None, "labels": [], "body": ""},
                            ],
                            "pulls": [],
                            "commits": [],
                        }
                    }
                }
            },
            "expected_state": {
                "gh": {
                    "repos": {
                        "acme/app": {
                            "issues": [
                                {"number": 1, "title": "Bug", "state": "open"},
                            ],
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
        (tasks_dir / "task.yaml").write_text(yaml.dump(data))

        agent = DummyAgent()
        runner = BenchmarkRunner(
            tasks_dir=tasks_dir, agent=agent, k=1,
            tool_adapters_dir=TOOL_ADAPTERS_DIR,
        )
        report = await runner.run_all()

        score = report.results[0].scores[0]
        # exploration defaults to 0.0 weight → no impact on total
        assert score.exploration == 0.5  # neutral for full mode
        # total should be unaffected by exploration (weight=0.0)
        expected_total = (
            score.outcome * 0.6
            + score.efficiency * 0.2
            + score.recovery * 0.2
        )
        assert score.total == pytest.approx(expected_total)
