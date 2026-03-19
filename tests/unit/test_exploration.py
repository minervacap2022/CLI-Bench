"""Tests for the tool exploration mode (Phases 1-5).

Covers:
- Phase 1: data model additions (doc_visibility, optimal_help_calls, exploration scoring)
- Phase 2: hierarchical help text generation
- Phase 3: help interception in base mock backend
- Phase 4: runner doc_visibility + adapter injection
- Phase 5: exploration scoring in evaluator
"""

from pathlib import Path

import pytest
import yaml

from cli_bench.harness.evaluator import Evaluator
from cli_bench.harness.runner import Runner, RunResult
from cli_bench.mock_backends.base import BaseMockBackend, MockResult
from cli_bench.mock_backends.github import GitHubMockBackend
from cli_bench.models.observation import Action, Observation
from cli_bench.models.scoring import ScoringWeights, TaskScore
from cli_bench.models.task import BenchTask, ScoringConfig
from cli_bench.models.tool_adapter import (
    AuthConfig,
    CommandArg,
    ToolAdapter,
    ToolCommand,
)
from cli_bench.agents.base import BenchAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gh_adapter() -> ToolAdapter:
    """Minimal GitHub-like adapter for testing."""
    return ToolAdapter(
        name="github-cli",
        description="GitHub CLI for repos and issues",
        binary="gh",
        auth=AuthConfig(type="env_var", key="GITHUB_TOKEN"),
        commands=[
            ToolCommand(
                name="issue list",
                description="List issues in a repository",
                args=[
                    CommandArg(name="repo", type="string", required=True, description="Repository"),
                    CommandArg(name="state", type="enum", required=False, description="Filter state", values=["open", "closed", "all"]),
                ],
                output_format="json",
                side_effects=False,
                example="gh issue list --repo owner/repo --state open",
            ),
            ToolCommand(
                name="issue create",
                description="Create a new issue",
                args=[
                    CommandArg(name="repo", type="string", required=True, description="Repository"),
                    CommandArg(name="title", type="string", required=True, description="Issue title"),
                ],
                output_format="json",
                side_effects=True,
                example="gh issue create --repo owner/repo --title 'Bug'",
            ),
            ToolCommand(
                name="pr list",
                description="List pull requests",
                args=[
                    CommandArg(name="repo", type="string", required=True, description="Repository"),
                ],
                output_format="json",
                side_effects=False,
            ),
        ],
    )


class SimpleMockBackend(BaseMockBackend):
    """Concrete mock backend for testing."""

    def route_command(self, command: list[str]) -> MockResult:
        return MockResult(stdout="ok", stderr="", exit_code=0)


class MockAgent(BenchAgent):
    """Simple mock agent that returns predefined actions."""

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


def _make_task(
    doc_visibility: str = "full",
    optimal_help_calls: int | None = None,
    scoring_exploration: float = 0.0,
    **kwargs,
) -> BenchTask:
    defaults = dict(
        id="test-explore",
        title="Exploration test",
        difficulty="easy",
        category="test",
        description="Test exploration",
        tools_provided=["gh"],
        initial_state={"gh": {"repos": {}}},
        expected_state={"gh": {"repos": {}}},
        max_turns=10,
        optimal_commands=1,
        doc_visibility=doc_visibility,
        optimal_help_calls=optimal_help_calls,
        scoring={
            "outcome": 0.5,
            "efficiency": 0.15,
            "recovery": 0.15,
            "exploration": scoring_exploration,
        },
    )
    defaults.update(kwargs)
    return BenchTask(**defaults)


def _make_run_result(action_log: list[dict] | None = None) -> RunResult:
    return RunResult(
        task_id="test-explore",
        turns=1,
        finished=True,
        final_state={},
        action_log=action_log or [],
        elapsed_ms=100,
        agent_result="Done",
    )


# =========================================================================
# Phase 1: Data model tests
# =========================================================================

class TestDataModelDefaults:
    def test_doc_visibility_defaults_full(self) -> None:
        """BenchTask.doc_visibility defaults to 'full'."""
        task = _make_task()
        assert task.doc_visibility == "full"

    def test_optimal_help_calls_defaults_none(self) -> None:
        """BenchTask.optimal_help_calls defaults to None."""
        task = _make_task()
        assert task.optimal_help_calls is None

    def test_scoring_config_exploration_default(self) -> None:
        """ScoringConfig.exploration defaults to 0.0."""
        cfg = ScoringConfig()
        assert cfg.exploration == 0.0

    def test_scoring_weights_exploration_default(self) -> None:
        """ScoringWeights.exploration defaults to 0.0."""
        w = ScoringWeights()
        assert w.exploration == 0.0

    def test_task_score_exploration_default(self) -> None:
        """TaskScore.exploration defaults to 0.0."""
        score = TaskScore.calculate(
            outcome=1.0, efficiency=1.0, recovery=1.0, weights=ScoringWeights()
        )
        assert score.exploration == 0.0

    def test_task_score_with_exploration(self) -> None:
        """TaskScore includes exploration in weighted total."""
        weights = ScoringWeights(
            outcome=0.5, efficiency=0.15, recovery=0.15, exploration=0.2
        )
        score = TaskScore.calculate(
            outcome=1.0, efficiency=1.0, recovery=1.0,
            exploration=0.8, weights=weights,
        )
        expected = 1.0 * 0.5 + 1.0 * 0.15 + 1.0 * 0.15 + 0.8 * 0.2
        assert score.total == pytest.approx(expected)
        assert score.exploration == 0.8

    def test_doc_visibility_from_yaml(self, tmp_path: Path) -> None:
        """BenchTask loads doc_visibility and optimal_help_calls from YAML."""
        data = {
            "id": "cb-test",
            "title": "Test",
            "difficulty": "easy",
            "category": "test",
            "description": "Test task",
            "tools_provided": ["kforge"],
            "initial_state": {"kforge": {}},
            "expected_state": {"kforge": {}},
            "doc_visibility": "name_only",
            "optimal_help_calls": 3,
            "max_turns": 5,
            "scoring": {"exploration": 0.2, "outcome": 0.5, "efficiency": 0.15, "recovery": 0.15},
        }
        p = tmp_path / "task.yaml"
        p.write_text(yaml.dump(data))
        task = BenchTask.from_yaml(p)
        assert task.doc_visibility == "name_only"
        assert task.optimal_help_calls == 3
        assert task.scoring.exploration == 0.2


# =========================================================================
# Phase 2: Hierarchical help text generation
# =========================================================================

class TestRootHelp:
    def test_to_root_help_sections(self) -> None:
        """Root help contains USAGE, AVAILABLE COMMANDS, and FLAGS sections."""
        adapter = _make_gh_adapter()
        text = adapter.to_root_help()
        assert "gh - GitHub CLI for repos and issues" in text
        assert "USAGE:" in text
        assert "AVAILABLE COMMANDS:" in text
        assert "FLAGS:" in text
        assert "issue" in text
        assert "pr" in text
        assert '--help' in text

    def test_to_root_help_hint(self) -> None:
        """Root help includes a 'Use ... --help' hint."""
        adapter = _make_gh_adapter()
        text = adapter.to_root_help()
        assert 'Use "gh <command> --help" for more information.' in text


class TestSubcommandHelp:
    def test_to_subcommand_help_lists_actions(self) -> None:
        """Subcommand help lists available actions."""
        adapter = _make_gh_adapter()
        text = adapter.to_subcommand_help("issue")
        assert "list" in text
        assert "create" in text
        assert "AVAILABLE COMMANDS:" in text

    def test_to_subcommand_help_unknown(self) -> None:
        """Unknown subcommand returns error message."""
        adapter = _make_gh_adapter()
        text = adapter.to_subcommand_help("nonexistent")
        assert "Error" in text
        assert "nonexistent" in text


class TestActionHelp:
    def test_to_action_help_shows_flags(self) -> None:
        """Action help shows FLAGS with types and descriptions."""
        adapter = _make_gh_adapter()
        text = adapter.to_action_help("issue list")
        assert "FLAGS:" in text
        assert "--repo" in text
        assert "--state" in text
        assert "<string>" in text
        assert "(required)" in text
        assert "EXAMPLES:" in text

    def test_to_action_help_unknown(self) -> None:
        """Unknown action returns error message."""
        adapter = _make_gh_adapter()
        text = adapter.to_action_help("issue nonexistent")
        assert "Error" in text

    def test_to_action_help_enum_values(self) -> None:
        """Action help includes enum values."""
        adapter = _make_gh_adapter()
        text = adapter.to_action_help("issue list")
        assert "open" in text
        assert "closed" in text


# =========================================================================
# Phase 3: Help interception in base mock backend
# =========================================================================

class TestHelpInterception:
    def test_help_intercepted_before_route(self) -> None:
        """Backend returns help text for ['gh', '--help'] instead of routing."""
        adapter = _make_gh_adapter()
        backend = SimpleMockBackend(initial_state={})
        backend.set_tool_adapter(adapter)
        result = backend.execute(["gh", "--help"])
        assert "AVAILABLE COMMANDS:" in result.stdout
        assert result.exit_code == 0

    def test_help_not_intercepted_without_flag(self) -> None:
        """Normal commands still go through route_command."""
        adapter = _make_gh_adapter()
        backend = SimpleMockBackend(initial_state={})
        backend.set_tool_adapter(adapter)
        result = backend.execute(["gh", "issue", "list"])
        assert result.stdout == "ok"

    def test_subcommand_help(self) -> None:
        """'gh issue --help' returns subcommand-level help."""
        adapter = _make_gh_adapter()
        backend = SimpleMockBackend(initial_state={})
        backend.set_tool_adapter(adapter)
        result = backend.execute(["gh", "issue", "--help"])
        assert "list" in result.stdout
        assert "create" in result.stdout

    def test_action_help(self) -> None:
        """'gh issue list --help' returns action-level help."""
        adapter = _make_gh_adapter()
        backend = SimpleMockBackend(initial_state={})
        backend.set_tool_adapter(adapter)
        result = backend.execute(["gh", "issue", "list", "--help"])
        assert "--repo" in result.stdout
        assert "FLAGS:" in result.stdout

    def test_help_short_flag(self) -> None:
        """'-h' works the same as '--help'."""
        adapter = _make_gh_adapter()
        backend = SimpleMockBackend(initial_state={})
        backend.set_tool_adapter(adapter)
        result = backend.execute(["gh", "-h"])
        assert "AVAILABLE COMMANDS:" in result.stdout

    def test_help_logged_in_history(self) -> None:
        """Help calls appear in command history like normal commands."""
        adapter = _make_gh_adapter()
        backend = SimpleMockBackend(initial_state={})
        backend.set_tool_adapter(adapter)
        backend.execute(["gh", "--help"])
        history = backend.get_command_history()
        assert len(history) == 1
        assert "gh --help" in history[0]

    def test_no_adapter_no_interception(self) -> None:
        """Without adapter set, help flags pass through to route_command."""
        backend = SimpleMockBackend(initial_state={})
        result = backend.execute(["gh", "--help"])
        # Falls through to route_command which returns "ok"
        assert result.stdout == "ok"


# =========================================================================
# Phase 4: Runner doc_visibility
# =========================================================================

class TestRunnerDocVisibility:
    @pytest.mark.asyncio
    async def test_name_only_hides_docs(self) -> None:
        """In name_only mode, tool prompts only have name + hint."""
        task = _make_task(doc_visibility="name_only")
        agent = MockAgent([Action.finish("done")])
        backend = GitHubMockBackend({"repos": {}})
        runner = Runner(
            agent=agent,
            backends={"gh": backend},
            tool_adapters_dir=Path("cli_bench/tool_adapters"),
        )
        # Capture observation by spying on agent
        observations: list[Observation] = []
        original_act = agent.act

        async def spy_act(obs: Observation) -> Action:
            observations.append(obs)
            return await original_act(obs)

        agent.act = spy_act
        await runner.run_task(task)

        assert len(observations) == 1
        tools = observations[0].tools
        assert len(tools) >= 1
        gh_tool = tools[0]
        assert gh_tool["name"] == "gh"
        assert "hint" in gh_tool
        assert "commands" not in gh_tool
        assert "full_documentation" not in gh_tool

    @pytest.mark.asyncio
    async def test_description_only_has_description(self) -> None:
        """In description_only mode, tool prompts have name + description + hint."""
        task = _make_task(doc_visibility="description_only")
        agent = MockAgent([Action.finish("done")])
        backend = GitHubMockBackend({"repos": {}})
        runner = Runner(
            agent=agent,
            backends={"gh": backend},
            tool_adapters_dir=Path("cli_bench/tool_adapters"),
        )
        observations: list[Observation] = []
        original_act = agent.act

        async def spy_act(obs: Observation) -> Action:
            observations.append(obs)
            return await original_act(obs)

        agent.act = spy_act
        await runner.run_task(task)

        gh_tool = observations[0].tools[0]
        assert gh_tool["name"] == "gh"
        assert "description" in gh_tool
        assert "hint" in gh_tool
        assert "commands" not in gh_tool

    @pytest.mark.asyncio
    async def test_full_mode_unchanged(self) -> None:
        """In full mode, tool prompts include commands and full_documentation."""
        task = _make_task(doc_visibility="full")
        agent = MockAgent([Action.finish("done")])
        backend = GitHubMockBackend({"repos": {}})
        runner = Runner(
            agent=agent,
            backends={"gh": backend},
            tool_adapters_dir=Path("cli_bench/tool_adapters"),
        )
        observations: list[Observation] = []
        original_act = agent.act

        async def spy_act(obs: Observation) -> Action:
            observations.append(obs)
            return await original_act(obs)

        agent.act = spy_act
        await runner.run_task(task)

        gh_tool = observations[0].tools[0]
        assert "commands" in gh_tool
        assert "full_documentation" in gh_tool


# =========================================================================
# Phase 5: Exploration scoring
# =========================================================================

class TestExplorationScoring:
    def test_exploration_full_mode_neutral(self) -> None:
        """When doc_visibility=full, exploration score is 0.5 (neutral)."""
        task = _make_task(doc_visibility="full", scoring_exploration=0.2)
        result = _make_run_result()
        evaluator = Evaluator()
        backend = GitHubMockBackend({"repos": {}})
        score = evaluator.evaluate(task, result, {"gh": backend})
        assert score.exploration == 0.5

    def test_exploration_proactive_help(self) -> None:
        """Help-first in name_only mode gets high exploration score."""
        task = _make_task(
            doc_visibility="name_only",
            optimal_help_calls=2,
            scoring_exploration=0.2,
        )
        action_log = [
            {"command": ["gh", "--help"], "stdout": "help text", "stderr": ""},
            {"command": ["gh", "issue", "--help"], "stdout": "help text", "stderr": ""},
            {"command": ["gh", "issue", "list", "--repo", "acme/app"], "stdout": "[]", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)
        evaluator = Evaluator()
        backend = GitHubMockBackend({"repos": {}})
        score = evaluator.evaluate(task, result, {"gh": backend})
        # proactive=1.0*0.4, efficiency=min(1,2/2)*0.3=0.3, reactive=0.5*0.3=0.15
        assert score.exploration == pytest.approx(0.85)

    def test_exploration_no_help(self) -> None:
        """No help calls in name_only mode gets 0.0 exploration."""
        task = _make_task(
            doc_visibility="name_only",
            scoring_exploration=0.2,
        )
        action_log = [
            {"command": ["gh", "issue", "list"], "stdout": "[]", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)
        evaluator = Evaluator()
        backend = GitHubMockBackend({"repos": {}})
        score = evaluator.evaluate(task, result, {"gh": backend})
        assert score.exploration == 0.0

    def test_exploration_reactive_recovery(self) -> None:
        """After error, consulting --help before retry earns reactive score."""
        task = _make_task(
            doc_visibility="name_only",
            optimal_help_calls=1,
            scoring_exploration=0.2,
        )
        action_log = [
            {"command": ["gh", "issue", "list"], "stdout": "", "stderr": "error: missing --repo"},
            {"command": ["gh", "issue", "list", "--help"], "stdout": "help text", "stderr": ""},
            {"command": ["gh", "issue", "list", "--repo", "acme/app"], "stdout": "[]", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)
        evaluator = Evaluator()
        backend = GitHubMockBackend({"repos": {}})
        score = evaluator.evaluate(task, result, {"gh": backend})
        # proactive=0 (help after non-help), efficiency=min(1,1/1)=1.0, reactive=1.0
        assert score.exploration == pytest.approx(0.0 * 0.4 + 1.0 * 0.3 + 1.0 * 0.3)

    def test_efficiency_excludes_help_in_exploration(self) -> None:
        """In exploration mode, help calls are excluded from efficiency count."""
        task = _make_task(
            doc_visibility="name_only",
            optimal_commands=1,
        )
        action_log = [
            {"command": ["gh", "--help"], "stdout": "help text", "stderr": ""},
            {"command": ["gh", "issue", "--help"], "stdout": "help text", "stderr": ""},
            {"command": ["gh", "issue", "list", "--repo", "acme/app"], "stdout": "[]", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)
        evaluator = Evaluator()
        backend = GitHubMockBackend({"repos": {}})
        score = evaluator.evaluate(task, result, {"gh": backend})
        # Only 1 non-help command, optimal_commands=1 → efficiency=1.0
        assert score.efficiency == 1.0

    def test_efficiency_unchanged_in_full_mode(self) -> None:
        """In full mode, all commands count toward efficiency."""
        task = _make_task(doc_visibility="full", optimal_commands=1)
        action_log = [
            {"command": ["gh", "--help"], "stdout": "help text", "stderr": ""},
            {"command": ["gh", "issue", "list", "--repo", "acme/app"], "stdout": "[]", "stderr": ""},
        ]
        result = _make_run_result(action_log=action_log)
        evaluator = Evaluator()
        backend = GitHubMockBackend({"repos": {}})
        score = evaluator.evaluate(task, result, {"gh": backend})
        # 2 total commands, optimal=1 → efficiency=0.5
        assert score.efficiency == pytest.approx(0.5)
