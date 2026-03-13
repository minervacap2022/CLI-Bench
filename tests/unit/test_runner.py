"""Tests for the Runner multi-turn agent execution loop."""

import pytest

from cli_bench.agents.dummy import DummyAgent, ScriptedAgent
from cli_bench.harness.runner import Runner, RunResult
from cli_bench.mock_backends.github import GitHubMockBackend
from cli_bench.models.observation import Action


class TestRunnerCompletesOnFinish:
    @pytest.mark.asyncio
    async def test_dummy_agent_finishes_in_one_turn(self) -> None:
        """DummyAgent finishes immediately, producing RunResult(finished=True, turns=1)."""
        from cli_bench.models.task import BenchTask

        task = BenchTask(
            id="test-001",
            title="Test task",
            difficulty="easy",
            category="test",
            description="Do something",
            tools_provided=["gh"],
            initial_state={"github": {"repos": {}}},
            expected_state={"github": {"repos": {}}},
            max_turns=5,
            optimal_commands=1,
        )

        agent = DummyAgent()
        backends = {"gh": GitHubMockBackend({"repos": {}})}
        runner = Runner(agent=agent, backends=backends)

        result = await runner.run_task(task)

        assert isinstance(result, RunResult)
        assert result.finished is True
        assert result.turns == 1
        assert result.task_id == "test-001"
        assert result.agent_result is not None


class TestRunnerRespectsMaxTurns:
    @pytest.mark.asyncio
    async def test_stops_at_max_turns(self) -> None:
        """Runner stops after max_turns even if agent keeps issuing commands."""
        from cli_bench.models.task import BenchTask

        task = BenchTask(
            id="test-002",
            title="Infinite loop",
            difficulty="easy",
            category="test",
            description="Agent never stops",
            tools_provided=["gh"],
            initial_state={"github": {"repos": {"acme/app": {"issues": [], "pulls": [], "commits": []}}}},
            expected_state={"github": {"repos": {"acme/app": {"issues": []}}}},
            max_turns=3,
            optimal_commands=1,
        )

        # ScriptedAgent that always issues commands (never finishes)
        actions = [
            Action.command(["gh", "issue", "list", "--repo", "acme/app"]),
        ] * 10  # More actions than max_turns

        agent = ScriptedAgent(actions=actions)
        backends = {"gh": GitHubMockBackend(task.initial_state["github"])}
        runner = Runner(agent=agent, backends=backends)

        result = await runner.run_task(task)

        assert result.finished is False
        assert result.turns == 3


class TestRunnerRoutesToBackend:
    @pytest.mark.asyncio
    async def test_command_routed_to_correct_backend(self) -> None:
        """ScriptedAgent issues gh commands, backend state is accessed."""
        from cli_bench.models.task import BenchTask

        task = BenchTask(
            id="test-003",
            title="Create issue",
            difficulty="easy",
            category="github",
            description="Create an issue",
            tools_provided=["gh"],
            initial_state={
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
            expected_state={
                "github": {
                    "repos": {
                        "acme/app": {
                            "issues": [{"title": "Bug report"}],
                        }
                    }
                }
            },
            max_turns=5,
            optimal_commands=1,
        )

        actions = [
            Action.command(["gh", "issue", "create", "--repo", "acme/app", "--title", "Bug report"]),
            Action.finish("Created issue"),
        ]
        agent = ScriptedAgent(actions=actions)
        backends = {"gh": GitHubMockBackend(task.initial_state["github"])}
        runner = Runner(agent=agent, backends=backends)

        result = await runner.run_task(task)

        assert result.finished is True
        assert result.turns == 2
        # Verify backend state was mutated
        state = backends["gh"].get_state_snapshot()
        assert len(state["repos"]["acme/app"]["issues"]) == 1
        assert state["repos"]["acme/app"]["issues"][0]["title"] == "Bug report"


class TestRunnerRecordsActionLog:
    @pytest.mark.asyncio
    async def test_action_log_entries(self) -> None:
        """Action log entries have command, stdout, stderr fields."""
        from cli_bench.models.task import BenchTask

        task = BenchTask(
            id="test-004",
            title="Log test",
            difficulty="easy",
            category="test",
            description="Test action logging",
            tools_provided=["gh"],
            initial_state={"github": {"repos": {"acme/app": {"issues": [], "pulls": [], "commits": []}}}},
            expected_state={"github": {}},
            max_turns=5,
            optimal_commands=1,
        )

        actions = [
            Action.command(["gh", "issue", "list", "--repo", "acme/app"]),
            Action.finish("Done"),
        ]
        agent = ScriptedAgent(actions=actions)
        backends = {"gh": GitHubMockBackend(task.initial_state["github"])}
        runner = Runner(agent=agent, backends=backends)

        result = await runner.run_task(task)

        assert len(result.action_log) == 1  # finish actions not logged as commands
        entry = result.action_log[0]
        assert "command" in entry
        assert "stdout" in entry
        assert "stderr" in entry
        assert entry["command"] == ["gh", "issue", "list", "--repo", "acme/app"]


class TestRunnerPassesStdoutToNextObservation:
    @pytest.mark.asyncio
    async def test_stdout_from_previous_turn_in_observation(self) -> None:
        """Agent receives stdout from previous command execution in the next observation."""
        from cli_bench.models.task import BenchTask

        task = BenchTask(
            id="test-005",
            title="Stdout test",
            difficulty="easy",
            category="test",
            description="Verify stdout forwarding",
            tools_provided=["gh"],
            initial_state={
                "github": {
                    "repos": {
                        "acme/app": {
                            "issues": [{"number": 1, "title": "Bug", "state": "open", "assignee": None, "labels": [], "body": ""}],
                            "pulls": [],
                            "commits": [],
                        }
                    }
                }
            },
            expected_state={"github": {}},
            max_turns=5,
            optimal_commands=1,
        )

        # We need an agent that records the observations it receives
        class ObservationRecordingAgent(ScriptedAgent):
            def __init__(self, actions: list[Action]):
                super().__init__(actions)
                self.observations: list = []

            async def act(self, observation):
                self.observations.append(observation)
                return await super().act(observation)

        actions = [
            Action.command(["gh", "issue", "list", "--repo", "acme/app"]),
            Action.finish("Done"),
        ]
        agent = ObservationRecordingAgent(actions=actions)
        backends = {"gh": GitHubMockBackend(task.initial_state["github"])}
        runner = Runner(agent=agent, backends=backends)

        await runner.run_task(task)

        # First observation should have empty stdout (no prior command)
        assert agent.observations[0].stdout == ""
        assert agent.observations[0].turn == 0

        # Second observation should have stdout from gh issue list
        assert agent.observations[1].stdout != ""
        assert agent.observations[1].turn == 1
