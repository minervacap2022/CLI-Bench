"""Tests for BenchAgent ABC, DummyAgent, and ScriptedAgent."""

import pytest

from cli_bench.agents.base import BenchAgent
from cli_bench.agents.dummy import DummyAgent, ScriptedAgent
from cli_bench.models.observation import Action, Observation


class TestBenchAgentIsAbstract:
    def test_bench_agent_cannot_be_instantiated(self) -> None:
        """BenchAgent is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BenchAgent()  # type: ignore[abstract]


class TestDummyAgent:
    @pytest.mark.asyncio
    async def test_dummy_finishes_immediately(self) -> None:
        """DummyAgent returns a finish action on any observation."""
        agent = DummyAgent()
        obs = Observation(
            task="Create an issue",
            tools=[{"name": "gh", "binary": "gh"}],
            turn=0,
        )
        action = await agent.act(obs)

        assert action.is_finish is True
        assert action.is_command is False
        assert action.result is not None

    @pytest.mark.asyncio
    async def test_dummy_reset_is_noop(self) -> None:
        """DummyAgent.reset() completes without error."""
        agent = DummyAgent()
        agent.reset()  # should not raise


class TestScriptedAgent:
    @pytest.mark.asyncio
    async def test_scripted_agent_returns_actions_in_order(self) -> None:
        """ScriptedAgent returns pre-configured actions in sequence."""
        actions = [
            Action.command(["gh", "issue", "list", "--repo", "acme/app"]),
            Action.command(["gh", "issue", "create", "--repo", "acme/app", "--title", "Bug"]),
            Action.finish("Done"),
        ]
        agent = ScriptedAgent(actions=actions)
        obs = Observation(task="test", tools=[])

        result_0 = await agent.act(obs)
        assert result_0.is_command is True
        assert result_0.cmd == ["gh", "issue", "list", "--repo", "acme/app"]

        result_1 = await agent.act(obs)
        assert result_1.is_command is True
        assert result_1.cmd == ["gh", "issue", "create", "--repo", "acme/app", "--title", "Bug"]

        result_2 = await agent.act(obs)
        assert result_2.is_finish is True
        assert result_2.result == "Done"

    @pytest.mark.asyncio
    async def test_scripted_agent_finishes_when_exhausted(self) -> None:
        """ScriptedAgent returns finish action when all scripted actions are used."""
        actions = [Action.command(["echo", "hello"])]
        agent = ScriptedAgent(actions=actions)
        obs = Observation(task="test", tools=[])

        await agent.act(obs)  # consume the only action
        exhausted = await agent.act(obs)

        assert exhausted.is_finish is True

    @pytest.mark.asyncio
    async def test_scripted_agent_reset_rewinds(self) -> None:
        """ScriptedAgent.reset() rewinds to the beginning of the action sequence."""
        actions = [Action.command(["echo", "first"]), Action.finish("done")]
        agent = ScriptedAgent(actions=actions)
        obs = Observation(task="test", tools=[])

        await agent.act(obs)  # consume first
        agent.reset()

        replayed = await agent.act(obs)
        assert replayed.is_command is True
        assert replayed.cmd == ["echo", "first"]
