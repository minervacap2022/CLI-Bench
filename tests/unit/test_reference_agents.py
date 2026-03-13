"""Tests for OpenAI and Anthropic reference agents.

Tests parsing, prompt construction, and reset — no real API calls.
"""

import pytest

from cli_bench.agents.base import BenchAgent
from cli_bench.models.observation import Action, Observation


class TestOpenAIAgent:
    """Tests for OpenAIAgent — parsing and prompt building only."""

    def _make_agent(self):
        from cli_bench.agents.openai_agent import OpenAIAgent

        return OpenAIAgent(model="gpt-4o")

    def test_openai_agent_is_bench_agent(self) -> None:
        """OpenAIAgent must be a subclass of BenchAgent."""
        agent = self._make_agent()
        assert isinstance(agent, BenchAgent)

    def test_openai_agent_stores_model(self) -> None:
        """OpenAIAgent stores the model name."""
        agent = self._make_agent()
        assert agent.model == "gpt-4o"

    def test_parse_action_command_block(self) -> None:
        """_parse_action extracts command from ```command blocks."""
        agent = self._make_agent()
        text = "I'll list the issues.\n```command\ngh issue list --repo acme/app\n```"
        action = agent._parse_action(text)
        assert action.is_command is True
        assert action.cmd == ["gh", "issue", "list", "--repo", "acme/app"]

    def test_parse_action_finish_block(self) -> None:
        """_parse_action extracts finish result from ```finish blocks."""
        agent = self._make_agent()
        text = "Task complete.\n```finish\nCreated issue #42\n```"
        action = agent._parse_action(text)
        assert action.is_finish is True
        assert action.result == "Created issue #42"

    def test_parse_action_no_block_defaults_to_finish(self) -> None:
        """_parse_action treats plain text as a finish action."""
        agent = self._make_agent()
        text = "I cannot complete this task."
        action = agent._parse_action(text)
        assert action.is_finish is True
        assert action.result == text

    def test_parse_action_empty_command_returns_finish(self) -> None:
        """_parse_action with empty command block returns finish."""
        agent = self._make_agent()
        text = "```command\n\n```"
        action = agent._parse_action(text)
        assert action.is_finish is True

    def test_build_system_prompt_includes_tools(self) -> None:
        """_build_system_prompt includes tool names and descriptions."""
        agent = self._make_agent()
        obs = Observation(
            task="Create an issue",
            tools=[
                {
                    "name": "gh",
                    "description": "GitHub CLI",
                    "commands": [
                        {"name": "issue create", "description": "Create a new issue"},
                    ],
                },
            ],
            turn=0,
        )
        prompt = agent._build_system_prompt(obs)
        assert "gh" in prompt
        assert "GitHub CLI" in prompt
        assert "issue create" in prompt

    def test_build_system_prompt_includes_memory_when_present(self) -> None:
        """_build_system_prompt includes memory context when provided."""
        agent = self._make_agent()
        obs = Observation(
            task="Send a message",
            tools=[{"name": "slack", "commands": []}],
            turn=0,
            memory={"preferred_channel": "#general", "timezone": "PST"},
        )
        prompt = agent._build_system_prompt(obs)
        assert "preferred_channel" in prompt
        assert "#general" in prompt

    def test_build_system_prompt_no_memory_section_when_none(self) -> None:
        """_build_system_prompt omits memory section when memory is None."""
        agent = self._make_agent()
        obs = Observation(
            task="Create an issue",
            tools=[{"name": "gh", "commands": []}],
            turn=0,
            memory=None,
        )
        prompt = agent._build_system_prompt(obs)
        assert "User Context" not in prompt

    def test_reset_clears_messages(self) -> None:
        """reset() clears internal message history."""
        agent = self._make_agent()
        agent._messages = [{"role": "system", "content": "test"}]
        agent.reset()
        assert agent._messages == []


class TestAnthropicAgent:
    """Tests for AnthropicAgent — parsing and prompt building only."""

    def _make_agent(self):
        from cli_bench.agents.anthropic_agent import AnthropicAgent

        return AnthropicAgent(model="claude-sonnet-4-20250514")

    def test_anthropic_agent_is_bench_agent(self) -> None:
        """AnthropicAgent must be a subclass of BenchAgent."""
        agent = self._make_agent()
        assert isinstance(agent, BenchAgent)

    def test_anthropic_agent_stores_model(self) -> None:
        """AnthropicAgent stores the model name."""
        agent = self._make_agent()
        assert agent.model == "claude-sonnet-4-20250514"

    def test_parse_action_command_block(self) -> None:
        """_parse_action extracts command from ```command blocks."""
        agent = self._make_agent()
        text = "Let me check.\n```command\nlinear issue list --team ENG\n```"
        action = agent._parse_action(text)
        assert action.is_command is True
        assert action.cmd == ["linear", "issue", "list", "--team", "ENG"]

    def test_parse_action_finish_block(self) -> None:
        """_parse_action extracts finish result from ```finish blocks."""
        agent = self._make_agent()
        text = "Done.\n```finish\nIssue created successfully\n```"
        action = agent._parse_action(text)
        assert action.is_finish is True
        assert action.result == "Issue created successfully"

    def test_parse_action_no_block_defaults_to_finish(self) -> None:
        """_parse_action treats plain text as a finish action."""
        agent = self._make_agent()
        text = "Cannot proceed with this task."
        action = agent._parse_action(text)
        assert action.is_finish is True

    def test_parse_action_empty_command_returns_finish(self) -> None:
        """_parse_action with empty command block returns finish."""
        agent = self._make_agent()
        text = "```command\n\n```"
        action = agent._parse_action(text)
        assert action.is_finish is True

    def test_build_system_prompt_includes_tools(self) -> None:
        """_build_system_prompt includes tool documentation."""
        agent = self._make_agent()
        obs = Observation(
            task="Create a Linear issue",
            tools=[
                {
                    "name": "linear",
                    "description": "Linear project management",
                    "commands": [
                        {"name": "issue create", "description": "Create issue"},
                    ],
                },
            ],
            turn=0,
        )
        prompt = agent._build_system_prompt(obs)
        assert "linear" in prompt
        assert "Linear project management" in prompt

    def test_build_system_prompt_includes_memory(self) -> None:
        """_build_system_prompt includes memory when present."""
        agent = self._make_agent()
        obs = Observation(
            task="Send update",
            tools=[{"name": "slack", "commands": []}],
            turn=0,
            memory={"team": "engineering"},
        )
        prompt = agent._build_system_prompt(obs)
        assert "team" in prompt
        assert "engineering" in prompt

    def test_reset_clears_messages(self) -> None:
        """reset() clears internal message history."""
        agent = self._make_agent()
        agent._messages = [{"role": "user", "content": "hello"}]
        agent.reset()
        assert agent._messages == []
