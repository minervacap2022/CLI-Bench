"""Tests for ToolAdapter, ToolCommand, CommandArg, AuthConfig models."""

from pathlib import Path

import pytest
import yaml

from cli_bench.models.tool_adapter import (
    AuthConfig,
    CommandArg,
    ToolAdapter,
    ToolCommand,
)


class TestAuthConfig:
    def test_valid_auth_config(self) -> None:
        auth = AuthConfig(type="env_var", key="GITHUB_TOKEN")
        assert auth.type == "env_var"
        assert auth.key == "GITHUB_TOKEN"

    def test_none_auth_type(self) -> None:
        auth = AuthConfig(type="none")
        assert auth.type == "none"
        assert auth.key is None


class TestCommandArg:
    def test_basic_string_arg(self) -> None:
        arg = CommandArg(
            name="repo",
            type="string",
            required=True,
            description="Repository name",
        )
        assert arg.name == "repo"
        assert arg.type == "string"
        assert arg.required is True
        assert arg.default is None
        assert arg.values is None

    def test_enum_arg_with_values(self) -> None:
        arg = CommandArg(
            name="priority",
            type="enum",
            required=False,
            description="Issue priority",
            values=["low", "medium", "high"],
        )
        assert arg.type == "enum"
        assert arg.values == ["low", "medium", "high"]

    def test_validates_enum_arg(self) -> None:
        """CommandArg with type='enum' but no values raises ValueError."""
        with pytest.raises(ValueError, match="enum"):
            CommandArg(
                name="status",
                type="enum",
                required=True,
                description="Status field",
            )

    def test_validates_enum_arg_empty_values(self) -> None:
        """CommandArg with type='enum' and empty values list raises ValueError."""
        with pytest.raises(ValueError, match="enum"):
            CommandArg(
                name="status",
                type="enum",
                required=True,
                description="Status field",
                values=[],
            )


class TestToolCommand:
    def test_basic_command(self) -> None:
        cmd = ToolCommand(
            name="list-issues",
            description="List all issues",
            args=[],
            output_format="json",
            side_effects=False,
        )
        assert cmd.name == "list-issues"
        assert cmd.output_format == "json"
        assert cmd.side_effects is False
        assert cmd.example is None

    def test_command_to_help_text(self) -> None:
        """ToolCommand generates readable help with args and (required) markers."""
        cmd = ToolCommand(
            name="create-issue",
            description="Create a new issue",
            args=[
                CommandArg(
                    name="title",
                    type="string",
                    required=True,
                    description="Issue title",
                ),
                CommandArg(
                    name="priority",
                    type="enum",
                    required=False,
                    description="Priority level",
                    values=["low", "medium", "high"],
                    default="medium",
                ),
            ],
            output_format="json",
            side_effects=True,
        )

        help_text = cmd.to_help_text()

        assert "create-issue" in help_text
        assert "Create a new issue" in help_text
        assert "title" in help_text
        assert "(required)" in help_text
        assert "priority" in help_text
        assert "low, medium, high" in help_text
        assert "default: medium" in help_text
        assert "Output: json" in help_text
        assert "Side effects: yes" in help_text

    def test_command_to_help_text_no_args(self) -> None:
        cmd = ToolCommand(
            name="whoami",
            description="Show current user",
            args=[],
            output_format="text",
            side_effects=False,
        )
        help_text = cmd.to_help_text()
        assert "whoami" in help_text
        assert "Show current user" in help_text
        assert "Side effects: no" in help_text


class TestToolAdapter:
    def _make_adapter(self) -> ToolAdapter:
        return ToolAdapter(
            name="github-cli",
            description="GitHub CLI for managing repos and issues",
            binary="gh",
            auth=AuthConfig(type="env_var", key="GITHUB_TOKEN"),
            commands=[
                ToolCommand(
                    name="list-repos",
                    description="List repositories",
                    args=[
                        CommandArg(
                            name="owner",
                            type="string",
                            required=True,
                            description="Repository owner",
                        ),
                    ],
                    output_format="json",
                    side_effects=False,
                ),
                ToolCommand(
                    name="create-issue",
                    description="Create an issue",
                    args=[
                        CommandArg(
                            name="title",
                            type="string",
                            required=True,
                            description="Issue title",
                        ),
                        CommandArg(
                            name="body",
                            type="string",
                            required=False,
                            description="Issue body",
                        ),
                    ],
                    output_format="json",
                    side_effects=True,
                ),
            ],
        )

    def test_adapter_command_lookup(self) -> None:
        """get_command returns correct command, None for nonexistent."""
        adapter = self._make_adapter()

        cmd = adapter.get_command("list-repos")
        assert cmd is not None
        assert cmd.name == "list-repos"

        cmd2 = adapter.get_command("create-issue")
        assert cmd2 is not None
        assert cmd2.name == "create-issue"

        assert adapter.get_command("nonexistent") is None

    def test_adapter_to_prompt(self) -> None:
        """Full adapter generates prompt with tool name, description, commands."""
        adapter = self._make_adapter()

        prompt = adapter.to_prompt()

        assert "github-cli" in prompt
        assert "GitHub CLI for managing repos and issues" in prompt
        assert "gh" in prompt
        assert "list-repos" in prompt
        assert "create-issue" in prompt
        assert "owner" in prompt
        assert "title" in prompt

    def test_load_from_yaml(self, tmp_path: Path) -> None:
        """Create a tmp YAML file, load it, verify all fields."""
        yaml_data = {
            "name": "slack-tool",
            "description": "Slack workspace CLI",
            "binary": "slk",
            "auth": {
                "type": "oauth",
                "key": "SLACK_OAUTH_TOKEN",
            },
            "commands": [
                {
                    "name": "send-message",
                    "description": "Send a message to a channel",
                    "args": [
                        {
                            "name": "channel",
                            "type": "string",
                            "required": True,
                            "description": "Channel name",
                        },
                        {
                            "name": "text",
                            "type": "string",
                            "required": True,
                            "description": "Message text",
                        },
                        {
                            "name": "format",
                            "type": "enum",
                            "required": False,
                            "description": "Message format",
                            "values": ["plain", "markdown", "blocks"],
                            "default": "plain",
                        },
                    ],
                    "output_format": "json",
                    "side_effects": True,
                    "example": "slk send-message --channel general --text 'Hello'",
                },
                {
                    "name": "list-channels",
                    "description": "List all channels",
                    "args": [],
                    "output_format": "json",
                    "side_effects": False,
                },
            ],
        }

        yaml_file = tmp_path / "slack-tool.yaml"
        yaml_file.write_text(yaml.dump(yaml_data))

        adapter = ToolAdapter.from_yaml(yaml_file)

        assert adapter.name == "slack-tool"
        assert adapter.description == "Slack workspace CLI"
        assert adapter.binary == "slk"
        assert adapter.auth.type == "oauth"
        assert adapter.auth.key == "SLACK_OAUTH_TOKEN"
        assert len(adapter.commands) == 2

        send_cmd = adapter.get_command("send-message")
        assert send_cmd is not None
        assert send_cmd.description == "Send a message to a channel"
        assert len(send_cmd.args) == 3
        assert send_cmd.output_format == "json"
        assert send_cmd.side_effects is True
        assert send_cmd.example == "slk send-message --channel general --text 'Hello'"

        format_arg = send_cmd.args[2]
        assert format_arg.name == "format"
        assert format_arg.type == "enum"
        assert format_arg.values == ["plain", "markdown", "blocks"]
        assert format_arg.default == "plain"

        list_cmd = adapter.get_command("list-channels")
        assert list_cmd is not None
        assert list_cmd.side_effects is False
