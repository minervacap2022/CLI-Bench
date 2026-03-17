"""Tests for all 12 tool adapter YAML files — 7 real + 5 fictional.

Validates that each YAML file:
- Loads successfully via ToolAdapter.from_yaml()
- Has at least 5 commands
- Has realistic arg patterns (types, required flags, enum values)
"""

from pathlib import Path

import pytest

from cli_bench.models.tool_adapter import ToolAdapter


TOOLS_DIR = Path(__file__).resolve().parent.parent.parent / "cli_bench" / "tool_adapters"

REAL_TOOLS = ["gh", "slack", "linear", "notion", "google", "jira", "microsoft"]
FICTIONAL_TOOLS = ["kforge", "flowctl", "meshctl", "datapipe", "alertmgr"]
OPEN_TOOLS = ["opencli"]
ALL_TOOLS = REAL_TOOLS + FICTIONAL_TOOLS + OPEN_TOOLS


class TestAllToolAdaptersLoad:
    """Every YAML file must load without error."""

    @pytest.fixture(params=ALL_TOOLS)
    def adapter(self, request: pytest.FixtureRequest) -> ToolAdapter:
        yaml_path = TOOLS_DIR / f"{request.param}.yaml"
        return ToolAdapter.from_yaml(yaml_path)

    def test_loads_successfully(self, adapter: ToolAdapter) -> None:
        """YAML parses and validates as a ToolAdapter."""
        assert adapter.name
        assert adapter.description
        assert adapter.binary
        assert adapter.auth is not None

    def test_has_at_least_5_commands(self, adapter: ToolAdapter) -> None:
        """Each tool defines at least 5 commands for meaningful benchmarking."""
        assert len(adapter.commands) >= 5, (
            f"{adapter.name} has only {len(adapter.commands)} commands, need >= 5"
        )

    def test_all_commands_have_names(self, adapter: ToolAdapter) -> None:
        """Every command has a non-empty name."""
        for cmd in adapter.commands:
            assert cmd.name, f"Command in {adapter.name} has empty name"

    def test_all_commands_have_descriptions(self, adapter: ToolAdapter) -> None:
        """Every command has a non-empty description."""
        for cmd in adapter.commands:
            assert cmd.description, (
                f"Command '{cmd.name}' in {adapter.name} has empty description"
            )

    def test_output_format_valid(self, adapter: ToolAdapter) -> None:
        """Every command declares a valid output format."""
        for cmd in adapter.commands:
            assert cmd.output_format in ("json", "text", "csv"), (
                f"Command '{cmd.name}' has invalid output_format: {cmd.output_format}"
            )

    def test_enum_args_have_values(self, adapter: ToolAdapter) -> None:
        """Every enum arg has a non-empty values list."""
        for cmd in adapter.commands:
            for arg in cmd.args:
                if arg.type == "enum":
                    assert arg.values and len(arg.values) >= 2, (
                        f"Enum arg '{arg.name}' in {adapter.name}/{cmd.name} "
                        f"needs at least 2 values"
                    )


class TestRealToolAdapters:
    """Additional checks specific to real tool adapters."""

    @pytest.fixture(params=REAL_TOOLS)
    def adapter(self, request: pytest.FixtureRequest) -> ToolAdapter:
        yaml_path = TOOLS_DIR / f"{request.param}.yaml"
        return ToolAdapter.from_yaml(yaml_path)

    def test_has_auth_config(self, adapter: ToolAdapter) -> None:
        """Real tools have auth configured (env_var, oauth, or token)."""
        assert adapter.auth.type in ("env_var", "oauth", "token"), (
            f"{adapter.name} should have real auth, got: {adapter.auth.type}"
        )

    def test_has_side_effect_commands(self, adapter: ToolAdapter) -> None:
        """Real tools have at least one command with side effects (create/send/etc)."""
        side_effect_cmds = [c for c in adapter.commands if c.side_effects]
        assert len(side_effect_cmds) >= 1, (
            f"{adapter.name} has no commands with side_effects=true"
        )

    def test_has_read_only_commands(self, adapter: ToolAdapter) -> None:
        """Real tools have at least one read-only command (list/get/etc)."""
        read_cmds = [c for c in adapter.commands if not c.side_effects]
        assert len(read_cmds) >= 1, (
            f"{adapter.name} has no read-only commands"
        )


class TestFictionalToolAdapters:
    """Additional checks specific to fictional tool adapters."""

    @pytest.fixture(params=FICTIONAL_TOOLS)
    def adapter(self, request: pytest.FixtureRequest) -> ToolAdapter:
        yaml_path = TOOLS_DIR / f"{request.param}.yaml"
        return ToolAdapter.from_yaml(yaml_path)

    def test_has_json_args(self, adapter: ToolAdapter) -> None:
        """Fictional tools should have at least one JSON-typed arg for complex input."""
        json_args = [
            arg
            for cmd in adapter.commands
            for arg in cmd.args
            if arg.type == "json"
        ]
        assert len(json_args) >= 1, (
            f"{adapter.name} should have at least one JSON arg for complexity"
        )

    def test_has_required_and_optional_args(self, adapter: ToolAdapter) -> None:
        """Fictional tools mix required and optional args."""
        required_args = [
            arg
            for cmd in adapter.commands
            for arg in cmd.args
            if arg.required
        ]
        optional_args = [
            arg
            for cmd in adapter.commands
            for arg in cmd.args
            if not arg.required
        ]
        assert len(required_args) >= 1, f"{adapter.name} needs required args"
        assert len(optional_args) >= 1, f"{adapter.name} needs optional args"

    def test_has_multiple_arg_types(self, adapter: ToolAdapter) -> None:
        """Fictional tools use at least 3 different arg types."""
        all_types = {
            arg.type
            for cmd in adapter.commands
            for arg in cmd.args
        }
        assert len(all_types) >= 3, (
            f"{adapter.name} only uses {all_types}, need >= 3 types"
        )

    def test_to_prompt_generates_output(self, adapter: ToolAdapter) -> None:
        """to_prompt() produces valid documentation string."""
        prompt = adapter.to_prompt()
        assert adapter.name in prompt
        assert adapter.binary in prompt
        assert len(prompt) > 200  # Should be substantial


class TestSpecificToolDetails:
    """Spot-check individual tools for expected commands."""

    def test_gh_has_issue_and_pr_commands(self) -> None:
        adapter = ToolAdapter.from_yaml(TOOLS_DIR / "gh.yaml")
        names = {c.name for c in adapter.commands}
        assert "issue list" in names
        assert "issue create" in names
        assert "pr list" in names

    def test_slack_has_channel_and_dm_commands(self) -> None:
        adapter = ToolAdapter.from_yaml(TOOLS_DIR / "slack.yaml")
        names = {c.name for c in adapter.commands}
        assert "channel list" in names
        assert "message send" in names
        assert "dm send" in names

    def test_kforge_has_artifact_and_pipeline_commands(self) -> None:
        adapter = ToolAdapter.from_yaml(TOOLS_DIR / "kforge.yaml")
        names = {c.name for c in adapter.commands}
        assert "artifact list" in names
        assert "pipeline trigger" in names
        assert "artifact promote" in names

    def test_flowctl_has_workflow_and_gate_commands(self) -> None:
        adapter = ToolAdapter.from_yaml(TOOLS_DIR / "flowctl.yaml")
        names = {c.name for c in adapter.commands}
        assert "workflow list" in names
        assert "gate create" in names
        assert "gate approve" in names
        assert "run trigger" in names

    def test_meshctl_has_service_and_policy_commands(self) -> None:
        adapter = ToolAdapter.from_yaml(TOOLS_DIR / "meshctl.yaml")
        names = {c.name for c in adapter.commands}
        assert "service list" in names
        assert "policy create" in names
        assert "traffic split" in names

    def test_datapipe_has_source_transform_sink_commands(self) -> None:
        adapter = ToolAdapter.from_yaml(TOOLS_DIR / "datapipe.yaml")
        names = {c.name for c in adapter.commands}
        assert "source list" in names
        assert "transform create" in names
        assert "sink create" in names
        assert "pipeline create" in names

    def test_alertmgr_has_alert_and_incident_commands(self) -> None:
        adapter = ToolAdapter.from_yaml(TOOLS_DIR / "alertmgr.yaml")
        names = {c.name for c in adapter.commands}
        assert "alert list" in names
        assert "alert acknowledge" in names
        assert "rule create" in names
        assert "incident create" in names
