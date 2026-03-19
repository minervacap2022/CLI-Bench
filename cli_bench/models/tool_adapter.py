"""Tool adapter schema: ToolAdapter, ToolCommand, CommandArg, AuthConfig.

Pydantic v2 models for defining CLI tool specifications loaded from YAML.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, model_validator


class AuthConfig(BaseModel):
    """Authentication configuration for a tool."""

    type: Literal["env_var", "oauth", "token", "none"]
    key: str | None = None


class CommandArg(BaseModel):
    """A single argument for a tool command."""

    name: str
    type: Literal["string", "int", "bool", "enum", "json", "datetime", "float"]
    required: bool
    description: str
    default: str | None = None
    values: list[str] | None = None

    @model_validator(mode="after")
    def validate_enum_has_values(self) -> "CommandArg":
        if self.type == "enum" and (not self.values):
            raise ValueError(
                f"CommandArg '{self.name}' has type='enum' but no values provided"
            )
        return self


class ToolCommand(BaseModel):
    """A command exposed by a tool."""

    name: str
    description: str
    args: list[CommandArg]
    output_format: Literal["json", "text", "csv"]
    side_effects: bool
    example: str | None = None

    def to_help_text(self) -> str:
        """Generate human-readable help text for this command."""
        lines: list[str] = []
        lines.append(f"Command: {self.name}")
        lines.append(f"  {self.description}")

        if self.args:
            lines.append("  Arguments:")
            for arg in self.args:
                required_marker = " (required)" if arg.required else ""
                line = f"    --{arg.name} [{arg.type}]{required_marker}: {arg.description}"
                if arg.values:
                    line += f" (values: {', '.join(arg.values)})"
                if arg.default is not None:
                    line += f" (default: {arg.default})"
                lines.append(line)

        lines.append(f"  Output: {self.output_format}")
        lines.append(f"  Side effects: {'yes' if self.side_effects else 'no'}")

        if self.example:
            lines.append(f"  Example: {self.example}")

        return "\n".join(lines)


class ToolAdapter(BaseModel):
    """Full tool adapter specification loaded from YAML."""

    name: str
    description: str
    binary: str
    auth: AuthConfig
    commands: list[ToolCommand]

    @classmethod
    def from_yaml(cls, path: Path) -> "ToolAdapter":
        """Load a ToolAdapter from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def get_command(self, name: str) -> ToolCommand | None:
        """Look up a command by name. Returns None if not found."""
        for cmd in self.commands:
            if cmd.name == name:
                return cmd
        return None

    def to_prompt(self) -> str:
        """Generate full prompt-ready documentation for this tool."""
        lines: list[str] = []
        lines.append(f"Tool: {self.name}")
        lines.append(f"Description: {self.description}")
        lines.append(f"Binary: {self.binary}")
        lines.append(f"Auth: {self.auth.type}" + (f" ({self.auth.key})" if self.auth.key else ""))
        lines.append("")
        lines.append("Commands:")
        for cmd in self.commands:
            lines.append("")
            lines.append(cmd.to_help_text())

        return "\n".join(lines)

    def _get_subcommand_groups(self) -> dict[str, list[ToolCommand]]:
        """Group commands by their subcommand prefix.

        Command names like ``"issue list"`` split into subcommand ``"issue"``
        and action ``"list"``.  Single-word commands are grouped under their
        own name.

        Returns a dict mapping subcommand name → list of ToolCommands.
        """
        groups: dict[str, list[ToolCommand]] = {}
        for cmd in self.commands:
            parts = cmd.name.split(None, 1)
            sub = parts[0]
            groups.setdefault(sub, []).append(cmd)
        return groups

    def to_root_help(self) -> str:
        """Generate root-level ``<binary> --help`` text.

        Lists subcommands grouped from command names, following real CLI
        conventions (USAGE / AVAILABLE COMMANDS / FLAGS).
        """
        groups = self._get_subcommand_groups()
        lines: list[str] = [
            f"{self.binary} - {self.description}",
            "",
            "USAGE:",
            f"  {self.binary} <command> [flags]",
            "",
            "AVAILABLE COMMANDS:",
        ]
        for sub, cmds in groups.items():
            # Build a short summary from the first command's description
            if len(cmds) == 1 and " " not in cmds[0].name:
                summary = cmds[0].description
            else:
                # Derive a generic summary from the subcommand name
                summary = f"Manage {sub}"
            lines.append(f"  {sub:<20s}{summary}")
        lines.extend([
            "",
            "FLAGS:",
            "  -h, --help          Show help for a command",
            "",
            f'Use "{self.binary} <command> --help" for more information.',
        ])
        return "\n".join(lines)

    def to_subcommand_help(self, subcommand: str) -> str:
        """Generate ``<binary> <subcommand> --help`` text.

        Lists actions available under *subcommand*.  For single-word commands
        that match exactly, falls back to full action help.
        """
        groups = self._get_subcommand_groups()
        cmds = groups.get(subcommand)
        if cmds is None:
            return f"Error: unknown command \"{subcommand}\" for \"{self.binary}\""

        # If there's only one command and it's a single-word name, show full help
        if len(cmds) == 1 and " " not in cmds[0].name:
            return self.to_action_help(cmds[0].name)

        lines: list[str] = [
            f"Manage {subcommand}",
            "",
            "USAGE:",
            f"  {self.binary} {subcommand} <command> [flags]",
            "",
            "AVAILABLE COMMANDS:",
        ]
        for cmd in cmds:
            parts = cmd.name.split(None, 1)
            action = parts[1] if len(parts) > 1 else parts[0]
            lines.append(f"  {action:<20s}{cmd.description}")
        lines.extend([
            "",
            "FLAGS:",
            "  -h, --help          Show help for a command",
            "",
            f'Use "{self.binary} {subcommand} <command> --help" for more information.',
        ])
        return "\n".join(lines)

    def to_action_help(self, name: str) -> str:
        """Generate ``<binary> <sub> <action> --help`` text.

        Shows full FLAGS + EXAMPLES for one command.
        """
        cmd = self.get_command(name)
        if cmd is None:
            return f"Error: unknown command \"{name}\" for \"{self.binary}\""

        lines: list[str] = [
            cmd.description,
            "",
            "USAGE:",
            f"  {self.binary} {cmd.name} [flags]",
        ]

        if cmd.args:
            lines.extend(["", "FLAGS:"])
            for arg in cmd.args:
                type_str = f"<{arg.type}>"
                req = " (required)" if arg.required else ""
                line = f"  --{arg.name} {type_str}{req}"
                line += f"\t{arg.description}"
                if arg.values:
                    line += f" (values: {', '.join(arg.values)})"
                if arg.default is not None:
                    line += f" (default: {arg.default})"
                lines.append(line)

        lines.extend([
            "",
            "  -h, --help\tShow help for this command",
        ])

        if cmd.example:
            lines.extend(["", "EXAMPLES:", f"  {cmd.example}"])

        return "\n".join(lines)
