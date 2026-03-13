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
