"""Generic mock backend for fictional CLI tools.

Routes commands based on a command registry and supports common
CRUD patterns (list, get/inspect/show, create, update, delete)
without needing per-tool handler code.
"""

import copy
import json
from collections.abc import Callable

from cli_bench.mock_backends.base import BaseMockBackend, MockResult


def _parse_args(args: list[str]) -> dict[str, list[str]]:
    """Parse CLI args into a dict mapping --flag to list of values.

    Supports repeated flags (e.g. --label bug --label urgent).
    Positional args are stored under the empty-string key.
    """
    parsed: dict[str, list[str]] = {"": []}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--"):
            key = arg
            if i + 1 < len(args) and not args[i + 1].startswith("--"):
                parsed.setdefault(key, []).append(args[i + 1])
                i += 2
            else:
                parsed.setdefault(key, []).append("")
                i += 1
        else:
            parsed[""].append(arg)
            i += 1
    return parsed


def _get_flag(parsed: dict[str, list[str]], flag: str) -> str | None:
    """Get the first value for a flag, or None if absent."""
    values = parsed.get(flag)
    if values:
        return values[0]
    return None


CommandHandler = Callable[["FictionalMockBackend", list[str]], MockResult]


class FictionalMockBackend(BaseMockBackend):
    """Generic mock for fictional CLI tools.

    Routes commands based on a command registry. If no custom handlers
    are provided, uses generic CRUD patterns based on state keys.

    State schema is flexible — the backend infers list/dict collections
    from the state keys and provides list/get/create operations automatically.

    Custom handlers can override any resource+action pair via command_handlers.
    """

    def __init__(
        self,
        initial_state: dict,
        tool_name: str,
        command_handlers: dict[str, CommandHandler] | None = None,
        id_prefix: str = "id",
    ) -> None:
        super().__init__(initial_state)
        self.tool_name = tool_name
        self._command_handlers: dict[str, CommandHandler] = command_handlers or {}
        self._id_prefix = id_prefix

    def route_command(self, command: list[str]) -> MockResult:
        """Route a command to the appropriate handler.

        Command format: <tool_name> <resource> <action> [args...]
        Looks up "{resource} {action}" in command_handlers first,
        then falls back to generic CRUD.
        """
        if not command or command[0] != self.tool_name:
            return MockResult(
                stdout="",
                stderr=f"unknown command: {' '.join(command)}",
                exit_code=1,
            )

        if len(command) < 3:
            return MockResult(
                stdout="",
                stderr=f"usage: {self.tool_name} <resource> <action> [args...]",
                exit_code=1,
            )

        resource = command[1]
        action = command[2]
        remaining = command[3:]
        handler_key = f"{resource} {action}"

        # Check custom handlers first
        handler = self._command_handlers.get(handler_key)
        if handler is not None:
            return handler(self, remaining)

        # Fall back to generic CRUD
        return self._generic_crud(resource, action, remaining)

    def _generic_crud(
        self, resource: str, action: str, args: list[str]
    ) -> MockResult:
        """Handle generic CRUD operations based on state structure."""
        # Pluralize resource name to find state key
        state_key = self._find_state_key(resource)
        if state_key is None:
            return MockResult(
                stdout="",
                stderr=f"unknown resource: {resource}",
                exit_code=1,
            )

        if action == "list":
            return self._generic_list(state_key, args)
        if action in ("get", "inspect", "show", "view", "status"):
            return self._generic_get(state_key, args)
        if action == "create":
            return self._generic_create(state_key, resource, args)
        if action in ("update", "edit"):
            return self._generic_update(state_key, args)
        if action in ("delete", "remove"):
            return self._generic_delete(state_key, args)

        return MockResult(
            stdout="",
            stderr=f"unknown action: {self.tool_name} {resource} {action}",
            exit_code=1,
        )

    def _find_state_key(self, resource: str) -> str | None:
        """Find the state key that matches a resource name.

        Tries: exact match, pluralized (+'s'), depluralized (-'s').
        """
        if resource in self.state:
            return resource
        # Standard plurals: +s
        plural_s = resource + "s"
        if plural_s in self.state:
            return plural_s
        # -y → -ies (e.g. registry → registries)
        if resource.endswith("y"):
            plural_ies = resource[:-1] + "ies"
            if plural_ies in self.state:
                return plural_ies
        # Depluralize: -s
        if resource.endswith("s") and resource[:-1] in self.state:
            return resource[:-1]
        # Depluralize: -ies → -y
        if resource.endswith("ies") and resource[:-3] + "y" in self.state:
            return resource[:-3] + "y"
        # Try with underscores for multi-word resources
        for key in self.state:
            if key.replace("_", "") == resource.replace("-", ""):
                return key
        return None

    def _generic_list(self, state_key: str, args: list[str]) -> MockResult:
        """List items from a state key with optional filtering."""
        items = self.state.get(state_key, [])
        if not isinstance(items, list):
            return MockResult(
                stdout=json.dumps(items),
                stderr="",
                exit_code=0,
            )

        parsed = _parse_args(args)
        filtered = list(items)

        # Apply any --<field> <value> filters
        for flag, values in parsed.items():
            if flag == "" or not flag.startswith("--"):
                continue
            field_name = flag[2:]  # strip --
            filter_val = values[0]
            if filter_val:
                filtered = [
                    item
                    for item in filtered
                    if str(item.get(field_name, "")) == filter_val
                ]

        return MockResult(
            stdout=json.dumps(filtered),
            stderr="",
            exit_code=0,
        )

    def _generic_get(self, state_key: str, args: list[str]) -> MockResult:
        """Get a single item by --id or --name."""
        parsed = _parse_args(args)
        item_id = _get_flag(parsed, "--id") or _get_flag(parsed, "--name")
        if not item_id:
            return MockResult(
                stdout="",
                stderr="--id or --name is required",
                exit_code=1,
            )

        items = self.state.get(state_key, [])
        if not isinstance(items, list):
            return MockResult(
                stdout="",
                stderr=f"{state_key} is not a list",
                exit_code=1,
            )

        for item in items:
            if item.get("id") == item_id or item.get("name") == item_id:
                return MockResult(
                    stdout=json.dumps(item),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"{state_key} item '{item_id}' not found",
            exit_code=1,
        )

    def _generic_create(
        self, state_key: str, resource: str, args: list[str]
    ) -> MockResult:
        """Create a new item with auto-generated ID."""
        parsed = _parse_args(args)
        items = self.state.get(state_key, [])
        if not isinstance(items, list):
            return MockResult(
                stdout="",
                stderr=f"{state_key} is not a list",
                exit_code=1,
            )

        # Auto-generate ID
        prefix = self._id_prefix
        max_num = 0
        for item in items:
            item_id = item.get("id", "")
            if isinstance(item_id, str) and "-" in item_id:
                parts = item_id.rsplit("-", 1)
                if parts[1].isdigit():
                    num = int(parts[1])
                    if num > max_num:
                        max_num = num
                    prefix = parts[0]

        new_item: dict = {"id": f"{prefix}-{max_num + 1}"}

        # Build item from flags
        for flag, values in parsed.items():
            if flag == "" or not flag.startswith("--"):
                continue
            field_name = flag[2:].replace("-", "_")
            value = values[0]

            # Try to parse JSON values
            if value.startswith("{") or value.startswith("["):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass

            new_item[field_name] = value

        items.append(new_item)
        self.state[state_key] = items

        return MockResult(
            stdout=json.dumps(new_item),
            stderr="",
            exit_code=0,
        )

    def _generic_update(self, state_key: str, args: list[str]) -> MockResult:
        """Update fields on an existing item by --id."""
        parsed = _parse_args(args)
        item_id = _get_flag(parsed, "--id") or _get_flag(parsed, "--name")
        if not item_id:
            return MockResult(
                stdout="",
                stderr="--id or --name is required",
                exit_code=1,
            )

        items = self.state.get(state_key, [])
        if not isinstance(items, list):
            return MockResult(
                stdout="",
                stderr=f"{state_key} is not a list",
                exit_code=1,
            )

        target = None
        for item in items:
            if item.get("id") == item_id or item.get("name") == item_id:
                target = item
                break

        if target is None:
            return MockResult(
                stdout="",
                stderr=f"{state_key} item '{item_id}' not found",
                exit_code=1,
            )

        # Apply mutations from flags (skip --id/--name)
        for flag, values in parsed.items():
            if flag in ("", "--id", "--name") or not flag.startswith("--"):
                continue
            field_name = flag[2:].replace("-", "_")
            value = values[0]

            # Try to parse JSON values
            if value.startswith("{") or value.startswith("["):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass

            target[field_name] = value

        return MockResult(
            stdout=json.dumps(target),
            stderr="",
            exit_code=0,
        )

    def _generic_delete(self, state_key: str, args: list[str]) -> MockResult:
        """Delete an item by --id."""
        parsed = _parse_args(args)
        item_id = _get_flag(parsed, "--id") or _get_flag(parsed, "--name")
        if not item_id:
            return MockResult(
                stdout="",
                stderr="--id or --name is required",
                exit_code=1,
            )

        items = self.state.get(state_key, [])
        if not isinstance(items, list):
            return MockResult(
                stdout="",
                stderr=f"{state_key} is not a list",
                exit_code=1,
            )

        for i, item in enumerate(items):
            if item.get("id") == item_id or item.get("name") == item_id:
                removed = items.pop(i)
                return MockResult(
                    stdout=json.dumps({"deleted": removed["id"]}),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"{state_key} item '{item_id}' not found",
            exit_code=1,
        )
