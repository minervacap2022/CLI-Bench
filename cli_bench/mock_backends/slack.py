"""Slack mock backend — simulates a Slack CLI tool.

Handles channel list/info/message, message send, dm send,
search message, and user list commands.
"""

import json
import time

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


class SlackMockBackend(BaseMockBackend):
    """Stateful mock for the Slack CLI tool (kslack style).

    State schema:
        {
            "channels": {
                "channel-name": {
                    "id": "C0123",
                    "topic": "Channel topic",
                    "messages": [{"from": str, "text": str, "ts": str}],
                    "pins": [],
                }
            },
            "dms": {
                "username": {
                    "messages": [{"from": str, "text": str, "ts": str}]
                }
            },
            "users": [{"id": str, "name": str, "email": str}]
        }
    """

    def route_command(self, command: list[str]) -> MockResult:
        """Route a slack CLI command to the appropriate handler."""
        if len(command) < 3 or command[0] != "slack":
            return MockResult(
                stdout="",
                stderr=f"unknown command: {' '.join(command)}",
                exit_code=1,
            )

        resource = command[1]
        action = command[2]
        remaining = command[3:]

        if resource == "channel":
            return self._handle_channel(action, remaining)
        if resource == "message":
            return self._handle_message(action, remaining)
        if resource == "dm":
            return self._handle_dm(action, remaining)
        if resource == "search":
            return self._handle_search(action, remaining)
        if resource == "user":
            return self._handle_user(action, remaining)

        return MockResult(
            stdout="",
            stderr=f"unknown command: slack {resource} {action}",
            exit_code=1,
        )

    def _channel_not_found(self, channel: str) -> MockResult:
        return MockResult(
            stdout="",
            stderr=f"channel '{channel}' not found",
            exit_code=1,
        )

    def _handle_channel(self, action: str, args: list[str]) -> MockResult:
        """Handle slack channel <action> ..."""
        if action == "list":
            return self._channel_list()
        if action == "message":
            return self._channel_messages(args)
        if action == "info":
            return self._channel_info(args)

        return MockResult(
            stdout="",
            stderr=f"unknown channel subcommand: {action}",
            exit_code=1,
        )

    def _channel_list(self) -> MockResult:
        """List all channels with their IDs."""
        channels = self.state.get("channels", {})
        result = [
            {"name": name, "id": data["id"], "topic": data["topic"]}
            for name, data in channels.items()
        ]
        return MockResult(
            stdout=json.dumps(result),
            stderr="",
            exit_code=0,
        )

    def _channel_messages(self, args: list[str]) -> MockResult:
        """List messages in a channel, optionally limited."""
        parsed = _parse_args(args)
        channel_name = _get_flag(parsed, "--channel")
        if not channel_name:
            return MockResult(stdout="", stderr="--channel is required", exit_code=1)

        channels = self.state.get("channels", {})
        if channel_name not in channels:
            return self._channel_not_found(channel_name)

        messages = list(channels[channel_name]["messages"])

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            messages = messages[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(messages),
            stderr="",
            exit_code=0,
        )

    def _channel_info(self, args: list[str]) -> MockResult:
        """Return details for a specific channel."""
        parsed = _parse_args(args)
        channel_name = _get_flag(parsed, "--channel")
        if not channel_name:
            return MockResult(stdout="", stderr="--channel is required", exit_code=1)

        channels = self.state.get("channels", {})
        if channel_name not in channels:
            return self._channel_not_found(channel_name)

        data = channels[channel_name]
        info = {
            "name": channel_name,
            "id": data["id"],
            "topic": data["topic"],
            "message_count": len(data["messages"]),
            "pins": data.get("pins", []),
        }
        return MockResult(
            stdout=json.dumps(info),
            stderr="",
            exit_code=0,
        )

    def _handle_message(self, action: str, args: list[str]) -> MockResult:
        """Handle slack message <action> ..."""
        if action == "send":
            return self._message_send(args)

        return MockResult(
            stdout="",
            stderr=f"unknown message subcommand: {action}",
            exit_code=1,
        )

    def _message_send(self, args: list[str]) -> MockResult:
        """Post a message to a channel (mutates state)."""
        parsed = _parse_args(args)
        channel_name = _get_flag(parsed, "--channel")
        if not channel_name:
            return MockResult(stdout="", stderr="--channel is required", exit_code=1)

        text = _get_flag(parsed, "--text")
        if not text:
            return MockResult(stdout="", stderr="--text is required", exit_code=1)

        channels = self.state.get("channels", {})
        if channel_name not in channels:
            return self._channel_not_found(channel_name)

        new_message = {
            "from": "current_user",
            "text": text,
            "ts": str(int(time.time())),
        }
        channels[channel_name]["messages"].append(new_message)

        return MockResult(
            stdout=json.dumps({"ok": True, "ts": new_message["ts"]}),
            stderr="",
            exit_code=0,
        )

    def _handle_dm(self, action: str, args: list[str]) -> MockResult:
        """Handle slack dm <action> ..."""
        if action == "send":
            return self._dm_send(args)

        return MockResult(
            stdout="",
            stderr=f"unknown dm subcommand: {action}",
            exit_code=1,
        )

    def _dm_send(self, args: list[str]) -> MockResult:
        """Send a DM to a user (mutates state, creates DM entry if needed)."""
        parsed = _parse_args(args)
        username = _get_flag(parsed, "--user")
        if not username:
            return MockResult(stdout="", stderr="--user is required", exit_code=1)

        text = _get_flag(parsed, "--text")
        if not text:
            return MockResult(stdout="", stderr="--text is required", exit_code=1)

        dms = self.state.setdefault("dms", {})
        if username not in dms:
            dms[username] = {"messages": []}

        new_message = {
            "from": "current_user",
            "text": text,
            "ts": str(int(time.time())),
        }
        dms[username]["messages"].append(new_message)

        return MockResult(
            stdout=json.dumps({"ok": True, "ts": new_message["ts"]}),
            stderr="",
            exit_code=0,
        )

    def _handle_search(self, action: str, args: list[str]) -> MockResult:
        """Handle slack search <action> ..."""
        if action == "message":
            return self._search_message(args)

        return MockResult(
            stdout="",
            stderr=f"unknown search subcommand: {action}",
            exit_code=1,
        )

    def _search_message(self, args: list[str]) -> MockResult:
        """Search messages across all channels by keyword."""
        parsed = _parse_args(args)
        query = _get_flag(parsed, "--query")
        if not query:
            return MockResult(stdout="", stderr="--query is required", exit_code=1)

        matches: list[dict] = []
        channels = self.state.get("channels", {})
        for channel_name, channel_data in channels.items():
            for msg in channel_data["messages"]:
                if query.lower() in msg["text"].lower():
                    matches.append({
                        "channel": channel_name,
                        "from": msg["from"],
                        "text": msg["text"],
                        "ts": msg["ts"],
                    })

        return MockResult(
            stdout=json.dumps(matches),
            stderr="",
            exit_code=0,
        )

    def _handle_user(self, action: str, args: list[str]) -> MockResult:
        """Handle slack user <action> ..."""
        if action == "list":
            return self._user_list()

        return MockResult(
            stdout="",
            stderr=f"unknown user subcommand: {action}",
            exit_code=1,
        )

    def _user_list(self) -> MockResult:
        """List all users."""
        users = self.state.get("users", [])
        return MockResult(
            stdout=json.dumps(users),
            stderr="",
            exit_code=0,
        )
