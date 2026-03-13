"""Google Workspace mock backend — simulates a Google CLI tool.

Handles calendar event list/create/delete, gmail send/search/list,
and drive list/search commands.
"""

import json

from cli_bench.mock_backends.base import BaseMockBackend, MockResult


def _parse_args(args: list[str]) -> dict[str, list[str]]:
    """Parse CLI args into a dict mapping --flag to list of values."""
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


class GoogleMockBackend(BaseMockBackend):
    """Stateful mock for the Google Workspace CLI tool.

    State schema:
        {
            "calendar": {
                "events": [{"id": str, "title": str, "start": str, "end": str,
                             "description": str, "attendees": list[str],
                             "location": str}]
            },
            "gmail": {
                "messages": [{"id": str, "from": str, "to": str, "subject": str,
                              "body": str, "labels": list[str], "read": bool}]
            },
            "drive": {
                "files": [{"id": str, "name": str, "type": str, "folder": str,
                            "size": int}]
            }
        }
    """

    def route_command(self, command: list[str]) -> MockResult:
        """Route a google CLI command to the appropriate handler."""
        if len(command) < 3 or command[0] != "google":
            return MockResult(
                stdout="",
                stderr=f"unknown command: {' '.join(command)}",
                exit_code=1,
            )

        service = command[1]

        if service == "calendar":
            return self._handle_calendar(command[2:])
        if service == "gmail":
            return self._handle_gmail(command[2:])
        if service == "drive":
            return self._handle_drive(command[2:])

        return MockResult(
            stdout="",
            stderr=f"unknown service: google {service}",
            exit_code=1,
        )

    def _handle_calendar(self, args: list[str]) -> MockResult:
        """Handle google calendar <resource> <action> ..."""
        if len(args) < 2:
            return MockResult(
                stdout="",
                stderr="usage: google calendar event <action>",
                exit_code=1,
            )

        resource = args[0]
        action = args[1]
        remaining = args[2:]

        if resource != "event":
            return MockResult(
                stdout="",
                stderr=f"unknown calendar resource: {resource}",
                exit_code=1,
            )

        if action == "list":
            return self._calendar_event_list(remaining)
        if action == "create":
            return self._calendar_event_create(remaining)
        if action == "delete":
            return self._calendar_event_delete(remaining)

        return MockResult(
            stdout="",
            stderr=f"unknown calendar event subcommand: {action}",
            exit_code=1,
        )

    def _calendar_event_list(self, args: list[str]) -> MockResult:
        """List calendar events with optional date filtering."""
        parsed = _parse_args(args)
        events = list(self.state.get("calendar", {}).get("events", []))

        start_filter = _get_flag(parsed, "--start")
        if start_filter:
            events = [e for e in events if e.get("start", "") >= start_filter]

        end_filter = _get_flag(parsed, "--end")
        if end_filter:
            events = [e for e in events if e.get("end", "") <= end_filter]

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            events = events[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(events),
            stderr="",
            exit_code=0,
        )

    def _calendar_event_create(self, args: list[str]) -> MockResult:
        """Create a new calendar event."""
        parsed = _parse_args(args)

        title = _get_flag(parsed, "--title")
        if not title:
            return MockResult(stdout="", stderr="--title is required", exit_code=1)

        start = _get_flag(parsed, "--start")
        if not start:
            return MockResult(stdout="", stderr="--start is required", exit_code=1)

        end = _get_flag(parsed, "--end")
        if not end:
            return MockResult(stdout="", stderr="--end is required", exit_code=1)

        calendar_data = self.state.setdefault("calendar", {"events": []})
        events = calendar_data.setdefault("events", [])

        max_num = 0
        for event in events:
            parts = event["id"].split("-")
            if len(parts) >= 2 and parts[-1].isdigit():
                num = int(parts[-1])
                if num > max_num:
                    max_num = num

        attendees_str = _get_flag(parsed, "--attendees")
        attendees = attendees_str.split(",") if attendees_str else []

        new_event = {
            "id": f"evt-{max_num + 1:03d}",
            "title": title,
            "start": start,
            "end": end,
            "description": _get_flag(parsed, "--description") or "",
            "attendees": attendees,
            "location": _get_flag(parsed, "--location") or "",
        }

        events.append(new_event)

        return MockResult(
            stdout=json.dumps(new_event),
            stderr="",
            exit_code=0,
        )

    def _calendar_event_delete(self, args: list[str]) -> MockResult:
        """Delete a calendar event by --id."""
        parsed = _parse_args(args)
        event_id = _get_flag(parsed, "--id")
        if not event_id:
            return MockResult(stdout="", stderr="--id is required", exit_code=1)

        events = self.state.get("calendar", {}).get("events", [])
        for i, event in enumerate(events):
            if event["id"] == event_id:
                removed = events.pop(i)
                return MockResult(
                    stdout=json.dumps({"deleted": removed["id"]}),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"event {event_id} not found",
            exit_code=1,
        )

    def _handle_gmail(self, args: list[str]) -> MockResult:
        """Handle google gmail <action> ..."""
        if not args:
            return MockResult(
                stdout="",
                stderr="usage: google gmail <action>",
                exit_code=1,
            )

        action = args[0]
        remaining = args[1:]

        if action == "send":
            return self._gmail_send(remaining)
        if action == "search":
            return self._gmail_search(remaining)
        if action == "list":
            return self._gmail_list(remaining)

        return MockResult(
            stdout="",
            stderr=f"unknown gmail subcommand: {action}",
            exit_code=1,
        )

    def _gmail_send(self, args: list[str]) -> MockResult:
        """Send an email (mutates state)."""
        parsed = _parse_args(args)

        to = _get_flag(parsed, "--to")
        if not to:
            return MockResult(stdout="", stderr="--to is required", exit_code=1)

        subject = _get_flag(parsed, "--subject")
        if not subject:
            return MockResult(stdout="", stderr="--subject is required", exit_code=1)

        body = _get_flag(parsed, "--body")
        if not body:
            return MockResult(stdout="", stderr="--body is required", exit_code=1)

        gmail_data = self.state.setdefault("gmail", {"messages": []})
        messages = gmail_data.setdefault("messages", [])

        max_num = 0
        for msg in messages:
            parts = msg["id"].split("-")
            if len(parts) >= 2 and parts[-1].isdigit():
                num = int(parts[-1])
                if num > max_num:
                    max_num = num

        new_msg = {
            "id": f"msg-{max_num + 1:03d}",
            "from": "current_user@example.com",
            "to": to,
            "subject": subject,
            "body": body,
            "labels": ["sent"],
            "read": True,
        }

        messages.append(new_msg)

        return MockResult(
            stdout=json.dumps(new_msg),
            stderr="",
            exit_code=0,
        )

    def _gmail_search(self, args: list[str]) -> MockResult:
        """Search emails by query."""
        parsed = _parse_args(args)
        query = _get_flag(parsed, "--query")
        if not query:
            return MockResult(stdout="", stderr="--query is required", exit_code=1)

        messages = self.state.get("gmail", {}).get("messages", [])
        query_lower = query.lower()

        matches = []
        for msg in messages:
            if (
                query_lower in msg.get("subject", "").lower()
                or query_lower in msg.get("body", "").lower()
                or query_lower in msg.get("from", "").lower()
                or query_lower in msg.get("to", "").lower()
            ):
                matches.append(msg)

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            matches = matches[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(matches),
            stderr="",
            exit_code=0,
        )

    def _gmail_list(self, args: list[str]) -> MockResult:
        """List recent emails."""
        parsed = _parse_args(args)
        messages = list(self.state.get("gmail", {}).get("messages", []))

        label_filter = _get_flag(parsed, "--label")
        if label_filter:
            messages = [
                m for m in messages if label_filter in m.get("labels", [])
            ]

        unread_str = _get_flag(parsed, "--unread")
        if unread_str is not None:
            messages = [m for m in messages if not m.get("read", True)]

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            messages = messages[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(messages),
            stderr="",
            exit_code=0,
        )

    def _handle_drive(self, args: list[str]) -> MockResult:
        """Handle google drive <action> ..."""
        if not args:
            return MockResult(
                stdout="",
                stderr="usage: google drive <action>",
                exit_code=1,
            )

        action = args[0]
        remaining = args[1:]

        if action == "list":
            return self._drive_list(remaining)
        if action == "search":
            return self._drive_search(remaining)

        return MockResult(
            stdout="",
            stderr=f"unknown drive subcommand: {action}",
            exit_code=1,
        )

    def _drive_list(self, args: list[str]) -> MockResult:
        """List files in Drive."""
        parsed = _parse_args(args)
        files = list(self.state.get("drive", {}).get("files", []))

        folder_filter = _get_flag(parsed, "--folder")
        if folder_filter:
            files = [f for f in files if f.get("folder") == folder_filter]

        type_filter = _get_flag(parsed, "--type")
        if type_filter:
            files = [f for f in files if f.get("type") == type_filter]

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            files = files[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(files),
            stderr="",
            exit_code=0,
        )

    def _drive_search(self, args: list[str]) -> MockResult:
        """Search for files in Drive."""
        parsed = _parse_args(args)
        query = _get_flag(parsed, "--query")
        if not query:
            return MockResult(stdout="", stderr="--query is required", exit_code=1)

        files = self.state.get("drive", {}).get("files", [])
        query_lower = query.lower()

        matches = [f for f in files if query_lower in f.get("name", "").lower()]

        type_filter = _get_flag(parsed, "--type")
        if type_filter:
            matches = [f for f in matches if f.get("type") == type_filter]

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            matches = matches[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(matches),
            stderr="",
            exit_code=0,
        )
