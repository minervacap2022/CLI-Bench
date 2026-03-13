"""Linear mock backend — simulates the klinear CLI tool.

Handles issue list/get/create/update/comment, project list, and team list.
"""

import json

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


def _get_all_flags(parsed: dict[str, list[str]], flag: str) -> list[str]:
    """Get all values for a repeated flag."""
    return parsed.get(flag, [])


class LinearMockBackend(BaseMockBackend):
    """Stateful mock for the Linear klinear CLI tool.

    State schema:
        {
            "issues": [{"id": str, "title": str, "assignee": str|None,
                         "status": str, "priority": int, "project": str|None,
                         "labels": list, "description": str}],
            "projects": [{"id": str, "name": str, "status": str, "lead": str|None}],
            "teams": [{"id": str, "name": str, "members": list[str]}],
            "comments": [{"id": str, "issue_id": str, "body": str, "author": str}],
        }
    """

    def route_command(self, command: list[str]) -> MockResult:
        """Route a linear CLI command to the appropriate handler."""
        if len(command) < 3 or command[0] != "linear":
            return MockResult(
                stdout="",
                stderr=f"unknown command: {' '.join(command)}",
                exit_code=1,
            )

        resource = command[1]
        action = command[2]
        remaining = command[3:]

        if resource == "issue":
            return self._handle_issue(action, remaining)
        if resource == "project":
            return self._handle_project(action)
        if resource == "team":
            return self._handle_team(action)

        return MockResult(
            stdout="",
            stderr=f"unknown command: linear {resource} {action}",
            exit_code=1,
        )

    def _handle_issue(self, action: str, args: list[str]) -> MockResult:
        """Handle linear issue <action> ..."""
        if action == "list":
            return self._issue_list(args)
        if action == "get":
            return self._issue_get(args)
        if action == "create":
            return self._issue_create(args)
        if action == "update":
            return self._issue_update(args)
        if action == "comment":
            return self._issue_comment(args)

        return MockResult(
            stdout="",
            stderr=f"unknown issue subcommand: {action}",
            exit_code=1,
        )

    def _issue_list(self, args: list[str]) -> MockResult:
        """List issues with optional filtering by project, assignee, status."""
        parsed = _parse_args(args)
        issues = list(self.state.get("issues", []))

        project_filter = _get_flag(parsed, "--project")
        if project_filter:
            issues = [i for i in issues if i.get("project") == project_filter]

        assignee_filter = _get_flag(parsed, "--assignee")
        if assignee_filter:
            issues = [i for i in issues if i.get("assignee") == assignee_filter]

        status_filter = _get_flag(parsed, "--status")
        if status_filter:
            issues = [i for i in issues if i.get("status") == status_filter]

        return MockResult(
            stdout=json.dumps(issues),
            stderr="",
            exit_code=0,
        )

    def _issue_get(self, args: list[str]) -> MockResult:
        """Get a single issue by --id."""
        parsed = _parse_args(args)
        issue_id = _get_flag(parsed, "--id")
        if not issue_id:
            return MockResult(stdout="", stderr="--id is required", exit_code=1)

        for issue in self.state.get("issues", []):
            if issue["id"] == issue_id:
                return MockResult(
                    stdout=json.dumps(issue),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"issue {issue_id} not found",
            exit_code=1,
        )

    def _issue_create(self, args: list[str]) -> MockResult:
        """Create a new issue with auto-generated ID."""
        parsed = _parse_args(args)

        title = _get_flag(parsed, "--title")
        if not title:
            return MockResult(stdout="", stderr="--title is required", exit_code=1)

        existing_issues = self.state.get("issues", [])
        max_num = 0
        for issue in existing_issues:
            # Extract numeric suffix from "ISS-N"
            parts = issue["id"].split("-")
            if len(parts) == 2 and parts[1].isdigit():
                num = int(parts[1])
                if num > max_num:
                    max_num = num

        priority_str = _get_flag(parsed, "--priority")
        priority = int(priority_str) if priority_str else 0

        new_issue = {
            "id": f"ISS-{max_num + 1}",
            "title": title,
            "assignee": _get_flag(parsed, "--assignee"),
            "status": "Todo",
            "priority": priority,
            "project": _get_flag(parsed, "--project"),
            "labels": _get_all_flags(parsed, "--label"),
            "description": _get_flag(parsed, "--description") or "",
        }

        existing_issues.append(new_issue)
        self.state["issues"] = existing_issues

        return MockResult(
            stdout=json.dumps(new_issue),
            stderr="",
            exit_code=0,
        )

    def _issue_update(self, args: list[str]) -> MockResult:
        """Update fields on an existing issue by --id."""
        parsed = _parse_args(args)
        issue_id = _get_flag(parsed, "--id")
        if not issue_id:
            return MockResult(stdout="", stderr="--id is required", exit_code=1)

        issue = None
        for i in self.state.get("issues", []):
            if i["id"] == issue_id:
                issue = i
                break

        if issue is None:
            return MockResult(
                stdout="",
                stderr=f"issue {issue_id} not found",
                exit_code=1,
            )

        new_assignee = _get_flag(parsed, "--assignee")
        if new_assignee is not None:
            issue["assignee"] = new_assignee

        new_status = _get_flag(parsed, "--status")
        if new_status is not None:
            issue["status"] = new_status

        new_priority = _get_flag(parsed, "--priority")
        if new_priority is not None:
            issue["priority"] = int(new_priority)

        new_title = _get_flag(parsed, "--title")
        if new_title is not None:
            issue["title"] = new_title

        return MockResult(
            stdout=json.dumps(issue),
            stderr="",
            exit_code=0,
        )

    def _issue_comment(self, args: list[str]) -> MockResult:
        """Add a comment to an issue."""
        parsed = _parse_args(args)

        issue_id = _get_flag(parsed, "--id")
        if not issue_id:
            return MockResult(stdout="", stderr="--id is required", exit_code=1)

        body = _get_flag(parsed, "--body")
        if not body:
            return MockResult(stdout="", stderr="--body is required", exit_code=1)

        author = _get_flag(parsed, "--author") or "current_user"

        existing_comments = self.state.get("comments", [])
        max_num = 0
        for comment in existing_comments:
            parts = comment["id"].split("-")
            if len(parts) == 2 and parts[1].isdigit():
                num = int(parts[1])
                if num > max_num:
                    max_num = num

        new_comment = {
            "id": f"COM-{max_num + 1}",
            "issue_id": issue_id,
            "body": body,
            "author": author,
        }

        existing_comments.append(new_comment)
        self.state["comments"] = existing_comments

        return MockResult(
            stdout=json.dumps(new_comment),
            stderr="",
            exit_code=0,
        )

    def _handle_project(self, action: str) -> MockResult:
        """Handle linear project <action>."""
        if action == "list":
            projects = self.state.get("projects", [])
            return MockResult(
                stdout=json.dumps(projects),
                stderr="",
                exit_code=0,
            )

        return MockResult(
            stdout="",
            stderr=f"unknown project subcommand: {action}",
            exit_code=1,
        )

    def _handle_team(self, action: str) -> MockResult:
        """Handle linear team <action>."""
        if action == "list":
            teams = self.state.get("teams", [])
            return MockResult(
                stdout=json.dumps(teams),
                stderr="",
                exit_code=0,
            )

        return MockResult(
            stdout="",
            stderr=f"unknown team subcommand: {action}",
            exit_code=1,
        )
