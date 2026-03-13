"""Jira mock backend — simulates a Jira CLI tool.

Handles issue list/create/update/view, project list, and search commands.
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


class JiraMockBackend(BaseMockBackend):
    """Stateful mock for the Jira CLI tool.

    State schema:
        {
            "projects": [{"key": str, "name": str, "lead": str}],
            "issues": [{"key": str, "project": str, "title": str,
                         "type": str, "status": str, "assignee": str|None,
                         "priority": str, "description": str,
                         "labels": list[str]}],
        }
    """

    def route_command(self, command: list[str]) -> MockResult:
        """Route a jira CLI command to the appropriate handler."""
        if len(command) < 3 or command[0] != "jira":
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
        if resource == "search":
            # search has no sub-action; the action position is actually the first arg
            return self._handle_search(command[2:])

        return MockResult(
            stdout="",
            stderr=f"unknown command: jira {resource}",
            exit_code=1,
        )

    def _handle_issue(self, action: str, args: list[str]) -> MockResult:
        """Handle jira issue <action> ..."""
        if action == "list":
            return self._issue_list(args)
        if action == "create":
            return self._issue_create(args)
        if action == "update":
            return self._issue_update(args)
        if action == "view":
            return self._issue_view(args)

        return MockResult(
            stdout="",
            stderr=f"unknown issue subcommand: {action}",
            exit_code=1,
        )

    def _issue_list(self, args: list[str]) -> MockResult:
        """List issues with optional filtering."""
        parsed = _parse_args(args)
        issues = list(self.state.get("issues", []))

        project_filter = _get_flag(parsed, "--project")
        if project_filter:
            issues = [i for i in issues if i.get("project") == project_filter]

        status_filter = _get_flag(parsed, "--status")
        if status_filter:
            issues = [i for i in issues if i.get("status") == status_filter]

        assignee_filter = _get_flag(parsed, "--assignee")
        if assignee_filter:
            issues = [i for i in issues if i.get("assignee") == assignee_filter]

        type_filter = _get_flag(parsed, "--type")
        if type_filter:
            issues = [i for i in issues if i.get("type") == type_filter]

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            issues = issues[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(issues),
            stderr="",
            exit_code=0,
        )

    def _issue_create(self, args: list[str]) -> MockResult:
        """Create a new issue in a project."""
        parsed = _parse_args(args)

        project = _get_flag(parsed, "--project")
        if not project:
            return MockResult(stdout="", stderr="--project is required", exit_code=1)

        title = _get_flag(parsed, "--title")
        if not title:
            return MockResult(stdout="", stderr="--title is required", exit_code=1)

        issue_type = _get_flag(parsed, "--type")
        if not issue_type:
            return MockResult(stdout="", stderr="--type is required", exit_code=1)

        # Find max issue number for this project
        issues = self.state.get("issues", [])
        max_num = 0
        for issue in issues:
            key = issue.get("key", "")
            if key.startswith(f"{project}-"):
                parts = key.split("-")
                if len(parts) == 2 and parts[1].isdigit():
                    num = int(parts[1])
                    if num > max_num:
                        max_num = num

        labels_str = _get_flag(parsed, "--labels")
        labels = labels_str.split(",") if labels_str else []

        new_issue = {
            "key": f"{project}-{max_num + 1}",
            "project": project,
            "title": title,
            "type": issue_type,
            "status": "To Do",
            "assignee": _get_flag(parsed, "--assignee"),
            "priority": _get_flag(parsed, "--priority") or "Medium",
            "description": _get_flag(parsed, "--description") or "",
            "labels": labels,
        }

        issues.append(new_issue)
        self.state["issues"] = issues

        return MockResult(
            stdout=json.dumps(new_issue),
            stderr="",
            exit_code=0,
        )

    def _issue_update(self, args: list[str]) -> MockResult:
        """Update fields on an existing issue by --key."""
        parsed = _parse_args(args)
        issue_key = _get_flag(parsed, "--key")
        if not issue_key:
            return MockResult(stdout="", stderr="--key is required", exit_code=1)

        issue = None
        for i in self.state.get("issues", []):
            if i["key"] == issue_key:
                issue = i
                break

        if issue is None:
            return MockResult(
                stdout="",
                stderr=f"issue {issue_key} not found",
                exit_code=1,
            )

        new_status = _get_flag(parsed, "--status")
        if new_status is not None:
            issue["status"] = new_status

        new_assignee = _get_flag(parsed, "--assignee")
        if new_assignee is not None:
            issue["assignee"] = new_assignee

        new_priority = _get_flag(parsed, "--priority")
        if new_priority is not None:
            issue["priority"] = new_priority

        new_title = _get_flag(parsed, "--title")
        if new_title is not None:
            issue["title"] = new_title

        return MockResult(
            stdout=json.dumps(issue),
            stderr="",
            exit_code=0,
        )

    def _issue_view(self, args: list[str]) -> MockResult:
        """View a single issue by --key."""
        parsed = _parse_args(args)
        issue_key = _get_flag(parsed, "--key")
        if not issue_key:
            return MockResult(stdout="", stderr="--key is required", exit_code=1)

        for issue in self.state.get("issues", []):
            if issue["key"] == issue_key:
                return MockResult(
                    stdout=json.dumps(issue),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"issue {issue_key} not found",
            exit_code=1,
        )

    def _handle_project(self, action: str) -> MockResult:
        """Handle jira project <action>."""
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

    def _handle_search(self, args: list[str]) -> MockResult:
        """Handle jira search --jql ..."""
        parsed = _parse_args(args)
        jql = _get_flag(parsed, "--jql")
        if not jql:
            return MockResult(stdout="", stderr="--jql is required", exit_code=1)

        # Simple JQL parsing: look for key=value patterns
        issues = list(self.state.get("issues", []))
        jql_lower = jql.lower()

        # Extract project filter
        if "project" in jql_lower:
            for token in jql.split():
                if token.startswith('"') or token.startswith("'"):
                    continue
                # Look for project = VALUE pattern
                parts = jql.split("project")
                if len(parts) > 1:
                    rest = parts[1].strip()
                    if rest.startswith("="):
                        val = rest[1:].strip().strip("'\"").split()[0]
                        issues = [
                            i for i in issues if i.get("project") == val
                        ]
                    break

        # Extract status filter
        if "status" in jql_lower:
            parts = jql.split("status")
            if len(parts) > 1:
                rest = parts[1].strip()
                if rest.startswith("="):
                    val = rest[1:].strip().strip("'\"").split(" AND")[0].split(" OR")[0].strip()
                    issues = [
                        i for i in issues if i.get("status") == val
                    ]

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            issues = issues[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(issues),
            stderr="",
            exit_code=0,
        )
