"""GitHub mock backend — simulates the gh CLI tool.

Handles issue list/create/edit/view, PR list, and API commits endpoints.
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


class GitHubMockBackend(BaseMockBackend):
    """Stateful mock for the GitHub gh CLI tool.

    State schema:
        {
            "repos": {
                "owner/repo": {
                    "issues": [{"number": int, "title": str, "state": str,
                                "assignee": str|None, "labels": list, "body": str}],
                    "pulls": [{"number": int, "title": str, "state": str, "author": str}],
                    "commits": [{"sha": str, "message": str, "author": str, "files": list[str]}],
                }
            }
        }
    """

    def route_command(self, command: list[str]) -> MockResult:
        """Route a gh CLI command to the appropriate handler."""
        if len(command) < 2 or command[0] != "gh":
            return MockResult(
                stdout="",
                stderr=f"unknown command: {' '.join(command)}",
                exit_code=1,
            )

        subcommand = command[1]

        if subcommand == "issue":
            return self._handle_issue(command[2:])
        if subcommand == "pr":
            return self._handle_pr(command[2:])
        if subcommand == "api":
            return self._handle_api(command[2:])

        return MockResult(
            stdout="",
            stderr=f"unknown command: gh {subcommand}",
            exit_code=1,
        )

    def _get_repo(self, repo_slug: str) -> dict | None:
        """Look up a repo by owner/name slug, return None if not found."""
        return self.state.get("repos", {}).get(repo_slug)

    def _repo_not_found(self, repo_slug: str) -> MockResult:
        return MockResult(
            stdout="",
            stderr=f"repository {repo_slug} not found",
            exit_code=1,
        )

    def _handle_issue(self, args: list[str]) -> MockResult:
        """Handle gh issue <action> ..."""
        if not args:
            return MockResult(stdout="", stderr="missing issue subcommand", exit_code=1)

        action = args[0]
        remaining = args[1:]

        if action == "list":
            return self._issue_list(remaining)
        if action == "create":
            return self._issue_create(remaining)
        if action == "edit":
            return self._issue_edit(remaining)
        if action == "view":
            return self._issue_view(remaining)

        return MockResult(
            stdout="",
            stderr=f"unknown issue subcommand: {action}",
            exit_code=1,
        )

    def _issue_list(self, args: list[str]) -> MockResult:
        parsed = _parse_args(args)
        repo_slug = _get_flag(parsed, "--repo")
        if not repo_slug:
            return MockResult(stdout="", stderr="--repo is required", exit_code=1)

        repo = self._get_repo(repo_slug)
        if repo is None:
            return self._repo_not_found(repo_slug)

        issues = list(repo.get("issues", []))

        # Filter by state (default: all when listing)
        state_filter = _get_flag(parsed, "--state")
        if state_filter and state_filter != "all":
            issues = [i for i in issues if i["state"] == state_filter]

        # Filter by label
        label_filter = _get_flag(parsed, "--label")
        if label_filter:
            issues = [i for i in issues if label_filter in i.get("labels", [])]

        return MockResult(
            stdout=json.dumps(issues),
            stderr="",
            exit_code=0,
        )

    def _issue_create(self, args: list[str]) -> MockResult:
        parsed = _parse_args(args)
        repo_slug = _get_flag(parsed, "--repo")
        if not repo_slug:
            return MockResult(stdout="", stderr="--repo is required", exit_code=1)

        repo = self._get_repo(repo_slug)
        if repo is None:
            return self._repo_not_found(repo_slug)

        title = _get_flag(parsed, "--title")
        if not title:
            return MockResult(stdout="", stderr="--title is required", exit_code=1)

        existing_issues = repo.get("issues", [])
        max_number = max((i["number"] for i in existing_issues), default=0)

        new_issue = {
            "number": max_number + 1,
            "title": title,
            "state": "open",
            "assignee": _get_flag(parsed, "--assignee"),
            "labels": _get_all_flags(parsed, "--label"),
            "body": _get_flag(parsed, "--body") or "",
        }

        existing_issues.append(new_issue)
        repo["issues"] = existing_issues

        return MockResult(
            stdout=json.dumps(new_issue),
            stderr="",
            exit_code=0,
        )

    def _issue_edit(self, args: list[str]) -> MockResult:
        parsed = _parse_args(args)
        positional = parsed.get("", [])
        if not positional:
            return MockResult(stdout="", stderr="issue number is required", exit_code=1)

        issue_number = int(positional[0])
        repo_slug = _get_flag(parsed, "--repo")
        if not repo_slug:
            return MockResult(stdout="", stderr="--repo is required", exit_code=1)

        repo = self._get_repo(repo_slug)
        if repo is None:
            return self._repo_not_found(repo_slug)

        issue = None
        for i in repo.get("issues", []):
            if i["number"] == issue_number:
                issue = i
                break

        if issue is None:
            return MockResult(
                stdout="",
                stderr=f"issue {issue_number} not found in {repo_slug}",
                exit_code=1,
            )

        # Apply mutations
        new_assignee = _get_flag(parsed, "--add-assignee")
        if new_assignee:
            issue["assignee"] = new_assignee

        new_title = _get_flag(parsed, "--title")
        if new_title:
            issue["title"] = new_title

        new_labels = _get_all_flags(parsed, "--add-label")
        for label in new_labels:
            if label not in issue.get("labels", []):
                issue.setdefault("labels", []).append(label)

        return MockResult(
            stdout=json.dumps(issue),
            stderr="",
            exit_code=0,
        )

    def _issue_view(self, args: list[str]) -> MockResult:
        parsed = _parse_args(args)
        positional = parsed.get("", [])
        if not positional:
            return MockResult(stdout="", stderr="issue number is required", exit_code=1)

        issue_number = int(positional[0])
        repo_slug = _get_flag(parsed, "--repo")
        if not repo_slug:
            return MockResult(stdout="", stderr="--repo is required", exit_code=1)

        repo = self._get_repo(repo_slug)
        if repo is None:
            return self._repo_not_found(repo_slug)

        for issue in repo.get("issues", []):
            if issue["number"] == issue_number:
                return MockResult(
                    stdout=json.dumps(issue),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"issue {issue_number} not found in {repo_slug}",
            exit_code=1,
        )

    def _handle_pr(self, args: list[str]) -> MockResult:
        """Handle gh pr <action> ..."""
        if not args:
            return MockResult(stdout="", stderr="missing pr subcommand", exit_code=1)

        action = args[0]
        remaining = args[1:]

        if action == "list":
            return self._pr_list(remaining)

        return MockResult(
            stdout="",
            stderr=f"unknown pr subcommand: {action}",
            exit_code=1,
        )

    def _pr_list(self, args: list[str]) -> MockResult:
        parsed = _parse_args(args)
        repo_slug = _get_flag(parsed, "--repo")
        if not repo_slug:
            return MockResult(stdout="", stderr="--repo is required", exit_code=1)

        repo = self._get_repo(repo_slug)
        if repo is None:
            return self._repo_not_found(repo_slug)

        pulls = repo.get("pulls", [])
        return MockResult(
            stdout=json.dumps(pulls),
            stderr="",
            exit_code=0,
        )

    def _handle_api(self, args: list[str]) -> MockResult:
        """Handle gh api <endpoint>."""
        if not args:
            return MockResult(stdout="", stderr="missing API endpoint", exit_code=1)

        endpoint = args[0]

        # Parse repos/OWNER/REPO/commits
        parts = endpoint.split("/")
        if len(parts) >= 4 and parts[0] == "repos" and parts[3] == "commits":
            repo_slug = f"{parts[1]}/{parts[2]}"
            repo = self._get_repo(repo_slug)
            if repo is None:
                return self._repo_not_found(repo_slug)

            commits = repo.get("commits", [])
            return MockResult(
                stdout=json.dumps(commits),
                stderr="",
                exit_code=0,
            )

        return MockResult(
            stdout="",
            stderr=f"unknown API endpoint: {endpoint}",
            exit_code=1,
        )
