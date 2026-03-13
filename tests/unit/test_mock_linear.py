"""Tests for LinearMockBackend — simulates the klinear CLI tool."""

import json

import pytest

from cli_bench.mock_backends.linear import LinearMockBackend


@pytest.fixture
def backend() -> LinearMockBackend:
    """Backend with 2 issues, 1 project, 1 team, and 1 comment."""
    return LinearMockBackend(
        initial_state={
            "issues": [
                {
                    "id": "ISS-1",
                    "title": "Fix auth flow",
                    "assignee": "alice",
                    "status": "In Progress",
                    "priority": 1,
                    "project": "Backend",
                    "labels": ["bug"],
                    "description": "Auth redirects fail on mobile",
                },
                {
                    "id": "ISS-2",
                    "title": "Add onboarding wizard",
                    "assignee": "bob",
                    "status": "Todo",
                    "priority": 2,
                    "project": "Frontend",
                    "labels": ["feature", "ux"],
                    "description": "New user onboarding flow",
                },
            ],
            "projects": [
                {"id": "PRJ-1", "name": "Backend", "status": "active", "lead": "alice"},
                {"id": "PRJ-2", "name": "Frontend", "status": "active", "lead": "bob"},
            ],
            "teams": [
                {"id": "TEAM-1", "name": "Engineering", "members": ["alice", "bob", "charlie"]},
            ],
            "comments": [
                {"id": "COM-1", "issue_id": "ISS-1", "body": "Investigating now", "author": "alice"},
            ],
        }
    )


class TestIssueList:
    def test_issue_list(self, backend: LinearMockBackend) -> None:
        """Lists all issues."""
        result = backend.execute(["linear", "issue", "list"])
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 2
        assert issues[0]["id"] == "ISS-1"
        assert issues[1]["id"] == "ISS-2"

    def test_issue_list_filter_assignee(self, backend: LinearMockBackend) -> None:
        """Filter issues by --assignee."""
        result = backend.execute(
            ["linear", "issue", "list", "--assignee", "alice"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["assignee"] == "alice"
        assert issues[0]["title"] == "Fix auth flow"

    def test_issue_list_filter_project(self, backend: LinearMockBackend) -> None:
        """Filter issues by --project."""
        result = backend.execute(
            ["linear", "issue", "list", "--project", "Frontend"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["project"] == "Frontend"
        assert issues[0]["title"] == "Add onboarding wizard"

    def test_issue_list_filter_status(self, backend: LinearMockBackend) -> None:
        """Filter issues by --status."""
        result = backend.execute(
            ["linear", "issue", "list", "--status", "Todo"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["status"] == "Todo"


class TestIssueCreate:
    def test_issue_create(self, backend: LinearMockBackend) -> None:
        """Creates issue with auto-generated ID, state mutated."""
        result = backend.execute(
            ["linear", "issue", "create", "--title", "New task",
             "--description", "Some details", "--assignee", "charlie",
             "--project", "Backend", "--priority", "3", "--label", "enhancement"]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["title"] == "New task"
        assert created["description"] == "Some details"
        assert created["assignee"] == "charlie"
        assert created["project"] == "Backend"
        assert created["priority"] == 3
        assert "enhancement" in created["labels"]
        # Auto-generated ID
        assert created["id"].startswith("ISS-")

        # Verify state mutated
        snapshot = backend.get_state_snapshot()
        assert len(snapshot["issues"]) == 3
        assert snapshot["issues"][-1]["title"] == "New task"

    def test_issue_create_minimal(self, backend: LinearMockBackend) -> None:
        """Creates issue with only --title, defaults applied."""
        result = backend.execute(
            ["linear", "issue", "create", "--title", "Minimal issue"]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["title"] == "Minimal issue"
        assert created["assignee"] is None
        assert created["project"] is None
        assert created["labels"] == []
        assert created["status"] == "Todo"

    def test_issue_create_missing_title(self, backend: LinearMockBackend) -> None:
        """Create without --title returns error."""
        result = backend.execute(["linear", "issue", "create"])
        assert result.exit_code == 1
        assert result.stderr != ""


class TestIssueUpdate:
    def test_issue_update_assignee(self, backend: LinearMockBackend) -> None:
        """Updates assignee on existing issue."""
        result = backend.execute(
            ["linear", "issue", "update", "--id", "ISS-1", "--assignee", "charlie"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        issue = next(i for i in snapshot["issues"] if i["id"] == "ISS-1")
        assert issue["assignee"] == "charlie"

    def test_issue_update_status(self, backend: LinearMockBackend) -> None:
        """Updates status on existing issue."""
        result = backend.execute(
            ["linear", "issue", "update", "--id", "ISS-1", "--status", "Done"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        issue = next(i for i in snapshot["issues"] if i["id"] == "ISS-1")
        assert issue["status"] == "Done"

    def test_issue_update_priority(self, backend: LinearMockBackend) -> None:
        """Updates priority on existing issue."""
        result = backend.execute(
            ["linear", "issue", "update", "--id", "ISS-1", "--priority", "0"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        issue = next(i for i in snapshot["issues"] if i["id"] == "ISS-1")
        assert issue["priority"] == 0

    def test_issue_update_title(self, backend: LinearMockBackend) -> None:
        """Updates title on existing issue."""
        result = backend.execute(
            ["linear", "issue", "update", "--id", "ISS-1", "--title", "Updated title"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        issue = next(i for i in snapshot["issues"] if i["id"] == "ISS-1")
        assert issue["title"] == "Updated title"

    def test_issue_update_not_found(self, backend: LinearMockBackend) -> None:
        """Update nonexistent issue returns error."""
        result = backend.execute(
            ["linear", "issue", "update", "--id", "ISS-999", "--status", "Done"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestIssueGet:
    def test_issue_get(self, backend: LinearMockBackend) -> None:
        """Get single issue by --id."""
        result = backend.execute(
            ["linear", "issue", "get", "--id", "ISS-1"]
        )
        assert result.exit_code == 0
        issue = json.loads(result.stdout)
        assert issue["id"] == "ISS-1"
        assert issue["title"] == "Fix auth flow"
        assert issue["assignee"] == "alice"
        assert issue["status"] == "In Progress"

    def test_issue_not_found(self, backend: LinearMockBackend) -> None:
        """Get nonexistent issue returns error."""
        result = backend.execute(
            ["linear", "issue", "get", "--id", "ISS-999"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestProjectList:
    def test_project_list(self, backend: LinearMockBackend) -> None:
        """Lists all projects."""
        result = backend.execute(["linear", "project", "list"])
        assert result.exit_code == 0
        projects = json.loads(result.stdout)
        assert len(projects) == 2
        names = {p["name"] for p in projects}
        assert names == {"Backend", "Frontend"}


class TestTeamList:
    def test_team_list(self, backend: LinearMockBackend) -> None:
        """Lists all teams."""
        result = backend.execute(["linear", "team", "list"])
        assert result.exit_code == 0
        teams = json.loads(result.stdout)
        assert len(teams) == 1
        assert teams[0]["name"] == "Engineering"
        assert "alice" in teams[0]["members"]


class TestIssueComment:
    def test_issue_comment(self, backend: LinearMockBackend) -> None:
        """Adds comment to existing issue."""
        result = backend.execute(
            ["linear", "issue", "comment", "--id", "ISS-2",
             "--body", "Looking into this", "--author", "charlie"]
        )
        assert result.exit_code == 0
        comment = json.loads(result.stdout)
        assert comment["issue_id"] == "ISS-2"
        assert comment["body"] == "Looking into this"
        assert comment["author"] == "charlie"
        assert comment["id"].startswith("COM-")

        # Verify state mutated
        snapshot = backend.get_state_snapshot()
        assert len(snapshot["comments"]) == 2
        assert snapshot["comments"][-1]["body"] == "Looking into this"

    def test_issue_comment_default_author(self, backend: LinearMockBackend) -> None:
        """Comment without --author defaults to current_user."""
        result = backend.execute(
            ["linear", "issue", "comment", "--id", "ISS-1", "--body", "On it"]
        )
        assert result.exit_code == 0
        comment = json.loads(result.stdout)
        assert comment["author"] == "current_user"

    def test_issue_comment_missing_body(self, backend: LinearMockBackend) -> None:
        """Comment without --body returns error."""
        result = backend.execute(
            ["linear", "issue", "comment", "--id", "ISS-1"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""

    def test_issue_comment_missing_id(self, backend: LinearMockBackend) -> None:
        """Comment without --id returns error."""
        result = backend.execute(
            ["linear", "issue", "comment", "--body", "Hello"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestUnknownCommand:
    def test_unknown_command(self, backend: LinearMockBackend) -> None:
        """Unknown linear subcommand returns exit_code=1."""
        result = backend.execute(["linear", "unknown", "subcommand"])
        assert result.exit_code == 1
        assert result.stderr != ""

    def test_not_linear_command(self, backend: LinearMockBackend) -> None:
        """Non-linear command returns exit_code=1."""
        result = backend.execute(["gh", "issue", "list"])
        assert result.exit_code == 1
        assert result.stderr != ""
