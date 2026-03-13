"""Tests for JiraMockBackend — simulates the Jira CLI tool."""

import json

import pytest

from cli_bench.mock_backends.jira import JiraMockBackend


@pytest.fixture
def backend() -> JiraMockBackend:
    """Backend with 2 projects and 3 issues."""
    return JiraMockBackend(
        initial_state={
            "projects": [
                {"key": "PROJ", "name": "Main Project", "lead": "alice"},
                {"key": "OPS", "name": "Operations", "lead": "bob"},
            ],
            "issues": [
                {
                    "key": "PROJ-1",
                    "project": "PROJ",
                    "title": "Fix login page",
                    "type": "Bug",
                    "status": "To Do",
                    "assignee": "alice",
                    "priority": "High",
                    "description": "Login button does not work on Safari",
                    "labels": ["frontend", "urgent"],
                },
                {
                    "key": "PROJ-2",
                    "project": "PROJ",
                    "title": "Add search feature",
                    "type": "Story",
                    "status": "In Progress",
                    "assignee": "bob",
                    "priority": "Medium",
                    "description": "Implement full-text search",
                    "labels": ["backend"],
                },
                {
                    "key": "OPS-1",
                    "project": "OPS",
                    "title": "Set up monitoring",
                    "type": "Task",
                    "status": "To Do",
                    "assignee": None,
                    "priority": "Medium",
                    "description": "Configure Prometheus and Grafana",
                    "labels": ["infrastructure"],
                },
            ],
        }
    )


class TestIssueList:
    def test_list_all_issues_in_project(self, backend: JiraMockBackend) -> None:
        """Lists issues filtered by project."""
        result = backend.execute(
            ["jira", "issue", "list", "--project", "PROJ"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 2
        for issue in issues:
            assert issue["project"] == "PROJ"

    def test_list_with_status_filter(self, backend: JiraMockBackend) -> None:
        """Filter by status."""
        result = backend.execute(
            ["jira", "issue", "list", "--project", "PROJ", "--status", "In Progress"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["title"] == "Add search feature"

    def test_list_with_assignee_filter(self, backend: JiraMockBackend) -> None:
        """Filter by assignee."""
        result = backend.execute(
            ["jira", "issue", "list", "--project", "PROJ", "--assignee", "alice"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["key"] == "PROJ-1"

    def test_list_with_type_filter(self, backend: JiraMockBackend) -> None:
        """Filter by issue type."""
        result = backend.execute(
            ["jira", "issue", "list", "--project", "PROJ", "--type", "Bug"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["type"] == "Bug"

    def test_list_with_limit(self, backend: JiraMockBackend) -> None:
        """Limit number of returned issues."""
        result = backend.execute(
            ["jira", "issue", "list", "--project", "PROJ", "--limit", "1"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1


class TestIssueCreate:
    def test_create_issue(self, backend: JiraMockBackend) -> None:
        """Creates issue, state mutated, auto-generates key."""
        result = backend.execute(
            [
                "jira", "issue", "create",
                "--project", "PROJ",
                "--title", "Implement caching",
                "--type", "Story",
                "--assignee", "charlie",
                "--priority", "High",
            ]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["key"] == "PROJ-3"
        assert created["title"] == "Implement caching"
        assert created["type"] == "Story"
        assert created["status"] == "To Do"
        assert created["assignee"] == "charlie"
        assert created["priority"] == "High"

        snapshot = backend.get_state_snapshot()
        assert len(snapshot["issues"]) == 4

    def test_create_issue_requires_project(self, backend: JiraMockBackend) -> None:
        result = backend.execute(
            ["jira", "issue", "create", "--title", "Test", "--type", "Task"]
        )
        assert result.exit_code == 1
        assert "--project" in result.stderr

    def test_create_issue_requires_title(self, backend: JiraMockBackend) -> None:
        result = backend.execute(
            ["jira", "issue", "create", "--project", "PROJ", "--type", "Task"]
        )
        assert result.exit_code == 1
        assert "--title" in result.stderr

    def test_create_issue_requires_type(self, backend: JiraMockBackend) -> None:
        result = backend.execute(
            ["jira", "issue", "create", "--project", "PROJ", "--title", "Test"]
        )
        assert result.exit_code == 1
        assert "--type" in result.stderr

    def test_create_issue_with_labels(self, backend: JiraMockBackend) -> None:
        """Creates issue with comma-separated labels."""
        result = backend.execute(
            [
                "jira", "issue", "create",
                "--project", "OPS",
                "--title", "Setup CI",
                "--type", "Task",
                "--labels", "devops,ci",
            ]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert "devops" in created["labels"]
        assert "ci" in created["labels"]


class TestIssueUpdate:
    def test_update_status(self, backend: JiraMockBackend) -> None:
        """Transitions issue status."""
        result = backend.execute(
            ["jira", "issue", "update", "--key", "PROJ-1", "--status", "In Progress"]
        )
        assert result.exit_code == 0
        updated = json.loads(result.stdout)
        assert updated["status"] == "In Progress"

        snapshot = backend.get_state_snapshot()
        issue = [i for i in snapshot["issues"] if i["key"] == "PROJ-1"][0]
        assert issue["status"] == "In Progress"

    def test_update_assignee(self, backend: JiraMockBackend) -> None:
        """Reassigns an issue."""
        result = backend.execute(
            ["jira", "issue", "update", "--key", "OPS-1", "--assignee", "charlie"]
        )
        assert result.exit_code == 0
        updated = json.loads(result.stdout)
        assert updated["assignee"] == "charlie"

    def test_update_priority(self, backend: JiraMockBackend) -> None:
        """Changes issue priority."""
        result = backend.execute(
            ["jira", "issue", "update", "--key", "PROJ-2", "--priority", "Highest"]
        )
        assert result.exit_code == 0
        updated = json.loads(result.stdout)
        assert updated["priority"] == "Highest"

    def test_update_not_found(self, backend: JiraMockBackend) -> None:
        """Update nonexistent issue returns error."""
        result = backend.execute(
            ["jira", "issue", "update", "--key", "PROJ-999", "--status", "Done"]
        )
        assert result.exit_code == 1
        assert "not found" in result.stderr


class TestIssueView:
    def test_view_issue(self, backend: JiraMockBackend) -> None:
        """View a single issue by key."""
        result = backend.execute(["jira", "issue", "view", "--key", "PROJ-1"])
        assert result.exit_code == 0
        issue = json.loads(result.stdout)
        assert issue["key"] == "PROJ-1"
        assert issue["title"] == "Fix login page"

    def test_view_not_found(self, backend: JiraMockBackend) -> None:
        result = backend.execute(["jira", "issue", "view", "--key", "PROJ-999"])
        assert result.exit_code == 1


class TestProjectList:
    def test_project_list(self, backend: JiraMockBackend) -> None:
        """Lists all projects."""
        result = backend.execute(["jira", "project", "list"])
        assert result.exit_code == 0
        projects = json.loads(result.stdout)
        assert len(projects) == 2
        keys = {p["key"] for p in projects}
        assert keys == {"PROJ", "OPS"}


class TestSearch:
    def test_search_by_project(self, backend: JiraMockBackend) -> None:
        """JQL search filters by project."""
        result = backend.execute(
            ["jira", "search", "--jql", "project = PROJ"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 2
        for issue in issues:
            assert issue["project"] == "PROJ"

    def test_search_by_status(self, backend: JiraMockBackend) -> None:
        """JQL search filters by status."""
        result = backend.execute(
            ["jira", "search", "--jql", 'project = PROJ AND status = "In Progress"']
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["status"] == "In Progress"

    def test_search_requires_jql(self, backend: JiraMockBackend) -> None:
        """Search without --jql returns error."""
        result = backend.execute(["jira", "search", "--limit", "10"])
        assert result.exit_code == 1
        assert "--jql" in result.stderr
