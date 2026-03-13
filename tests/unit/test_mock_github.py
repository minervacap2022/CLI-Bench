"""Tests for GitHubMockBackend — simulates the gh CLI tool."""

import json

import pytest

from cli_bench.mock_backends.github import GitHubMockBackend


@pytest.fixture
def backend() -> GitHubMockBackend:
    """Backend with 2 issues and 1 commit in owner/repo."""
    return GitHubMockBackend(
        initial_state={
            "repos": {
                "owner/repo": {
                    "issues": [
                        {
                            "number": 1,
                            "title": "Bug in login",
                            "state": "open",
                            "assignee": None,
                            "labels": ["bug"],
                            "body": "Login fails on Safari",
                        },
                        {
                            "number": 2,
                            "title": "Add dark mode",
                            "state": "closed",
                            "assignee": "alice",
                            "labels": ["enhancement"],
                            "body": "Users want dark mode",
                        },
                    ],
                    "pulls": [
                        {
                            "number": 10,
                            "title": "Fix login bug",
                            "state": "open",
                            "author": "bob",
                        },
                    ],
                    "commits": [
                        {
                            "sha": "abc123",
                            "message": "Initial commit",
                            "author": "alice",
                            "files": ["README.md"],
                        },
                    ],
                }
            }
        }
    )


class TestIssueList:
    def test_issue_list(self, backend: GitHubMockBackend) -> None:
        """Lists all issues and returns JSON."""
        result = backend.execute(
            ["gh", "issue", "list", "--repo", "owner/repo", "--json",
             "number,title,state,assignee,labels,body"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 2
        assert issues[0]["number"] == 1
        assert issues[0]["title"] == "Bug in login"
        assert issues[1]["number"] == 2

    def test_issue_list_with_state_filter(self, backend: GitHubMockBackend) -> None:
        """Filter issues by --state open."""
        result = backend.execute(
            ["gh", "issue", "list", "--repo", "owner/repo", "--state", "open"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["state"] == "open"
        assert issues[0]["title"] == "Bug in login"

    def test_issue_list_state_closed(self, backend: GitHubMockBackend) -> None:
        """Filter issues by --state closed."""
        result = backend.execute(
            ["gh", "issue", "list", "--repo", "owner/repo", "--state", "closed"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["state"] == "closed"

    def test_issue_list_state_all(self, backend: GitHubMockBackend) -> None:
        """--state all returns everything."""
        result = backend.execute(
            ["gh", "issue", "list", "--repo", "owner/repo", "--state", "all"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 2

    def test_issue_list_with_label_filter(self, backend: GitHubMockBackend) -> None:
        """Filter issues by --label."""
        result = backend.execute(
            ["gh", "issue", "list", "--repo", "owner/repo", "--label", "bug"]
        )
        assert result.exit_code == 0
        issues = json.loads(result.stdout)
        assert len(issues) == 1
        assert issues[0]["title"] == "Bug in login"


class TestIssueCreate:
    def test_issue_create(self, backend: GitHubMockBackend) -> None:
        """Creates issue, state mutated, auto-increments number."""
        result = backend.execute(
            ["gh", "issue", "create", "--repo", "owner/repo",
             "--title", "New feature request", "--body", "Please add X"]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["number"] == 3
        assert created["title"] == "New feature request"
        assert created["body"] == "Please add X"
        assert created["state"] == "open"

        # Verify state mutated
        snapshot = backend.get_state_snapshot()
        issues = snapshot["repos"]["owner/repo"]["issues"]
        assert len(issues) == 3
        assert issues[-1]["number"] == 3

    def test_issue_create_with_labels(self, backend: GitHubMockBackend) -> None:
        """Create issue with --label flag."""
        result = backend.execute(
            ["gh", "issue", "create", "--repo", "owner/repo",
             "--title", "Labeled issue", "--label", "bug",
             "--label", "urgent"]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["number"] == 3
        assert "bug" in created["labels"]
        assert "urgent" in created["labels"]

    def test_issue_create_with_assignee(self, backend: GitHubMockBackend) -> None:
        """Create issue with --assignee flag."""
        result = backend.execute(
            ["gh", "issue", "create", "--repo", "owner/repo",
             "--title", "Assigned issue", "--assignee", "charlie"]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["assignee"] == "charlie"


class TestIssueEdit:
    def test_issue_edit_assignee(self, backend: GitHubMockBackend) -> None:
        """Edit adds assignee to existing issue."""
        result = backend.execute(
            ["gh", "issue", "edit", "1", "--repo", "owner/repo",
             "--add-assignee", "charlie"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        issue = snapshot["repos"]["owner/repo"]["issues"][0]
        assert issue["assignee"] == "charlie"

    def test_issue_edit_title(self, backend: GitHubMockBackend) -> None:
        """Edit changes issue title."""
        result = backend.execute(
            ["gh", "issue", "edit", "1", "--repo", "owner/repo",
             "--title", "Updated title"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        issue = snapshot["repos"]["owner/repo"]["issues"][0]
        assert issue["title"] == "Updated title"

    def test_issue_edit_add_label(self, backend: GitHubMockBackend) -> None:
        """Edit adds label to existing issue."""
        result = backend.execute(
            ["gh", "issue", "edit", "1", "--repo", "owner/repo",
             "--add-label", "critical"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        issue = snapshot["repos"]["owner/repo"]["issues"][0]
        assert "critical" in issue["labels"]
        assert "bug" in issue["labels"]  # original label preserved

    def test_issue_edit_not_found(self, backend: GitHubMockBackend) -> None:
        """Edit nonexistent issue number returns error."""
        result = backend.execute(
            ["gh", "issue", "edit", "999", "--repo", "owner/repo",
             "--title", "nope"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestIssueView:
    def test_issue_view(self, backend: GitHubMockBackend) -> None:
        """View single issue by number."""
        result = backend.execute(
            ["gh", "issue", "view", "1", "--repo", "owner/repo"]
        )
        assert result.exit_code == 0
        issue = json.loads(result.stdout)
        assert issue["number"] == 1
        assert issue["title"] == "Bug in login"
        assert issue["body"] == "Login fails on Safari"

    def test_issue_view_not_found(self, backend: GitHubMockBackend) -> None:
        """View nonexistent issue returns error."""
        result = backend.execute(
            ["gh", "issue", "view", "999", "--repo", "owner/repo"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestRepoNotFound:
    def test_repo_not_found(self, backend: GitHubMockBackend) -> None:
        """Nonexistent repo returns exit_code=1."""
        result = backend.execute(
            ["gh", "issue", "list", "--repo", "nonexistent/repo"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestPRList:
    def test_pr_list(self, backend: GitHubMockBackend) -> None:
        """List pull requests for a repo."""
        result = backend.execute(
            ["gh", "pr", "list", "--repo", "owner/repo"]
        )
        assert result.exit_code == 0
        prs = json.loads(result.stdout)
        assert len(prs) == 1
        assert prs[0]["number"] == 10
        assert prs[0]["title"] == "Fix login bug"
        assert prs[0]["author"] == "bob"


class TestCommitLog:
    def test_commit_log(self, backend: GitHubMockBackend) -> None:
        """gh api repos/.../commits returns commits."""
        result = backend.execute(
            ["gh", "api", "repos/owner/repo/commits"]
        )
        assert result.exit_code == 0
        commits = json.loads(result.stdout)
        assert len(commits) == 1
        assert commits[0]["sha"] == "abc123"
        assert commits[0]["message"] == "Initial commit"


class TestUnknownCommand:
    def test_unknown_command(self, backend: GitHubMockBackend) -> None:
        """Unknown gh subcommand returns exit_code=1 and stderr."""
        result = backend.execute(["gh", "unknown", "subcommand"])
        assert result.exit_code == 1
        assert result.stderr != ""
