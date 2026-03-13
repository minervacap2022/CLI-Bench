"""Tests for GoogleMockBackend — simulates Google Workspace CLI tool."""

import json

import pytest

from cli_bench.mock_backends.google import GoogleMockBackend


@pytest.fixture
def backend() -> GoogleMockBackend:
    """Backend with calendar events, gmail messages, and drive files."""
    return GoogleMockBackend(
        initial_state={
            "calendar": {
                "events": [
                    {
                        "id": "evt-001",
                        "title": "Team Standup",
                        "start": "2026-03-10T09:00:00Z",
                        "end": "2026-03-10T09:30:00Z",
                        "description": "Daily standup",
                        "attendees": ["alice@co.com", "bob@co.com"],
                        "location": "Room A",
                    },
                    {
                        "id": "evt-002",
                        "title": "Sprint Review",
                        "start": "2026-03-12T14:00:00Z",
                        "end": "2026-03-12T15:00:00Z",
                        "description": "End of sprint review",
                        "attendees": ["team@co.com"],
                        "location": "",
                    },
                    {
                        "id": "evt-003",
                        "title": "1:1 with Manager",
                        "start": "2026-03-15T10:00:00Z",
                        "end": "2026-03-15T10:30:00Z",
                        "description": "",
                        "attendees": ["manager@co.com"],
                        "location": "Office",
                    },
                ]
            },
            "gmail": {
                "messages": [
                    {
                        "id": "msg-001",
                        "from": "alice@co.com",
                        "to": "current_user@co.com",
                        "subject": "Project Update",
                        "body": "Here is the latest project status",
                        "labels": ["inbox"],
                        "read": False,
                    },
                    {
                        "id": "msg-002",
                        "from": "bob@co.com",
                        "to": "current_user@co.com",
                        "subject": "Meeting Notes",
                        "body": "Notes from yesterday's meeting",
                        "labels": ["inbox"],
                        "read": True,
                    },
                    {
                        "id": "msg-003",
                        "from": "current_user@co.com",
                        "to": "charlie@co.com",
                        "subject": "Follow Up",
                        "body": "Following up on our discussion",
                        "labels": ["sent"],
                        "read": True,
                    },
                ]
            },
            "drive": {
                "files": [
                    {
                        "id": "file-001",
                        "name": "Q1 Report.docx",
                        "type": "document",
                        "folder": "root",
                        "size": 15000,
                    },
                    {
                        "id": "file-002",
                        "name": "Budget 2026.xlsx",
                        "type": "spreadsheet",
                        "folder": "root",
                        "size": 32000,
                    },
                    {
                        "id": "file-003",
                        "name": "Architecture.pdf",
                        "type": "pdf",
                        "folder": "engineering",
                        "size": 1200000,
                    },
                ]
            },
        }
    )


class TestCalendarEventList:
    def test_list_all_events(self, backend: GoogleMockBackend) -> None:
        """Lists all calendar events."""
        result = backend.execute(["google", "calendar", "event", "list"])
        assert result.exit_code == 0
        events = json.loads(result.stdout)
        assert len(events) == 3

    def test_list_events_with_date_filter(self, backend: GoogleMockBackend) -> None:
        """Filter events by date range."""
        result = backend.execute(
            [
                "google", "calendar", "event", "list",
                "--start", "2026-03-11T00:00:00Z",
                "--end", "2026-03-13T23:59:59Z",
            ]
        )
        assert result.exit_code == 0
        events = json.loads(result.stdout)
        assert len(events) == 1
        assert events[0]["title"] == "Sprint Review"

    def test_list_events_with_limit(self, backend: GoogleMockBackend) -> None:
        """Limit event results."""
        result = backend.execute(
            ["google", "calendar", "event", "list", "--limit", "2"]
        )
        assert result.exit_code == 0
        events = json.loads(result.stdout)
        assert len(events) == 2


class TestCalendarEventCreate:
    def test_create_event(self, backend: GoogleMockBackend) -> None:
        """Creates a new calendar event."""
        result = backend.execute(
            [
                "google", "calendar", "event", "create",
                "--title", "New Meeting",
                "--start", "2026-03-20T10:00:00Z",
                "--end", "2026-03-20T11:00:00Z",
                "--attendees", "alice@co.com,bob@co.com",
                "--location", "Room B",
            ]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["title"] == "New Meeting"
        assert created["id"] == "evt-004"
        assert "alice@co.com" in created["attendees"]
        assert created["location"] == "Room B"

        snapshot = backend.get_state_snapshot()
        assert len(snapshot["calendar"]["events"]) == 4

    def test_create_event_requires_title(self, backend: GoogleMockBackend) -> None:
        result = backend.execute(
            [
                "google", "calendar", "event", "create",
                "--start", "2026-03-20T10:00:00Z",
                "--end", "2026-03-20T11:00:00Z",
            ]
        )
        assert result.exit_code == 1
        assert "--title" in result.stderr

    def test_create_event_requires_start(self, backend: GoogleMockBackend) -> None:
        result = backend.execute(
            [
                "google", "calendar", "event", "create",
                "--title", "Test",
                "--end", "2026-03-20T11:00:00Z",
            ]
        )
        assert result.exit_code == 1
        assert "--start" in result.stderr


class TestCalendarEventDelete:
    def test_delete_event(self, backend: GoogleMockBackend) -> None:
        """Deletes an event, state mutated."""
        result = backend.execute(
            ["google", "calendar", "event", "delete", "--id", "evt-002"]
        )
        assert result.exit_code == 0
        deleted = json.loads(result.stdout)
        assert deleted["deleted"] == "evt-002"

        snapshot = backend.get_state_snapshot()
        assert len(snapshot["calendar"]["events"]) == 2

    def test_delete_event_not_found(self, backend: GoogleMockBackend) -> None:
        result = backend.execute(
            ["google", "calendar", "event", "delete", "--id", "evt-999"]
        )
        assert result.exit_code == 1


class TestGmailSend:
    def test_send_email(self, backend: GoogleMockBackend) -> None:
        """Sends email, state mutated."""
        result = backend.execute(
            [
                "google", "gmail", "send",
                "--to", "alice@co.com",
                "--subject", "Test Subject",
                "--body", "Test body content",
            ]
        )
        assert result.exit_code == 0
        sent = json.loads(result.stdout)
        assert sent["to"] == "alice@co.com"
        assert sent["subject"] == "Test Subject"
        assert "sent" in sent["labels"]

        snapshot = backend.get_state_snapshot()
        assert len(snapshot["gmail"]["messages"]) == 4

    def test_send_requires_to(self, backend: GoogleMockBackend) -> None:
        result = backend.execute(
            [
                "google", "gmail", "send",
                "--subject", "Test",
                "--body", "Test",
            ]
        )
        assert result.exit_code == 1
        assert "--to" in result.stderr


class TestGmailSearch:
    def test_search_by_subject(self, backend: GoogleMockBackend) -> None:
        """Search emails by keyword in subject."""
        result = backend.execute(
            ["google", "gmail", "search", "--query", "project"]
        )
        assert result.exit_code == 0
        matches = json.loads(result.stdout)
        assert len(matches) >= 1
        assert any("Project" in m["subject"] for m in matches)

    def test_search_no_results(self, backend: GoogleMockBackend) -> None:
        result = backend.execute(
            ["google", "gmail", "search", "--query", "nonexistent_xyz"]
        )
        assert result.exit_code == 0
        assert json.loads(result.stdout) == []


class TestGmailList:
    def test_list_all(self, backend: GoogleMockBackend) -> None:
        """Lists all emails."""
        result = backend.execute(["google", "gmail", "list"])
        assert result.exit_code == 0
        messages = json.loads(result.stdout)
        assert len(messages) == 3

    def test_list_by_label(self, backend: GoogleMockBackend) -> None:
        """Filter by label."""
        result = backend.execute(
            ["google", "gmail", "list", "--label", "inbox"]
        )
        assert result.exit_code == 0
        messages = json.loads(result.stdout)
        assert len(messages) == 2

    def test_list_unread(self, backend: GoogleMockBackend) -> None:
        """Filter unread messages."""
        result = backend.execute(
            ["google", "gmail", "list", "--unread", "true"]
        )
        assert result.exit_code == 0
        messages = json.loads(result.stdout)
        assert len(messages) == 1
        assert messages[0]["subject"] == "Project Update"


class TestDriveList:
    def test_list_all_files(self, backend: GoogleMockBackend) -> None:
        """Lists all drive files."""
        result = backend.execute(["google", "drive", "list"])
        assert result.exit_code == 0
        files = json.loads(result.stdout)
        assert len(files) == 3

    def test_list_by_folder(self, backend: GoogleMockBackend) -> None:
        """Filter by folder."""
        result = backend.execute(
            ["google", "drive", "list", "--folder", "root"]
        )
        assert result.exit_code == 0
        files = json.loads(result.stdout)
        assert len(files) == 2

    def test_list_by_type(self, backend: GoogleMockBackend) -> None:
        """Filter by file type."""
        result = backend.execute(
            ["google", "drive", "list", "--type", "spreadsheet"]
        )
        assert result.exit_code == 0
        files = json.loads(result.stdout)
        assert len(files) == 1
        assert files[0]["name"] == "Budget 2026.xlsx"


class TestDriveSearch:
    def test_search_files(self, backend: GoogleMockBackend) -> None:
        """Search files by name."""
        result = backend.execute(
            ["google", "drive", "search", "--query", "report"]
        )
        assert result.exit_code == 0
        files = json.loads(result.stdout)
        assert len(files) == 1
        assert files[0]["name"] == "Q1 Report.docx"

    def test_search_with_type_filter(self, backend: GoogleMockBackend) -> None:
        """Search filtered by type."""
        result = backend.execute(
            ["google", "drive", "search", "--query", "budget", "--type", "spreadsheet"]
        )
        assert result.exit_code == 0
        files = json.loads(result.stdout)
        assert len(files) == 1

    def test_search_no_results(self, backend: GoogleMockBackend) -> None:
        result = backend.execute(
            ["google", "drive", "search", "--query", "nonexistent_xyz"]
        )
        assert result.exit_code == 0
        assert json.loads(result.stdout) == []
