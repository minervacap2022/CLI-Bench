"""Tests for SlackMockBackend — simulates a Slack CLI tool."""

import json

import pytest

from cli_bench.mock_backends.slack import SlackMockBackend


@pytest.fixture
def backend() -> SlackMockBackend:
    """Backend with 2 channels, 1 DM thread, and 3 users."""
    return SlackMockBackend(
        initial_state={
            "channels": {
                "general": {
                    "id": "C001",
                    "topic": "General discussion",
                    "messages": [
                        {"from": "alice", "text": "Hello everyone!", "ts": "1710000001"},
                        {"from": "bob", "text": "Hey alice, welcome!", "ts": "1710000002"},
                    ],
                    "pins": [],
                },
                "engineering": {
                    "id": "C002",
                    "topic": "Engineering team",
                    "messages": [
                        {"from": "charlie", "text": "Deploy is done", "ts": "1710000010"},
                    ],
                    "pins": [],
                },
            },
            "dms": {
                "bob": {
                    "messages": [
                        {"from": "alice", "text": "Can you review my PR?", "ts": "1710000050"},
                    ]
                }
            },
            "users": [
                {"id": "U001", "name": "alice", "email": "alice@example.com"},
                {"id": "U002", "name": "bob", "email": "bob@example.com"},
                {"id": "U003", "name": "charlie", "email": "charlie@example.com"},
            ],
        }
    )


class TestChannelList:
    def test_channel_list(self, backend: SlackMockBackend) -> None:
        """Lists channels with IDs."""
        result = backend.execute(["slack", "channel", "list"])
        assert result.exit_code == 0
        channels = json.loads(result.stdout)
        assert len(channels) == 2
        names = {ch["name"] for ch in channels}
        assert names == {"general", "engineering"}
        # Each channel entry should have id
        for ch in channels:
            assert "id" in ch
            assert "name" in ch


class TestSendMessage:
    def test_send_message(self, backend: SlackMockBackend) -> None:
        """Posts to channel, state mutated."""
        result = backend.execute(
            ["slack", "message", "send", "--channel", "general", "--text", "New message here"]
        )
        assert result.exit_code == 0

        # Verify state mutated
        snapshot = backend.get_state_snapshot()
        messages = snapshot["channels"]["general"]["messages"]
        assert len(messages) == 3
        assert messages[-1]["text"] == "New message here"


class TestDmSend:
    def test_dm_send(self, backend: SlackMockBackend) -> None:
        """Sends DM, creates DM entry if not exists."""
        # Send to existing DM thread
        result = backend.execute(
            ["slack", "dm", "send", "--user", "bob", "--text", "Thanks for the review!"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        dm_messages = snapshot["dms"]["bob"]["messages"]
        assert len(dm_messages) == 2
        assert dm_messages[-1]["text"] == "Thanks for the review!"

    def test_dm_send_creates_entry(self, backend: SlackMockBackend) -> None:
        """Sends DM to user with no existing DM entry — creates it."""
        result = backend.execute(
            ["slack", "dm", "send", "--user", "charlie", "--text", "Hi Charlie!"]
        )
        assert result.exit_code == 0

        snapshot = backend.get_state_snapshot()
        assert "charlie" in snapshot["dms"]
        dm_messages = snapshot["dms"]["charlie"]["messages"]
        assert len(dm_messages) == 1
        assert dm_messages[0]["text"] == "Hi Charlie!"


class TestSearchMessage:
    def test_search_message(self, backend: SlackMockBackend) -> None:
        """Finds messages by keyword across all channels."""
        result = backend.execute(
            ["slack", "search", "message", "--query", "Deploy"]
        )
        assert result.exit_code == 0
        matches = json.loads(result.stdout)
        assert len(matches) >= 1
        texts = [m["text"] for m in matches]
        assert any("Deploy" in t for t in texts)

    def test_search_message_no_results(self, backend: SlackMockBackend) -> None:
        """Search with no matching messages returns empty list."""
        result = backend.execute(
            ["slack", "search", "message", "--query", "nonexistent_xyz"]
        )
        assert result.exit_code == 0
        matches = json.loads(result.stdout)
        assert len(matches) == 0


class TestChannelNotFound:
    def test_channel_not_found(self, backend: SlackMockBackend) -> None:
        """Message to nonexistent channel returns error."""
        result = backend.execute(
            ["slack", "message", "send", "--channel", "nonexistent", "--text", "hello"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestChannelMessages:
    def test_channel_messages(self, backend: SlackMockBackend) -> None:
        """Lists messages in channel."""
        result = backend.execute(
            ["slack", "channel", "message", "--channel", "general"]
        )
        assert result.exit_code == 0
        messages = json.loads(result.stdout)
        assert len(messages) == 2
        assert messages[0]["from"] == "alice"
        assert messages[1]["from"] == "bob"

    def test_channel_messages_with_limit(self, backend: SlackMockBackend) -> None:
        """Lists messages with --limit."""
        result = backend.execute(
            ["slack", "channel", "message", "--channel", "general", "--limit", "1"]
        )
        assert result.exit_code == 0
        messages = json.loads(result.stdout)
        assert len(messages) == 1

    def test_channel_messages_not_found(self, backend: SlackMockBackend) -> None:
        """Messages from nonexistent channel returns error."""
        result = backend.execute(
            ["slack", "channel", "message", "--channel", "nonexistent"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestUserList:
    def test_user_list(self, backend: SlackMockBackend) -> None:
        """Returns users."""
        result = backend.execute(["slack", "user", "list"])
        assert result.exit_code == 0
        users = json.loads(result.stdout)
        assert len(users) == 3
        names = {u["name"] for u in users}
        assert names == {"alice", "bob", "charlie"}


class TestChannelInfo:
    def test_channel_info(self, backend: SlackMockBackend) -> None:
        """Channel details returned."""
        result = backend.execute(
            ["slack", "channel", "info", "--channel", "general"]
        )
        assert result.exit_code == 0
        info = json.loads(result.stdout)
        assert info["id"] == "C001"
        assert info["topic"] == "General discussion"
        assert info["name"] == "general"

    def test_channel_info_not_found(self, backend: SlackMockBackend) -> None:
        """Info for nonexistent channel returns error."""
        result = backend.execute(
            ["slack", "channel", "info", "--channel", "nonexistent"]
        )
        assert result.exit_code == 1
        assert result.stderr != ""


class TestUnknownCommand:
    def test_unknown_command(self, backend: SlackMockBackend) -> None:
        """Unknown slack subcommand returns exit_code=1."""
        result = backend.execute(["slack", "unknown", "subcommand"])
        assert result.exit_code == 1
        assert result.stderr != ""
