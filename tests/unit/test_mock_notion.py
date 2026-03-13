"""Tests for NotionMockBackend — simulates the Notion CLI tool."""

import json

import pytest

from cli_bench.mock_backends.notion import NotionMockBackend


@pytest.fixture
def backend() -> NotionMockBackend:
    """Backend with pages and a database."""
    return NotionMockBackend(
        initial_state={
            "pages": [
                {
                    "id": "page-001",
                    "title": "Meeting Notes Q1",
                    "content": "Discussion about roadmap priorities",
                    "parent": "db-001",
                    "archived": False,
                    "blocks": [
                        {"type": "paragraph", "content": "First item discussed"},
                    ],
                },
                {
                    "id": "page-002",
                    "title": "Sprint Planning",
                    "content": "Sprint 42 planning session",
                    "parent": "db-001",
                    "archived": False,
                    "blocks": [],
                },
                {
                    "id": "page-003",
                    "title": "Archived Doc",
                    "content": "Old archived content",
                    "parent": None,
                    "archived": True,
                    "blocks": [],
                },
            ],
            "databases": [
                {
                    "id": "db-001",
                    "title": "Team Wiki",
                    "rows": [
                        {"id": "row-001", "name": "Feature A", "status": "active"},
                        {"id": "row-002", "name": "Feature B", "status": "done"},
                        {"id": "row-003", "name": "Feature C", "status": "active"},
                    ],
                },
            ],
        }
    )


class TestPageList:
    def test_list_pages(self, backend: NotionMockBackend) -> None:
        """Lists non-archived pages."""
        result = backend.execute(["notion", "page", "list"])
        assert result.exit_code == 0
        pages = json.loads(result.stdout)
        assert len(pages) == 2  # archived excluded
        titles = {p["title"] for p in pages}
        assert "Meeting Notes Q1" in titles
        assert "Sprint Planning" in titles

    def test_list_pages_with_parent_filter(self, backend: NotionMockBackend) -> None:
        """Filter pages by parent."""
        result = backend.execute(
            ["notion", "page", "list", "--parent", "db-001"]
        )
        assert result.exit_code == 0
        pages = json.loads(result.stdout)
        assert len(pages) == 2
        for p in pages:
            assert p.get("parent") == "db-001"

    def test_list_pages_with_limit(self, backend: NotionMockBackend) -> None:
        """Limit number of returned pages."""
        result = backend.execute(["notion", "page", "list", "--limit", "1"])
        assert result.exit_code == 0
        pages = json.loads(result.stdout)
        assert len(pages) == 1


class TestPageCreate:
    def test_create_page(self, backend: NotionMockBackend) -> None:
        """Creates page, state mutated."""
        result = backend.execute(
            [
                "notion", "page", "create",
                "--title", "Design Doc",
                "--parent", "db-001",
                "--content", "Architecture overview",
            ]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["title"] == "Design Doc"
        assert created["parent"] == "db-001"
        assert created["content"] == "Architecture overview"
        assert created["id"] == "page-004"

        # Verify state mutated
        snapshot = backend.get_state_snapshot()
        assert len(snapshot["pages"]) == 4

    def test_create_page_requires_title(self, backend: NotionMockBackend) -> None:
        """Create without --title returns error."""
        result = backend.execute(
            ["notion", "page", "create", "--parent", "db-001"]
        )
        assert result.exit_code == 1
        assert "--title" in result.stderr

    def test_create_page_requires_parent(self, backend: NotionMockBackend) -> None:
        """Create without --parent returns error."""
        result = backend.execute(
            ["notion", "page", "create", "--title", "Test"]
        )
        assert result.exit_code == 1
        assert "--parent" in result.stderr


class TestPageUpdate:
    def test_update_page_title(self, backend: NotionMockBackend) -> None:
        """Updates page title."""
        result = backend.execute(
            ["notion", "page", "update", "--id", "page-001", "--title", "Updated Title"]
        )
        assert result.exit_code == 0
        updated = json.loads(result.stdout)
        assert updated["title"] == "Updated Title"

    def test_update_page_content(self, backend: NotionMockBackend) -> None:
        """Updates page content."""
        result = backend.execute(
            ["notion", "page", "update", "--id", "page-001", "--content", "New content"]
        )
        assert result.exit_code == 0
        updated = json.loads(result.stdout)
        assert updated["content"] == "New content"

    def test_archive_page(self, backend: NotionMockBackend) -> None:
        """Archives a page."""
        result = backend.execute(
            ["notion", "page", "update", "--id", "page-001", "--archived", "true"]
        )
        assert result.exit_code == 0
        updated = json.loads(result.stdout)
        assert updated["archived"] is True

    def test_update_not_found(self, backend: NotionMockBackend) -> None:
        """Update nonexistent page returns error."""
        result = backend.execute(
            ["notion", "page", "update", "--id", "page-999", "--title", "nope"]
        )
        assert result.exit_code == 1


class TestPageGet:
    def test_get_page(self, backend: NotionMockBackend) -> None:
        """Gets a page with full content."""
        result = backend.execute(["notion", "page", "get", "--id", "page-001"])
        assert result.exit_code == 0
        page = json.loads(result.stdout)
        assert page["id"] == "page-001"
        assert page["title"] == "Meeting Notes Q1"
        assert len(page["blocks"]) == 1

    def test_get_page_not_found(self, backend: NotionMockBackend) -> None:
        """Get nonexistent page returns error."""
        result = backend.execute(["notion", "page", "get", "--id", "page-999"])
        assert result.exit_code == 1


class TestDbQuery:
    def test_query_database(self, backend: NotionMockBackend) -> None:
        """Queries a database, returns all rows."""
        result = backend.execute(["notion", "db", "query", "--id", "db-001"])
        assert result.exit_code == 0
        rows = json.loads(result.stdout)
        assert len(rows) == 3

    def test_query_with_filter(self, backend: NotionMockBackend) -> None:
        """Filters database rows by JSON filter."""
        result = backend.execute(
            [
                "notion", "db", "query",
                "--id", "db-001",
                "--filter", '{"status": "active"}',
            ]
        )
        assert result.exit_code == 0
        rows = json.loads(result.stdout)
        assert len(rows) == 2
        for row in rows:
            assert row["status"] == "active"

    def test_query_not_found(self, backend: NotionMockBackend) -> None:
        """Query nonexistent database returns error."""
        result = backend.execute(["notion", "db", "query", "--id", "db-999"])
        assert result.exit_code == 1


class TestBlockAppend:
    def test_append_block(self, backend: NotionMockBackend) -> None:
        """Appends a block to a page."""
        result = backend.execute(
            [
                "notion", "block", "append",
                "--page-id", "page-001",
                "--content", "New paragraph",
                "--type", "paragraph",
            ]
        )
        assert result.exit_code == 0
        block = json.loads(result.stdout)
        assert block["type"] == "paragraph"
        assert block["content"] == "New paragraph"

        # Verify state mutated
        snapshot = backend.get_state_snapshot()
        page = [p for p in snapshot["pages"] if p["id"] == "page-001"][0]
        assert len(page["blocks"]) == 2

    def test_append_block_page_not_found(self, backend: NotionMockBackend) -> None:
        """Append to nonexistent page returns error."""
        result = backend.execute(
            [
                "notion", "block", "append",
                "--page-id", "page-999",
                "--content", "Test",
            ]
        )
        assert result.exit_code == 1


class TestSearch:
    def test_search_pages(self, backend: NotionMockBackend) -> None:
        """Searches across pages by title/content."""
        result = backend.execute(["notion", "search", "--query", "meeting"])
        assert result.exit_code == 0
        results = json.loads(result.stdout)
        assert len(results) >= 1
        assert any(r["title"] == "Meeting Notes Q1" for r in results)

    def test_search_databases(self, backend: NotionMockBackend) -> None:
        """Searches databases by title."""
        result = backend.execute(["notion", "search", "--query", "wiki"])
        assert result.exit_code == 0
        results = json.loads(result.stdout)
        assert len(results) >= 1
        assert any(r["type"] == "database" for r in results)

    def test_search_with_type_filter(self, backend: NotionMockBackend) -> None:
        """Filter search results by type."""
        result = backend.execute(
            ["notion", "search", "--query", "meeting", "--type", "page"]
        )
        assert result.exit_code == 0
        results = json.loads(result.stdout)
        for r in results:
            assert r["type"] == "page"

    def test_search_excludes_archived(self, backend: NotionMockBackend) -> None:
        """Archived pages are excluded from search."""
        result = backend.execute(["notion", "search", "--query", "archived"])
        assert result.exit_code == 0
        results = json.loads(result.stdout)
        assert len(results) == 0

    def test_search_no_results(self, backend: NotionMockBackend) -> None:
        """Search with no matches returns empty list."""
        result = backend.execute(
            ["notion", "search", "--query", "nonexistent_xyz"]
        )
        assert result.exit_code == 0
        results = json.loads(result.stdout)
        assert len(results) == 0
