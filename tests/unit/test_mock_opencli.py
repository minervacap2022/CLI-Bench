"""Tests for OpenCLIMockBackend -- simulates the opencli CLI tool."""

import json

import pytest

from cli_bench.mock_backends.opencli import OpenCLIMockBackend


@pytest.fixture
def backend() -> OpenCLIMockBackend:
    """Backend seeded with data for all sites."""
    return OpenCLIMockBackend(
        initial_state={
            "hackernews": {
                "top": [
                    {"rank": 1, "title": "Show HN: A new database", "score": 312, "author": "pg", "comments": 89, "url": "https://example.com/db"},
                    {"rank": 2, "title": "Rust is awesome", "score": 245, "author": "dtolnay", "comments": 134, "url": "https://example.com/rust"},
                    {"rank": 3, "title": "Why I left Google", "score": 198, "author": "exgoogler", "comments": 256, "url": "https://example.com/google"},
                ],
            },
            "reddit": {
                "hot": [
                    {"rank": 1, "title": "Python 4.0 released", "subreddit": "programming", "score": 5432, "comments": 876, "author": "guido", "url": "https://reddit.com/r/programming/1"},
                    {"rank": 2, "title": "My cat learned to code", "subreddit": "funny", "score": 12345, "comments": 432, "author": "catperson", "url": "https://reddit.com/r/funny/1"},
                    {"rank": 3, "title": "Best practices for Go", "subreddit": "programming", "score": 2345, "comments": 123, "author": "robpike", "url": "https://reddit.com/r/programming/2"},
                ],
            },
            "github": {
                "search": [
                    {"name": "tensorflow", "description": "An open source machine learning framework", "stars": 178000, "language": "Python", "url": "https://github.com/tensorflow/tensorflow"},
                    {"name": "react", "description": "A JavaScript library for building user interfaces", "stars": 210000, "language": "JavaScript", "url": "https://github.com/facebook/react"},
                    {"name": "rustlings", "description": "Small exercises to get you used to reading and writing Rust code", "stars": 45000, "language": "Rust", "url": "https://github.com/rust-lang/rustlings"},
                ],
                "trending": [
                    {"name": "awesome-llm", "description": "Curated list of LLM resources", "stars": 5000, "language": "Python", "url": "https://github.com/example/awesome-llm"},
                    {"name": "fast-api-template", "description": "Production-ready FastAPI template", "stars": 3200, "language": "Python", "url": "https://github.com/example/fast-api-template"},
                    {"name": "go-microservices", "description": "Microservices patterns in Go", "stars": 2100, "language": "Go", "url": "https://github.com/example/go-microservices"},
                ],
            },
            "bbc": {
                "news": [
                    {"rank": 1, "title": "Global climate summit begins", "category": "world", "summary": "Leaders gather for annual summit", "url": "https://bbc.com/news/1"},
                    {"rank": 2, "title": "New AI breakthrough announced", "category": "technology", "summary": "Researchers achieve milestone", "url": "https://bbc.com/news/2"},
                    {"rank": 3, "title": "Economy shows signs of recovery", "category": "business", "summary": "Markets rally on positive data", "url": "https://bbc.com/news/3"},
                ],
            },
            "v2ex": {
                "hot": [
                    {"rank": 1, "title": "MacBook Pro M4 review", "node": "apple", "author": "techfan", "replies": 89},
                    {"rank": 2, "title": "Best VPN for developers", "node": "programmer", "author": "devops_guy", "replies": 156},
                ],
            },
        }
    )


class TestCoreRouting:
    def test_unknown_binary(self, backend: OpenCLIMockBackend) -> None:
        """Non-opencli binary returns error."""
        result = backend.execute(["notopencli", "hackernews", "top"])
        assert result.exit_code == 1
        assert result.stderr != ""

    def test_no_args(self, backend: OpenCLIMockBackend) -> None:
        """opencli with no args returns error."""
        result = backend.execute(["opencli"])
        assert result.exit_code == 1
        assert result.stderr != ""

    def test_unknown_site(self, backend: OpenCLIMockBackend) -> None:
        """Unknown site returns error."""
        result = backend.execute(["opencli", "twitter", "trending"])
        assert result.exit_code == 1
        assert result.stderr != ""

    def test_unknown_command_for_site(self, backend: OpenCLIMockBackend) -> None:
        """Known site but unknown command returns error."""
        result = backend.execute(["opencli", "hackernews", "search"])
        assert result.exit_code == 1
        assert result.stderr != ""


class TestListCommand:
    def test_list_returns_all_sites(self, backend: OpenCLIMockBackend) -> None:
        """opencli list returns all available sites and their commands."""
        result = backend.execute(["opencli", "list"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "hackernews" in data
        assert "reddit" in data
        assert "github" in data
        assert "bbc" in data
        assert "v2ex" in data


class TestHackernewsTop:
    def test_returns_all(self, backend: OpenCLIMockBackend) -> None:
        """Returns all top stories without limit."""
        result = backend.execute(["opencli", "hackernews", "top"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 3
        assert items[0]["title"] == "Show HN: A new database"

    def test_with_limit(self, backend: OpenCLIMockBackend) -> None:
        """--limit restricts the number of results."""
        result = backend.execute(["opencli", "hackernews", "top", "--limit", "2"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 2

    def test_limit_exceeds_available(self, backend: OpenCLIMockBackend) -> None:
        """--limit greater than available returns all items."""
        result = backend.execute(["opencli", "hackernews", "top", "--limit", "100"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 3


class TestRedditHot:
    def test_returns_all(self, backend: OpenCLIMockBackend) -> None:
        """Returns all hot posts without filter."""
        result = backend.execute(["opencli", "reddit", "hot"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 3

    def test_subreddit_filter(self, backend: OpenCLIMockBackend) -> None:
        """--subreddit filters to matching subreddit."""
        result = backend.execute(["opencli", "reddit", "hot", "--subreddit", "programming"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 2
        assert all(i["subreddit"] == "programming" for i in items)

    def test_subreddit_with_r_prefix(self, backend: OpenCLIMockBackend) -> None:
        """--subreddit handles r/ prefix form."""
        result = backend.execute(["opencli", "reddit", "hot", "--subreddit", "r/programming"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 2
        assert all(i["subreddit"] == "programming" for i in items)

    def test_with_limit(self, backend: OpenCLIMockBackend) -> None:
        """--limit restricts the number of results."""
        result = backend.execute(["opencli", "reddit", "hot", "--limit", "1"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1


class TestRedditSearch:
    def test_query_required(self, backend: OpenCLIMockBackend) -> None:
        """reddit search without --query returns error."""
        result = backend.execute(["opencli", "reddit", "search"])
        assert result.exit_code == 1
        assert result.stderr != ""

    def test_search_by_title(self, backend: OpenCLIMockBackend) -> None:
        """Searches reddit hot items by title (case-insensitive)."""
        result = backend.execute(["opencli", "reddit", "search", "--query", "python"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1
        assert items[0]["title"] == "Python 4.0 released"

    def test_no_match_returns_empty(self, backend: OpenCLIMockBackend) -> None:
        """Search with no matches returns empty list."""
        result = backend.execute(["opencli", "reddit", "search", "--query", "nonexistent_xyz"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 0


class TestGithubSearch:
    def test_query_required(self, backend: OpenCLIMockBackend) -> None:
        """github search without --query returns error."""
        result = backend.execute(["opencli", "github", "search"])
        assert result.exit_code == 1
        assert result.stderr != ""

    def test_search_by_name(self, backend: OpenCLIMockBackend) -> None:
        """Matches query against repo name."""
        result = backend.execute(["opencli", "github", "search", "--query", "react"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1
        assert items[0]["name"] == "react"

    def test_search_by_description(self, backend: OpenCLIMockBackend) -> None:
        """Matches query against repo description."""
        result = backend.execute(["opencli", "github", "search", "--query", "machine learning"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1
        assert items[0]["name"] == "tensorflow"

    def test_language_filter(self, backend: OpenCLIMockBackend) -> None:
        """--language filters search results."""
        result = backend.execute(["opencli", "github", "search", "--query", "rust", "--language", "Rust"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1
        assert items[0]["name"] == "rustlings"

    def test_with_limit(self, backend: OpenCLIMockBackend) -> None:
        """--limit restricts search results."""
        result = backend.execute(["opencli", "github", "search", "--query", "e", "--limit", "1"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1


class TestGithubTrending:
    def test_returns_all(self, backend: OpenCLIMockBackend) -> None:
        """Returns all trending repos."""
        result = backend.execute(["opencli", "github", "trending"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 3

    def test_language_filter(self, backend: OpenCLIMockBackend) -> None:
        """--language filters trending repos."""
        result = backend.execute(["opencli", "github", "trending", "--language", "Go"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1
        assert items[0]["name"] == "go-microservices"


class TestBbcNews:
    def test_returns_all(self, backend: OpenCLIMockBackend) -> None:
        """Returns all news items."""
        result = backend.execute(["opencli", "bbc", "news"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 3

    def test_category_filter(self, backend: OpenCLIMockBackend) -> None:
        """--category filters news by category."""
        result = backend.execute(["opencli", "bbc", "news", "--category", "technology"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1
        assert items[0]["title"] == "New AI breakthrough announced"

    def test_with_limit(self, backend: OpenCLIMockBackend) -> None:
        """--limit restricts news results."""
        result = backend.execute(["opencli", "bbc", "news", "--limit", "2"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 2


class TestV2exHot:
    def test_returns_all(self, backend: OpenCLIMockBackend) -> None:
        """Returns all hot topics."""
        result = backend.execute(["opencli", "v2ex", "hot"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 2

    def test_with_limit(self, backend: OpenCLIMockBackend) -> None:
        """--limit restricts results."""
        result = backend.execute(["opencli", "v2ex", "hot", "--limit", "1"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1


class TestStateDiff:
    def test_diff_matches_expected_state(self, backend: OpenCLIMockBackend) -> None:
        """diff() returns matches=True when state matches expected."""
        expected = backend.get_state_snapshot()
        diff = backend.diff(expected)
        assert diff.matches is True
        assert diff.score == 1.0

    def test_reset_restores_initial(self, backend: OpenCLIMockBackend) -> None:
        """reset() restores state to initial snapshot."""
        initial = backend.get_state_snapshot()
        # Mutate state by executing a command (state is read-only for opencli, but reset should still work)
        backend.execute(["opencli", "hackernews", "top"])
        backend.reset()
        restored = backend.get_state_snapshot()
        assert restored == initial
