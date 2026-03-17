# OpenCLI Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add OpenCLI as a new semi-fictional tool in CLI-Bench with mock backend, tool adapter YAML, and 10 benchmark tasks testing information retrieval and cross-tool orchestration.

**Architecture:** Custom `OpenCLIMockBackend` extending `BaseMockBackend` (not `FictionalMockBackend` — OpenCLI uses `<site> <command>` routing, not CRUD `<resource> <action>`). State organized by site (hackernews, reddit, github, bbc, v2ex). Read-only operations with per-site filtering logic. New `info_retrieval` task category.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, pytest + pytest-asyncio

---

### Task 1: OpenCLI Mock Backend — Core routing + `hackernews top`

**Files:**
- Create: `cli_bench/mock_backends/opencli.py`
- Test: `tests/unit/test_mock_opencli.py`

**Step 1: Write failing tests for core routing and hackernews top**

Create `tests/unit/test_mock_opencli.py`:

```python
"""Tests for OpenCLIMockBackend — simulates the opencli web data retrieval tool."""

import json

import pytest

from cli_bench.mock_backends.opencli import OpenCLIMockBackend


@pytest.fixture
def backend() -> OpenCLIMockBackend:
    """Backend with seeded data for hackernews, reddit, github, bbc, v2ex."""
    return OpenCLIMockBackend(
        initial_state={
            "hackernews": {
                "top": [
                    {"rank": 1, "title": "Show HN: AI Code Review Tool", "score": 487, "author": "techdev", "comments": 142, "url": "https://example.com/ai-review"},
                    {"rank": 2, "title": "PostgreSQL 18 Released", "score": 352, "author": "pgfan", "comments": 98, "url": "https://example.com/pg18"},
                    {"rank": 3, "title": "Rust in the Linux Kernel", "score": 291, "author": "kernelhacker", "comments": 67, "url": "https://example.com/rust-kernel"},
                    {"rank": 4, "title": "WebAssembly 3.0 Spec", "score": 203, "author": "wasmdev", "comments": 45, "url": "https://example.com/wasm3"},
                    {"rank": 5, "title": "Open Source LLM Benchmark", "score": 189, "author": "mlresearch", "comments": 76, "url": "https://example.com/llm-bench"},
                ],
            },
            "reddit": {
                "hot": [
                    {"rank": 1, "title": "Best practices for microservices", "subreddit": "r/programming", "score": 1205, "comments": 342, "author": "devops_pro", "url": "https://reddit.com/r/programming/1"},
                    {"rank": 2, "title": "My home lab setup 2026", "subreddit": "r/homelab", "score": 890, "comments": 156, "author": "homelabber", "url": "https://reddit.com/r/homelab/2"},
                    {"rank": 3, "title": "TypeScript 6.0 announced", "subreddit": "r/typescript", "score": 675, "comments": 89, "author": "ts_fan", "url": "https://reddit.com/r/typescript/3"},
                ],
            },
            "github": {
                "search": [
                    {"name": "fastapi/fastapi", "description": "FastAPI framework", "stars": 78000, "language": "Python", "url": "https://github.com/fastapi/fastapi"},
                    {"name": "astral-sh/uv", "description": "Python package manager", "stars": 52000, "language": "Rust", "url": "https://github.com/astral-sh/uv"},
                    {"name": "microsoft/typescript", "description": "TypeScript language", "stars": 101000, "language": "TypeScript", "url": "https://github.com/microsoft/typescript"},
                ],
                "trending": [
                    {"name": "new-ai-tool/agent", "description": "AI agent framework", "stars": 3200, "language": "Python", "url": "https://github.com/new-ai-tool/agent"},
                    {"name": "cool-project/db", "description": "Embedded database", "stars": 1500, "language": "Rust", "url": "https://github.com/cool-project/db"},
                ],
            },
            "bbc": {
                "news": [
                    {"rank": 1, "title": "Global Climate Summit Opens", "category": "world", "summary": "Leaders gather for climate talks", "url": "https://bbc.com/1"},
                    {"rank": 2, "title": "Tech Giants Report Earnings", "category": "business", "summary": "Q1 results exceed expectations", "url": "https://bbc.com/2"},
                    {"rank": 3, "title": "New Space Mission Announced", "category": "science", "summary": "Mars sample return mission planned", "url": "https://bbc.com/3"},
                ],
            },
            "v2ex": {
                "hot": [
                    {"rank": 1, "title": "Remote work policy changes", "node": "career", "author": "v2user1", "replies": 87},
                    {"rank": 2, "title": "Best VPS providers 2026", "node": "cloud", "author": "v2user2", "replies": 64},
                ],
            },
        }
    )


class TestCoreRouting:
    def test_unknown_binary_returns_error(self, backend: OpenCLIMockBackend) -> None:
        """Non-opencli binary returns error."""
        result = backend.execute(["curl", "https://example.com"])
        assert result.exit_code == 1
        assert "unknown command" in result.stderr

    def test_no_args_returns_error(self, backend: OpenCLIMockBackend) -> None:
        """opencli with no subcommand returns error."""
        result = backend.execute(["opencli"])
        assert result.exit_code == 1
        assert "usage" in result.stderr

    def test_unknown_site_returns_error(self, backend: OpenCLIMockBackend) -> None:
        """opencli with unknown site returns error."""
        result = backend.execute(["opencli", "fakeSite", "top"])
        assert result.exit_code == 1
        assert "unknown site" in result.stderr

    def test_unknown_command_for_site(self, backend: OpenCLIMockBackend) -> None:
        """opencli <valid_site> <bad_command> returns error."""
        result = backend.execute(["opencli", "hackernews", "nonexistent"])
        assert result.exit_code == 1
        assert "unknown command" in result.stderr


class TestListCommand:
    def test_list_returns_all_sites(self, backend: OpenCLIMockBackend) -> None:
        """opencli list returns available sites and commands."""
        result = backend.execute(["opencli", "list"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "hackernews" in data
        assert "reddit" in data
        assert "github" in data
        assert "bbc" in data
        assert "v2ex" in data


class TestHackernewsTop:
    def test_returns_all_stories(self, backend: OpenCLIMockBackend) -> None:
        """opencli hackernews top returns all seeded stories."""
        result = backend.execute(["opencli", "hackernews", "top"])
        assert result.exit_code == 0
        stories = json.loads(result.stdout)
        assert len(stories) == 5
        assert stories[0]["title"] == "Show HN: AI Code Review Tool"
        assert stories[0]["score"] == 487

    def test_limit_flag(self, backend: OpenCLIMockBackend) -> None:
        """--limit restricts number of results."""
        result = backend.execute(["opencli", "hackernews", "top", "--limit", "2"])
        assert result.exit_code == 0
        stories = json.loads(result.stdout)
        assert len(stories) == 2
        assert stories[0]["rank"] == 1
        assert stories[1]["rank"] == 2

    def test_limit_exceeds_available(self, backend: OpenCLIMockBackend) -> None:
        """--limit larger than data returns all available."""
        result = backend.execute(["opencli", "hackernews", "top", "--limit", "100"])
        assert result.exit_code == 0
        stories = json.loads(result.stdout)
        assert len(stories) == 5
```

**Step 2: Run tests to verify they fail**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -m pytest tests/unit/test_mock_opencli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli_bench.mock_backends.opencli'`

**Step 3: Write minimal implementation for core routing + hackernews**

Create `cli_bench/mock_backends/opencli.py`:

```python
"""OpenCLI mock backend — simulates the opencli web data retrieval tool.

Routes commands in the form: opencli <site> <command> [--flags]
State organized by site → command → list of items.
Read-only operations with per-site filtering (--limit, --subreddit, --query, etc.).
"""

import json

from cli_bench.mock_backends.base import BaseMockBackend, MockResult


def _parse_args(args: list[str]) -> dict[str, str]:
    """Parse --flag value pairs into a dict. Last value wins."""
    parsed: dict[str, str] = {}
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith("--") and i + 1 < len(args) and not args[i + 1].startswith("--"):
            parsed[arg[2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    return parsed


def _apply_limit(items: list[dict], args: dict[str, str]) -> list[dict]:
    """Apply --limit to a list of items."""
    limit_str = args.get("limit")
    if limit_str is not None:
        limit = int(limit_str)
        return items[:limit]
    return items


class OpenCLIMockBackend(BaseMockBackend):
    """Stateful mock for the opencli web data retrieval tool.

    State schema:
        {
            "hackernews": {"top": [{"rank": int, "title": str, "score": int, ...}]},
            "reddit": {"hot": [...], "search": [...]},
            "github": {"search": [...], "trending": [...]},
            "bbc": {"news": [...]},
            "v2ex": {"hot": [...]},
        }
    """

    # Map of site → {command → handler_method_name}
    _SITE_COMMANDS: dict[str, list[str]] = {
        "hackernews": ["top"],
        "reddit": ["hot", "search"],
        "github": ["search", "trending"],
        "bbc": ["news"],
        "v2ex": ["hot"],
    }

    def route_command(self, command: list[str]) -> MockResult:
        """Route opencli <site> <command> [--flags] to handler."""
        if not command or command[0] != "opencli":
            return MockResult(
                stdout="",
                stderr=f"unknown command: {' '.join(command)}",
                exit_code=1,
            )

        if len(command) < 2:
            return MockResult(
                stdout="",
                stderr="usage: opencli <site> <command> [--flags] or opencli list",
                exit_code=1,
            )

        # opencli list
        if command[1] == "list":
            return self._handle_list()

        if len(command) < 3:
            return MockResult(
                stdout="",
                stderr=f"usage: opencli {command[1]} <command> [--flags]",
                exit_code=1,
            )

        site = command[1]
        subcmd = command[2]
        flags = command[3:]

        if site not in self._SITE_COMMANDS:
            return MockResult(
                stdout="",
                stderr=f"unknown site: {site}. Available: {', '.join(sorted(self._SITE_COMMANDS))}",
                exit_code=1,
            )

        valid_cmds = self._SITE_COMMANDS[site]
        if subcmd not in valid_cmds:
            return MockResult(
                stdout="",
                stderr=f"unknown command: opencli {site} {subcmd}. Available: {', '.join(valid_cmds)}",
                exit_code=1,
            )

        handler_name = f"_handle_{site}_{subcmd}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            return self._generic_site_command(site, subcmd, flags)

        return handler(flags)

    def _handle_list(self) -> MockResult:
        """Return available sites and their commands."""
        site_commands: dict[str, list[str]] = {}
        for site in sorted(self.state):
            site_commands[site] = sorted(self.state[site].keys())
        return MockResult(
            stdout=json.dumps(site_commands),
            stderr="",
            exit_code=0,
        )

    def _generic_site_command(
        self, site: str, subcmd: str, flags: list[str]
    ) -> MockResult:
        """Generic handler: return items from state[site][subcmd] with --limit."""
        site_data = self.state.get(site, {})
        items = site_data.get(subcmd, [])
        if not isinstance(items, list):
            return MockResult(stdout=json.dumps(items), stderr="", exit_code=0)

        args = _parse_args(flags)
        result_items = list(items)
        result_items = _apply_limit(result_items, args)

        return MockResult(
            stdout=json.dumps(result_items),
            stderr="",
            exit_code=0,
        )

    # --- Hackernews ---

    def _handle_hackernews_top(self, flags: list[str]) -> MockResult:
        """opencli hackernews top [--limit N]"""
        return self._generic_site_command("hackernews", "top", flags)

    # --- Reddit ---

    def _handle_reddit_hot(self, flags: list[str]) -> MockResult:
        """opencli reddit hot [--subreddit NAME] [--limit N]"""
        args = _parse_args(flags)
        items = list(self.state.get("reddit", {}).get("hot", []))

        subreddit = args.get("subreddit")
        if subreddit:
            prefix = f"r/{subreddit}" if not subreddit.startswith("r/") else subreddit
            items = [i for i in items if i.get("subreddit") == prefix]

        items = _apply_limit(items, args)
        return MockResult(stdout=json.dumps(items), stderr="", exit_code=0)

    def _handle_reddit_search(self, flags: list[str]) -> MockResult:
        """opencli reddit search --query QUERY [--subreddit NAME] [--limit N]"""
        args = _parse_args(flags)
        query = args.get("query")
        if not query:
            return MockResult(
                stdout="", stderr="--query is required", exit_code=1
            )

        items = list(self.state.get("reddit", {}).get("hot", []))

        # Search in title (case-insensitive)
        query_lower = query.lower()
        items = [i for i in items if query_lower in i.get("title", "").lower()]

        subreddit = args.get("subreddit")
        if subreddit:
            prefix = f"r/{subreddit}" if not subreddit.startswith("r/") else subreddit
            items = [i for i in items if i.get("subreddit") == prefix]

        items = _apply_limit(items, args)
        return MockResult(stdout=json.dumps(items), stderr="", exit_code=0)

    # --- GitHub ---

    def _handle_github_search(self, flags: list[str]) -> MockResult:
        """opencli github search --query QUERY [--language LANG] [--limit N]"""
        args = _parse_args(flags)
        query = args.get("query")
        if not query:
            return MockResult(
                stdout="", stderr="--query is required", exit_code=1
            )

        items = list(self.state.get("github", {}).get("search", []))

        query_lower = query.lower()
        items = [
            i for i in items
            if query_lower in i.get("name", "").lower()
            or query_lower in i.get("description", "").lower()
        ]

        language = args.get("language")
        if language:
            items = [i for i in items if i.get("language", "").lower() == language.lower()]

        items = _apply_limit(items, args)
        return MockResult(stdout=json.dumps(items), stderr="", exit_code=0)

    def _handle_github_trending(self, flags: list[str]) -> MockResult:
        """opencli github trending [--language LANG] [--since daily|weekly|monthly] [--limit N]"""
        args = _parse_args(flags)
        items = list(self.state.get("github", {}).get("trending", []))

        language = args.get("language")
        if language:
            items = [i for i in items if i.get("language", "").lower() == language.lower()]

        items = _apply_limit(items, args)
        return MockResult(stdout=json.dumps(items), stderr="", exit_code=0)

    # --- BBC ---

    def _handle_bbc_news(self, flags: list[str]) -> MockResult:
        """opencli bbc news [--category CATEGORY] [--limit N]"""
        args = _parse_args(flags)
        items = list(self.state.get("bbc", {}).get("news", []))

        category = args.get("category")
        if category:
            items = [i for i in items if i.get("category", "").lower() == category.lower()]

        items = _apply_limit(items, args)
        return MockResult(stdout=json.dumps(items), stderr="", exit_code=0)

    # --- V2EX ---

    def _handle_v2ex_hot(self, flags: list[str]) -> MockResult:
        """opencli v2ex hot [--limit N]"""
        return self._generic_site_command("v2ex", "hot", flags)
```

**Step 4: Run tests to verify they pass**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -m pytest tests/unit/test_mock_opencli.py -v`
Expected: 10 tests PASS

**Step 5: Commit**

```bash
cd /opt/klik_benchmark/CLI-Bench
git add cli_bench/mock_backends/opencli.py tests/unit/test_mock_opencli.py
git commit -m "feat: add OpenCLI mock backend with core routing and hackernews top"
```

---

### Task 2: OpenCLI Mock Backend — All site commands + full test coverage

**Files:**
- Modify: `tests/unit/test_mock_opencli.py`

**Step 1: Add failing tests for reddit, github, bbc, v2ex commands**

Append to `tests/unit/test_mock_opencli.py`:

```python
class TestRedditHot:
    def test_returns_all_posts(self, backend: OpenCLIMockBackend) -> None:
        """opencli reddit hot returns all seeded posts."""
        result = backend.execute(["opencli", "reddit", "hot"])
        assert result.exit_code == 0
        posts = json.loads(result.stdout)
        assert len(posts) == 3

    def test_subreddit_filter(self, backend: OpenCLIMockBackend) -> None:
        """--subreddit filters by subreddit."""
        result = backend.execute(["opencli", "reddit", "hot", "--subreddit", "programming"])
        assert result.exit_code == 0
        posts = json.loads(result.stdout)
        assert len(posts) == 1
        assert posts[0]["subreddit"] == "r/programming"

    def test_subreddit_filter_with_prefix(self, backend: OpenCLIMockBackend) -> None:
        """--subreddit r/homelab works with r/ prefix."""
        result = backend.execute(["opencli", "reddit", "hot", "--subreddit", "r/homelab"])
        assert result.exit_code == 0
        posts = json.loads(result.stdout)
        assert len(posts) == 1
        assert posts[0]["title"] == "My home lab setup 2026"

    def test_limit_flag(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "reddit", "hot", "--limit", "1"])
        assert result.exit_code == 0
        posts = json.loads(result.stdout)
        assert len(posts) == 1


class TestRedditSearch:
    def test_query_required(self, backend: OpenCLIMockBackend) -> None:
        """opencli reddit search without --query fails."""
        result = backend.execute(["opencli", "reddit", "search"])
        assert result.exit_code == 1
        assert "--query" in result.stderr

    def test_search_by_title(self, backend: OpenCLIMockBackend) -> None:
        """--query matches titles case-insensitively."""
        result = backend.execute(["opencli", "reddit", "search", "--query", "microservices"])
        assert result.exit_code == 0
        posts = json.loads(result.stdout)
        assert len(posts) == 1
        assert "microservices" in posts[0]["title"].lower()

    def test_search_no_match(self, backend: OpenCLIMockBackend) -> None:
        """Search with no matches returns empty list."""
        result = backend.execute(["opencli", "reddit", "search", "--query", "nonexistent_xyz"])
        assert result.exit_code == 0
        posts = json.loads(result.stdout)
        assert len(posts) == 0


class TestGithubSearch:
    def test_query_required(self, backend: OpenCLIMockBackend) -> None:
        """opencli github search without --query fails."""
        result = backend.execute(["opencli", "github", "search"])
        assert result.exit_code == 1
        assert "--query" in result.stderr

    def test_search_by_name(self, backend: OpenCLIMockBackend) -> None:
        """--query matches repo name."""
        result = backend.execute(["opencli", "github", "search", "--query", "fastapi"])
        assert result.exit_code == 0
        repos = json.loads(result.stdout)
        assert len(repos) == 1
        assert repos[0]["name"] == "fastapi/fastapi"

    def test_search_by_description(self, backend: OpenCLIMockBackend) -> None:
        """--query matches description."""
        result = backend.execute(["opencli", "github", "search", "--query", "package manager"])
        assert result.exit_code == 0
        repos = json.loads(result.stdout)
        assert len(repos) == 1
        assert repos[0]["name"] == "astral-sh/uv"

    def test_language_filter(self, backend: OpenCLIMockBackend) -> None:
        """--language filters by programming language."""
        result = backend.execute(["opencli", "github", "search", "--query", "a", "--language", "Rust"])
        assert result.exit_code == 0
        repos = json.loads(result.stdout)
        assert all(r["language"] == "Rust" for r in repos)

    def test_limit_flag(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "github", "search", "--query", "a", "--limit", "1"])
        assert result.exit_code == 0
        repos = json.loads(result.stdout)
        assert len(repos) == 1


class TestGithubTrending:
    def test_returns_all(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "github", "trending"])
        assert result.exit_code == 0
        repos = json.loads(result.stdout)
        assert len(repos) == 2

    def test_language_filter(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "github", "trending", "--language", "Python"])
        assert result.exit_code == 0
        repos = json.loads(result.stdout)
        assert len(repos) == 1
        assert repos[0]["language"] == "Python"


class TestBbcNews:
    def test_returns_all(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "bbc", "news"])
        assert result.exit_code == 0
        articles = json.loads(result.stdout)
        assert len(articles) == 3

    def test_category_filter(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "bbc", "news", "--category", "science"])
        assert result.exit_code == 0
        articles = json.loads(result.stdout)
        assert len(articles) == 1
        assert articles[0]["category"] == "science"

    def test_limit_flag(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "bbc", "news", "--limit", "1"])
        assert result.exit_code == 0
        articles = json.loads(result.stdout)
        assert len(articles) == 1


class TestV2exHot:
    def test_returns_all(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "v2ex", "hot"])
        assert result.exit_code == 0
        topics = json.loads(result.stdout)
        assert len(topics) == 2

    def test_limit_flag(self, backend: OpenCLIMockBackend) -> None:
        result = backend.execute(["opencli", "v2ex", "hot", "--limit", "1"])
        assert result.exit_code == 0
        topics = json.loads(result.stdout)
        assert len(topics) == 1


class TestStateDiff:
    def test_diff_matches_expected(self, backend: OpenCLIMockBackend) -> None:
        """State diff works for opencli backend."""
        expected = {
            "hackernews": {
                "top": [
                    {"rank": 1, "title": "Show HN: AI Code Review Tool"},
                ]
            }
        }
        diff = backend.diff(expected)
        assert diff.score > 0.0

    def test_reset_restores_initial(self, backend: OpenCLIMockBackend) -> None:
        """Reset restores state to initial_state."""
        # Mutate state (not typical for opencli, but backend supports it)
        backend.state["hackernews"]["top"] = []
        backend.reset()
        assert len(backend.state["hackernews"]["top"]) == 5
```

**Step 2: Run tests to verify they pass** (implementation already exists from Task 1)

Run: `cd /opt/klik_benchmark/CLI-Bench && python -m pytest tests/unit/test_mock_opencli.py -v`
Expected: All ~30 tests PASS

**Step 3: Commit**

```bash
cd /opt/klik_benchmark/CLI-Bench
git add tests/unit/test_mock_opencli.py
git commit -m "test: add full test coverage for OpenCLI mock backend"
```

---

### Task 3: OpenCLI Tool Adapter YAML

**Files:**
- Create: `cli_bench/tool_adapters/opencli.yaml`

**Step 1: Write the tool adapter YAML**

Create `cli_bench/tool_adapters/opencli.yaml`:

```yaml
name: opencli
description: CLI tool for querying web platforms — fetches news, posts, repositories, and articles from HackerNews, Reddit, GitHub, BBC, and V2EX
binary: opencli
auth:
  type: none
commands:
  - name: hackernews top
    description: Get top stories from Hacker News ranked by score
    args:
      - name: limit
        type: int
        required: false
        description: Maximum number of stories to return
    output_format: json
    side_effects: false
    example: "opencli hackernews top --limit 10"

  - name: reddit hot
    description: Get hot posts from Reddit, optionally filtered by subreddit
    args:
      - name: subreddit
        type: string
        required: false
        description: Subreddit name to filter (e.g. programming). Empty for frontpage
      - name: limit
        type: int
        required: false
        description: Maximum number of posts to return
    output_format: json
    side_effects: false
    example: "opencli reddit hot --subreddit programming --limit 10"

  - name: reddit search
    description: Search Reddit posts by keyword across all subreddits or within a specific one
    args:
      - name: query
        type: string
        required: true
        description: Search query to match against post titles
      - name: subreddit
        type: string
        required: false
        description: Subreddit to search within
      - name: limit
        type: int
        required: false
        description: Maximum number of results
    output_format: json
    side_effects: false
    example: "opencli reddit search --query 'machine learning' --subreddit datascience"

  - name: github search
    description: Search GitHub repositories by name or description with optional language filter
    args:
      - name: query
        type: string
        required: true
        description: Search query to match against repo names and descriptions
      - name: language
        type: string
        required: false
        description: Filter by programming language (e.g. Python, Rust, TypeScript)
      - name: limit
        type: int
        required: false
        description: Maximum number of results
    output_format: json
    side_effects: false
    example: "opencli github search --query 'web framework' --language Python --limit 5"

  - name: github trending
    description: Get trending GitHub repositories with optional language and time range filters
    args:
      - name: language
        type: string
        required: false
        description: Filter by programming language
      - name: since
        type: enum
        required: false
        description: Time range for trending calculation
        values: ["daily", "weekly", "monthly"]
      - name: limit
        type: int
        required: false
        description: Maximum number of results
    output_format: json
    side_effects: false
    example: "opencli github trending --language Python --since weekly"

  - name: bbc news
    description: Get BBC news headlines with optional category filter
    args:
      - name: category
        type: enum
        required: false
        description: News category to filter by
        values: ["world", "business", "technology", "science", "health", "entertainment"]
      - name: limit
        type: int
        required: false
        description: Maximum number of articles
    output_format: json
    side_effects: false
    example: "opencli bbc news --category technology --limit 5"

  - name: v2ex hot
    description: Get hot topics from V2EX tech community forum
    args:
      - name: limit
        type: int
        required: false
        description: Maximum number of topics
    output_format: json
    side_effects: false
    example: "opencli v2ex hot --limit 10"

  - name: list
    description: List all available sites and their commands
    args: []
    output_format: json
    side_effects: false
    example: "opencli list"
```

**Step 2: Run existing tool adapter validation tests**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -m pytest tests/unit/test_tool_adapters.py -v`

The existing `test_tool_adapters.py` needs its `ALL_TOOLS` list updated. Update it:

**Step 3: Update test_tool_adapters.py to include opencli**

In `tests/unit/test_tool_adapters.py`, add `"opencli"` to the lists:

```python
# Line 17-18: Add opencli to REAL_TOOLS (it's a real tool, not fictional)
REAL_TOOLS = ["gh", "slack", "linear", "notion", "google", "jira", "microsoft", "opencli"]
```

Wait — OpenCLI has 8 commands, which passes the `>= 5` check. Good.

**Step 4: Run tests**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -m pytest tests/unit/test_tool_adapters.py -v`
Expected: All tests pass including opencli adapter validation

**Step 5: Commit**

```bash
cd /opt/klik_benchmark/CLI-Bench
git add cli_bench/tool_adapters/opencli.yaml tests/unit/test_tool_adapters.py
git commit -m "feat: add OpenCLI tool adapter YAML with 8 commands across 5 sites"
```

---

### Task 4: Register OpenCLI in BenchmarkRunner

**Files:**
- Modify: `cli_bench/harness/benchmark.py` (lines 15-33)

**Step 1: Write failing integration test**

The existing `test_benchmark.py` uses `github` tasks. Add a test for opencli:

Append to `tests/unit/test_benchmark.py`:

```python
class TestBenchmarkWithOpenCLI:
    @pytest.mark.asyncio
    async def test_opencli_task_produces_result(self, tmp_path: Path) -> None:
        """An opencli task creates backend and runs through pipeline."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        data = {
            "id": "test-opencli-001",
            "title": "Fetch HN top stories",
            "difficulty": "easy",
            "category": "info_retrieval",
            "description": "Fetch the top stories from Hacker News.",
            "tools_provided": ["opencli"],
            "initial_state": {
                "opencli": {
                    "hackernews": {
                        "top": [
                            {"rank": 1, "title": "Test Story", "score": 100, "author": "test", "comments": 10, "url": "https://example.com"},
                        ]
                    },
                    "reddit": {"hot": []},
                    "github": {"search": [], "trending": []},
                    "bbc": {"news": []},
                    "v2ex": {"hot": []},
                }
            },
            "expected_state": {
                "opencli": {
                    "hackernews": {
                        "top": [{"rank": 1, "title": "Test Story"}]
                    }
                }
            },
            "max_turns": 3,
            "optimal_commands": 1,
            "scoring": {"outcome": 0.6, "efficiency": 0.2, "recovery": 0.2},
        }
        (tasks_dir / "test.yaml").write_text(yaml.dump(data))

        agent = DummyAgent()
        runner = BenchmarkRunner(tasks_dir=tasks_dir, agent=agent, k=1)
        report = await runner.run_all()

        assert len(report.results) == 1
        assert report.results[0].task_id == "test-opencli-001"
```

**Step 2: Run test to verify it fails**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -m pytest tests/unit/test_benchmark.py::TestBenchmarkWithOpenCLI -v`
Expected: FAIL — opencli backend not registered

**Step 3: Register OpenCLI in benchmark.py**

Add import and registry entries to `cli_bench/harness/benchmark.py`:

```python
# Add import at line 17 (after linear import):
from cli_bench.mock_backends.opencli import OpenCLIMockBackend

# Add to _BACKEND_REGISTRY (line 26):
"opencli": OpenCLIMockBackend,

# Add to _TOOL_TO_SERVICE (line 33):
"opencli": "opencli",
```

**Step 4: Run tests to verify they pass**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -m pytest tests/unit/test_benchmark.py -v`
Expected: All pass

**Step 5: Commit**

```bash
cd /opt/klik_benchmark/CLI-Bench
git add cli_bench/harness/benchmark.py tests/unit/test_benchmark.py
git commit -m "feat: register OpenCLI backend in BenchmarkRunner"
```

---

### Task 5: Easy benchmark tasks (cb-041, cb-042, cb-043)

**Files:**
- Create: `data/tasks/cb-041.yaml`, `data/tasks/cb-042.yaml`, `data/tasks/cb-043.yaml`

**Step 1: Create cb-041 — Fetch top HN stories**

```yaml
id: cb-041
title: "Fetch top 5 Hacker News stories"
difficulty: easy
category: info_retrieval
description: |
  Use the opencli tool to fetch the top 5 stories currently trending on
  Hacker News. Display the title and score for each story.
tools_provided:
  - opencli
initial_state:
  opencli:
    hackernews:
      top:
        - {rank: 1, title: "Show HN: AI-powered code review", score: 487, author: "techdev", comments: 142, url: "https://example.com/1"}
        - {rank: 2, title: "PostgreSQL 18 Released", score: 352, author: "pgfan", comments: 98, url: "https://example.com/2"}
        - {rank: 3, title: "Rust in the Linux Kernel", score: 291, author: "kernelhacker", comments: 67, url: "https://example.com/3"}
        - {rank: 4, title: "WebAssembly 3.0 Spec", score: 203, author: "wasmdev", comments: 45, url: "https://example.com/4"}
        - {rank: 5, title: "Open Source LLM Benchmark", score: 189, author: "mlresearch", comments: 76, url: "https://example.com/5"}
        - {rank: 6, title: "New CSS Container Queries", score: 145, author: "cssdev", comments: 32, url: "https://example.com/6"}
        - {rank: 7, title: "Kubernetes 2.0 Roadmap", score: 134, author: "k8sfan", comments: 55, url: "https://example.com/7"}
    reddit: {hot: [], search: []}
    github: {search: [], trending: []}
    bbc: {news: []}
    v2ex: {hot: []}
expected_state:
  opencli:
    command_history:
      - pattern: "opencli hackernews top.*--limit 5"
    output_contains:
      - "AI-powered code review"
      - "PostgreSQL 18"
      - "Rust in the Linux"
      - "WebAssembly 3.0"
      - "Open Source LLM"
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 3
optimal_commands: 1
timeout_seconds: 30
```

**Step 2: Create cb-042 — Search GitHub repos by language**

```yaml
id: cb-042
title: "Search GitHub for Python web frameworks"
difficulty: easy
category: info_retrieval
description: |
  Use the opencli tool to search GitHub for repositories related to
  "web framework" that are written in Python. List the repository name
  and star count for each result.
tools_provided:
  - opencli
initial_state:
  opencli:
    hackernews: {top: []}
    reddit: {hot: [], search: []}
    github:
      search:
        - {name: "fastapi/fastapi", description: "FastAPI web framework for Python", stars: 78000, language: "Python", url: "https://github.com/fastapi/fastapi"}
        - {name: "django/django", description: "The web framework for perfectionists", stars: 82000, language: "Python", url: "https://github.com/django/django"}
        - {name: "pallets/flask", description: "Lightweight web framework", stars: 68000, language: "Python", url: "https://github.com/pallets/flask"}
        - {name: "expressjs/express", description: "Fast web framework for Node.js", stars: 65000, language: "JavaScript", url: "https://github.com/expressjs/express"}
        - {name: "actix/actix-web", description: "Rust web framework", stars: 22000, language: "Rust", url: "https://github.com/actix/actix-web"}
      trending: []
    bbc: {news: []}
    v2ex: {hot: []}
expected_state:
  opencli:
    command_history:
      - pattern: "opencli github search.*--query.*--language Python"
    output_contains:
      - "fastapi"
      - "django"
      - "flask"
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 3
optimal_commands: 1
timeout_seconds: 30
```

**Step 3: Create cb-043 — Reddit hot posts from subreddit**

```yaml
id: cb-043
title: "Get hot posts from r/programming"
difficulty: easy
category: info_retrieval
description: |
  Use the opencli tool to fetch the hot posts from the "programming"
  subreddit on Reddit. Show the title and score for each post.
tools_provided:
  - opencli
initial_state:
  opencli:
    hackernews: {top: []}
    reddit:
      hot:
        - {rank: 1, title: "Best practices for error handling in Go", subreddit: "r/programming", score: 1205, comments: 342, author: "gopher_fan", url: "https://reddit.com/1"}
        - {rank: 2, title: "My home lab network diagram", subreddit: "r/homelab", score: 890, comments: 156, author: "homelabber", url: "https://reddit.com/2"}
        - {rank: 3, title: "TypeScript 6.0 announced", subreddit: "r/programming", score: 675, comments: 89, author: "ts_fan", url: "https://reddit.com/3"}
        - {rank: 4, title: "Show reddit: built a CLI tool", subreddit: "r/programming", score: 432, comments: 67, author: "clidev", url: "https://reddit.com/4"}
        - {rank: 5, title: "GPU prices dropping", subreddit: "r/hardware", score: 1100, comments: 234, author: "hw_watcher", url: "https://reddit.com/5"}
      search: []
    github: {search: [], trending: []}
    bbc: {news: []}
    v2ex: {hot: []}
expected_state:
  opencli:
    command_history:
      - pattern: "opencli reddit hot.*--subreddit.*programming"
    output_contains:
      - "error handling in Go"
      - "TypeScript 6.0"
      - "CLI tool"
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 3
optimal_commands: 1
timeout_seconds: 30
```

**Step 4: Validate all 3 task YAMLs load**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -c "from cli_bench.models.task import BenchTask; [BenchTask.from_yaml(p) for p in sorted(__import__('pathlib').Path('data/tasks').glob('cb-04[1-3].yaml'))] and print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
cd /opt/klik_benchmark/CLI-Bench
git add data/tasks/cb-041.yaml data/tasks/cb-042.yaml data/tasks/cb-043.yaml
git commit -m "feat: add 3 easy OpenCLI benchmark tasks (cb-041 to cb-043)"
```

---

### Task 6: Medium benchmark tasks (cb-044, cb-045, cb-046)

**Files:**
- Create: `data/tasks/cb-044.yaml`, `data/tasks/cb-045.yaml`, `data/tasks/cb-046.yaml`

**Step 1: Create cb-044 — Cross-site comparison (opencli only)**

```yaml
id: cb-044
title: "Compare trending topics across Hacker News and Reddit"
difficulty: medium
category: info_retrieval
description: |
  Use the opencli tool to gather trending content from both Hacker News
  and Reddit's r/programming subreddit. Identify topics that appear on
  both platforms (overlapping themes). Report the overlapping topics
  and their respective scores on each platform.
tools_provided:
  - opencli
initial_state:
  opencli:
    hackernews:
      top:
        - {rank: 1, title: "Rust memory safety improvements in 2026", score: 520, author: "rustfan", comments: 180, url: "https://hn.com/1"}
        - {rank: 2, title: "GPT-5 leaked benchmark results", score: 480, author: "aitracker", comments: 310, url: "https://hn.com/2"}
        - {rank: 3, title: "PostgreSQL vs MySQL performance in 2026", score: 290, author: "dbexpert", comments: 95, url: "https://hn.com/3"}
        - {rank: 4, title: "WebAssembly for server-side rendering", score: 210, author: "wasmdev", comments: 42, url: "https://hn.com/4"}
    reddit:
      hot:
        - {rank: 1, title: "GPT-5 benchmarks are out - discussion", subreddit: "r/programming", score: 2100, comments: 890, author: "ai_enthusiast", url: "https://reddit.com/1"}
        - {rank: 2, title: "Rust 2026 edition preview", subreddit: "r/programming", score: 1500, comments: 430, author: "rust_dev", url: "https://reddit.com/2"}
        - {rank: 3, title: "New React Server Components update", subreddit: "r/programming", score: 980, comments: 210, author: "react_dev", url: "https://reddit.com/3"}
        - {rank: 4, title: "PostgreSQL 18 performance deep dive", subreddit: "r/programming", score: 870, comments: 156, author: "pg_admin", url: "https://reddit.com/4"}
      search: []
    github: {search: [], trending: []}
    bbc: {news: []}
    v2ex: {hot: []}
expected_state:
  opencli:
    command_history:
      - pattern: "opencli hackernews top"
      - pattern: "opencli reddit hot"
    output_contains:
      - "Rust"
      - "GPT-5"
      - "PostgreSQL"
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 5
optimal_commands: 2
timeout_seconds: 60
```

**Step 2: Create cb-045 — Fetch BBC news → share in Slack (cross-tool)**

```yaml
id: cb-045
title: "Fetch tech news from BBC and share summary in Slack"
difficulty: medium
category: composite
description: |
  Use opencli to fetch the latest technology news from BBC. Then send a
  message to the #tech-news Slack channel summarizing the top 3 headlines
  with their links.
tools_provided:
  - opencli
  - slack
initial_state:
  opencli:
    hackernews: {top: []}
    reddit: {hot: [], search: []}
    github: {search: [], trending: []}
    bbc:
      news:
        - {rank: 1, title: "AI Regulation Bill Passes Senate", category: "technology", summary: "New legislation sets guidelines for AI development", url: "https://bbc.com/tech/1"}
        - {rank: 2, title: "Quantum Computing Milestone Achieved", category: "technology", summary: "1000-qubit processor demonstrated", url: "https://bbc.com/tech/2"}
        - {rank: 3, title: "Cybersecurity Threats Rising in 2026", category: "technology", summary: "Report warns of increased state-sponsored attacks", url: "https://bbc.com/tech/3"}
        - {rank: 4, title: "Global Markets Rally on Trade Deal", category: "business", summary: "Markets respond to new trade agreement", url: "https://bbc.com/biz/1"}
    v2ex: {hot: []}
  slack:
    channels:
      - {id: "C100", name: "tech-news", messages: []}
      - {id: "C101", name: "general", messages: []}
    users: []
expected_state:
  opencli:
    command_history:
      - pattern: "opencli bbc news.*--category.*technology"
  slack:
    channels:
      - id: "C100"
        name: "tech-news"
        messages:
          - contains: ["AI Regulation", "Quantum Computing", "Cybersecurity"]
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 5
optimal_commands: 2
timeout_seconds: 60
```

**Step 3: Create cb-046 — Search GitHub → create issue (cross-tool)**

```yaml
id: cb-046
title: "Search GitHub repos and create evaluation issue"
difficulty: medium
category: composite
description: |
  Use opencli to search GitHub for repositories related to "vector database"
  written in Rust. Then create an issue in the "acme-corp/infra" repo to
  evaluate the top result, including the repo name, star count, and URL
  in the issue body.
tools_provided:
  - opencli
  - gh
initial_state:
  opencli:
    hackernews: {top: []}
    reddit: {hot: [], search: []}
    github:
      search:
        - {name: "qdrant/qdrant", description: "High-performance vector database", stars: 21000, language: "Rust", url: "https://github.com/qdrant/qdrant"}
        - {name: "meilisearch/meilisearch", description: "Lightning fast search engine", stars: 48000, language: "Rust", url: "https://github.com/meilisearch/meilisearch"}
        - {name: "chroma-core/chroma", description: "AI-native embedding database", stars: 16000, language: "Python", url: "https://github.com/chroma-core/chroma"}
        - {name: "lancedb/lance", description: "Columnar vector database", stars: 4200, language: "Rust", url: "https://github.com/lancedb/lance"}
      trending: []
    bbc: {news: []}
    v2ex: {hot: []}
  gh:
    repos:
      acme-corp/infra:
        issues: []
        pulls: []
        commits: []
expected_state:
  opencli:
    command_history:
      - pattern: "opencli github search.*--query.*vector.*--language.*Rust"
  gh:
    repos:
      acme-corp/infra:
        issues:
          - title_contains: "qdrant"
            state: open
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 5
optimal_commands: 2
timeout_seconds: 60
```

**Step 4: Validate all 3 task YAMLs load**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -c "from cli_bench.models.task import BenchTask; [BenchTask.from_yaml(p) for p in sorted(__import__('pathlib').Path('data/tasks').glob('cb-04[4-6].yaml'))] and print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
cd /opt/klik_benchmark/CLI-Bench
git add data/tasks/cb-044.yaml data/tasks/cb-045.yaml data/tasks/cb-046.yaml
git commit -m "feat: add 3 medium OpenCLI benchmark tasks (cb-044 to cb-046)"
```

---

### Task 7: Hard benchmark tasks (cb-047, cb-048, cb-049, cb-050)

**Files:**
- Create: `data/tasks/cb-047.yaml` through `data/tasks/cb-050.yaml`

**Step 1: Create cb-047 — Research workflow: HN + Reddit → Slack → Linear**

```yaml
id: cb-047
title: "Research trending AI topics and create tracking workflow"
difficulty: hard
category: composite
description: |
  Research current AI trends across multiple platforms, then set up a
  tracking workflow:
  1. Fetch top stories from Hacker News and filter for AI-related content
  2. Search Reddit r/machinelearning for related discussions
  3. Post a summary of findings to the #ai-research Slack channel
  4. Create a Linear issue to track follow-up research on the most
     discussed topic
tools_provided:
  - opencli
  - slack
  - linear
initial_state:
  opencli:
    hackernews:
      top:
        - {rank: 1, title: "GPT-5 architecture leaked by insider", score: 890, author: "ai_insider", comments: 456, url: "https://hn.com/1"}
        - {rank: 2, title: "New open-source speech model beats Whisper", score: 670, author: "ml_dev", comments: 234, url: "https://hn.com/2"}
        - {rank: 3, title: "PostgreSQL 18 release notes", score: 520, author: "pgfan", comments: 89, url: "https://hn.com/3"}
        - {rank: 4, title: "Anthropic publishes AI safety research paper", score: 480, author: "safety_first", comments: 312, url: "https://hn.com/4"}
        - {rank: 5, title: "New JavaScript runtime benchmarks", score: 340, author: "jsdev", comments: 67, url: "https://hn.com/5"}
    reddit:
      hot:
        - {rank: 1, title: "GPT-5 discussion thread - what we know so far", subreddit: "r/machinelearning", score: 3400, comments: 1200, author: "ml_researcher", url: "https://reddit.com/ml/1"}
        - {rank: 2, title: "Open source speech recognition comparison 2026", subreddit: "r/machinelearning", score: 1800, comments: 560, author: "speech_dev", url: "https://reddit.com/ml/2"}
        - {rank: 3, title: "AI safety alignment approaches - survey", subreddit: "r/machinelearning", score: 1200, comments: 340, author: "alignment_fan", url: "https://reddit.com/ml/3"}
      search: []
    github: {search: [], trending: []}
    bbc: {news: []}
    v2ex: {hot: []}
  slack:
    channels:
      - {id: "C200", name: "ai-research", messages: []}
    users: []
  linear:
    teams:
      - id: "TEAM-1"
        name: "Research"
        issues: []
expected_state:
  opencli:
    command_history:
      - pattern: "opencli hackernews top"
      - pattern: "opencli reddit"
  slack:
    channels:
      - id: "C200"
        name: "ai-research"
        messages:
          - contains: ["GPT-5"]
  linear:
    teams:
      - id: "TEAM-1"
        issues:
          - title_contains: "research"
            state: open
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 10
optimal_commands: 4
timeout_seconds: 90
```

**Step 2: Create cb-048 — Trending repos → filter → create issues**

```yaml
id: cb-048
title: "Find trending Rust repos and create evaluation issues"
difficulty: hard
category: composite
description: |
  Use opencli to find trending GitHub repositories written in Rust. For
  each repository with more than 2000 stars, create a separate issue in
  "acme-corp/evaluations" with the repo name, description, and star count.
  At least 2 issues should be created.
tools_provided:
  - opencli
  - gh
initial_state:
  opencli:
    hackernews: {top: []}
    reddit: {hot: [], search: []}
    github:
      search: []
      trending:
        - {name: "oxide-computer/hubris", description: "Firmware for Oxide rack servers", stars: 3100, language: "Rust", url: "https://github.com/oxide-computer/hubris"}
        - {name: "paradigm-ai/srt", description: "Streaming runtime for AI inference", stars: 5200, language: "Rust", url: "https://github.com/paradigm-ai/srt"}
        - {name: "user123/tiny-project", description: "Small utility tool", stars: 150, language: "Rust", url: "https://github.com/user123/tiny-project"}
        - {name: "quantum-db/quasar", description: "Distributed vector store", stars: 2800, language: "Rust", url: "https://github.com/quantum-db/quasar"}
        - {name: "webtools/mini-css", description: "CSS minifier", stars: 800, language: "Rust", url: "https://github.com/webtools/mini-css"}
    bbc: {news: []}
    v2ex: {hot: []}
  gh:
    repos:
      acme-corp/evaluations:
        issues: []
        pulls: []
        commits: []
expected_state:
  opencli:
    command_history:
      - pattern: "opencli github trending"
  gh:
    repos:
      acme-corp/evaluations:
        issues:
          - title_contains: "hubris"
          - title_contains: "srt"
          - title_contains: "quasar"
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 10
optimal_commands: 4
timeout_seconds: 90
```

**Step 3: Create cb-049 — Multi-site aggregation**

```yaml
id: cb-049
title: "Aggregate tech news across 3 platforms"
difficulty: hard
category: info_retrieval
description: |
  Gather technology-related content from three different platforms using
  opencli: Hacker News top stories, BBC technology news, and V2EX hot
  topics. Compile a unified list of the top item from each platform,
  reporting the platform name, title, and engagement metric (score/replies).
  The agent should fetch from all three sources and present a combined view.
tools_provided:
  - opencli
initial_state:
  opencli:
    hackernews:
      top:
        - {rank: 1, title: "Breakthrough in quantum error correction", score: 720, author: "qc_researcher", comments: 189, url: "https://hn.com/1"}
        - {rank: 2, title: "New Rust async runtime", score: 450, author: "async_dev", comments: 76, url: "https://hn.com/2"}
    reddit: {hot: [], search: []}
    github: {search: [], trending: []}
    bbc:
      news:
        - {rank: 1, title: "EU passes comprehensive AI Act", category: "technology", summary: "Regulation covers all AI systems", url: "https://bbc.com/1"}
        - {rank: 2, title: "Tech layoffs slow as hiring rebounds", category: "technology", summary: "Q1 hiring up 23%", url: "https://bbc.com/2"}
    v2ex:
      hot:
        - {rank: 1, title: "Remote work policies in China tech 2026", node: "career", author: "v2_user", replies: 156}
        - {rank: 2, title: "Best cloud GPU providers comparison", node: "cloud", author: "gpu_user", replies: 98}
expected_state:
  opencli:
    command_history:
      - pattern: "opencli hackernews top"
      - pattern: "opencli bbc news"
      - pattern: "opencli v2ex hot"
    output_contains:
      - "quantum error correction"
      - "AI Act"
      - "Remote work"
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 8
optimal_commands: 3
timeout_seconds: 90
```

**Step 4: Create cb-050 — Full pipeline: discover → cross-reference → notify → track**

```yaml
id: cb-050
title: "Full research pipeline: discover trends, notify team, track follow-ups"
difficulty: hard
category: composite
description: |
  Execute a complete research and notification pipeline:
  1. Search GitHub for repositories related to "agent framework" in Python
  2. Check Hacker News for any discussions about the top result
  3. Post the findings to #engineering Slack channel with repo details
  4. Create an issue in "acme-corp/research" to evaluate the top
     repository, including its name, stars, and description
  The workflow requires combining information across platforms and tools.
tools_provided:
  - opencli
  - slack
  - gh
initial_state:
  opencli:
    hackernews:
      top:
        - {rank: 1, title: "CrewAI vs AutoGen: agent framework comparison", score: 650, author: "agent_dev", comments: 234, url: "https://hn.com/1"}
        - {rank: 2, title: "PostgreSQL replication deep dive", score: 420, author: "dba", comments: 67, url: "https://hn.com/2"}
        - {rank: 3, title: "LangGraph for production agent workflows", score: 380, author: "langchain_user", comments: 145, url: "https://hn.com/3"}
    reddit: {hot: [], search: []}
    github:
      search:
        - {name: "crewai/crewai", description: "Framework for orchestrating AI agents", stars: 24000, language: "Python", url: "https://github.com/crewai/crewai"}
        - {name: "microsoft/autogen", description: "Multi-agent conversation framework", stars: 35000, language: "Python", url: "https://github.com/microsoft/autogen"}
        - {name: "langchain-ai/langgraph", description: "Build stateful agent workflows", stars: 8500, language: "Python", url: "https://github.com/langchain-ai/langgraph"}
      trending: []
    bbc: {news: []}
    v2ex: {hot: []}
  slack:
    channels:
      - {id: "C300", name: "engineering", messages: []}
    users: []
  gh:
    repos:
      acme-corp/research:
        issues: []
        pulls: []
        commits: []
expected_state:
  opencli:
    command_history:
      - pattern: "opencli github search.*--query.*agent.*--language.*Python"
      - pattern: "opencli hackernews top"
  slack:
    channels:
      - id: "C300"
        name: "engineering"
        messages:
          - contains: ["autogen"]
  gh:
    repos:
      acme-corp/research:
        issues:
          - title_contains: "autogen"
            state: open
scoring:
  outcome: 0.6
  efficiency: 0.2
  recovery: 0.2
max_turns: 12
optimal_commands: 4
timeout_seconds: 120
```

**Step 5: Validate all 4 task YAMLs load**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -c "from cli_bench.models.task import BenchTask; [BenchTask.from_yaml(p) for p in sorted(__import__('pathlib').Path('data/tasks').glob('cb-04[7-9].yaml')) or sorted(__import__('pathlib').Path('data/tasks').glob('cb-050.yaml'))] and print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
cd /opt/klik_benchmark/CLI-Bench
git add data/tasks/cb-047.yaml data/tasks/cb-048.yaml data/tasks/cb-049.yaml data/tasks/cb-050.yaml
git commit -m "feat: add 4 hard OpenCLI benchmark tasks (cb-047 to cb-050)"
```

---

### Task 8: Full integration validation

**Files:** None (verification only)

**Step 1: Run complete test suite**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -m pytest tests/ -v --tb=short`
Expected: All existing + new tests pass

**Step 2: Validate all 50 task YAMLs load**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -c "
from pathlib import Path
from cli_bench.models.task import BenchTask
tasks = [BenchTask.from_yaml(p) for p in sorted(Path('data/tasks').glob('*.yaml'))]
print(f'Loaded {len(tasks)} tasks')
categories = set(t.category for t in tasks)
difficulties = {}
for t in tasks:
    difficulties.setdefault(t.difficulty, 0)
    difficulties[t.difficulty] += 1
print(f'Categories: {sorted(categories)}')
print(f'Difficulties: {difficulties}')
opencli_tasks = [t for t in tasks if 'opencli' in t.tools_provided]
print(f'OpenCLI tasks: {len(opencli_tasks)}')
"`

Expected output:
```
Loaded 50 tasks
Categories: ['communication', 'composite', 'custom_cli', 'data_ops', 'devops', 'info_retrieval', 'project_mgmt']
Difficulties: {'easy': 23, 'medium': 13, 'hard': 14}
OpenCLI tasks: 10
```

**Step 3: Validate tool adapter loads**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -c "
from pathlib import Path
from cli_bench.models.tool_adapter import ToolAdapter
a = ToolAdapter.from_yaml(Path('cli_bench/tool_adapters/opencli.yaml'))
print(f'{a.name}: {len(a.commands)} commands, binary={a.binary}')
for c in a.commands:
    print(f'  {c.name} ({len(c.args)} args, {c.output_format})')
"`

Expected:
```
opencli: 8 commands, binary=opencli
  hackernews top (1 args, json)
  reddit hot (2 args, json)
  reddit search (3 args, json)
  github search (3 args, json)
  github trending (3 args, json)
  bbc news (2 args, json)
  v2ex hot (1 args, json)
  list (0 args, json)
```

**Step 4: Run DummyAgent on one new task to verify pipeline**

Run: `cd /opt/klik_benchmark/CLI-Bench && python -c "
import asyncio
from pathlib import Path
from cli_bench.agents.dummy import DummyAgent
from cli_bench.harness.benchmark import BenchmarkRunner
runner = BenchmarkRunner(tasks_dir=Path('data/tasks'), agent=DummyAgent(), k=1)
result = asyncio.run(runner.run_single('cb-041'))
print(f'Task: {result.task_id}, mean_score: {result.mean_score:.3f}, pass_k: {result.pass_k:.1f}')
"`

Expected: Runs without error (score will be low since DummyAgent does nothing, which is expected)

**Step 5: Final commit (if any fixups needed)**

```bash
cd /opt/klik_benchmark/CLI-Bench
git add -A && git status
# Only commit if there are changes
```
