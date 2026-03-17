"""OpenCLI mock backend -- simulates the opencli CLI tool.

Handles site-based commands: hackernews, reddit, github, bbc, v2ex, and list.
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


def _apply_limit(items: list[dict], parsed: dict[str, list[str]]) -> list[dict]:
    """Apply --limit flag to restrict the number of results."""
    limit_str = _get_flag(parsed, "--limit")
    if limit_str is not None:
        limit = int(limit_str)
        return items[:limit]
    return items


class OpenCLIMockBackend(BaseMockBackend):
    """Stateful mock for the OpenCLI tool.

    State schema:
        {
            "hackernews": {"top": [{"rank": int, "title": str, "score": int, "author": str, "comments": int, "url": str}]},
            "reddit": {"hot": [{"rank": int, "title": str, "subreddit": str, "score": int, "comments": int, "author": str, "url": str}]},
            "github": {
                "search": [{"name": str, "description": str, "stars": int, "language": str, "url": str}],
                "trending": [{"name": str, "description": str, "stars": int, "language": str, "url": str}],
            },
            "bbc": {"news": [{"rank": int, "title": str, "category": str, "summary": str, "url": str}]},
            "v2ex": {"hot": [{"rank": int, "title": str, "node": str, "author": str, "replies": int}]},
        }
    """

    def route_command(self, command: list[str]) -> MockResult:
        """Route an opencli command to the appropriate handler."""
        if len(command) < 2 or command[0] != "opencli":
            return MockResult(
                stdout="",
                stderr=f"unknown command: {' '.join(command)}",
                exit_code=1,
            )

        site = command[1]

        if site == "list":
            return self._handle_list()

        if site not in self.state:
            return MockResult(
                stdout="",
                stderr=f"unknown site: {site}",
                exit_code=1,
            )

        if len(command) < 3:
            return MockResult(
                stdout="",
                stderr=f"missing command for site: {site}",
                exit_code=1,
            )

        cmd = command[2]
        remaining = command[3:]

        handler_name = f"_handle_{site}_{cmd}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            return MockResult(
                stdout="",
                stderr=f"unknown command for {site}: {cmd}",
                exit_code=1,
            )

        return handler(remaining)

    def _handle_list(self) -> MockResult:
        """Return all available sites and their commands."""
        sites: dict[str, list[str]] = {}
        for site_name, site_data in self.state.items():
            sites[site_name] = list(site_data.keys())
        return MockResult(
            stdout=json.dumps(sites),
            stderr="",
            exit_code=0,
        )

    def _handle_hackernews_top(self, args: list[str]) -> MockResult:
        """Handle opencli hackernews top [--limit N]."""
        parsed = _parse_args(args)
        items = list(self.state["hackernews"]["top"])
        items = _apply_limit(items, parsed)
        return MockResult(
            stdout=json.dumps(items),
            stderr="",
            exit_code=0,
        )

    def _handle_reddit_hot(self, args: list[str]) -> MockResult:
        """Handle opencli reddit hot [--subreddit X] [--limit N]."""
        parsed = _parse_args(args)
        items = list(self.state["reddit"]["hot"])

        subreddit = _get_flag(parsed, "--subreddit")
        if subreddit is not None:
            if subreddit.startswith("r/"):
                subreddit = subreddit[2:]
            items = [i for i in items if i["subreddit"] == subreddit]

        items = _apply_limit(items, parsed)
        return MockResult(
            stdout=json.dumps(items),
            stderr="",
            exit_code=0,
        )

    def _handle_reddit_search(self, args: list[str]) -> MockResult:
        """Handle opencli reddit search --query Q [--subreddit X] [--limit N]."""
        parsed = _parse_args(args)
        query = _get_flag(parsed, "--query")
        if not query:
            return MockResult(
                stdout="",
                stderr="--query is required",
                exit_code=1,
            )

        items = list(self.state["reddit"]["hot"])

        subreddit = _get_flag(parsed, "--subreddit")
        if subreddit is not None:
            if subreddit.startswith("r/"):
                subreddit = subreddit[2:]
            items = [i for i in items if i["subreddit"] == subreddit]

        query_lower = query.lower()
        items = [i for i in items if query_lower in i["title"].lower()]

        items = _apply_limit(items, parsed)
        return MockResult(
            stdout=json.dumps(items),
            stderr="",
            exit_code=0,
        )

    def _handle_github_search(self, args: list[str]) -> MockResult:
        """Handle opencli github search --query Q [--language L] [--limit N]."""
        parsed = _parse_args(args)
        query = _get_flag(parsed, "--query")
        if not query:
            return MockResult(
                stdout="",
                stderr="--query is required",
                exit_code=1,
            )

        items = list(self.state["github"]["search"])

        query_lower = query.lower()
        items = [
            i for i in items
            if query_lower in i["name"].lower() or query_lower in i["description"].lower()
        ]

        language = _get_flag(parsed, "--language")
        if language is not None:
            items = [i for i in items if i["language"] == language]

        items = _apply_limit(items, parsed)
        return MockResult(
            stdout=json.dumps(items),
            stderr="",
            exit_code=0,
        )

    def _handle_github_trending(self, args: list[str]) -> MockResult:
        """Handle opencli github trending [--language L] [--since S] [--limit N]."""
        parsed = _parse_args(args)
        items = list(self.state["github"]["trending"])

        language = _get_flag(parsed, "--language")
        if language is not None:
            items = [i for i in items if i["language"] == language]

        items = _apply_limit(items, parsed)
        return MockResult(
            stdout=json.dumps(items),
            stderr="",
            exit_code=0,
        )

    def _handle_bbc_news(self, args: list[str]) -> MockResult:
        """Handle opencli bbc news [--category C] [--limit N]."""
        parsed = _parse_args(args)
        items = list(self.state["bbc"]["news"])

        category = _get_flag(parsed, "--category")
        if category is not None:
            items = [i for i in items if i["category"] == category]

        items = _apply_limit(items, parsed)
        return MockResult(
            stdout=json.dumps(items),
            stderr="",
            exit_code=0,
        )

    def _handle_v2ex_hot(self, args: list[str]) -> MockResult:
        """Handle opencli v2ex hot [--limit N]."""
        parsed = _parse_args(args)
        items = list(self.state["v2ex"]["hot"])
        items = _apply_limit(items, parsed)
        return MockResult(
            stdout=json.dumps(items),
            stderr="",
            exit_code=0,
        )
