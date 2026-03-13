"""Notion mock backend — simulates a Notion CLI tool.

Handles page list/create/update/get, db query, block append, and search commands.
"""

import json

from cli_bench.mock_backends.base import BaseMockBackend, MockResult


def _parse_args(args: list[str]) -> dict[str, list[str]]:
    """Parse CLI args into a dict mapping --flag to list of values."""
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


class NotionMockBackend(BaseMockBackend):
    """Stateful mock for the Notion CLI tool.

    State schema:
        {
            "pages": [{"id": str, "title": str, "content": str,
                        "parent": str|None, "archived": bool,
                        "blocks": [{"type": str, "content": str}]}],
            "databases": [{"id": str, "title": str,
                           "rows": [{"id": str, ...fields}]}],
        }
    """

    def route_command(self, command: list[str]) -> MockResult:
        """Route a notion CLI command to the appropriate handler."""
        if len(command) < 2 or command[0] != "notion":
            return MockResult(
                stdout="",
                stderr=f"unknown command: {' '.join(command)}",
                exit_code=1,
            )

        resource = command[1]
        action = command[2] if len(command) > 2 else ""
        remaining = command[3:] if len(command) > 3 else []

        if resource == "page":
            return self._handle_page(action, remaining)
        if resource == "db":
            return self._handle_db(action, remaining)
        if resource == "block":
            return self._handle_block(action, remaining)
        if resource == "search":
            # search has no sub-action, args start at command[2:]
            return self._handle_search(command[2:])

        return MockResult(
            stdout="",
            stderr=f"unknown command: notion {resource}",
            exit_code=1,
        )

    def _handle_page(self, action: str, args: list[str]) -> MockResult:
        """Handle notion page <action> ..."""
        if action == "list":
            return self._page_list(args)
        if action == "create":
            return self._page_create(args)
        if action == "update":
            return self._page_update(args)
        if action == "get":
            return self._page_get(args)

        return MockResult(
            stdout="",
            stderr=f"unknown page subcommand: {action}",
            exit_code=1,
        )

    def _page_list(self, args: list[str]) -> MockResult:
        """List pages with optional parent filter."""
        parsed = _parse_args(args)
        pages = list(self.state.get("pages", []))

        parent_filter = _get_flag(parsed, "--parent")
        if parent_filter:
            pages = [p for p in pages if p.get("parent") == parent_filter]

        # Exclude archived by default
        pages = [p for p in pages if not p.get("archived", False)]

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            pages = pages[: int(limit_str)]

        # Return summary without blocks
        result = []
        for p in pages:
            result.append({
                "id": p["id"],
                "title": p["title"],
                "parent": p.get("parent"),
            })

        return MockResult(
            stdout=json.dumps(result),
            stderr="",
            exit_code=0,
        )

    def _page_create(self, args: list[str]) -> MockResult:
        """Create a new page."""
        parsed = _parse_args(args)

        title = _get_flag(parsed, "--title")
        if not title:
            return MockResult(stdout="", stderr="--title is required", exit_code=1)

        parent = _get_flag(parsed, "--parent")
        if not parent:
            return MockResult(stdout="", stderr="--parent is required", exit_code=1)

        pages = self.state.get("pages", [])
        max_num = 0
        for page in pages:
            parts = page["id"].split("-")
            if len(parts) >= 2 and parts[-1].isdigit():
                num = int(parts[-1])
                if num > max_num:
                    max_num = num

        content = _get_flag(parsed, "--content") or ""
        icon = _get_flag(parsed, "--icon")

        new_page: dict = {
            "id": f"page-{max_num + 1:03d}",
            "title": title,
            "content": content,
            "parent": parent,
            "archived": False,
            "blocks": [],
        }
        if icon:
            new_page["icon"] = icon

        pages.append(new_page)
        self.state["pages"] = pages

        return MockResult(
            stdout=json.dumps(new_page),
            stderr="",
            exit_code=0,
        )

    def _page_update(self, args: list[str]) -> MockResult:
        """Update a page by --id."""
        parsed = _parse_args(args)
        page_id = _get_flag(parsed, "--id")
        if not page_id:
            return MockResult(stdout="", stderr="--id is required", exit_code=1)

        for page in self.state.get("pages", []):
            if page["id"] == page_id:
                new_title = _get_flag(parsed, "--title")
                if new_title is not None:
                    page["title"] = new_title

                new_content = _get_flag(parsed, "--content")
                if new_content is not None:
                    page["content"] = new_content

                archived_str = _get_flag(parsed, "--archived")
                if archived_str is not None:
                    page["archived"] = archived_str.lower() in ("true", "1", "yes")

                return MockResult(
                    stdout=json.dumps(page),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"page {page_id} not found",
            exit_code=1,
        )

    def _page_get(self, args: list[str]) -> MockResult:
        """Get a single page by --id."""
        parsed = _parse_args(args)
        page_id = _get_flag(parsed, "--id")
        if not page_id:
            return MockResult(stdout="", stderr="--id is required", exit_code=1)

        for page in self.state.get("pages", []):
            if page["id"] == page_id:
                return MockResult(
                    stdout=json.dumps(page),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"page {page_id} not found",
            exit_code=1,
        )

    def _handle_db(self, action: str, args: list[str]) -> MockResult:
        """Handle notion db <action> ..."""
        if action == "query":
            return self._db_query(args)

        return MockResult(
            stdout="",
            stderr=f"unknown db subcommand: {action}",
            exit_code=1,
        )

    def _db_query(self, args: list[str]) -> MockResult:
        """Query a database by --id with optional --filter."""
        parsed = _parse_args(args)
        db_id = _get_flag(parsed, "--id")
        if not db_id:
            return MockResult(stdout="", stderr="--id is required", exit_code=1)

        for db in self.state.get("databases", []):
            if db["id"] == db_id:
                rows = list(db.get("rows", []))

                # Apply simple JSON filter
                filter_str = _get_flag(parsed, "--filter")
                if filter_str:
                    try:
                        filter_obj = json.loads(filter_str)
                        rows = [
                            row
                            for row in rows
                            if all(
                                row.get(k) == v for k, v in filter_obj.items()
                            )
                        ]
                    except json.JSONDecodeError:
                        pass

                limit_str = _get_flag(parsed, "--limit")
                if limit_str is not None:
                    rows = rows[: int(limit_str)]

                return MockResult(
                    stdout=json.dumps(rows),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"database {db_id} not found",
            exit_code=1,
        )

    def _handle_block(self, action: str, args: list[str]) -> MockResult:
        """Handle notion block <action> ..."""
        if action == "append":
            return self._block_append(args)

        return MockResult(
            stdout="",
            stderr=f"unknown block subcommand: {action}",
            exit_code=1,
        )

    def _block_append(self, args: list[str]) -> MockResult:
        """Append a block to a page."""
        parsed = _parse_args(args)
        page_id = _get_flag(parsed, "--page-id")
        if not page_id:
            return MockResult(stdout="", stderr="--page-id is required", exit_code=1)

        content = _get_flag(parsed, "--content")
        if not content:
            return MockResult(stdout="", stderr="--content is required", exit_code=1)

        block_type = _get_flag(parsed, "--type") or "paragraph"

        for page in self.state.get("pages", []):
            if page["id"] == page_id:
                block = {"type": block_type, "content": content}
                page.setdefault("blocks", []).append(block)
                return MockResult(
                    stdout=json.dumps(block),
                    stderr="",
                    exit_code=0,
                )

        return MockResult(
            stdout="",
            stderr=f"page {page_id} not found",
            exit_code=1,
        )

    def _handle_search(self, args: list[str]) -> MockResult:
        """Search across pages and databases."""
        parsed = _parse_args(args)
        query = _get_flag(parsed, "--query")
        if not query:
            return MockResult(stdout="", stderr="--query is required", exit_code=1)

        type_filter = _get_flag(parsed, "--type")
        results: list[dict] = []

        query_lower = query.lower()

        if type_filter is None or type_filter == "page":
            for page in self.state.get("pages", []):
                if page.get("archived", False):
                    continue
                if (
                    query_lower in page.get("title", "").lower()
                    or query_lower in page.get("content", "").lower()
                ):
                    results.append({
                        "type": "page",
                        "id": page["id"],
                        "title": page["title"],
                    })

        if type_filter is None or type_filter == "database":
            for db in self.state.get("databases", []):
                if query_lower in db.get("title", "").lower():
                    results.append({
                        "type": "database",
                        "id": db["id"],
                        "title": db["title"],
                    })

        limit_str = _get_flag(parsed, "--limit")
        if limit_str is not None:
            results = results[: int(limit_str)]

        return MockResult(
            stdout=json.dumps(results),
            stderr="",
            exit_code=0,
        )
