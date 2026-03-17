# OpenCLI Integration into CLI-Bench

## Decision

Add OpenCLI (github.com/jackwener/opencli) as a new tool in CLI-Bench with:
- Custom mock backend (not FictionalMockBackend — OpenCLI's `<site> <command>` pattern differs from fictional CRUD `<resource> <action>`)
- Tool adapter YAML with 8 commands across 5 sites
- 10 benchmark tasks (3 easy, 3 medium, 4 hard)
- New category: `info_retrieval`

## Why OpenCLI

- **Semi-fictional**: Real tool, low LLM training exposure → tests genuine tool-learning without full memorization
- **New capability axis**: Information retrieval from web → complements existing devops/project_mgmt/communication
- **Cross-tool composition**: "Fetch data with opencli → act on it with gh/slack/linear" tasks

## Architecture

### Command Interface

```
opencli <site> <command> [--flag value]
```

Sites: hackernews, reddit, github, bbc, v2ex
Output: JSON (matches CLI-Bench's mock result format)

### Mock Backend Design

`OpenCLIMockBackend(BaseMockBackend)` — custom backend (not fictional) because:
- Command format is `opencli <site> <command>` not `opencli <resource> <action>`
- State is organized by site, not by CRUD resource
- Read-only operations (no create/update/delete)
- Filtering logic differs per site (subreddit, query, limit)

State schema:
```yaml
hackernews:
  top: [{rank, title, score, author, comments, url}]
reddit:
  hot: [{rank, title, subreddit, score, comments, author, url}]
  search: [{rank, title, subreddit, score, comments, author, url}]
github:
  search: [{name, description, stars, language, url}]
  trending: [{name, description, stars, language, url}]
bbc:
  news: [{rank, title, category, summary, url}]
v2ex:
  hot: [{rank, title, node, author, replies}]
```

### Tool Adapter (opencli.yaml)

8 commands:
1. `hackernews top` — top HN stories (--limit)
2. `reddit hot` — hot posts (--subreddit, --limit)
3. `reddit search` — search posts (--query, --subreddit, --limit)
4. `github search` — search repos (--query, --language, --limit)
5. `github trending` — trending repos (--language, --since, --limit)
6. `bbc news` — BBC news headlines (--category, --limit)
7. `v2ex hot` — V2EX hot topics (--limit)
8. `list` — list available sites/commands

### Tasks (cb-041 to cb-050)

| ID | Difficulty | Tools | Description |
|----|-----------|-------|-------------|
| cb-041 | easy | opencli | Fetch top 5 HN stories |
| cb-042 | easy | opencli | Search GitHub repos by language |
| cb-043 | easy | opencli | Get Reddit hot posts from specific subreddit |
| cb-044 | medium | opencli | Cross-site: compare HN and Reddit trending |
| cb-045 | medium | opencli, slack | Fetch BBC news → share summary in Slack |
| cb-046 | medium | opencli, gh | Search GitHub repos → create issue to evaluate top result |
| cb-047 | hard | opencli, slack, linear | Research workflow: HN + Reddit → Slack notify → Linear track |
| cb-048 | hard | opencli, gh | Multi-step: trending repos → filter by stars → create issues |
| cb-049 | hard | opencli | Multi-site aggregation with filtering and ranking |
| cb-050 | hard | opencli, slack, gh | Full pipeline: discover trends → cross-reference → notify → track |

### Registration

```python
# benchmark.py
_BACKEND_REGISTRY["opencli"] = OpenCLIMockBackend
_TOOL_TO_SERVICE["opencli"] = "opencli"
```

## Test Plan

- Unit tests for OpenCLIMockBackend: each command, filtering, error cases
- Tool adapter YAML validation (existing test_tool_adapters.py covers this)
- Task YAML validation (existing BenchTask.from_yaml covers this)
- Integration: run DummyAgent on new tasks, verify scoring works
