# Contributing to CLI-Bench

Thank you for your interest in contributing to CLI-Bench. This document outlines the process for contributing tasks, tools, mock backends, and bug fixes.

## Development Setup

```bash
git clone https://github.com/minervacap2022/CLI-Bench.git
cd CLI-Bench
pip install -e ".[dev]"
pytest tests/ -v --tb=short
```

## Adding a New CLI Tool

1. **Write a YAML tool adapter** in `cli_bench/tool_adapters/<tool>.yaml`. Follow the `ToolAdapter` schema (see existing adapters for reference). Each tool must have at least 5 commands with a mix of read-only and side-effect operations.

2. **Implement a mock backend** in `cli_bench/mock_backends/<tool>.py`. Subclass `BaseMockBackend` and implement `route_command()`. For fictional tools, use `FictionalMockBackend` which provides generic CRUD automatically.

3. **Write tests** in `tests/unit/test_mock_<tool>.py`. Test all commands, error handling, state mutation, and state diffing.

4. **Create tasks** in `data/tasks/` using the new tool. Follow the `BenchTask` YAML schema.

## Adding New Tasks

Task YAMLs must include:
- `id`: Unique identifier (e.g., `cb-041`)
- `title`: Human-readable title
- `difficulty`: One of `easy`, `medium`, `hard`
- `category`: One of `devops`, `project_mgmt`, `communication`, `data_ops`, `custom_cli`, `composite`
- `description`: Natural language task description (what the agent sees)
- `tools_provided`: List of tool binary names available to the agent
- `initial_state`: Starting state for each mock backend
- `expected_state`: Target state to evaluate against
- `max_turns`: Maximum agent turns
- `optimal_commands`: Minimum commands needed by an expert
- `scoring`: Weight configuration for outcome/efficiency/recovery

## Running Tests

```bash
pytest tests/ -v --tb=short
```

All tests must pass before submitting a pull request.

## Code Style

- Python 3.12+, type hints throughout
- Pydantic v2 for data models
- Dataclasses for lightweight runtime types
- Async/await for agent execution
- No external service dependencies in tests

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request with a clear description
