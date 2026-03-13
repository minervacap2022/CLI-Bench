# CLI-Bench: Benchmarking AI Agents on Command-Line Tool Orchestration

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Dataset on HF](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow)](https://huggingface.co/datasets/ChengyiX/CLI-Bench)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/main.pdf)

## Abstract

CLI-Bench is an evaluation benchmark for measuring AI agents' ability to learn and use command-line interface (CLI) tools to complete real-world tasks. Unlike existing benchmarks that test general coding ability or narrow tool-use scenarios, CLI-Bench evaluates **tool-agnostic CLI orchestration** -- the capacity to read tool documentation, plan multi-step workflows, execute commands, interpret outputs, recover from errors, and achieve desired end states across diverse service domains.

The benchmark comprises 40 tasks spanning six categories (DevOps, project management, communication, data operations, custom CLI, and composite workflows) across 12 CLI tools. Tasks are grounded in stateful mock backends that simulate real services (GitHub, Slack, Linear, Notion, Google Workspace, Jira) with deterministic execution, enabling reproducible evaluation without live API dependencies. Each tool is defined via a declarative YAML adapter specification, making the benchmark trivially extensible to new tools.

A key contribution is the inclusion of **five fictional CLI tools** (kforge, flowctl, meshctl, datapipe, alertmgr) that no language model has encountered during training. These tools follow realistic CLI conventions but implement novel domain semantics, providing a memorization-proof evaluation of genuine tool-learning capability rather than pattern recall. Evaluation uses state-diffing against expected outcomes, efficiency measurement against optimal command counts, error recovery analysis, and a pass^k consistency metric adapted from tau-bench.

## Key Features

- **Tool-agnostic via YAML adapters** -- Any CLI tool can be added by writing a YAML specification and a mock backend. No hardcoded tool knowledge in the harness.
- **Fictional tools for memorization-proof evaluation** -- Five novel CLI tools (kforge, flowctl, meshctl, datapipe, alertmgr) test genuine tool learning, not training data recall.
- **Multi-turn execution** -- Agents operate in a realistic loop: observe task and tool docs, issue commands, receive stdout/stderr, iterate until completion or timeout.
- **State-diffing evaluation** -- Scoring compares actual service state against expected state using deep recursive comparison with partial credit (0.0--1.0).
- **pass^k consistency metric** -- Measures reliability across k independent runs, not just peak performance. An agent must succeed on all k runs to score pass^k = 1.0.
- **Deterministic mock backends** -- All 7 service simulators (GitHub, Slack, Linear, Notion, Google, Jira, plus a generic fictional backend) are fully stateful and deterministic.

## Benchmark Statistics

| Dimension | Value |
|-----------|-------|
| Total tasks | 40 |
| Easy / Medium / Hard | 20 / 10 / 10 |
| Real-world CLI tools | 7 (gh, slack, linear, notion, google, jira, microsoft) |
| Fictional CLI tools | 5 (kforge, flowctl, meshctl, datapipe, alertmgr) |
| Task categories | 6 (devops, project_mgmt, communication, data_ops, custom_cli, composite) |
| Commands per tool | >= 5 |
| Max turns per task | 3--15 |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Outcome** (default weight: 0.6) | State-diff score: fraction of expected state matched after execution |
| **Efficiency** (default weight: 0.2) | `min(1.0, optimal_commands / actual_commands)` |
| **Recovery** (default weight: 0.2) | 1.0 if errors encountered and recovered; 0.5 if no errors; 0.0 if errors with no recovery |
| **pass^k** | 1.0 if outcome >= 0.5 on all k runs, else 0.0. Measures consistency. |

## Installation

```bash
pip install git+https://github.com/minervacap2022/CLI-Bench.git

# Or clone and install in development mode
git clone https://github.com/minervacap2022/CLI-Bench.git
cd CLI-Bench
pip install -e ".[dev]"
```

## Quick Start

```python
import asyncio
from pathlib import Path
from cli_bench.agents.dummy import DummyAgent
from cli_bench.harness.benchmark import BenchmarkRunner

async def main():
    agent = DummyAgent()
    runner = BenchmarkRunner(
        tasks_dir=Path("data/tasks"),
        agent=agent,
        k=1,
    )
    report = await runner.run_all()
    print(f"Overall score: {report.overall_score:.3f}")
    print(f"Pass^k: {report.overall_pass_k:.3f}")

asyncio.run(main())
```

Or via the CLI:

```bash
python scripts/run_benchmark.py --agent dummy --k 1
```

## Task Categories

| Category | Description | Example Task |
|----------|-------------|--------------|
| `devops` | CI/CD, deployment, infrastructure management | Trigger a deployment pipeline and verify status |
| `project_mgmt` | Issue tracking, sprint planning, team coordination | Create and assign issues across projects |
| `communication` | Messaging, notifications, search | Send targeted messages based on channel context |
| `data_ops` | ETL pipelines, data transformation, monitoring | Build a data pipeline from source to sink |
| `custom_cli` | Fictional tool operations (memorization-proof) | Manage artifacts in kforge registry |
| `composite` | Multi-tool workflows spanning categories | Create issue in Linear, notify team in Slack, schedule review in Calendar |

## Evaluation Metrics

### Outcome (State Diffing)

The primary metric compares the actual state of mock backends against the task's expected state using deep recursive comparison. Dict keys are checked individually with partial credit; list membership is verified order-independently. The resulting score is a float in [0.0, 1.0].

### Efficiency

Measures command economy: `min(1.0, optimal_commands / actual_commands)`. An agent that uses exactly the optimal number of commands scores 1.0; using twice as many scores 0.5.

### Recovery

Evaluates error handling:
- **1.0**: Errors encountered during execution AND the agent successfully recovered (issued a successful command after the last error)
- **0.5**: No errors encountered (neutral baseline)
- **0.0**: Errors encountered but the agent failed to recover

### pass^k

Adapted from [tau-bench](https://github.com/sierra-research/tau-bench). Given k independent runs of the same task, pass^k = 1.0 only if **all** k runs achieve outcome >= 0.5. This measures consistency and reliability, not just peak performance.

## Adding Custom Tools

### 1. Write a YAML Tool Adapter

Create `cli_bench/tool_adapters/<tool_name>.yaml`:

```yaml
name: my-tool
description: "Description of the tool"
binary: mytool
auth:
  type: env_var
  key: MYTOOL_API_KEY
commands:
  - name: resource list
    description: "List all resources"
    args:
      - name: filter
        type: string
        required: false
        description: "Filter expression"
    output_format: json
    side_effects: false
  - name: resource create
    description: "Create a new resource"
    args:
      - name: name
        type: string
        required: true
        description: "Resource name"
    output_format: json
    side_effects: true
```

### 2. Implement a Mock Backend

For real tools, subclass `BaseMockBackend`. For fictional tools, use `FictionalMockBackend` which provides generic CRUD operations automatically:

```python
from cli_bench.mock_backends.fictional import FictionalMockBackend

backend = FictionalMockBackend(
    initial_state={"resources": [{"id": "res-1", "name": "alpha"}]},
    tool_name="mytool",
)
```

### 3. Write Tasks

Create task YAMLs in `data/tasks/` following the `BenchTask` schema (see existing tasks for examples).

## Leaderboard

Results and model comparisons are hosted on HuggingFace Spaces:

**[https://huggingface.co/datasets/ChengyiX/CLI-Bench](https://huggingface.co/datasets/ChengyiX/CLI-Bench)**

## Citation

If you use CLI-Bench in your research, please cite:

```bibtex
@misc{cli-bench-2026,
  title={CLI-Bench: Benchmarking AI Agents on Command-Line Tool Orchestration},
  author={{KLIK Team}},
  year={2026},
  url={https://github.com/minervacap2022/CLI-Bench},
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
