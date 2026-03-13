#!/usr/bin/env python3
"""Run CLI-Bench evaluation."""

import argparse
import asyncio
import json
from pathlib import Path

from cli_bench.agents.dummy import DummyAgent
from cli_bench.harness.benchmark import BenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Run CLI-Bench evaluation")
    parser.add_argument(
        "--agent",
        choices=["dummy", "openai", "anthropic"],
        default="dummy",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for LLM agents",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of runs per task for pass^k",
    )
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("data/tasks"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/cli_bench"),
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run single task by ID",
    )
    args = parser.parse_args()

    agent = _create_agent(args.agent, args.model)
    runner = BenchmarkRunner(
        tasks_dir=args.tasks_dir,
        agent=agent,
        k=args.k,
    )

    if args.task_id:
        result = asyncio.run(runner.run_single(args.task_id))
        _save_result(args.output, result)
    else:
        report = asyncio.run(runner.run_all())
        _save_report(args.output, report)


def _create_agent(agent_type: str, model: str | None):
    """Create a benchmark agent by type name."""
    if agent_type == "dummy":
        return DummyAgent()
    elif agent_type == "openai":
        from cli_bench.agents.openai_agent import OpenAIAgent

        return OpenAIAgent(model=model or "gpt-4o")
    elif agent_type == "anthropic":
        from cli_bench.agents.anthropic_agent import AnthropicAgent

        return AnthropicAgent(model=model or "claude-sonnet-4-20250514")
    raise ValueError(f"Unknown agent type: {agent_type}")


def _save_report(output_dir: Path, report) -> None:
    """Write benchmark report as JSON and print summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "report.json", "w") as f:
        json.dump(
            report.__dict__ if hasattr(report, "__dict__") else report,
            f,
            indent=2,
            default=str,
        )
    print(f"\n=== CLI-Bench Results ===")
    print(f"Overall Score: {report.overall_score:.3f}")
    print(f"Pass^k: {report.overall_pass_k:.3f}")
    print(f"By Difficulty:")
    for diff, score in report.by_difficulty.items():
        print(f"  {diff}: {score:.3f}")


def _save_result(output_dir: Path, result) -> None:
    """Write single task result as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{result.task_id}.json", "w") as f:
        json.dump(
            {
                "task_id": result.task_id,
                "mean_score": result.mean_score,
                "pass_k": result.pass_k,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
