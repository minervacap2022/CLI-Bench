"""BenchmarkRunner — full load-run-score-report pipeline.

Loads tasks from YAML, runs each task k times with an agent,
scores each run, and produces an aggregate BenchmarkReport.
"""

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path

from cli_bench.agents.base import BenchAgent
from cli_bench.harness.evaluator import Evaluator
from cli_bench.harness.runner import Runner
from cli_bench.mock_backends.base import BaseMockBackend
from cli_bench.mock_backends.fictional import FictionalMockBackend
from cli_bench.mock_backends.github import GitHubMockBackend
from cli_bench.mock_backends.google import GoogleMockBackend
from cli_bench.mock_backends.jira import JiraMockBackend
from cli_bench.mock_backends.linear import LinearMockBackend
from cli_bench.mock_backends.notion import NotionMockBackend
from cli_bench.mock_backends.opencli import OpenCLIMockBackend
from cli_bench.mock_backends.slack import SlackMockBackend
from cli_bench.models.scoring import TaskScore
from cli_bench.models.task import BenchTask
from cli_bench.scoring.reliability import ReliabilityMetrics, analyze_reliability

# Map service names to backend classes.
# Also includes binary names (e.g. ``gh``) so that tasks whose
# ``initial_state`` keys use binary names resolve correctly.
_BACKEND_REGISTRY: dict[str, type[BaseMockBackend]] = {
    "github": GitHubMockBackend,
    "gh": GitHubMockBackend,
    "slack": SlackMockBackend,
    "linear": LinearMockBackend,
    "opencli": OpenCLIMockBackend,
    "jira": JiraMockBackend,
    "notion": NotionMockBackend,
    "google": GoogleMockBackend,
    "kforge": FictionalMockBackend,
    "flowctl": FictionalMockBackend,
    "meshctl": FictionalMockBackend,
    "datapipe": FictionalMockBackend,
    "alertmgr": FictionalMockBackend,
}

# Fictional tools share FictionalMockBackend and require tool_name on init
_FICTIONAL_TOOLS: set[str] = {"kforge", "flowctl", "meshctl", "datapipe", "alertmgr"}

# Map tool binary names to their service name for backend routing
_TOOL_TO_SERVICE: dict[str, str] = {
    "gh": "github",
    "slack": "slack",
    "linear": "linear",
    "opencli": "opencli",
    "jira": "jira",
    "notion": "notion",
    "google": "google",
    "kforge": "kforge",
    "flowctl": "flowctl",
    "meshctl": "meshctl",
    "datapipe": "datapipe",
    "alertmgr": "alertmgr",
}


@dataclass
class TaskResult:
    """Aggregated result for a single task run k times."""

    task_id: str
    scores: list[TaskScore]
    mean_score: float
    pass_k: float


@dataclass
class BenchmarkReport:
    """Full benchmark report across all tasks."""

    results: list[TaskResult]
    overall_score: float
    overall_pass_k: float
    by_difficulty: dict[str, float]
    by_category: dict[str, float]
    total_cost_usd: float
    total_time_ms: int
    reliability: ReliabilityMetrics | None = None


class BenchmarkRunner:
    """Full pipeline: load tasks, run agent k times each, score, report."""

    def __init__(
        self,
        tasks_dir: Path,
        agent: BenchAgent,
        k: int = 5,
        tool_adapters_dir: Path | None = None,
    ) -> None:
        self._tasks = self._load_tasks(tasks_dir)
        self._agent = agent
        self._k = k
        self._evaluator = Evaluator()
        self._tool_adapters_dir = (
            tool_adapters_dir
            if tool_adapters_dir is not None
            else Path("cli_bench/tool_adapters")
        )

    async def run_all(self) -> BenchmarkReport:
        """Run all tasks k times each, aggregate into BenchmarkReport."""
        start_ms = _now_ms()
        results: list[TaskResult] = []

        for task in self._tasks:
            task_result = await self._run_task_k_times(task)
            results.append(task_result)

        total_time = _now_ms() - start_ms

        return self._build_report(results, total_time)

    async def run_single(self, task_id: str) -> TaskResult:
        """Run one task k times by ID."""
        task = self._find_task(task_id)
        return await self._run_task_k_times(task)

    async def _run_task_k_times(self, task: BenchTask) -> TaskResult:
        """Run a single task k times, return TaskResult."""
        scores: list[TaskScore] = []

        for _ in range(self._k):
            self._agent.reset()
            backends = self._create_backends(task)
            tool_backends = self._map_tools_to_backends(task, backends)

            runner = Runner(
                agent=self._agent,
                backends=tool_backends,
                tool_adapters_dir=self._tool_adapters_dir,
            )
            run_result = await runner.run_task(task)
            score = self._evaluator.evaluate(task, run_result, backends)
            scores.append(score)

        mean_score = sum(s.total for s in scores) / len(scores)
        pass_k = TaskScore.pass_k(scores)

        return TaskResult(
            task_id=task.id,
            scores=scores,
            mean_score=mean_score,
            pass_k=pass_k,
        )

    def _create_backends(self, task: BenchTask) -> dict[str, BaseMockBackend]:
        """Create fresh mock backends from task's initial_state."""
        backends: dict[str, BaseMockBackend] = {}
        for service_name, initial_state in task.initial_state.items():
            backend_cls = _BACKEND_REGISTRY.get(service_name)
            if backend_cls is not None:
                state = copy.deepcopy(initial_state)
                if service_name in _FICTIONAL_TOOLS:
                    backends[service_name] = backend_cls(state, tool_name=service_name)
                else:
                    backends[service_name] = backend_cls(state)
        return backends

    def _map_tools_to_backends(
        self,
        task: BenchTask,
        backends: dict[str, BaseMockBackend],
    ) -> dict[str, BaseMockBackend]:
        """Map tool binary names to their corresponding backends for the Runner."""
        tool_backends: dict[str, BaseMockBackend] = {}
        for tool_name in task.tools_provided:
            service_name = _TOOL_TO_SERVICE.get(tool_name)
            if service_name and service_name in backends:
                tool_backends[tool_name] = backends[service_name]
        return tool_backends

    def _find_task(self, task_id: str) -> BenchTask:
        """Look up a task by ID."""
        for task in self._tasks:
            if task.id == task_id:
                return task
        raise KeyError(f"Task not found: {task_id}")

    def _build_report(
        self,
        results: list[TaskResult],
        total_time_ms: int,
    ) -> BenchmarkReport:
        """Aggregate task results into a BenchmarkReport."""
        if not results:
            return BenchmarkReport(
                results=[],
                overall_score=0.0,
                overall_pass_k=0.0,
                by_difficulty={},
                by_category={},
                total_cost_usd=0.0,
                total_time_ms=total_time_ms,
                reliability=None,
            )

        overall_score = sum(r.mean_score for r in results) / len(results)
        overall_pass_k = sum(r.pass_k for r in results) / len(results)

        # Group by difficulty
        by_difficulty: dict[str, list[float]] = {}
        by_category: dict[str, list[float]] = {}

        task_map = {t.id: t for t in self._tasks}
        for result in results:
            task = task_map[result.task_id]
            by_difficulty.setdefault(task.difficulty, []).append(result.mean_score)
            by_category.setdefault(task.category, []).append(result.mean_score)

        # Compute reliability metrics from per-task run scores
        # Shape: [n_tasks, k_runs] — each inner list is scores for one task
        run_outcomes = [
            [s.total for s in result.scores] for result in results
        ]
        reliability = analyze_reliability(run_outcomes)

        return BenchmarkReport(
            results=results,
            overall_score=overall_score,
            overall_pass_k=overall_pass_k,
            by_difficulty={k: sum(v) / len(v) for k, v in by_difficulty.items()},
            by_category={k: sum(v) / len(v) for k, v in by_category.items()},
            total_cost_usd=0.0,
            total_time_ms=total_time_ms,
            reliability=reliability,
        )

    @staticmethod
    def _load_tasks(tasks_dir: Path) -> list[BenchTask]:
        """Load all task YAML files from a directory."""
        tasks: list[BenchTask] = []
        for yaml_path in sorted(tasks_dir.glob("*.yaml")):
            tasks.append(BenchTask.from_yaml(yaml_path))
        return tasks


def _now_ms() -> int:
    """Current time in milliseconds."""
    return time.monotonic_ns() // 1_000_000
