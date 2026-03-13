"""Base mock backend ABC with stateful execution and deep state comparison.

Provides BaseMockBackend for subclasses to implement route_command(),
and _deep_diff() for recursive state comparison with partial scoring.
"""

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class MockResult:
    """Result of executing a command against a mock backend."""

    stdout: str
    stderr: str
    exit_code: int


@dataclass
class Action:
    """A recorded command execution with its result."""

    command: list[str]
    result: MockResult
    timestamp_ms: int = 0


@dataclass
class StateDiff:
    """Result of comparing actual state against expected state."""

    matches: bool
    score: float
    missing: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)


def _deep_diff(actual: object, expected: object, path: str = "") -> StateDiff:
    """Deep recursive comparison with partial scoring.

    For dicts: compare each expected key, compute fraction matched.
    For lists: check if each expected item exists in actual (order-independent).
    For scalars: exact equality.

    Returns StateDiff with score 0.0-1.0 and detailed missing/mismatched paths.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _diff_dicts(actual, expected, path)

    if isinstance(expected, list) and isinstance(actual, list):
        return _diff_lists(actual, expected, path)

    # Scalar comparison
    if actual == expected:
        return StateDiff(matches=True, score=1.0)

    label = path if path else repr(expected)
    return StateDiff(matches=False, score=0.0, mismatched=[label])


def _diff_dicts(
    actual: dict, expected: dict, path: str
) -> StateDiff:
    """Compare two dicts, scoring based on expected keys matched."""
    if not expected:
        extra = [
            f"{path}.{k}" if path else k for k in actual if k not in expected
        ]
        return StateDiff(matches=True, score=1.0, extra=extra)

    missing: list[str] = []
    extra: list[str] = []
    mismatched: list[str] = []
    matched_count = 0

    for key in expected:
        child_path = f"{path}.{key}" if path else key
        if key not in actual:
            missing.append(child_path)
            continue

        child_diff = _deep_diff(actual[key], expected[key], child_path)
        matched_count += child_diff.score
        missing.extend(child_diff.missing)
        extra.extend(child_diff.extra)
        mismatched.extend(child_diff.mismatched)

    # Extra keys in actual that are not in expected
    for key in actual:
        if key not in expected:
            child_path = f"{path}.{key}" if path else key
            extra.append(child_path)

    score = matched_count / len(expected)
    matches = score == 1.0 and not missing and not mismatched

    return StateDiff(
        matches=matches,
        score=score,
        missing=missing,
        extra=extra,
        mismatched=mismatched,
    )


def _diff_lists(
    actual: list, expected: list, path: str
) -> StateDiff:
    """Compare two lists order-independently, scoring by fraction of expected items found."""
    if not expected:
        return StateDiff(matches=True, score=1.0)

    found = 0
    missing: list[str] = []

    for i, item in enumerate(expected):
        if item in actual:
            found += 1
        else:
            label = f"{path}[{i}]" if path else f"[{i}]"
            missing.append(label)

    score = found / len(expected)
    matches = score == 1.0

    return StateDiff(matches=matches, score=score, missing=missing)


class BaseMockBackend(ABC):
    """Abstract base for stateful mock service backends.

    Subclasses implement route_command() to handle specific commands.
    The base class manages state, action logging, reset, and diffing.
    """

    def __init__(self, initial_state: dict) -> None:
        self._initial_state = copy.deepcopy(initial_state)
        self.state: dict = copy.deepcopy(initial_state)
        self._action_log: list[Action] = []

    def execute(self, command: list[str]) -> MockResult:
        """Execute a command, log the action, and return the result."""
        result = self.route_command(command)
        self._action_log.append(
            Action(command=command, result=result)
        )
        return result

    @abstractmethod
    def route_command(self, command: list[str]) -> MockResult:
        """Route a command to the appropriate handler. Subclasses must implement."""
        ...

    def get_state_snapshot(self) -> dict:
        """Return a deep copy of the current state."""
        return copy.deepcopy(self.state)

    def get_action_log(self) -> list[Action]:
        """Return the list of recorded actions."""
        return list(self._action_log)

    def reset(self) -> None:
        """Restore state to initial and clear the action log."""
        self.state = copy.deepcopy(self._initial_state)
        self._action_log.clear()

    def diff(self, expected_state: dict) -> StateDiff:
        """Compare current state against expected using deep recursive comparison."""
        return _deep_diff(self.state, expected_state)
