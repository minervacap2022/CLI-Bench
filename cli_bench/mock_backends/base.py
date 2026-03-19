"""Base mock backend ABC with stateful execution and deep state comparison.

Provides BaseMockBackend for subclasses to implement route_command(),
and _deep_diff() for recursive state comparison with partial scoring.
Supports semantic assertion keys in expected state:
  - ``*_contains``: substring check against the field without the suffix
  - ``contains``: list of substrings that must all be present
  - ``pattern``: regex match assertion (used inside command_history items)
  - ``command_history``: matched against the backend's recorded command log
"""

from __future__ import annotations

import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli_bench.models.tool_adapter import ToolAdapter


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


_ASSERTION_SUFFIXES = ("_contains", "_contain")

# Keys that are assertion-only (never correspond to real state fields).
_ASSERTION_ONLY_KEYS = {"pattern", "command_history", "output_contains", "contains"}


def _is_assertion_key(key: str) -> bool:
    """Return True if *key* is a special assertion key rather than a plain field."""
    if key in _ASSERTION_ONLY_KEYS:
        return True
    for suffix in _ASSERTION_SUFFIXES:
        if key.endswith(suffix):
            return True
    return False


def _check_contains(actual_value: object, substring: str) -> bool:
    """Return True if *substring* appears in a reasonable string form of *actual_value*."""
    if isinstance(actual_value, str):
        return substring in actual_value
    if isinstance(actual_value, list):
        # Check if substring appears in any element (stringified).
        return any(substring in str(item) for item in actual_value)
    return substring in str(actual_value)


def _match_assertion_key(
    actual_item: dict, key: str, value: object
) -> bool:
    """Evaluate a single assertion key/value against *actual_item*.

    Returns True when the assertion passes.
    """
    # --- *_contains / *_contain: substring check on the base field ---
    for suffix in _ASSERTION_SUFFIXES:
        if key.endswith(suffix):
            base_field = key[: -len(suffix)]
            if base_field not in actual_item:
                return False
            return _check_contains(actual_item[base_field], str(value))

    # --- pattern: regex match against the full string of actual_item ---
    if key == "pattern":
        text = str(actual_item) if not isinstance(actual_item, str) else actual_item
        return bool(re.search(str(value), text))

    # --- contains: list of substrings, all must be present ---
    if key == "contains" and isinstance(value, list):
        text = str(actual_item)
        return all(sub in text for sub in value)

    return False


def _match_item(actual_item: object, expected_item: dict) -> bool:
    """Return True if *actual_item* satisfies all fields & assertions in *expected_item*."""
    if not isinstance(actual_item, dict):
        # If expected_item has only a 'pattern' key, match against the string form.
        if set(expected_item.keys()) == {"pattern"}:
            text = str(actual_item) if not isinstance(actual_item, str) else actual_item
            return bool(re.search(str(expected_item["pattern"]), text))
        return False

    for key, value in expected_item.items():
        if _is_assertion_key(key):
            if not _match_assertion_key(actual_item, key, value):
                return False
        else:
            # Exact field comparison (recurse for nested structures).
            if key not in actual_item:
                return False
            child = _deep_diff(actual_item[key], value)
            if not child.matches:
                return False
    return True


def _match_command_history(
    actual_commands: list[str], expected_patterns: list[dict]
) -> StateDiff:
    """Match a list of command-history pattern dicts against actual command strings."""
    if not expected_patterns:
        return StateDiff(matches=True, score=1.0)

    found = 0
    missing: list[str] = []

    for i, pat_item in enumerate(expected_patterns):
        pattern = pat_item.get("pattern", "") if isinstance(pat_item, dict) else str(pat_item)
        if any(re.search(pattern, cmd) for cmd in actual_commands):
            found += 1
        else:
            missing.append(f"command_history[{i}] (pattern={pattern!r})")

    score = found / len(expected_patterns)
    return StateDiff(matches=score == 1.0, score=score, missing=missing)


def _deep_diff(
    actual: object,
    expected: object,
    path: str = "",
    command_history: list[str] | None = None,
) -> StateDiff:
    """Deep recursive comparison with partial scoring.

    For dicts: compare each expected key, compute fraction matched.
      - Keys ending with ``_contains`` / ``_contain`` trigger substring checks.
      - ``command_history`` is matched against *command_history* argument.
      - ``output_contains`` / ``contains`` lists trigger substring presence checks.
    For lists: check if each expected item exists in actual (order-independent).
      - List items that are dicts may use assertion keys for flexible matching.
    For scalars: exact equality.

    Returns StateDiff with score 0.0-1.0 and detailed missing/mismatched paths.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        return _diff_dicts(actual, expected, path, command_history=command_history)

    if isinstance(expected, list) and isinstance(actual, list):
        return _diff_lists(actual, expected, path, command_history=command_history)

    # Scalar comparison
    if actual == expected:
        return StateDiff(matches=True, score=1.0)

    label = path if path else repr(expected)
    return StateDiff(matches=False, score=0.0, mismatched=[label])


def _diff_dicts(
    actual: dict,
    expected: dict,
    path: str,
    command_history: list[str] | None = None,
) -> StateDiff:
    """Compare two dicts, scoring based on expected keys matched.

    Handles special assertion keys transparently.
    """
    if not expected:
        extra = [
            f"{path}.{k}" if path else k for k in actual if k not in expected
        ]
        return StateDiff(matches=True, score=1.0, extra=extra)

    missing: list[str] = []
    extra: list[str] = []
    mismatched: list[str] = []
    matched_count = 0
    total_keys = 0  # track denominator (some assertion keys may be skipped)

    for key in expected:
        child_path = f"{path}.{key}" if path else key

        # ---- command_history: match against recorded commands ----
        if key == "command_history":
            total_keys += 1
            cmds = command_history if command_history is not None else []
            ch_diff = _match_command_history(cmds, expected[key])
            matched_count += ch_diff.score
            missing.extend(ch_diff.missing)
            mismatched.extend(ch_diff.mismatched)
            continue

        # ---- output_contains: skip (evaluated elsewhere by runner) ----
        if key == "output_contains":
            # These are checked against captured stdout by the runner, not state.
            # Count as matched so they don't penalise the state diff score.
            total_keys += 1
            matched_count += 1
            continue

        # ---- *_contains / *_contain: substring assertion on base field ----
        handled_as_assertion = False
        for suffix in _ASSERTION_SUFFIXES:
            if key.endswith(suffix):
                total_keys += 1
                base_field = key[: -len(suffix)]
                if base_field not in actual:
                    missing.append(child_path)
                elif _check_contains(actual[base_field], str(expected[key])):
                    matched_count += 1
                else:
                    mismatched.append(child_path)
                handled_as_assertion = True
                break
        if handled_as_assertion:
            continue

        # ---- contains (list of substrings) ----
        if key == "contains" and isinstance(expected[key], list):
            total_keys += 1
            text = str(actual)
            if all(sub in text for sub in expected[key]):
                matched_count += 1
            else:
                mismatched.append(child_path)
            continue

        # ---- Regular key: must exist in actual ----
        total_keys += 1
        if key not in actual:
            missing.append(child_path)
            continue

        child_diff = _deep_diff(
            actual[key], expected[key], child_path, command_history=command_history
        )
        matched_count += child_diff.score
        missing.extend(child_diff.missing)
        extra.extend(child_diff.extra)
        mismatched.extend(child_diff.mismatched)

    # Extra keys in actual that are not in expected
    for key in actual:
        if key not in expected:
            child_path = f"{path}.{key}" if path else key
            extra.append(child_path)

    score = matched_count / total_keys if total_keys else 1.0
    matches = score == 1.0 and not missing and not mismatched

    return StateDiff(
        matches=matches,
        score=score,
        missing=missing,
        extra=extra,
        mismatched=mismatched,
    )


def _diff_lists(
    actual: list,
    expected: list,
    path: str,
    command_history: list[str] | None = None,
) -> StateDiff:
    """Compare two lists order-independently, scoring by fraction of expected items found.

    When expected items are dicts that contain assertion keys, flexible matching
    via ``_match_item`` is used instead of exact equality.
    """
    if not expected:
        return StateDiff(matches=True, score=1.0)

    found = 0
    missing: list[str] = []
    mismatched: list[str] = []

    # Track which actual items have been consumed so each is used at most once.
    used: set[int] = set()

    for i, exp_item in enumerate(expected):
        label = f"{path}[{i}]" if path else f"[{i}]"

        # --- Dict expected item with assertion keys: flexible match ---
        if isinstance(exp_item, dict) and any(_is_assertion_key(k) for k in exp_item):
            matched = False
            best_score = 0.0
            for j, act_item in enumerate(actual):
                if j in used:
                    continue
                if _match_item(act_item, exp_item):
                    found += 1
                    used.add(j)
                    matched = True
                    break
            if not matched:
                # Try to get a partial score from the best candidate.
                for j, act_item in enumerate(actual):
                    if j in used:
                        continue
                    if isinstance(act_item, dict):
                        # Count how many assertion/field checks pass.
                        passes = 0
                        total = len(exp_item)
                        for key, value in exp_item.items():
                            if _is_assertion_key(key):
                                if _match_assertion_key(act_item, key, value):
                                    passes += 1
                            elif key in act_item:
                                cd = _deep_diff(act_item[key], value)
                                passes += cd.score
                        candidate_score = passes / total if total else 0.0
                        if candidate_score > best_score:
                            best_score = candidate_score
                found += best_score
                if best_score < 1.0:
                    missing.append(label)
            continue

        # --- Dict expected item without assertion keys: recurse ---
        if isinstance(exp_item, dict):
            matched = False
            best_score = 0.0
            best_diff: StateDiff | None = None
            for j, act_item in enumerate(actual):
                if j in used:
                    continue
                if isinstance(act_item, dict):
                    cd = _deep_diff(
                        act_item, exp_item, label, command_history=command_history
                    )
                    if cd.matches:
                        found += 1
                        used.add(j)
                        matched = True
                        break
                    if cd.score > best_score:
                        best_score = cd.score
                        best_diff = cd
            if not matched:
                found += best_score
                if best_diff:
                    missing.extend(best_diff.missing)
                    mismatched.extend(best_diff.mismatched)
                else:
                    missing.append(label)
            continue

        # --- Scalar expected item: exact equality ---
        if exp_item in actual:
            found += 1
        else:
            missing.append(label)

    score = found / len(expected)
    matches = score == 1.0 and not missing and not mismatched

    return StateDiff(matches=matches, score=score, missing=missing, mismatched=mismatched)


class BaseMockBackend(ABC):
    """Abstract base for stateful mock service backends.

    Subclasses implement route_command() to handle specific commands.
    The base class manages state, action logging, reset, and diffing.
    """

    def __init__(self, initial_state: dict) -> None:
        self._initial_state = copy.deepcopy(initial_state)
        self.state: dict = copy.deepcopy(initial_state)
        self._action_log: list[Action] = []
        self._command_history: list[str] = []
        self._tool_adapter: ToolAdapter | None = None

    def set_tool_adapter(self, adapter: ToolAdapter) -> None:
        """Attach a tool adapter so that ``--help`` / ``-h`` requests are intercepted."""
        self._tool_adapter = adapter

    def _try_handle_help(self, command: list[str]) -> MockResult | None:
        """Intercept ``--help`` / ``-h`` flags and return help text.

        Returns ``None`` when the command is not a help request.
        """
        if self._tool_adapter is None:
            return None

        if "--help" not in command and "-h" not in command:
            return None

        # Strip binary name and help flags to determine scope
        parts = [
            p for p in command
            if p not in ("--help", "-h") and p != self._tool_adapter.binary
        ]

        adapter = self._tool_adapter
        if len(parts) == 0:
            # Root help: e.g. ``gh --help``
            text = adapter.to_root_help()
        elif len(parts) == 1:
            # Subcommand help: e.g. ``gh issue --help``
            text = adapter.to_subcommand_help(parts[0])
        else:
            # Action help: e.g. ``gh issue list --help``
            action_name = f"{parts[0]} {parts[1]}"
            text = adapter.to_action_help(action_name)

        return MockResult(stdout=text, stderr="", exit_code=0)

    def execute(self, command: list[str]) -> MockResult:
        """Execute a command, log the action, and return the result.

        Help requests (``--help`` / ``-h``) are intercepted before
        ``route_command()`` when a tool adapter is attached.
        """
        help_result = self._try_handle_help(command)
        if help_result is not None:
            result = help_result
        else:
            result = self.route_command(command)
        self._action_log.append(
            Action(command=command, result=result)
        )
        # Record the raw command string for command_history matching.
        self._command_history.append(" ".join(command))
        return result

    @abstractmethod
    def route_command(self, command: list[str]) -> MockResult:
        """Route a command to the appropriate handler. Subclasses must implement."""
        ...

    def get_state_snapshot(self) -> dict:
        """Return a deep copy of the current state, including command history."""
        snapshot = copy.deepcopy(self.state)
        snapshot["command_history"] = list(self._command_history)
        return snapshot

    def get_action_log(self) -> list[Action]:
        """Return the list of recorded actions."""
        return list(self._action_log)

    def reset(self) -> None:
        """Restore state to initial and clear the action log and command history."""
        self.state = copy.deepcopy(self._initial_state)
        self._action_log.clear()
        self._command_history.clear()

    def get_command_history(self) -> list[str]:
        """Return a copy of the recorded command strings."""
        return list(self._command_history)

    def diff(self, expected_state: dict) -> StateDiff:
        """Compare current state against expected using deep recursive comparison.

        Passes the recorded command history so that ``command_history`` assertion
        keys in *expected_state* can be evaluated.
        """
        return _deep_diff(
            self.state, expected_state, command_history=self._command_history
        )
