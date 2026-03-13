"""Tests for BaseMockBackend ABC and state diffing."""

import pytest

from cli_bench.mock_backends.base import (
    Action,
    BaseMockBackend,
    MockResult,
    StateDiff,
    _deep_diff,
)


class SimpleMockBackend(BaseMockBackend):
    """Concrete mock backend for testing."""

    def route_command(self, command: list[str]) -> MockResult:
        if command[0] == "list":
            return MockResult(
                stdout=str(self.state.get("items", [])),
                stderr="",
                exit_code=0,
            )
        elif command[0] == "add":
            self.state.setdefault("items", []).append(command[1])
            return MockResult(
                stdout=f"Added {command[1]}",
                stderr="",
                exit_code=0,
            )
        return MockResult(
            stdout="",
            stderr=f"Unknown: {command[0]}",
            exit_code=1,
        )


class TestExecuteRecordsAction:
    def test_execute_records_action(self) -> None:
        """Execute adds to the action log with command and result."""
        backend = SimpleMockBackend(initial_state={"items": []})
        result = backend.execute(["list"])

        log = backend.get_action_log()
        assert len(log) == 1
        assert log[0].command == ["list"]
        assert log[0].result == result
        assert log[0].result.exit_code == 0

    def test_multiple_executions_recorded(self) -> None:
        """Multiple executions all appear in the action log."""
        backend = SimpleMockBackend(initial_state={})
        backend.execute(["add", "foo"])
        backend.execute(["add", "bar"])
        backend.execute(["list"])

        log = backend.get_action_log()
        assert len(log) == 3
        assert log[0].command == ["add", "foo"]
        assert log[1].command == ["add", "bar"]
        assert log[2].command == ["list"]


class TestStateMutation:
    def test_state_mutation(self) -> None:
        """Adding items mutates the state, visible in snapshot."""
        backend = SimpleMockBackend(initial_state={"items": []})
        backend.execute(["add", "alpha"])
        backend.execute(["add", "beta"])

        snapshot = backend.get_state_snapshot()
        assert snapshot["items"] == ["alpha", "beta"]

    def test_snapshot_is_deep_copy(self) -> None:
        """Modifying a snapshot does not affect backend state."""
        backend = SimpleMockBackend(initial_state={"items": ["x"]})
        snapshot = backend.get_state_snapshot()
        snapshot["items"].append("y")

        assert backend.get_state_snapshot()["items"] == ["x"]


class TestInitialStateNotMutated:
    def test_initial_state_not_mutated(self) -> None:
        """Original dict passed to __init__ is not affected by mutations."""
        original = {"items": ["a"]}
        backend = SimpleMockBackend(initial_state=original)
        backend.execute(["add", "b"])

        assert original == {"items": ["a"]}
        assert backend.get_state_snapshot()["items"] == ["a", "b"]


class TestStateDiffMatch:
    def test_state_diff_match(self) -> None:
        """Exact match between state and expected yields score=1.0."""
        backend = SimpleMockBackend(initial_state={"items": ["x", "y"]})
        diff = backend.diff({"items": ["x", "y"]})

        assert diff.matches is True
        assert diff.score == 1.0
        assert diff.missing == []
        assert diff.extra == []
        assert diff.mismatched == []


class TestStateDiffPartial:
    def test_state_diff_partial_missing_key(self) -> None:
        """Missing key in state results in partial score."""
        backend = SimpleMockBackend(initial_state={"items": ["x"]})
        diff = backend.diff({"items": ["x"], "count": 1})

        assert diff.matches is False
        assert 0.0 < diff.score < 1.0
        assert "count" in diff.missing

    def test_state_diff_partial_wrong_value(self) -> None:
        """Mismatched scalar value results in partial score."""
        backend = SimpleMockBackend(initial_state={"count": 5})
        diff = backend.diff({"count": 10})

        assert diff.matches is False
        assert diff.score == 0.0
        assert "count" in diff.mismatched

    def test_state_diff_partial_list_mismatch(self) -> None:
        """Missing items in a list result in partial score."""
        backend = SimpleMockBackend(initial_state={"items": ["a"]})
        diff = backend.diff({"items": ["a", "b"]})

        assert diff.matches is False
        assert 0.0 < diff.score < 1.0

    def test_state_diff_nested(self) -> None:
        """Nested dict differences are reported with dotted paths."""
        backend = SimpleMockBackend(
            initial_state={"repo": {"name": "test", "stars": 5}}
        )
        diff = backend.diff({"repo": {"name": "test", "stars": 10}})

        assert diff.matches is False
        assert any("repo.stars" in m for m in diff.mismatched)


class TestUnknownCommand:
    def test_unknown_command_returns_error(self) -> None:
        """Unknown command returns exit_code=1 and stderr with error."""
        backend = SimpleMockBackend(initial_state={})
        result = backend.execute(["nonexistent"])

        assert result.exit_code == 1
        assert "Unknown" in result.stderr


class TestReset:
    def test_reset(self) -> None:
        """After mutations, reset restores original state."""
        backend = SimpleMockBackend(initial_state={"items": []})
        backend.execute(["add", "a"])
        backend.execute(["add", "b"])
        assert backend.get_state_snapshot()["items"] == ["a", "b"]

        backend.reset()

        assert backend.get_state_snapshot()["items"] == []
        assert backend.get_action_log() == []

    def test_reset_does_not_affect_original(self) -> None:
        """Reset restores state without linking to original dict."""
        original = {"items": ["x"]}
        backend = SimpleMockBackend(initial_state=original)
        backend.execute(["add", "y"])
        backend.reset()

        # Mutate post-reset state
        backend.execute(["add", "z"])
        assert original == {"items": ["x"]}


class TestDeepDiff:
    def test_scalar_match(self) -> None:
        diff = _deep_diff(42, 42)
        assert diff.matches is True
        assert diff.score == 1.0

    def test_scalar_mismatch(self) -> None:
        diff = _deep_diff(42, 99, path="val")
        assert diff.matches is False
        assert diff.score == 0.0
        assert "val" in diff.mismatched

    def test_dict_extra_key_in_actual(self) -> None:
        """Extra keys in actual (not in expected) are reported but don't reduce score."""
        diff = _deep_diff({"a": 1, "b": 2}, {"a": 1})
        assert diff.matches is True
        assert diff.score == 1.0
        assert "b" in diff.extra

    def test_dict_missing_key(self) -> None:
        diff = _deep_diff({"a": 1}, {"a": 1, "b": 2})
        assert diff.matches is False
        assert diff.score == 0.5
        assert "b" in diff.missing

    def test_list_order_independent(self) -> None:
        """List comparison is order-independent."""
        diff = _deep_diff([1, 2, 3], [3, 1, 2])
        assert diff.matches is True
        assert diff.score == 1.0

    def test_list_partial(self) -> None:
        """Only some expected items present in actual list."""
        diff = _deep_diff(["a", "b"], ["a", "b", "c"])
        assert diff.matches is False
        assert diff.score == pytest.approx(2.0 / 3.0)

    def test_empty_expected(self) -> None:
        """Empty expected dict is a trivial match."""
        diff = _deep_diff({"a": 1}, {})
        assert diff.matches is True
        assert diff.score == 1.0


class TestActionDataclass:
    def test_action_fields(self) -> None:
        result = MockResult(stdout="ok", stderr="", exit_code=0)
        action = Action(command=["test"], result=result, timestamp_ms=100)
        assert action.command == ["test"]
        assert action.result.stdout == "ok"
        assert action.timestamp_ms == 100

    def test_action_default_timestamp(self) -> None:
        result = MockResult(stdout="", stderr="", exit_code=0)
        action = Action(command=["x"], result=result)
        assert action.timestamp_ms == 0


class TestMockResult:
    def test_mock_result_fields(self) -> None:
        r = MockResult(stdout="hello", stderr="warn", exit_code=2)
        assert r.stdout == "hello"
        assert r.stderr == "warn"
        assert r.exit_code == 2
