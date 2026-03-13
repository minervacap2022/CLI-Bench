"""Tests for FictionalMockBackend — generic CRUD mock for fictional CLI tools."""

import json

import pytest

from cli_bench.mock_backends.fictional import FictionalMockBackend


@pytest.fixture
def kforge_backend() -> FictionalMockBackend:
    """Backend simulating kforge with artifacts, pipelines, and registries."""
    return FictionalMockBackend(
        initial_state={
            "artifacts": [
                {
                    "id": "art-001",
                    "name": "api-server",
                    "namespace": "team-backend",
                    "type": "docker",
                    "version": "1.2.0",
                    "status": "published",
                },
                {
                    "id": "art-002",
                    "name": "web-client",
                    "namespace": "team-frontend",
                    "type": "npm",
                    "version": "3.0.1",
                    "status": "published",
                },
                {
                    "id": "art-003",
                    "name": "ml-model",
                    "namespace": "team-backend",
                    "type": "binary",
                    "version": "0.9.0",
                    "status": "staging",
                },
            ],
            "pipelines": [
                {
                    "id": "pipe-001",
                    "artifact": "art-001",
                    "target": "staging",
                    "status": "completed",
                },
            ],
            "registries": [
                {
                    "id": "reg-001",
                    "namespace": "team-backend",
                    "description": "Backend team artifacts",
                    "visibility": "internal",
                },
            ],
        },
        tool_name="kforge",
    )


class TestGenericList:
    def test_list_all_artifacts(self, kforge_backend: FictionalMockBackend) -> None:
        """Lists all items from a state key."""
        result = kforge_backend.execute(["kforge", "artifact", "list"])
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 3

    def test_list_with_filter(self, kforge_backend: FictionalMockBackend) -> None:
        """Filters by --<field> <value>."""
        result = kforge_backend.execute(
            ["kforge", "artifact", "list", "--namespace", "team-backend"]
        )
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 2
        for item in items:
            assert item["namespace"] == "team-backend"

    def test_list_with_type_filter(self, kforge_backend: FictionalMockBackend) -> None:
        """Filters artifacts by type."""
        result = kforge_backend.execute(
            ["kforge", "artifact", "list", "--type", "docker"]
        )
        assert result.exit_code == 0
        items = json.loads(result.stdout)
        assert len(items) == 1
        assert items[0]["name"] == "api-server"


class TestGenericGet:
    def test_get_by_id(self, kforge_backend: FictionalMockBackend) -> None:
        """Gets a single item by --id."""
        result = kforge_backend.execute(
            ["kforge", "artifact", "inspect", "--id", "art-001"]
        )
        assert result.exit_code == 0
        item = json.loads(result.stdout)
        assert item["id"] == "art-001"
        assert item["name"] == "api-server"

    def test_get_not_found(self, kforge_backend: FictionalMockBackend) -> None:
        """Returns error for nonexistent ID."""
        result = kforge_backend.execute(
            ["kforge", "artifact", "inspect", "--id", "art-999"]
        )
        assert result.exit_code == 1
        assert "not found" in result.stderr

    def test_get_pipeline_status(self, kforge_backend: FictionalMockBackend) -> None:
        """Gets pipeline by --id using 'status' action (maps to generic get)."""
        result = kforge_backend.execute(
            ["kforge", "pipeline", "status", "--id", "pipe-001"]
        )
        assert result.exit_code == 0
        item = json.loads(result.stdout)
        assert item["id"] == "pipe-001"
        assert item["status"] == "completed"


class TestGenericCreate:
    def test_create_artifact(self, kforge_backend: FictionalMockBackend) -> None:
        """Creates a new item with auto-generated ID."""
        result = kforge_backend.execute(
            [
                "kforge", "artifact", "create",
                "--name", "new-service",
                "--namespace", "team-platform",
                "--type", "docker",
            ]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["id"] == "art-4"
        assert created["name"] == "new-service"
        assert created["namespace"] == "team-platform"

        # Verify state mutated
        snapshot = kforge_backend.get_state_snapshot()
        assert len(snapshot["artifacts"]) == 4

    def test_create_registry(self, kforge_backend: FictionalMockBackend) -> None:
        """Creates a registry in a different state key."""
        result = kforge_backend.execute(
            [
                "kforge", "registry", "create",
                "--namespace", "ml-team",
                "--description", "ML artifacts",
            ]
        )
        assert result.exit_code == 0
        created = json.loads(result.stdout)
        assert created["id"] == "reg-2"
        assert created["namespace"] == "ml-team"


class TestGenericUpdate:
    def test_update_artifact(self, kforge_backend: FictionalMockBackend) -> None:
        """Updates fields on an existing item."""
        result = kforge_backend.execute(
            [
                "kforge", "artifact", "update",
                "--id", "art-001",
                "--version", "1.3.0",
                "--status", "promoted",
            ]
        )
        assert result.exit_code == 0
        updated = json.loads(result.stdout)
        assert updated["version"] == "1.3.0"
        assert updated["status"] == "promoted"
        assert updated["name"] == "api-server"  # unchanged field preserved

    def test_update_not_found(self, kforge_backend: FictionalMockBackend) -> None:
        """Update nonexistent item returns error."""
        result = kforge_backend.execute(
            ["kforge", "artifact", "update", "--id", "art-999", "--version", "2.0"]
        )
        assert result.exit_code == 1
        assert "not found" in result.stderr


class TestGenericDelete:
    def test_delete_artifact(self, kforge_backend: FictionalMockBackend) -> None:
        """Deletes an item by --id."""
        result = kforge_backend.execute(
            ["kforge", "artifact", "delete", "--id", "art-002"]
        )
        assert result.exit_code == 0
        deleted = json.loads(result.stdout)
        assert deleted["deleted"] == "art-002"

        # Verify state mutated
        snapshot = kforge_backend.get_state_snapshot()
        assert len(snapshot["artifacts"]) == 2
        ids = {a["id"] for a in snapshot["artifacts"]}
        assert "art-002" not in ids

    def test_delete_not_found(self, kforge_backend: FictionalMockBackend) -> None:
        """Delete nonexistent item returns error."""
        result = kforge_backend.execute(
            ["kforge", "artifact", "delete", "--id", "art-999"]
        )
        assert result.exit_code == 1
        assert "not found" in result.stderr


class TestUnknownCommands:
    def test_unknown_tool(self, kforge_backend: FictionalMockBackend) -> None:
        """Wrong tool name returns error."""
        result = kforge_backend.execute(["wrongtool", "artifact", "list"])
        assert result.exit_code == 1

    def test_unknown_resource(self, kforge_backend: FictionalMockBackend) -> None:
        """Unknown resource returns error."""
        result = kforge_backend.execute(["kforge", "nonexistent", "list"])
        assert result.exit_code == 1

    def test_unknown_action(self, kforge_backend: FictionalMockBackend) -> None:
        """Unknown action returns error."""
        result = kforge_backend.execute(["kforge", "artifact", "explode"])
        assert result.exit_code == 1

    def test_too_few_args(self, kforge_backend: FictionalMockBackend) -> None:
        """Too few args returns usage error."""
        result = kforge_backend.execute(["kforge", "artifact"])
        assert result.exit_code == 1


class TestCustomHandlers:
    def test_custom_handler_overrides_generic(self) -> None:
        """Custom handler takes precedence over generic CRUD."""

        def custom_list(backend: FictionalMockBackend, args: list[str]) -> "MockResult":
            from cli_bench.mock_backends.base import MockResult
            return MockResult(
                stdout=json.dumps({"custom": True}),
                stderr="",
                exit_code=0,
            )

        backend = FictionalMockBackend(
            initial_state={"items": []},
            tool_name="custom",
            command_handlers={"item list": custom_list},
        )

        result = backend.execute(["custom", "item", "list"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["custom"] is True


class TestResetAndDiff:
    def test_reset_restores_state(self, kforge_backend: FictionalMockBackend) -> None:
        """Reset restores initial state after mutations."""
        kforge_backend.execute(
            ["kforge", "artifact", "delete", "--id", "art-001"]
        )
        assert len(kforge_backend.state["artifacts"]) == 2

        kforge_backend.reset()
        assert len(kforge_backend.state["artifacts"]) == 3

    def test_action_log(self, kforge_backend: FictionalMockBackend) -> None:
        """Actions are recorded in the log."""
        kforge_backend.execute(["kforge", "artifact", "list"])
        kforge_backend.execute(
            ["kforge", "artifact", "inspect", "--id", "art-001"]
        )

        log = kforge_backend.get_action_log()
        assert len(log) == 2
        assert log[0].command == ["kforge", "artifact", "list"]
        assert log[1].command == ["kforge", "artifact", "inspect", "--id", "art-001"]
