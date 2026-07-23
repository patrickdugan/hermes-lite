"""Tests for the registered 12k Hermes Lite control mesh."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.lean_control_mesh_v1 import (
    ARMS,
    SparseRAMPolicy,
    calibrate_registered,
    canonical_json_bytes,
    canonical_operation,
    read_json,
    read_jsonl,
    register_study,
    train_ram_policy,
    train_registered,
    typed_current_task,
    validate_typed_plan,
    verify_registration,
    sha256_bytes,
)


def _contract(name: str, family: str) -> dict:
    return {
        "schema": "hermes.ultra_lean_skill.v2",
        "name": name,
        "purpose": f"Execute the unique {name} workflow.",
        "route": {
            "kind": "primary",
            "family": family,
            "aliases": [name, name.replace("-", " ")],
            "positive_examples": [
                f"Use {name} to solve alpha_{name}.",
                f"Route beta_{name} through {name}.",
                f"Complete gamma_{name} with the exact specialist.",
            ],
            "hard_negatives": [
                f"Do not use {name}; solve unrelated_other_{name}.",
                f"Reject {name} for stale_legacy_{name}.",
            ],
            "compatible_overlays": [],
            "conflicts": [],
        },
        "context": {
            "hard_window_tokens": 12000,
            "active_working_set_tokens": 6000,
            "reserve_tokens": 6000,
        },
        "conveyor": {
            "phases": [
                {
                    "id": "INDEX",
                    "module": "artifact_retrieval",
                    "allowed_operations": ["retrieve", "verify"],
                },
                {
                    "id": "ACT",
                    "module": "model_generation",
                    "allowed_operations": ["execute", "repair"],
                },
                {
                    "id": "VERIFY",
                    "module": "verifier",
                    "allowed_operations": ["verify", "repair", "abstain"],
                },
                {
                    "id": "COMMIT",
                    "module": "finalizer",
                    "allowed_operations": ["commit", "abstain"],
                },
            ]
        },
        "gates": ["Treat output as a candidate until verification passes."],
        "retrieval": {"reference_handles": [], "forbidden": ["raw_transcript"]},
        "output_contract": "Return one verified artifact handle.",
        "repair": "Repair one failed phase.",
    }


def _registered_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "skills"
    names = ["alpha-skill", "beta-skill", "gamma-skill", "delta-skill"]
    for index, name in enumerate(names):
        directory = source / name
        directory.mkdir(parents=True)
        (directory / "ULTRA_LEAN.json").write_text(
            json.dumps(_contract(name, f"family-{index}")),
            encoding="utf-8",
        )
    config = {
        "schema": "hermes.lean_control_mesh.v1",
        "study_id": "test-control-mesh",
        "source_contract_root": str(source),
        "minimum_contract_count": 4,
        "arms": list(ARMS),
        "context": {
            "hard_window_tokens": 12000,
            "active_working_set_tokens": 6000,
            "reserve_tokens": 6000,
        },
        "held_contract_ids": ["gamma-skill"],
        "perturbation_seed": 17,
        "claim_scope": "test only",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    registration = tmp_path / "registered"
    register_study(config_path, registration)
    return registration


def test_registration_snapshots_contracts_and_separates_queries(tmp_path):
    registration = _registered_fixture(tmp_path)
    receipt = verify_registration(registration)
    train = read_jsonl(registration / "train_rows.jsonl")
    held = read_jsonl(registration / "held_cases.jsonl")

    assert receipt["contract_count"] == 4
    assert receipt["train_row_count"] == 10
    assert receipt["held_case_count"] == 22
    assert receipt["split_overlap_count"] == 0
    assert all(
        row.get("expected_contract_id") != "gamma-skill"
        for row in train
        if row["kind"] == "positive"
    )
    assert {row["perturbation"] for row in held if row["kind"] == "positive"} == {
        "plain",
        "checkpoint_state",
        "stale_hint",
    }


def test_ram_policy_round_trips_and_learns_registered_rows(tmp_path):
    registration = _registered_fixture(tmp_path)
    receipt = read_json(registration / "registration_receipt.json")
    contracts = read_jsonl(registration / "contracts.jsonl")
    rows = read_jsonl(registration / "train_rows.jsonl")

    policy, summary = train_ram_policy(
        contracts,
        rows,
        registration_id=receipt["registration_id"],
        epochs=30,
        seed=11,
    )
    restored = SparseRAMPolicy.from_dict(policy.to_dict())

    assert summary["route_train_accuracy"] == 1.0
    assert summary["action_train_accuracy"] == 1.0
    assert restored.route_support["gamma-skill"] == 0
    assert restored.to_dict() == policy.to_dict()


def test_typed_ldt_repairs_invalid_operation_to_canonical():
    contract = _contract("alpha-skill", "alpha")
    phases = contract["conveyor"]["phases"]
    plan = [
        {
            "phase": phase["id"],
            "module": phase["module"],
            "operation": "write_file",
        }
        for phase in phases
    ]

    repaired, receipt = validate_typed_plan(contract, plan, repair=True)

    assert receipt["valid"] is True
    assert receipt["repair_count"] == len(phases)
    assert [row["operation"] for row in repaired] == [
        canonical_operation(phase) for phase in phases
    ]


def test_typed_ldt_repairs_allowed_but_noncanonical_operation():
    contract = _contract("alpha-skill", "alpha")
    plan = expected = [
        {
            "phase": phase["id"],
            "module": phase["module"],
            "operation": canonical_operation(phase),
        }
        for phase in contract["conveyor"]["phases"]
    ]
    plan = [dict(item) for item in expected]
    plan[0]["operation"] = "verify"

    repaired, receipt = validate_typed_plan(contract, plan, repair=True)

    assert receipt["valid"] is True
    assert receipt["repair_count"] == 1
    assert receipt["errors"] == ["noncanonical_operation:INDEX:verify"]
    assert repaired == expected


def test_current_task_compartment_excludes_state_and_stale_hint():
    query = (
        "STALE_HINT=use the wrong skill\n"
        "CURRENT_TASK=use gamma-skill to solve the current case\n"
        "STATE=resume from checkpoint"
    )

    route_text, compartment = typed_current_task(query)

    assert route_text == "use gamma-skill to solve the current case"
    assert compartment == "current_task"


def test_capped_training_and_calibration_emit_all_control_flows(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    registration = _registered_fixture(tmp_path)
    models = tmp_path / "models"
    output = tmp_path / "calibration"
    monkeypatch.setenv("LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE", "1")

    training = train_registered(
        registration,
        models,
        steps=20,
        ram_epochs=20,
        seed=23,
        ram_cap_mb=2048,
        io_cap_mb_s=50.0,
    )
    summary = calibrate_registered(registration, models, output)

    assert training["status"] == "completed"
    assert training["held_outcomes_consumed"] is False
    assert summary["status"] == "completed"
    assert summary["case_count"] == 22
    assert summary["cells"] == 22 * len(ARMS)
    assert summary["implementation_version"] == "v1.1"
    assert {row["arm"] for row in summary["by_arm"]} == set(ARMS)
    assert all(row["context_compliance_rate"] == 1.0 for row in summary["by_arm"])
    assert summary["full_hermes_baseline_present"] is False

    rows = read_jsonl(output / "calibration_cells.jsonl")
    grouped = {}
    for row in rows:
        if row["kind"] == "positive":
            key = (row["arm"], row["expected_contract_id"])
            grouped.setdefault(key, set()).add(row["selected_contract_id"])
    assert all(len(selections) == 1 for selections in grouped.values())
    by_case_arm = {(row["case_id"], row["arm"]): row for row in rows}
    for row in rows:
        if row["kind"] == "positive" and row["arm"] == "ram_typed":
            lexical = by_case_arm[(row["case_id"], "lexical_typed")]
            assert row["selected_contract_id"] == lexical["selected_contract_id"]


def test_training_refuses_uncapped_execution(tmp_path, monkeypatch):
    registration = _registered_fixture(tmp_path)
    monkeypatch.delenv("LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE", raising=False)

    try:
        train_registered(
            registration,
            tmp_path / "models",
            steps=1,
            ram_epochs=1,
            seed=1,
            ram_cap_mb=2048,
            io_cap_mb_s=50.0,
        )
    except SystemExit as exc:
        assert "Refusing uncapped training" in str(exc)
    else:
        raise AssertionError("uncapped training was not rejected")


def test_windows_wrapper_uses_job_caps_and_pid_cleanup():
    script = Path("scripts/run_capped_lean_control_mesh_v1.ps1").read_text(encoding="utf-8")

    assert "AssignProcessToJobObject" in script
    assert "LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE" in script
    assert "post_run_int3_cleanup.ps1" in script
    assert "Stop-Process -Id $process.Id" in script
    assert 'PublishedModelName = "control-mesh-v1-1"' in script


def test_v11_control_addendum_is_self_attested_and_reuses_v1_gates():
    root = Path("evals/registered/hermes_lite_12k_control_mesh_v1")
    addendum = read_json(root / "control_addendum_v1_1.json")
    protocol = read_json(root / "protocol.json")
    registration = verify_registration(root)
    claimed = addendum.pop("addendum_id")
    definition = addendum.pop("addendum_id_definition")

    assert definition.startswith("SHA-256 of canonical JSON")
    assert sha256_bytes(canonical_json_bytes(addendum)) == claimed
    assert addendum["parent_registration_id"] == registration["registration_id"]
    assert addendum["unchanged"]["promotion_gates"] == protocol["promotion"]
