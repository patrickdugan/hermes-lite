import argparse
import json
from pathlib import Path

import pytest

from agent.bitagent_mesh_benchmark_v0 import (
    ARMS,
    RAMClassifier,
    _augment_semantic_ldt,
    candidate_from_labels,
    candidate_signature,
    dense_features,
    descriptor,
    evaluate,
    label_catalog,
    lexical_labels,
    make_trm,
    read_jsonl,
    train_models,
    train_ram,
)
from agent.bitagent_control_mesh_v0 import canonical_hash, load_protocol, validate_ldt_candidate
from agent.bitagent_role_adapters import load_config, route_role


REGISTRATION = Path("evals/registered/bitagent_hermes_control_mesh_v0")


def test_ram_round_trip_uses_only_registered_train_rows():
    rows = read_jsonl(REGISTRATION / "train_cases.jsonl")
    policy = train_ram(rows, epochs=10, seed=26072026)
    payload = policy.to_dict("registration", 26072026)
    restored = RAMClassifier.from_dict(payload)
    assert len(rows) == 40
    assert restored.labels == policy.labels
    assert all(row["split"] == "train" for row in rows)


def test_tiny_recursive_classifier_stays_below_parameter_cap():
    labels = label_catalog(read_jsonl(REGISTRATION / "train_cases.jsonl"))
    model = make_trm(len(labels))
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000
    assert tuple(dense_features("deposit bitcoin").shape) == (96,)


def test_lexical_observable_policy_solves_registered_held_cases():
    held = read_jsonl(REGISTRATION / "held_cases.jsonl")
    assert len(held) == 10
    for row in held:
        assert lexical_labels(row["task_card"]) == row["labels"]


def test_ldt_rejects_learned_candidate_that_misses_held_precondition():
    protocol = load_protocol(REGISTRATION / "protocol.json")
    role_config = load_config(protocol["role_adapter_config"])
    held = next(row for row in read_jsonl(REGISTRATION / "held_cases.jsonl") if row["id"] == "strategy-09")
    task = held["task_card"]
    route = {"role": route_role(task), "owner": "deterministic_host"}
    evidence = {
        "state_sha256": "1" * 64,
        "tool_contracts_sha256": "2" * 64,
        "route_sha256": canonical_hash(route),
    }
    wrong = {
        "intent": "starter_strategy",
        "tool": "bitagent.strategy.simulate",
        "missing": None,
        "prohibited": False,
        "role": "utxo_tradelayer_specialist",
    }
    candidate = candidate_from_labels(wrong, task, evidence)
    decision = validate_ldt_candidate(role_config, protocol, task, candidate, evidence)
    assert decision["decision"] == "candidate_valid"
    decision = _augment_semantic_ldt(decision, candidate, lexical_labels(task))
    assert decision["decision"] == "rejected"
    assert "host_precondition_mismatch" in decision["reason_codes"]
    assert candidate_signature(candidate)["missing"] is None
    assert lexical_labels(task)["missing"] == "confirmed_deposit"


def test_training_refuses_uncapped_execution(monkeypatch, tmp_path):
    monkeypatch.delenv("LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE", raising=False)
    args = argparse.Namespace(
        registration_dir=str(REGISTRATION),
        output_dir=str(tmp_path),
        steps=1,
        ram_epochs=1,
        seed=26072026,
        ram_cap_mb=2048,
        io_cap_mb_s=50,
        wall_seconds=900,
        checkpoint_steps=1,
    )
    with pytest.raises(SystemExit, match="Refusing uncapped"):
        train_models(args)


def test_wrapper_exposes_registered_bitagent_training_kind():
    wrapper = Path("scripts/run_capped_lean_control_mesh_v1.ps1").read_text()
    assert '"bitagent_mesh_v0"' in wrapper
    assert "agent.bitagent_mesh_benchmark_v0" in wrapper
    assert "post_run_int3_cleanup.ps1" in wrapper
    assert set(ARMS) == {"lexical_ldt", "ram_ldt", "trm_ldt", "adaptive_mesh_ldt"}


def test_controller_addendum_is_frozen_and_binds_implementation():
    addendum = json.loads((REGISTRATION / "controller_addendum_v0_1.json").read_text())
    assert addendum["status"] == "frozen_before_model_outcomes"
    assert addendum["adaptive"]["threshold"] == 0.6
    assert addendum["training"]["held_case_reads_per_training_run"] == 0
    for binding in ("benchmark_module", "cap_wrapper"):
        item = addendum["source_bindings"][binding]
        import hashlib

        assert hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest() == item["sha256"]
