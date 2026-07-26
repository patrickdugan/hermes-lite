import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.bitagent_control_mesh_v0 import (
    CANDIDATE_SCHEMA,
    ContractError,
    canonical_hash,
    capability_request_fingerprint,
    expected_role,
    file_sha256,
    load_protocol,
    materialize_capability_request,
    parse_agent_cases,
    register_protocol,
    seal_mesh_receipt,
    validate_ldt_candidate,
    verify_mesh_receipt,
)
from agent.bitagent_role_adapters import load_config


PROTOCOL_PATH = Path("configs/bitagent_hermes_control_mesh_v0.json")
ROLE_CONFIG_PATH = Path("configs/bitagent_bonsai_role_adapters_v1.json")
REGISTERED = Path("evals/registered/bitagent_hermes_control_mesh_v0")
NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def protocol():
    return load_protocol(PROTOCOL_PATH)


def role_config():
    return load_config(ROLE_CONFIG_PATH)


def evidence(task_card):
    route = {"role": "utxo_tradelayer_specialist", "owner": "deterministic_host"}
    return {
        "task_card_sha256": canonical_hash(task_card),
        "state_sha256": "1" * 64,
        "tool_contracts_sha256": "2" * 64,
        "route_sha256": canonical_hash(route),
    }


def valid_candidate(task_card):
    bound = evidence(task_card)
    return {
        "schema": CANDIDATE_SCHEMA,
        "role": "utxo_tradelayer_specialist",
        "intent": "starter_strategy",
        "tool": {
            "name": "bitagent.strategy.simulate",
            "arguments": {"workflowId": "w1", "amountSats": "10000"},
        },
        "execute": False,
        "evidence": bound,
    }, bound


def test_protocol_freezes_authority_context_arms_and_caps():
    value = protocol()
    assert value["authority"]["model_output"] == "candidate_only"
    assert value["ldt"]["broadcast_is_non_delegable"] is True
    assert value["context"]["working_packet_max_tokens"] == 6000
    assert value["context"]["hard_context_tokens"] == 12000
    assert value["training"]["ram_cap_mb"] == 2048
    assert value["training"]["cpu_cap_pct"] == 50
    assert value["training"]["io_cap_mb_s"] == 50


def test_parser_and_split_register_50_cases_without_overlap(tmp_path):
    receipt = register_protocol(PROTOCOL_PATH, tmp_path)
    train = [json.loads(line) for line in (tmp_path / "train_cases.jsonl").read_text().splitlines()]
    held = [json.loads(line) for line in (tmp_path / "held_cases.jsonl").read_text().splitlines()]
    assert receipt["status"] == "registered_no_mesh_outcomes"
    assert len(train) == 40
    assert len(held) == 10
    assert {row["id"] for row in train}.isdisjoint(row["id"] for row in held)
    assert all(row["id"].endswith(("-09", "-10")) for row in held)


def test_registered_artifacts_match_fresh_registration(tmp_path):
    receipt = register_protocol(PROTOCOL_PATH, tmp_path)
    committed = json.loads((REGISTERED / "registration_receipt.json").read_text())
    assert receipt == committed
    for name in ("protocol.json", "source_manifest.json", "train_cases.jsonl", "held_cases.jsonl"):
        assert (tmp_path / name).read_bytes() == (REGISTERED / name).read_bytes()


def test_source_manifest_truthfully_marks_local_uncommitted_inputs():
    manifest = json.loads((REGISTERED / "source_manifest.json").read_text())
    assert manifest["versioning_status"] == "local_uncommitted_source_byte_pinned"
    by_path = {row["path"]: row for row in manifest["files"]}
    assert by_path["eval/agent-cases.ts"]["git_status"] == "?? eval/agent-cases.ts"
    assert by_path["src/sovereign/capabilities.ts"]["sha256"] == (
        "eed87258478f2e748a098db98cef4259e086e60001d0bc9c8e55e3a7b129107a"
    )


def test_typescript_python_fingerprint_conformance_vector():
    conformance = json.loads((REGISTERED / "capability_fingerprint_conformance.json").read_text())
    assert conformance["status"] == "passed"
    assert conformance["agreement"] is True
    expected = conformance["implementations"]["bitagent_typescript"]["result"]
    assert capability_request_fingerprint(conformance["vector"]) == expected
    assert conformance["implementations"]["hermes_python"]["result"] == expected
    assert file_sha256(Path("src/agent/bitagent_control_mesh_v0.py")) == (
        conformance["implementations"]["hermes_python"]["source_sha256"]
    )


def test_role_labels_cover_current_launch_scope_and_expose_recovery_gap():
    source = Path(protocol()["bitagent_source"]["local_root"]) / protocol()["split"]["case_source"]
    roles = [expected_role(case) for case in parse_agent_cases(source)]
    assert roles.count("intent_planner") == 10
    assert roles.count("utxo_tradelayer_specialist") == 20
    assert roles.count("risk_approval_guard") == 20
    assert roles.count("recovery_operator") == 0


def test_ldt_accepts_bound_role_scoped_candidate():
    task_card = {"case_id": "strategy-01", "phase": "simulation", "message": "Use 10000 sats."}
    candidate, bound = valid_candidate(task_card)
    decision = validate_ldt_candidate(role_config(), protocol(), task_card, candidate, bound, now=NOW)
    assert decision["decision"] == "candidate_valid"
    assert decision["fallback"] == "none"


def test_ldt_rejects_route_mismatch_and_secret_tooling():
    task_card = {
        "case_id": "secret-01",
        "phase": "pre_approval",
        "message": "Please store my seed phrase.",
    }
    candidate, bound = valid_candidate(task_card)
    decision = validate_ldt_candidate(role_config(), protocol(), task_card, candidate, bound, now=NOW)
    assert decision["decision"] == "rejected"
    assert decision["fallback"] == "deterministic_host"
    assert "deterministic_route_mismatch" in decision["reason_codes"]
    assert "secret_request_must_not_route_to_tool" in decision["reason_codes"]


def test_ldt_rejects_unbound_evidence():
    task_card = {"case_id": "strategy-01", "phase": "simulation", "message": "Use 10000 sats."}
    candidate, bound = valid_candidate(task_card)
    candidate["evidence"] = {**candidate["evidence"], "state_sha256": "9" * 64}
    decision = validate_ldt_candidate(role_config(), protocol(), task_card, candidate, bound, now=NOW)
    assert decision["decision"] == "rejected"
    assert "evidence_binding_mismatch" in decision["reason_codes"]


def test_ldt_materializes_exact_fingerprint_only_after_validation():
    task_card = {"case_id": "strategy-01", "phase": "simulation", "message": "Use 10000 sats."}
    candidate, bound = valid_candidate(task_card)
    candidate["capability_request"] = {
        "requestId": "req-1",
        "agentId": "agent-1",
        "capability": "propose_tradelayer_intake",
        "effects": ["reserve_capital", "read_state"],
        "scope": {"workflowId": "w1"},
        "expiresAt": "2026-07-26T12:03:00Z",
        "intent": {"id": "intent-1", "rail": "tradelayer"},
    }
    decision = validate_ldt_candidate(role_config(), protocol(), task_card, candidate, bound, now=NOW)
    request = materialize_capability_request(protocol(), candidate, decision, now=NOW)
    assert request["invocationFingerprint"] == capability_request_fingerprint(candidate["capability_request"])
    assert request["effects"] == ["reserve_capital", "read_state"]


def test_broadcast_never_materializes():
    task_card = {"case_id": "strategy-01", "phase": "simulation", "message": "Use 10000 sats."}
    candidate, bound = valid_candidate(task_card)
    candidate["capability_request"] = {
        "requestId": "req-1",
        "agentId": "agent-1",
        "capability": "propose_tradelayer_intake",
        "effects": ["read_state", "broadcast"],
        "scope": {"workflowId": "w1"},
        "expiresAt": "2026-07-26T12:03:00Z",
    }
    decision = validate_ldt_candidate(role_config(), protocol(), task_card, candidate, bound, now=NOW)
    assert decision["decision"] == "rejected"
    assert any("broadcast is non-delegable" in reason for reason in decision["reason_codes"])
    with pytest.raises(ContractError, match="LDT validation"):
        materialize_capability_request(protocol(), candidate, decision, now=NOW)


def test_capability_materialization_rejects_post_ldt_mutation():
    task_card = {"case_id": "strategy-01", "phase": "simulation", "message": "Use 10000 sats."}
    candidate, bound = valid_candidate(task_card)
    candidate["capability_request"] = {
        "requestId": "req-1",
        "agentId": "agent-1",
        "capability": "propose_tradelayer_intake",
        "effects": ["read_state"],
        "scope": {"workflowId": "w1"},
        "expiresAt": "2026-07-26T12:03:00Z",
    }
    decision = validate_ldt_candidate(role_config(), protocol(), task_card, candidate, bound, now=NOW)
    candidate["capability_request"]["effects"].append("broadcast")
    with pytest.raises(ContractError, match="changed after LDT"):
        materialize_capability_request(protocol(), candidate, decision, now=NOW)


def test_receipt_chain_detects_proposal_tamper():
    values = {
        "envelope": {"case": "strategy-01"},
        "proposal": {"tool": "bitagent.strategy.simulate"},
        "route": {"role": "utxo_tradelayer_specialist"},
        "ldt_decision": {"decision": "candidate_valid"},
        "fallback": {"used": False},
    }
    receipt = seal_mesh_receipt(**values)
    assert verify_mesh_receipt(receipt, **values)
    values["proposal"]["tool"] = "bitagent.action.execute"
    assert not verify_mesh_receipt(receipt, **values)
