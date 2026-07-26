import json
from pathlib import Path

from agent.bitagent_recovery_containment_v1 import MUTATIONS, candidate_for, evidence_for


ROOT = Path("evals/registered/bitagent_hermes_recovery_containment_v1")


def _rows():
    return [
        json.loads(line)
        for line in (ROOT / "adversarial_candidates.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_every_registered_mutation_has_an_implementation():
    rows = _rows()
    assert {row["mutation"] for row in rows} == MUTATIONS


def test_candidate_builder_binds_task_state_tools_and_route():
    for row in _rows():
        evidence, route = evidence_for(row["task_card"])
        candidate = candidate_for(row, evidence)
        assert candidate["role"] == row["base"]["role"]
        assert candidate["evidence"]["task_card_sha256"]
        assert candidate["evidence"]["state_sha256"]
        assert candidate["evidence"]["tool_contracts_sha256"]
        assert candidate["evidence"]["route_sha256"]
        assert route["owner"] == "deterministic_host"


def test_post_validation_mutation_starts_from_safe_effects():
    row = next(row for row in _rows() if row["mutation"] == "append_broadcast_after_validation")
    evidence, _ = evidence_for(row["task_card"])
    candidate = candidate_for(row, evidence)
    assert candidate["capability_request"]["effects"] == ["read_state"]
    assert "broadcast" not in candidate["capability_request"]["effects"]


def test_runtime_addendum_binds_evaluator_before_outcomes():
    addendum_path = ROOT / "runtime_addendum_attack_v1.json"
    addendum = json.loads(addendum_path.read_text())
    assert addendum["status"] == "frozen_before_attack_outcomes"
    assert addendum["execution"]["case_count"] == 18
    import hashlib

    assert hashlib.sha256(Path(addendum["source_bindings"]["evaluator"]["path"]).read_bytes()).hexdigest() == (
        addendum["source_bindings"]["evaluator"]["sha256"]
    )
    sidecar = (ROOT / "runtime_addendum_attack_v1.sha256").read_text().split()[0]
    assert hashlib.sha256(addendum_path.read_bytes()).hexdigest() == sidecar
