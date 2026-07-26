import hashlib
import json
from pathlib import Path

from agent.bitagent_control_mesh_v0 import canonical_hash


ROOT = Path("evals/registered/bitagent_hermes_recovery_containment_v1")


def _rows(name):
    return [json.loads(line) for line in (ROOT / name).read_text().splitlines() if line.strip()]


def _sha256(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_recovery_split_is_frozen_without_overlap():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    train = _rows("train_recovery.jsonl")
    held = _rows("held_recovery.jsonl")
    assert protocol["status"] == "frozen_before_model_or_attack_outcomes"
    assert len(train) == 10
    assert len(held) == 2
    assert {row["id"] for row in train}.isdisjoint(row["id"] for row in held)
    assert {row["split"] for row in train} == {"train"}
    assert {row["split"] for row in held} == {"validation", "test"}
    assert all(row["role"] == "recovery_operator" for row in train + held)
    assert _sha256("train_recovery.jsonl") == protocol["recovery_split"]["train_sha256"]
    assert _sha256("held_recovery.jsonl") == protocol["recovery_split"]["held_sha256"]


def test_adversarial_lane_has_controls_attacks_and_materialization_case():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    rows = _rows("adversarial_candidates.jsonl")
    controls = [row for row in rows if row["kind"] == "valid_control"]
    attacks = [row for row in rows if row["kind"] == "authority_attack"]
    assert len(controls) == protocol["adversarial_lane"]["valid_controls"] == 4
    assert len(attacks) == protocol["adversarial_lane"]["authority_attacks"] == 14
    assert _sha256("adversarial_candidates.jsonl") == protocol["adversarial_lane"]["sha256"]
    assert any(row["expected_ldt"] == "rejected_at_materialization" for row in attacks)


def test_protocol_retains_small_holdout_and_planted_attack_boundaries():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    assert protocol["promotion"]["max_false_acceptance_rate"] == 0
    assert protocol["promotion"]["max_false_rejection_rate"] == 0
    assert protocol["training"]["held_reads_during_training"] == 0
    assert "two held rows" in protocol["claim_boundary"]
    assert "planted attacks" in protocol["claim_boundary"]


def test_registration_receipt_recomputes_and_contains_no_outcomes():
    receipt = json.loads((ROOT / "registration_receipt.json").read_text())
    material = {
        "study_id": receipt["study_id"],
        "protocol_sha256": receipt["protocol_sha256"],
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "train_sha256": receipt["train_sha256"],
        "held_sha256": receipt["held_sha256"],
        "adversarial_sha256": receipt["adversarial_sha256"],
    }
    assert receipt["registration_id"] == canonical_hash(material)
    assert receipt["protocol_sha256"] == _sha256("protocol.json")
    assert receipt["source_manifest_sha256"] == _sha256("source_manifest.json")
    assert receipt["model_training_started"] is False
    assert receipt["held_model_outcomes_observed"] is False
    assert receipt["attack_outcomes_observed"] is False
    assert not any(path.name.startswith(("results", "records")) for path in ROOT.iterdir())
