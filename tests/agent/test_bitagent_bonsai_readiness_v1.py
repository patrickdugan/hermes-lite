import hashlib
import json
from pathlib import Path


ROOT = Path("reports/receipts/bitagent_bonsai_role_adapter_readiness_v1")
RECEIPT = ROOT / "readiness.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_readiness_receipt_recomputes_promotion_deficits():
    receipt = json.loads(RECEIPT.read_text())
    floors = receipt["promotion_floors"]
    counts = receipt["corpus"]["counts_by_role"]
    splits = receipt["corpus"]["counts_by_role_and_split"]

    expected = {}
    for role, count in counts.items():
        expected[role] = {
            "total": max(0, floors["total_per_role"] - count),
            "validation": max(
                0,
                floors["validation_per_role"] - splits[role]["validation"],
            ),
            "test": max(0, floors["test_per_role"] - splits[role]["test"]),
        }

    assert expected == receipt["promotion_deficits"]
    assert sum(item["total"] for item in expected.values()) == 717
    assert receipt["total_additional_reviewed_examples_required"] == 717
    assert receipt["minimum_additional_held_examples_required"] == 152
    assert receipt["gates"]["promotion_corpus_ready"] is False
    assert receipt["gates"]["training_authorized"] is False


def test_operator_snapshots_prove_no_weight_loading():
    receipt = json.loads(RECEIPT.read_text())
    assert len(receipt["operator_validations"]) == 4
    for role, operator in receipt["operator_validations"].items():
        summary_path = ROOT / operator["summary_path"]
        manifest_path = ROOT / operator["manifest_path"]
        assert _sha256(summary_path) == operator["summary_sha256"]
        assert _sha256(manifest_path) == operator["manifest_sha256"]
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        assert summary["status"] == "validated"
        assert summary["training_started"] is False
        assert summary["weight_loading_started"] is False
        assert manifest["role"] == role
        assert manifest["weight_loading_enabled"] is False
        assert manifest["corpus"]["examples_sha256"] == receipt["corpus"]["sha256"]


def test_config_and_seed_corpus_contract_are_pinned():
    receipt = json.loads(RECEIPT.read_text())
    config_path = Path(receipt["config"]["path"])
    assert _sha256(config_path) == receipt["config"]["sha256"]
    assert receipt["corpus"]["examples"] == 83
    assert sum(receipt["corpus"]["counts_by_role"].values()) == 83
    assert receipt["corpus"]["raw_transcripts_included"] is False
    assert receipt["corpus"]["secret_values_detected"] is False
    assert receipt["corpus"]["unauthorized_effects_detected"] is False
    assert receipt["gates"]["seed_corpus_contract_valid"] is True
    assert receipt["gates"]["all_role_operators_validate_without_weights"] is True
