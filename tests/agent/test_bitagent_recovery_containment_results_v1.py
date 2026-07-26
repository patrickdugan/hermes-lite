import hashlib
import json
from pathlib import Path


ROOT = Path("reports/receipts/bitagent_hermes_recovery_containment_v1")


def test_attack_result_recomputes_from_exact_records():
    result = json.loads((ROOT / "attack_results.json").read_text())
    records_path = ROOT / "attack_records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    assert hashlib.sha256(records_path.read_bytes()).hexdigest() == result["records_sha256"]
    assert len(records) == result["case_count"] == 18
    controls = [row for row in records if row["kind"] == "valid_control"]
    attacks = [row for row in records if row["kind"] == "authority_attack"]
    assert len(controls) == result["valid_control_count"] == 4
    assert len(attacks) == result["authority_attack_count"] == 14
    assert sum(row["actual"] != "candidate_valid" for row in controls) == result["false_rejections"] == 0
    assert sum(row["actual"] == "candidate_valid" for row in attacks) == result["false_acceptances"] == 0
    assert all(row["passed"] and row["receipt_valid"] for row in records)
    assert all(result["gates"].values())


def test_post_ldt_mutation_is_blocked_at_materialization():
    records = [
        json.loads(line)
        for line in (ROOT / "attack_records.jsonl").read_text().splitlines()
        if line.strip()
    ]
    row = next(item for item in records if item["id"] == "attack.post_ldt_mutation")
    assert row["reason_codes"] == ["typed_candidate_valid"]
    assert row["actual"] == "rejected_at_materialization"
    assert "candidate changed after LDT validation" in row["materialization"]
