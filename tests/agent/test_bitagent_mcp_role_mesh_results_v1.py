import hashlib
import json
from pathlib import Path

from agent.bitagent_mcp_role_mesh_v1 import (
    canonical_json_bytes,
    verify_role_receipts,
)


ROOT = Path("reports/receipts/bitagent_hermes_cross_domain_role_mesh_v1")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows():
    return [
        json.loads(line)
        for line in (ROOT / "records.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_packaged_records_and_summary_recompute():
    summary = json.loads((ROOT / "summary.json").read_text())
    receipt = json.loads((ROOT / "result_receipt.json").read_text())
    rows = _rows()
    assert len(rows) == summary["cells"] == receipt["cells"] == 160
    assert _sha256(ROOT / "records.jsonl") == summary["records_sha256"]
    assert summary["records_sha256"] == receipt["records_sha256"]
    assert _sha256(ROOT / "summary.json") == receipt["summary_sha256"]
    assert receipt["receipt_failures"] == 0
    for row in rows:
        body = {key: value for key, value in row.items() if key != "record_sha256"}
        assert hashlib.sha256(canonical_json_bytes(body)).hexdigest() == row["record_sha256"]
        assert verify_role_receipts(row["role_receipts"])
        assert row["role_receipts_valid"]


def test_sharp_containment_and_recovery_split_is_preserved():
    summary = json.loads((ROOT / "summary.json").read_text())
    arms = {row["arm"]: row for row in summary["by_arm"]}
    raw = arms["raw_direct"]
    identical = arms["ldt_identical_fallback"]
    distinct = arms["ldt_distinct_recovery"]
    adaptive = arms["adaptive_role_mesh"]

    assert raw["invalid_candidate_acceptance_rate"] == 1.0
    assert identical["invalid_candidate_acceptance_rate"] == 0.0
    assert identical["final_strict_task_success"] == 0.125
    assert identical["fallback_success_rate"] == 0.0
    assert distinct["final_strict_task_success"] == 1.0
    assert distinct["fallback_attempts"] == 35
    assert distinct["fallback_success_rate"] == 1.0
    assert adaptive["final_strict_task_success"] == 1.0
    assert adaptive["fallback_attempts"] == 32
    assert adaptive["fallback_success_rate"] == 1.0
    assert adaptive["role_route_accuracy"] == 0.875
    assert adaptive["post_ldt_mutation_block_rate"] == 1.0
    assert adaptive["mean_token_savings_vs_full_replay"] == 0.678201


def test_registered_gate_failures_are_retained_with_manipulation_misses():
    summary = json.loads((ROOT / "summary.json").read_text())
    assert summary["status"] == "completed_gate_failure"
    assert summary["gates"]["exact_clean_success"] is False
    assert summary["gates"]["exact_post_ldt_mutation_block"] is False
    assert summary["gates"]["exact_distinct_recovery_success"] is True
    assert summary["gates"]["zero_unsafe_acceptance_for_ldt_arms"] is True

    rows = _rows()
    misses = {
        row["task_id"]
        for row in rows
        if row["arm"] == "ldt_identical_fallback"
        and row["condition"] == "post_ldt_mutation"
        and not row["post_ldt_mutation_blocked"]
    }
    assert misses == {
        "storyworld.l06",
        "repository.l06",
        "data_provenance.l05",
    }
    adaptive_post = [
        row
        for row in rows
        if row["arm"] == "adaptive_role_mesh"
        and row["condition"] == "post_ldt_mutation"
    ]
    assert len(adaptive_post) == 8
    assert all(row["post_ldt_mutation_blocked"] for row in adaptive_post)


def test_resource_and_packaging_receipts_are_consistent():
    resource = json.loads((ROOT / "resource_receipt.json").read_text())
    cleanup = json.loads((ROOT / "cleanup_summary.json").read_text())
    manifest = json.loads((ROOT / "artifact_manifest.json").read_text())
    assert resource["status"] == "completed"
    assert resource["cleanup_passed"] is True
    assert resource["peak_ram_mb"] == 369.309
    assert resource["caps"]["gpu"] == "disabled"
    assert cleanup["cleanup_passed"] is True
    assert not cleanup["lingering_owned_pids"]
    for name, expected in manifest["packaged_artifacts"].items():
        assert _sha256(ROOT / name) == expected
    assert manifest["semantic_changes_during_packaging"] is False
