import hashlib
import json
from pathlib import Path


ROOT = Path("reports/receipts/bitagent_hermes_control_mesh_v0")
SEEDS = (26072026, 26072027, 26072028)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_training_receipts_preserve_caps_split_and_cleanup():
    for seed in SEEDS:
        training = _json(ROOT / f"training_seed_{seed}.json")
        wrapper = _json(ROOT / f"wrapper_seed_{seed}.json")
        cleanup = _json(ROOT / f"cleanup_seed_{seed}.json")
        assert training["status"] == "completed"
        assert training["steps_completed"] == 400
        assert training["train_case_reads"] == 40
        assert training["held_case_reads"] == 0
        assert all(training["protocol_training_caps_match"].values())
        assert wrapper["status"] == "completed"
        assert wrapper["cleanup_passed"] is True
        assert cleanup["cleanup_passed"] is True
        assert not cleanup["lingering_owned_pids"]


def test_result_receipts_recompute_from_per_case_records():
    for seed in SEEDS:
        result = _json(ROOT / f"results_seed_{seed}.json")
        records_path = ROOT / f"records_seed_{seed}.jsonl"
        rows = _rows(records_path)
        assert result["records_sha256"] == _sha256(records_path)
        assert len(rows) == 40
        assert all(row["receipt_valid"] for row in rows)
        assert not any(row["unsafe_candidate_accepted"] for row in rows)
        for arm, summary in result["arms"].items():
            arm_rows = [row for row in rows if row["arm"] == arm]
            assert len(arm_rows) == 10
            assert sum(row["final_correct"] for row in arm_rows) / 10 == summary["held_task_success"]
            assert sum(row["proposal_correct"] for row in arm_rows) / 10 == summary["proposal_success"]
            assert sum(row["fallback_used"] for row in arm_rows) / 10 == summary["fallback_rate"]


def test_aggregate_matches_records_and_retains_claim_boundary():
    aggregate = _json(ROOT / "aggregate_results.json")
    rows = [row for seed in SEEDS for row in _rows(ROOT / f"records_seed_{seed}.jsonl")]
    assert len(rows) == aggregate["evaluation"]["arm_case_receipts"] == 120
    for arm, summary in aggregate["evaluation"]["arms"].items():
        arm_rows = [row for row in rows if row["arm"] == arm]
        assert sum(row["proposal_correct"] for row in arm_rows) == summary["proposal_correct"]
        assert sum(row["final_correct"] for row in arm_rows) == summary["final_correct"]
        assert sum(row["fallback_used"] for row in arm_rows) == summary["fallbacks"]
    assert "no recovery role" in aggregate["claim_boundary"]
    assert "not learned-proposer competence" in aggregate["claim_boundary"]


def test_host_baseline_is_source_bound_and_failed_orchestration_is_retained():
    passed = _json(ROOT / "host_baseline.json")
    failed = _json(ROOT / "host_baseline_orchestration_failure.json")
    assert passed["status"] == "passed"
    assert passed["critical_test_count"] == 36
    assert passed["launch_case_count"] == 50
    assert passed["launch_passed"] == 50
    assert len(set(passed["scores"].values())) == 1
    assert next(iter(passed["scores"].values())) == 1
    assert failed["status"] == "failed"
    assert failed["commands"] == {"launch": 1}
