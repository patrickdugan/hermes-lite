import json
from pathlib import Path

from agent.bitagent_mcp_role_mesh_v1 import (
    MATRIX_SCHEMA,
    verify_registration,
)


ROOT = Path("evals/registered/bitagent_hermes_cross_domain_role_mesh_v1")


def test_registration_recomputes_before_role_mesh_outcomes():
    receipt = verify_registration(ROOT)
    assert receipt["status"] == "registered_no_role_mesh_outcomes"
    assert receipt["held_source_tasks"] == 8
    assert receipt["fault_conditions"] == 5
    assert receipt["matrix_rows"] == 40
    assert len(receipt["arms"]) == 4
    assert receipt["domains"] == [
        "data_provenance",
        "logic",
        "repository",
        "storyworld",
    ]
    assert "blind task generalization" in receipt["claim_scope"]


def test_matrix_has_exact_crossing_and_no_outcomes():
    rows = [
        json.loads(line)
        for line in (ROOT / "matrix.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len({row["matrix_id"] for row in rows}) == 40
    assert {row["schema"] for row in rows} == {MATRIX_SCHEMA}
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task_id"], set()).add(row["condition"])
        assert "actual" not in row
        assert "passed" not in row
        assert row["split"] == "held_perturbation"
    assert len(by_task) == 8
    assert all(len(conditions) == 5 for conditions in by_task.values())


def test_controller_artifacts_are_snapshotted():
    receipt = verify_registration(ROOT)
    artifacts = ROOT / "controller_artifacts"
    assert (artifacts / "base_ram.json").is_file()
    assert (artifacts / "trm_router.pt").is_file()
    assert (artifacts / "domain_ram.json").is_file()
    assert receipt["component_hashes"]["trm_router_sha256"] == (
        "ce8fc00f23830b9650ab06044667f65af751ff467ffabfb6434ab5f5f20b6543"
    )
