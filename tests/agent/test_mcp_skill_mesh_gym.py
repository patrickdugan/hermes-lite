"""Tests for the sealed domain-general MCP retrieval mesh gym."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from agent.mcp_skill_mesh_gym import (
    MESH_ARMS,
    _extract_json_object,
    _confirmation_arms,
    _score_live_response,
    build_mesh_packet,
    calibrate_registered,
    materialize_cases,
    read_json,
    register_study,
    score_packet,
    select_mesh_resources,
    verify_registration,
)


CONFIG_PATH = Path("configs/mcp_skill_mesh_bonsai_v0.json")


def test_materialize_cases_crosses_domains_and_complexity_levels():
    config = read_json(CONFIG_PATH)

    cases = materialize_cases(config)

    assert len(cases) == 24
    assert {case["domain"] for case in cases} == {
        "storyworld",
        "logic",
        "repository",
        "data_provenance",
    }
    assert {case["complexity"]["level"] for case in cases} == {1, 2, 3, 4, 5, 6}
    assert len({case["task_id"] for case in cases}) == 24


def test_adaptive_hybrid_retrieves_current_required_resources_only():
    config = read_json(CONFIG_PATH)
    case = next(case for case in materialize_cases(config) if case["task_id"] == "storyworld.l06")

    selected, control = select_mesh_resources(
        case,
        "adaptive_hybrid",
        static_top_k=config["static_top_k"],
        trm_top_k=config["trm_top_k"],
    )

    assert {row["resource_id"] for row in selected} == set(case["expected"]["required_resource_ids"])
    assert all(not row["stale"] for row in selected)
    assert control["contract_coverage"] == 1.0
    assert control["abstain"] is False


def test_ldt_verifier_abstains_when_typed_candidate_is_stale():
    config = read_json(CONFIG_PATH)
    case = deepcopy(next(case for case in materialize_cases(config) if case["task_id"] == "logic.l04"))
    required_type = case["task"]["interface_contract"]["required_resource_types"][0]
    for resource in case["resources"]:
        if resource["resource_type"] == required_type:
            resource["content"] = case["task"]["instruction"] if resource["stale"] else ""

    selected, control = select_mesh_resources(
        case,
        "ldt_verified",
        static_top_k=config["static_top_k"],
        trm_top_k=config["trm_top_k"],
    )

    assert all(not row["stale"] for row in selected)
    assert control["rejected_resource_ids"]
    assert control["abstain"] is True


def test_registration_is_stable_and_detects_component_tampering(tmp_path):
    first = register_study(CONFIG_PATH, tmp_path / "first")
    second = register_study(CONFIG_PATH, tmp_path / "second")

    assert first["registration_id"] == second["registration_id"]
    assert first["component_hashes"]["cases_sha256"] == second["component_hashes"]["cases_sha256"]
    assert first["split_overlap_count"] == 0
    assert verify_registration(tmp_path / "first")["registration_id"] == first["registration_id"]

    cases_path = tmp_path / "first" / "cases.jsonl"
    cases_path.write_text(cases_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_registration(tmp_path / "first")


def test_calibration_covers_every_case_arm_and_reports_token_savings(tmp_path):
    registration_dir = tmp_path / "registration"
    register_study(CONFIG_PATH, registration_dir)

    summary = calibrate_registered(registration_dir, tmp_path / "calibration")

    assert summary["cell_count"] == 24 * len(MESH_ARMS)
    by_arm = {row["arm"]: row for row in summary["by_arm"]}
    assert by_arm["full_context"]["token_savings_vs_full_context"] == 0.0
    assert by_arm["adaptive_hybrid"]["token_savings_vs_full_context"] > 0.2
    assert by_arm["adaptive_hybrid"]["mean_retrieval_recall"] == 1.0
    assert by_arm["adaptive_hybrid"]["mean_wrong_skill_activation_rate"] == 0.0
    assert (tmp_path / "calibration" / "result_receipt.json").exists()


def test_packet_scoring_keeps_bonsai_performance_out_of_calibration():
    config = read_json(CONFIG_PATH)
    case = materialize_cases(config)[0]
    packet, data = build_mesh_packet(
        case,
        "adaptive_hybrid",
        working_budget_tokens=config["working_budget_tokens"],
        hard_context_tokens=config["hard_context_tokens"],
        static_top_k=config["static_top_k"],
        trm_top_k=config["trm_top_k"],
    )

    score = score_packet(case, packet, data)

    assert score["proxy_success"] is True
    assert "task_success" not in score
    assert score["packet_tokens_est"] <= config["working_budget_tokens"]


def test_live_response_parser_and_exact_scorer():
    config = read_json(CONFIG_PATH)
    case = materialize_cases(config)[0]
    response = {
        "task_id": case["task_id"],
        "resource_ids": case["expected"]["required_resource_ids"],
        "action_sequence": case["expected"]["action_sequence"],
        "abstain": False,
    }
    text = "prefix " + json.dumps(response) + " suffix"

    parsed = _extract_json_object(text)
    result = _score_live_response(case, parsed)

    assert parsed == response
    assert result["task_success"] is True
    assert result["model_retrieval_recall"] == 1.0


def test_confirmation_requires_completed_screening_receipt(tmp_path):
    config = read_json(CONFIG_PATH)

    with pytest.raises(ValueError, match="requires a completed screening"):
        _confirmation_arms(tmp_path, config, "registered-id")


def test_confirmation_promotes_only_arms_passing_frozen_gates(tmp_path):
    config = read_json(CONFIG_PATH)
    summary = {
        "registration_id": "registered-id",
        "status": "completed",
        "by_arm": [
            {
                "arm": "adaptive_hybrid",
                "task_success_rate": 0.75,
                "mean_model_wrong_skill_activation_rate": 0.0,
                "token_savings_vs_full_context": 0.5,
            },
            {
                "arm": "typed_packet",
                "task_success_rate": 0.75,
                "mean_model_wrong_skill_activation_rate": 0.2,
                "token_savings_vs_full_context": 0.5,
            },
        ],
    }
    (tmp_path / "live_screening_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    assert _confirmation_arms(tmp_path, config, "registered-id") == ["adaptive_hybrid"]
