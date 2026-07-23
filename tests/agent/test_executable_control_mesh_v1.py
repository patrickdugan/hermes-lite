"""Regression tests for the executable Hermes Lite/full-Hermes bridge."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.executable_control_mesh_v1 import (
    _build_full_packet,
    _build_local_packet,
    _canonical_action_tokens,
    _held_cases,
    _load_contracts,
    _load_protocol_cases,
    _split_cases,
    prepare_full_hermes,
    read_json,
    register_study,
    train_domain_ram,
    verify_registration,
)
from agent.lean_control_mesh_v1 import SparseRAMPolicy
from scripts.run_full_hermes_baseline_v1 import (
    AgentResultError,
    _extract_agent_response,
    _load_api_key,
)


CONFIG = Path("configs/hermes_lite_executable_bridge_v1.json")


def _registration(tmp_path: Path) -> Path:
    output = tmp_path / "registered"
    register_study(CONFIG, output)
    return output


def test_registration_freezes_16_train_and_8_held_cases(tmp_path):
    registration = _registration(tmp_path)
    receipt = verify_registration(registration)
    split = read_json(registration / "split_manifest.json")

    assert receipt["train_case_count"] == 16
    assert receipt["held_case_count"] == 8
    assert not set(split["train_task_ids"]) & set(split["held_task_ids"])
    assert not set(split["train_query_sha256"]) & set(split["held_query_sha256"])


def test_domain_ram_training_is_capped_and_does_not_consume_held(tmp_path, monkeypatch):
    registration = _registration(tmp_path)
    monkeypatch.setenv("LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE", "1")

    summary = train_domain_ram(
        registration,
        tmp_path / "model",
        epochs=40,
        seed=8819,
        ram_cap_mb=2048,
        io_cap_mb_s=50.0,
    )

    assert summary["status"] == "completed"
    assert summary["held_outcomes_consumed"] is False
    assert summary["metrics"]["route_train_accuracy"] == 1.0
    assert summary["metrics"]["supported_labels"] == 4


def test_domain_ram_refuses_uncapped_training(tmp_path, monkeypatch):
    registration = _registration(tmp_path)
    monkeypatch.delenv("LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE", raising=False)

    with pytest.raises(SystemExit, match="Refusing uncapped"):
        train_domain_ram(
            registration,
            tmp_path / "model",
            epochs=1,
            seed=1,
            ram_cap_mb=2048,
            io_cap_mb_s=50.0,
        )


def test_action_namespace_uses_current_canonical_tokens_only():
    protocol = read_json(CONFIG)
    cases = _load_protocol_cases(protocol)
    resources = cases[0]["resources"]

    tokens = _canonical_action_tokens(resources)

    assert "validate_world_invariants" in tokens
    assert all(not token.startswith("legacy_") for token in tokens)


def test_full_packet_has_no_expected_answer_and_fits_160k(tmp_path):
    registration = _registration(tmp_path)
    receipt = verify_registration(registration)
    output = tmp_path / "prompts.jsonl"

    summary = prepare_full_hermes(
        registration,
        output,
        confirm_registration_id=receipt["registration_id"],
    )
    text = output.read_text(encoding="utf-8")

    assert summary["prompts"] == 8
    assert summary["max_packet_tokens_est"] < 160000
    assert '"expected"' not in text
    assert '"allowed_tools"' not in text


def test_adaptive_domain_ram_selects_registered_specialists(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    registration = _registration(tmp_path)
    monkeypatch.setenv("LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE", "1")
    domain_dir = tmp_path / "domain"
    train_domain_ram(
        registration,
        domain_dir,
        epochs=40,
        seed=8819,
        ram_cap_mb=2048,
        io_cap_mb_s=50.0,
    )
    protocol = read_json(registration / "protocol.json")
    contracts = _load_contracts(protocol)
    domain_ram = SparseRAMPolicy.from_dict(read_json(domain_dir / "domain_ram_policy.json"))

    from agent.executable_control_mesh_v1 import _load_router_models

    base_ram, trm = _load_router_models(Path.home() / ".hermes-lite/models/control-mesh-v1-1")
    matches = 0
    held = _held_cases(protocol)
    for case in held:
        packet, packet_data = _build_local_packet(
            case,
            "adaptive_mesh",
            protocol,
            contracts,
            base_ram,
            trm,
            domain_ram,
        )
        matches += packet_data["control"]["selected_contract"] == protocol["domain_specialists"][
            case["domain"]
        ]
        assert '"expected"' not in packet
        assert '"allowed_tools"' not in packet
        assert len(packet) // 4 < 6000
    assert matches / len(held) >= 0.75


def test_full_hermes_runner_pins_context_and_disables_tools():
    script = Path("scripts/run_full_hermes_baseline_v1.py").read_text(encoding="utf-8")

    assert 'enabled_toolsets=["__no_tools__"]' in script
    assert "context_length_override=context_tokens" in script
    assert "minimum_context_length=context_tokens" in script
    assert "full Hermes prompt batch hash mismatch" in script
    assert "print(key)" not in script


def test_full_hermes_runner_rejects_empty_or_failed_agent_results():
    with pytest.raises(AgentResultError, match="empty_final_response"):
        _extract_agent_response({"final_response": "", "completed": True})
    with pytest.raises(AgentResultError, match="agent_result_failed"):
        _extract_agent_response(
            {
                "final_response": None,
                "completed": False,
                "failed": True,
                "error": "provider detail must not enter the receipt",
            }
        )


def test_full_hermes_runner_extracts_response_and_usage():
    content, usage = _extract_agent_response(
        {
            "final_response": "  {\"actions\": []}  ",
            "completed": True,
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        }
    )

    assert content == '{"actions": []}'
    assert usage["input_tokens"] == 11
    assert usage["total_tokens"] == 18


def test_full_hermes_runner_loads_named_dotenv_key(tmp_path):
    credential = tmp_path / ".env"
    credential.write_text(
        "# fixture\nOPENROUTER_API_KEY='not-a-real-key'\n",
        encoding="utf-8",
    )

    assert _load_api_key(credential, "OPENROUTER_API_KEY") == "not-a-real-key"


def test_cap_wrappers_have_explicit_executable_bridge_modes():
    training = Path("scripts/run_capped_lean_control_mesh_v1.ps1").read_text(
        encoding="utf-8"
    )
    inference = Path("scripts/run_capped_bonsai_mcp_mesh.ps1").read_text(encoding="utf-8")

    assert '"executable_domain_ram"' in training
    assert '"agent.executable_control_mesh_v1", "train"' in training
    assert '"executable_bridge_v1"' in inference
    assert '"agent.executable_control_mesh_v1"' in inference
    assert "--domain-model-dir" in inference
