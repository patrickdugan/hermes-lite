import json
from pathlib import Path

import pytest

from agent.bitagent_role_adapters import (
    ContractError,
    ROLES,
    build_packet,
    load_config,
    route_role,
    validate_candidate,
    validate_corpus,
)


CONFIG_PATH = Path("configs/bitagent_bonsai_role_adapters_v1.json")


def config():
    return load_config(CONFIG_PATH)


def example(role: str, index: int, split: str):
    cfg = config()
    candidate = {
        "role": role,
        "action": "collect_or_explain",
        "execute": False,
        "truthfulStateRequired": True,
    }
    return {
        "schema": "hermes.bitagent_role_example.v1",
        "id": f"{role}-{index}",
        "role": role,
        "split": split,
        "source": {"path": "fixture.json", "recordId": str(index), "sha256": "0" * 64},
        "authority": {
            "proposeOnly": True,
            "allowedTools": cfg["roles"][role]["allowed_tools"],
            "forbiddenEffects": cfg["authority"]["forbidden_model_effects"],
        },
        "messages": [
            {"role": "system", "content": f"ROLE={role}; candidate only"},
            {"role": "user", "content": json.dumps({"task": f"fixture {index}"})},
            {"role": "assistant", "content": json.dumps(candidate)},
        ],
        "tags": ["fixture"],
    }


def test_config_pins_separate_training_and_q1_runtime_bases():
    cfg = config()
    assert cfg["models"]["training_base"]["repo"] == "prism-ml/Bonsai-8B-unpacked"
    assert cfg["models"]["inference_base"]["repo"] == "prism-ml/Bonsai-8B-gguf"
    assert len(cfg["models"]["training_base"]["revision"]) == 40
    assert cfg["routing"]["model_may_select_role"] is False


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ({"phase": "conversation"}, "intent_planner"),
        ({"phase": "simulation"}, "utxo_tradelayer_specialist"),
        ({"phase": "pre_approval"}, "risk_approval_guard"),
        ({"status": "submitted"}, "recovery_operator"),
        ({"error_code": "simulation_stale"}, "recovery_operator"),
    ],
)
def test_deterministic_host_role_routing(task, expected):
    assert route_role(task) == expected


def test_candidate_rejects_unauthorized_execution_and_tools():
    cfg = config()
    with pytest.raises(ContractError, match="execute=true"):
        validate_candidate(cfg, "risk_approval_guard", {"execute": True})
    with pytest.raises(ContractError, match="may not propose tool"):
        validate_candidate(
            cfg,
            "intent_planner",
            {"tool": {"name": "bitagent.action.execute", "arguments": {}}},
        )


@pytest.mark.parametrize(
    "candidate",
    [
        {"action": "request_seed_phrase"},
        {"status": "approved"},
        {"tool": {"name": "terminal", "arguments": {}}},
    ],
)
def test_candidate_rejects_secret_authority_and_non_role_tool_bypasses(candidate):
    with pytest.raises(ContractError):
        validate_candidate(config(), "intent_planner", candidate)


def test_candidate_accepts_role_scoped_simulation_proposal():
    cfg = config()
    candidate = {
        "action": "validate_then_propose",
        "tool": {
            "name": "bitagent.strategy.simulate",
            "arguments": {"workflowId": "w1", "amountSats": "10000"},
        },
        "execute": False,
    }
    assert validate_candidate(cfg, "utxo_tradelayer_specialist", candidate) == candidate


def test_packet_filters_tools_and_carries_at_most_one_matching_replay():
    cfg = config()
    packet, receipt = build_packet(
        cfg,
        task_card={"phase": "simulation", "operation": "simulate", "intent": "starter_strategy"},
        workflow_state={"wallet": {"confirmedBalanceSats": "100000"}},
        tool_contracts=[
            {"name": "bitagent.strategy.simulate", "schema": {"type": "object"}},
            {"name": "bitagent.action.execute", "schema": {"type": "object"}},
        ],
        replay_candidates=[
            {"role": "utxo_tradelayer_specialist", "error_code": "utxo_unconfirmed"},
            {"role": "utxo_tradelayer_specialist", "error_code": "risk_rejected"},
        ],
    )
    assert receipt["role"] == "utxo_tradelayer_specialist"
    assert receipt["replay_hints"] == 1
    assert receipt["estimated_tokens"] <= 6000
    assert [row["name"] for row in packet["typed_tool_contracts"]] == ["bitagent.strategy.simulate"]


def test_seed_corpus_validator_enforces_roles_splits_and_authority(tmp_path):
    rows = []
    for role in sorted(ROLES):
        rows.extend(
            example(role, index, "validation" if index == 0 else "test" if index == 1 else "train")
            for index in range(5)
        )
    corpus = tmp_path / "examples.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    result = validate_corpus(config(), corpus)
    assert result["status"] == "passed"
    assert result["examples"] == 20
    assert result["unauthorized_effects_detected"] is False


def test_promotion_gate_rejects_seed_sized_corpus(tmp_path):
    rows = []
    for role in sorted(ROLES):
        rows.extend(
            example(role, index, "validation" if index == 0 else "test" if index == 1 else "train")
            for index in range(5)
        )
    corpus = tmp_path / "examples.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="minimum_promotion_examples_per_role"):
        validate_corpus(config(), corpus, promotion=True)
