"""Tests for compact TRM/LDT retrieval packet construction."""

import json

from agent.loop_driver import _build_api_messages
from agent.retrieval_packet import RetrievalPacketBuilder


def test_retrieval_packet_selects_same_task_failure(tmp_path):
    root = tmp_path / ".hermes" / "trm"
    root.mkdir(parents=True)
    (root / "current_task.json").write_text(json.dumps({
        "task_id": "trm_002",
        "instruction": "Choose one action.",
    }))
    (root / "self_model.json").write_text(json.dumps({
        "recent_pass_rate": 0.5,
        "recurring_failure_modes": ["format_error"],
    }))
    (root / "replay_candidates.jsonl").write_text(
        json.dumps({"task_id": "other", "score": 0, "failure": "wrong task"}) + "\n"
        + json.dumps({"task_id": "trm_002", "score": 0, "passed": False, "failure": "regex_miss", "output": "left"}) + "\n"
    )

    builder = RetrievalPacketBuilder(cwd=tmp_path, enabled=True, budget_tokens=300)
    packet = builder.build(user_message="retry trm_002")

    assert packet.startswith("# TRM/LDT Retrieval Packet")
    assert "trm_002" in packet
    assert "regex_miss" in packet
    assert "wrong task" not in packet
    assert len(packet) <= 300 * 4 + len("# TRM/LDT Retrieval Packet\n")


def test_retrieval_packet_disabled_returns_empty(tmp_path):
    root = tmp_path / ".hermes" / "trm"
    root.mkdir(parents=True)
    (root / "current_task.json").write_text("{}")

    builder = RetrievalPacketBuilder(cwd=tmp_path, enabled=False)

    assert builder.build(user_message="anything") == ""


def test_retrieval_packet_can_carry_active_skill_without_artifacts(tmp_path):
    builder = RetrievalPacketBuilder(cwd=tmp_path, enabled=True, budget_tokens=300)
    builder.set_skill_contract({
        "schema": "hermes.ultra_lean_skill.v1",
        "name": "trm-test",
        "conveyor": {"phases": ["ROUTE", "VERIFY", "COMMIT"]},
        "gates": ["Verify before commit."],
    })

    packet = builder.build(user_message="run it")

    assert "trm-test" in packet
    assert len(packet) <= 300 * 4 + len("# TRM/LDT Retrieval Packet\n")


def test_retrieval_packet_without_artifacts_or_active_skill_is_empty(tmp_path):
    builder = RetrievalPacketBuilder(cwd=tmp_path, enabled=True)
    assert builder.build(user_message="anything") == ""


def test_ollama_bonsai_auto_enables_for_small_context(tmp_path):
    builder = RetrievalPacketBuilder.from_config(
        {},
        model="digitsflow/bonsai-8b",
        context_length=12000,
        cwd=tmp_path,
    )

    assert builder.enabled is True


def test_llamacpp_12288_context_uses_12k_planning_profile(tmp_path):
    builder = RetrievalPacketBuilder.from_config(
        {},
        model="local/bonsai-8b",
        context_length=12288,
        cwd=tmp_path,
    )

    assert builder.enabled is True
    assert builder.hard_context_tokens == 12000


def test_build_api_messages_injects_retrieval_packet():
    class Agent:
        ephemeral_system_prompt = ""
        _honcho_context = ""
        _needs_tool_adapter = False
        tools = []
        prefill_messages = []
        _use_prompt_caching = False
        _cache_ttl = "5m"

        def _build_retrieval_packet(self, messages):
            return "# TRM/LDT Retrieval Packet\n{\"task\":{\"task_id\":\"x\"}}"

    messages = [{"role": "user", "content": "next"}]
    api_messages = _build_api_messages(Agent(), messages, "base system")

    assert api_messages[0]["role"] == "system"
    assert "base system" in api_messages[0]["content"]
    assert "TRM/LDT Retrieval Packet" in api_messages[0]["content"]
