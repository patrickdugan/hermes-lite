import json
from pathlib import Path

from agent.lean_contracts import GENERAL_LEAN_CONTRACT, HybridSkillRouter, contract_phases, deterministic_route_score, phase_projection
from agent.lean_filesystem_mcp import QUERIES, RESOURCES, retrieve
from agent.lean_packet import LEAN_KERNEL, StepPacketBuilder, TokenCounter
from agent.lean_runtime import LeanSkillRuntime, should_use_lean_runtime
from agent.lean_state import ArtifactBroker, LeanStateStore


def _contract(name="logic-skill", family="logic"):
    return {
        "schema": "hermes.ultra_lean_skill.v2",
        "name": name,
        "purpose": "Solve tent logic grids with a verifier.",
        "route": {
            "kind": "primary",
            "family": family,
            "aliases": [name, "tent logic"],
            "positive_examples": ["solve a tent logic grid", "verify trees and tents", "run a campsite puzzle"],
            "hard_negatives": ["write a storyworld", "train a model"],
            "compatible_overlays": [],
        },
        "conveyor": {
            "phases": [
                {"id": "GET", "module": "artifact_retrieval", "allowed_operations": ["retrieve"], "retrieval_slots": 1, "tool_profile": [], "max_output_tokens": 256},
                {"id": "GENERATE", "module": "model_generation", "allowed_operations": ["execute"], "retrieval_slots": 1, "tool_profile": [], "max_output_tokens": 256},
                {"id": "VERIFY", "module": "verifier", "allowed_operations": ["verify"], "retrieval_slots": 0, "tool_profile": [], "max_output_tokens": 256},
                {"id": "FINAL", "module": "finalizer", "allowed_operations": ["commit"], "retrieval_slots": 0, "tool_profile": [], "max_output_tokens": 256},
            ]
        },
        "gates": ["Candidate must exist."],
        "retrieval": {"reference_handles": []},
        "output_contract": "Return one answer.",
        "repair": "Retry once.",
    }


def test_auto_runtime_only_targets_configured_small_model():
    config = {"runtime": {"mode": "auto"}, "skills": {"ultra_lean_models": ["local/bonsai-8b"]}}
    assert should_use_lean_runtime(config, "local/bonsai-8b", 12000)
    assert not should_use_lean_runtime(config, "local/bonsai-8b", 32000)
    assert not should_use_lean_runtime(config, "cloud/large", 12000)


def test_router_prefers_matching_contract_and_has_general_fallback():
    logic = _contract()
    story = _contract("story-skill", "storyworld")
    story["purpose"] = "Build narrative storyworld encounters."
    story["route"]["aliases"] = ["storyworld"]
    story["route"]["positive_examples"] = ["build a storyworld", "write encounters", "rebalance narrative paths"]
    router = HybridSkillRouter([logic, story])
    candidates = router.candidates("solve this tent logic campsite grid")
    assert candidates[0].contract["name"] == "logic-skill"
    assert deterministic_route_score("tent logic", logic) > deterministic_route_score("tent logic", story)
    assert GENERAL_LEAN_CONTRACT["name"] == "general-lean"


def test_phase_projection_contains_only_current_phase():
    contract = _contract()
    projection = phase_projection(contract, 1)
    assert projection["phase"]["id"] == "GENERATE"
    assert "route" not in projection
    assert len(contract_phases(contract)) == 4


def test_packet_budget_is_structured_and_under_limit():
    builder = StepPacketBuilder(counter=TokenCounter(), max_input_tokens=8000)
    packet, receipt = builder.build(
        task_card={"instruction": "x" * 1000},
        phase=phase_projection(_contract(), 0),
        state={"phase_index": 0},
        evidence=[{"preview": "y" * 3000}],
        replay_hint="z" * 1000,
    )
    assert packet.startswith("STEP_PACKET=")
    json.loads(packet.split("=", 1)[1])
    assert receipt["input_tokens"] <= 8000
    assert receipt["kernel_tokens"] < 500
    assert len(LEAN_KERNEL) < 2000


def test_state_and_artifacts_replay(tmp_path):
    store = LeanStateStore(tmp_path, "task-1")
    broker = ArtifactBroker(store)
    ref = broker.put_text("candidate", "answer")
    store.append("phase_result", contract_id="logic-skill", phase="VERIFY", phase_index=2, candidate_ref=ref, status="running")
    snapshot = store.rebuild_snapshot()
    assert snapshot["contract_id"] == "logic-skill"
    assert snapshot["candidate_ref"] == ref
    assert broker.read(ref) == "answer"


def test_runtime_executes_stateless_phase_packets(monkeypatch, tmp_path):
    contract = _contract()
    monkeypatch.setattr("agent.lean_runtime.load_contracts", lambda: [contract])

    class Agent:
        model = "local/bonsai-8b"
        base_url = ""
        api_key = "local"
        _session_messages = []

    config = {
        "runtime": {
            "lean": {
                "state_dir": str(tmp_path),
                "max_input_tokens": 8000,
                "max_repair_attempts": 1,
                "router_confidence": 0.8,
                "router_margin": 0.15,
                "router_checkpoint": str(tmp_path / "missing-router.pt"),
            }
        },
    }
    runtime = LeanSkillRuntime(agent=Agent(), config=config)
    replies = iter([
        {"choices": [{"message": {"content": '{"phase":"GET","operation":"retrieve","gate":"pending"}'}}], "usage": {}},
        {"choices": [{"message": {"content": '{"phase":"GENERATE","operation":"execute","gate":"pending"}'}}], "usage": {}},
        {"choices": [{"message": {"content": "verified answer"}}], "usage": {}},
        {"choices": [{"message": {"content": '{"phase":"VERIFY","operation":"verify","gate":"pending"}'}}], "usage": {}},
        {"choices": [{"message": {"content": '{"phase":"FINAL","operation":"commit","gate":"pending"}'}}], "usage": {}},
    ])
    monkeypatch.setattr(runtime, "_post", lambda *args, **kwargs: next(replies))

    result = runtime.run("solve a tent logic campsite grid", task_id="test-run")

    assert result["completed"] is True
    assert result["final_response"] == "verified answer"
    assert result["lean_runtime"]["contract_id"] == "logic-skill"
    assert (Path(result["lean_runtime"]["state_dir"]) / "events.jsonl").exists()


def test_filesystem_descriptor_router_hits_all_held_queries():
    descriptors = [{"uri": item["uri"], "label": item["label"]} for item in RESOURCES]
    assert all(retrieve(query, descriptors) == expected for query, expected in QUERIES)


def test_runtime_never_finalizes_without_verifier_pass(monkeypatch, tmp_path):
    contract = _contract()
    contract["conveyor"]["phases"] = [
        {"id": "VERIFY", "module": "verifier", "allowed_operations": ["verify"], "retrieval_slots": 0, "tool_profile": [], "max_output_tokens": 64},
        {"id": "FINAL", "module": "finalizer", "allowed_operations": ["commit"], "retrieval_slots": 0, "tool_profile": [], "max_output_tokens": 64},
    ]
    monkeypatch.setattr("agent.lean_runtime.load_contracts", lambda: [contract])

    class Agent:
        model = "local/bonsai-8b"
        base_url = ""
        api_key = "local"
        _session_messages = []

    config = {"runtime": {"lean": {"state_dir": str(tmp_path), "router_checkpoint": str(tmp_path / "missing.pt")}}}
    runtime = LeanSkillRuntime(agent=Agent(), config=config)
    replies = iter([
        {"choices": [{"message": {"content": '{"phase":"VERIFY","operation":"verify","gate":"pending"}'}}], "usage": {}},
        {"choices": [{"message": {"content": '{"phase":"FINAL","operation":"commit","gate":"pending"}'}}], "usage": {}},
    ])
    monkeypatch.setattr(runtime, "_post", lambda *args, **kwargs: next(replies))

    result = runtime.run("solve a tent logic campsite grid", task_id="unverified")

    assert result["completed"] is False
    assert "no verifier-passed final candidate" in result["final_response"]
