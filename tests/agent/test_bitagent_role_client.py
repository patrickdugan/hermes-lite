import pytest

from agent.bitagent_role_adapters import ContractError, load_config
from agent.bitagent_role_client import build_request, load_runtime, map_adapter_ids


CONFIG = "configs/bitagent_bonsai_role_adapters_v1.json"
RUNTIME = "configs/bitagent_bonsai_runtime_v1.json"


def test_runtime_requires_host_owned_single_adapter_activation():
    runtime = load_runtime(RUNTIME)
    assert runtime["activation"]["owner"] == "deterministic_host"
    assert runtime["activation"]["active_adapter_count"] == 1


def test_adapter_ids_map_by_exact_filename():
    runtime = load_runtime(RUNTIME)
    rows = [
        {"id": index, "path": f"C:/adapters/{filename}", "scale": 0}
        for index, filename in enumerate(runtime["adapter_files"].values())
    ]
    mapped = map_adapter_ids(runtime, rows)
    assert set(mapped) == set(runtime["adapter_files"])
    assert len(set(mapped.values())) == 4


def test_adapter_mapping_rejects_missing_or_duplicate_artifacts():
    runtime = load_runtime(RUNTIME)
    with pytest.raises(ContractError, match="exactly one"):
        map_adapter_ids(runtime, [])


def test_request_activates_only_host_routed_role():
    config = load_config(CONFIG)
    runtime = load_runtime(RUNTIME)
    adapter_ids = {
        "intent_planner": 0,
        "utxo_tradelayer_specialist": 1,
        "risk_approval_guard": 2,
        "recovery_operator": 3,
    }
    role, payload, receipt = build_request(
        config,
        runtime,
        adapter_ids,
        task_card={"phase": "pre_approval", "operation": "check_risk"},
        workflow_state={"simulation": {"feeSats": "100"}},
    )
    assert role == "risk_approval_guard"
    assert payload["lora"] == [{"id": 2, "scale": 1.0}]
    assert receipt["replay_hints"] == 0
    assert "tools" not in payload
