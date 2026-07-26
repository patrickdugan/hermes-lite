import io
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.bitagent_control_mesh_v0 import CANDIDATE_SCHEMA, load_protocol
from agent.bitagent_role_adapters import ContractError, load_config
from agent.bitagent_sidecar_receipt_v1 import build_interface_receipt
from agent.bitagent_sidecar_v1 import (
    REQUEST_SCHEMA,
    load_sidecar_config,
    process_line,
    process_request,
    serve_stdio,
    validate_preflight,
)


CONFIG = Path("configs/bitagent_hermes_sidecar_v1.json")
ROLE_CONFIG = Path("configs/bitagent_bonsai_role_adapters_v1.json")
PROTOCOL = Path("configs/bitagent_hermes_control_mesh_v0.json")


def _configs():
    return load_config(ROLE_CONFIG), load_protocol(PROTOCOL)


def _request(operation="packetize", request_id=None):
    return {
        "schema": REQUEST_SCHEMA,
        "request_id": request_id or f"request-{operation}",
        "operation": operation,
        "task_card": {
            "case_id": "strategy-sidecar",
            "phase": "simulation",
            "operation": "simulate",
            "message": "Simulate a 10000 sat strategy.",
        },
        "workflow_state": {"workflowId": "w1", "status": "draft"},
        "tool_contracts": [
            {
                "name": "bitagent.strategy.simulate",
                "arguments": {"workflowId": "string", "amountSats": "string"},
            }
        ],
        "current_evidence": [{"kind": "wallet_state", "fresh": True}],
    }


def _packetized():
    role_config, protocol = _configs()
    return process_request(
        _request(),
        role_config=role_config,
        protocol=protocol,
    )


def _candidate(binding, *, with_capability=False):
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "role": "utxo_tradelayer_specialist",
        "intent": "starter_strategy",
        "tool": {
            "name": "bitagent.strategy.simulate",
            "arguments": {"workflowId": "w1", "amountSats": "10000"},
        },
        "execute": False,
        "evidence": binding,
    }
    if with_capability:
        candidate["capability_request"] = {
            "requestId": "sidecar-req-1",
            "agentId": "bitagent-host",
            "capability": "propose_tradelayer_intake",
            "effects": ["read_state", "reserve_capital"],
            "scope": {"workflowId": "w1"},
            "expiresAt": (
                datetime.now(timezone.utc) + timedelta(minutes=3)
            ).isoformat(),
            "intent": {"id": "intent-sidecar", "rail": "tradelayer"},
        }
    return candidate


def _run_stdio(lines, *, max_line_bytes=1_048_576, entries=256):
    role_config, protocol = _configs()
    output = io.BytesIO()
    serve_stdio(
        role_config=role_config,
        protocol=protocol,
        max_line_bytes=max_line_bytes,
        idempotency_entries=entries,
        input_stream=io.BytesIO(lines),
        output_stream=output,
    )
    return [
        json.loads(line)
        for line in output.getvalue().decode("utf-8").splitlines()
    ]


def test_preflight_is_deterministic_and_reports_live_runtime_separately():
    result = validate_preflight(CONFIG)
    assert result["status"] == "ready_deterministic_sidecar"
    assert result["live_adapter_runtime_ready"] is False
    assert result["weight_loading_started"] is False
    assert result["candidate_only"] is True
    assert result["tool_execution_enabled"] is False


def test_config_rejects_authority_or_cache_drift(tmp_path):
    config = load_sidecar_config(CONFIG)
    config["authority"]["execution"] = True
    path = tmp_path / "authority.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ContractError, match="authority boundary drift"):
        load_sidecar_config(path)

    config = load_sidecar_config(CONFIG)
    config["transport"]["idempotency_entries"] = 0
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ContractError, match="idempotency cache"):
        load_sidecar_config(path)


def test_import_and_preflight_do_not_import_torch():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    code = (
        "import sys; import agent.bitagent_sidecar_v1 as s; "
        f"s.validate_preflight(r'{CONFIG}'); "
        "assert 'torch' not in sys.modules; print('no_weights')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "no_weights"


def test_packetize_routes_and_binds_exact_inputs():
    response = _packetized()
    assert response["ok"] is True
    result = response["result"]
    assert result["role"] == "utxo_tradelayer_specialist"
    assert result["route"]["owner"] == "deterministic_host"
    assert result["packet_receipt"]["estimated_tokens"] < 6000
    assert len(result["evidence_binding"]["state_sha256"]) == 64
    assert result["packet"]["output_contract"][
        "may_approve_sign_broadcast_or_execute"
    ] is False


def test_validate_returns_ldt_and_hash_linked_receipt():
    role_config, protocol = _configs()
    request = _request("validate")
    request["candidate"] = _candidate(
        _packetized()["result"]["evidence_binding"]
    )
    response = process_request(
        request,
        role_config=role_config,
        protocol=protocol,
    )
    assert response["result"]["ldt_decision"]["decision"] == "candidate_valid"
    assert len(response["result"]["mesh_receipt"]["decision_sha256"]) == 64
    assert len(response["response_sha256"]) == 64


def test_materializes_fingerprinted_data_without_authority():
    role_config, protocol = _configs()
    request = _request("materialize_capability_request")
    request["candidate"] = _candidate(
        _packetized()["result"]["evidence_binding"],
        with_capability=True,
    )
    response = process_request(
        request,
        role_config=role_config,
        protocol=protocol,
    )
    result = response["result"]
    assert len(result["capability_request"]["invocationFingerprint"]) == 64
    assert result["authorization"] is False
    assert result["execution"] is False
    assert response["authority"] == (
        "candidate_only_or_capability_material_no_execution"
    )


def test_secret_bearing_input_is_rejected_without_echo():
    role_config, protocol = _configs()
    request = _request()
    request["workflow_state"]["private_key"] = "not-a-real-but-forbidden-value"
    response, _ = process_line(
        json.dumps(request).encode(),
        role_config=role_config,
        protocol=protocol,
    )
    assert response["ok"] is False
    assert response["error"]["code"] == "contract_rejected"
    assert "not-a-real" not in json.dumps(response)

    request = _request(request_id="request-wif-value")
    request["workflow_state"]["wallet_material"] = (
        "K" + "1" * 51
    )
    response, _ = process_line(
        json.dumps(request).encode(),
        role_config=role_config,
        protocol=protocol,
    )
    assert response["error"]["code"] == "contract_rejected"


def test_stdio_continues_after_malformed_and_oversized_lines():
    good = json.dumps(_request()).encode() + b"\n"
    responses = _run_stdio(
        b"{bad json}\n" + b"x" * 1500 + b"\n" + good,
        max_line_bytes=1024,
    )
    assert responses[0]["error"]["code"] == "invalid_json"
    assert responses[1]["error"]["code"] == "request_too_large"
    assert responses[2]["ok"] is True


def test_stdio_replays_identical_request_and_rejects_id_collision():
    request = _request(request_id="stable-id")
    same = json.dumps(request).encode() + b"\n"
    changed = dict(request)
    changed["workflow_state"] = {"workflowId": "w2", "status": "draft"}
    responses = _run_stdio(same + same + json.dumps(changed).encode() + b"\n")
    assert responses[0] == responses[1]
    assert responses[2]["error"]["code"] == "request_id_conflict"


def test_stdio_node_client_cross_process_roundtrip():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    result = subprocess.run(
        [node, "--test", "tests/node/bitagent_hermes_stdio_sidecar.test.mjs"],
        env={**os.environ, "BITAGENT_TEST_PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_schema_declares_only_stdio_control_contracts():
    schema = json.loads(
        Path("contracts/bitagent_hermes_sidecar_v1.schema.json").read_text()
    )
    assert set(schema["$defs"]) == {
        "request",
        "successResponse",
        "errorResponse",
    }


def test_sealed_interface_receipt_matches_current_bytes():
    expected = build_interface_receipt(Path("."))
    actual = json.loads(
        Path(
            "evals/registered/bitagent_hermes_paired_sidecar_v1/"
            "interface_receipt.json"
        ).read_text(encoding="utf-8")
    )
    assert actual == expected
    assert actual["capability_fingerprint_cross_language_agreement"] is True
    assert actual["control_mesh_source_status"] == (
        "local_uncommitted_source_byte_pinned"
    )


def test_result_receipt_reverifies_all_declared_artifacts():
    receipt = json.loads(
        Path("reports/bitagent_hermes_stdio_sidecar_v1_result.json").read_text()
    )
    for path, expected in receipt["artifact_canonical_lf_sha256"].items():
        canonical = Path(path).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(canonical).hexdigest() == expected
