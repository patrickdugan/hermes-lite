"""Process-owned stdio bridge between a BitAgent Node host and Hermes Lite."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, BinaryIO

from agent.bitagent_control_mesh_v0 import (
    CANDIDATE_SCHEMA,
    canonical_hash,
    load_protocol,
    materialize_capability_request,
    seal_mesh_receipt,
    validate_ldt_candidate,
)
from agent.bitagent_role_adapters import (
    ContractError,
    SECRET_ACTION_PATTERN,
    SECRET_VALUE_PATTERNS,
    build_packet,
    load_config as load_role_config,
    route_role,
)
from agent.bitagent_role_client import load_runtime


CONFIG_SCHEMA = "hermes.bitagent_sidecar_config.v1"
REQUEST_SCHEMA = "hermes.bitagent_sidecar_request.v1"
RESPONSE_SCHEMA = "hermes.bitagent_sidecar_response.v1"
OPERATIONS = {
    "packetize",
    "validate",
    "materialize_capability_request",
}
REQUEST_KEYS = {
    "schema",
    "request_id",
    "operation",
    "task_card",
    "workflow_state",
    "tool_contracts",
    "current_evidence",
    "replay_candidates",
    "candidate",
}
SECRET_KEYS = {
    "api_key",
    "apikey",
    "auth_token",
    "authorization",
    "mnemonic",
    "password",
    "private_key",
    "privatekey",
    "seed_phrase",
    "secret",
    "token",
    "wallet_seed",
}
SECRET_VALUE = re.compile(
    r"(?:-----BEGIN [A-Z ]*PRIVATE KEY-----|\bxprv[a-zA-Z0-9]{20,}\b)",
    re.IGNORECASE,
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def load_sidecar_config(path: str | Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ContractError("invalid BitAgent/Hermes sidecar config")
    transport = config.get("transport", {})
    if transport.get("kind") != "owned_child_stdio_jsonl":
        raise ContractError("sidecar transport must be process-owned stdio JSONL")
    if int(transport.get("max_line_bytes", 0)) <= 0:
        raise ContractError("sidecar line cap must be positive")
    if not 1 <= int(transport.get("idempotency_entries", 0)) <= 4096:
        raise ContractError("sidecar idempotency cache must contain 1 to 4096 entries")
    if set(config.get("operations", [])) != OPERATIONS:
        raise ContractError("sidecar operation set drift")
    authority = config.get("authority", {})
    if (
        authority.get("candidate_only") is not True
        or authority.get("capability_material_is_not_authorization") is not True
        or any(
            authority.get(effect) is not False
            for effect in ("approval", "signing", "broadcast", "execution")
        )
    ):
        raise ContractError("sidecar authority boundary drift")
    return config


def _resolve_config_paths(
    sidecar_config_path: str | Path,
    *,
    role_config_override: str = "",
    protocol_override: str = "",
) -> tuple[dict[str, Any], Path, Path, Path]:
    sidecar_config_path = Path(sidecar_config_path)
    sidecar = load_sidecar_config(sidecar_config_path)
    repo_root = sidecar_config_path.resolve().parents[1]
    role_path = repo_root / (
        role_config_override or sidecar["role_config_path"]
    )
    protocol_path = repo_root / (
        protocol_override or sidecar["protocol_path"]
    )
    runtime_path = repo_root / sidecar["live_runtime_path"]
    return sidecar, role_path, protocol_path, runtime_path


def validate_preflight(
    sidecar_config_path: str | Path,
    *,
    role_config_override: str = "",
    protocol_override: str = "",
) -> dict[str, Any]:
    sidecar, role_path, protocol_path, runtime_path = _resolve_config_paths(
        sidecar_config_path,
        role_config_override=role_config_override,
        protocol_override=protocol_override,
    )
    role_config = load_role_config(role_path)
    protocol = load_protocol(protocol_path)
    runtime = load_runtime(runtime_path)
    return {
        "schema": "hermes.bitagent_sidecar_preflight.v1",
        "status": "ready_deterministic_sidecar",
        "transport": sidecar["transport"]["kind"],
        "operations": sorted(OPERATIONS),
        "role_count": len(role_config["roles"]),
        "working_packet_max_tokens": protocol["context"]["working_packet_max_tokens"],
        "hard_context_tokens": protocol["context"]["hard_context_tokens"],
        "candidate_only": True,
        "capability_material_is_authorization": False,
        "tool_execution_enabled": False,
        "weight_loading_started": False,
        "live_adapter_runtime_status": runtime["status"],
        "live_adapter_runtime_ready": runtime["status"] == "ready",
    }


def _validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != REQUEST_SCHEMA:
        raise ContractError("invalid sidecar request schema")
    unknown = set(value) - REQUEST_KEYS
    if unknown:
        raise ContractError(f"unknown sidecar request fields: {sorted(unknown)}")
    request_id = value.get("request_id")
    if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
        raise ContractError("request_id must contain 1 to 128 characters")
    if value.get("operation") not in OPERATIONS:
        raise ContractError("unknown sidecar operation")
    for key in ("task_card", "workflow_state"):
        if not isinstance(value.get(key), dict):
            raise ContractError(f"{key} must be an object")
    for key in ("tool_contracts", "current_evidence", "replay_candidates"):
        if key in value and not isinstance(value[key], list):
            raise ContractError(f"{key} must be an array")
    if value["operation"] != "packetize" and not isinstance(
        value.get("candidate"), dict
    ):
        raise ContractError("candidate must be supplied for this operation")
    if _contains_secret(value):
        raise ContractError("secret-bearing input is forbidden")
    return value


def _contains_secret(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")
            if normalized in SECRET_KEYS and item not in (None, "", [], {}):
                return True
            if _contains_secret(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    return isinstance(value, str) and (
        SECRET_VALUE.search(value) is not None
        or SECRET_ACTION_PATTERN.search(value) is not None
        or any(pattern.search(value) for pattern in SECRET_VALUE_PATTERNS)
    )


def _evidence_binding(
    *,
    task_card: dict[str, Any],
    workflow_state: dict[str, Any],
    tool_contracts: list[dict[str, Any]],
    role: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    route = {
        "owner": "deterministic_host",
        "role": role,
    }
    evidence = {
        "task_card_sha256": canonical_hash(task_card),
        "state_sha256": canonical_hash(workflow_state),
        "tool_contracts_sha256": canonical_hash(tool_contracts),
        "route_sha256": canonical_hash(route),
    }
    return evidence, route


def process_request(
    value: Any,
    *,
    role_config: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    request_value = _validate_request(value)
    request_id = request_value["request_id"]
    operation = request_value["operation"]
    task_card = request_value["task_card"]
    workflow_state = request_value["workflow_state"]
    tool_contracts = request_value.get("tool_contracts", [])
    current_evidence = request_value.get("current_evidence", [])
    replay_candidates = request_value.get("replay_candidates", [])
    role = route_role(task_card)
    evidence_binding, route = _evidence_binding(
        task_card=task_card,
        workflow_state=workflow_state,
        tool_contracts=tool_contracts,
        role=role,
    )

    if operation == "packetize":
        packet, packet_receipt = build_packet(
            role_config,
            task_card=task_card,
            workflow_state=workflow_state,
            tool_contracts=tool_contracts,
            evidence=current_evidence,
            replay_candidates=replay_candidates,
            role=role,
        )
        result = {
            "role": role,
            "packet": packet,
            "packet_receipt": packet_receipt,
            "evidence_binding": evidence_binding,
            "route": route,
        }
        authority = "no_effect"
    else:
        candidate = request_value["candidate"]
        ldt_decision = validate_ldt_candidate(
            role_config,
            protocol,
            task_card,
            candidate,
            evidence_binding,
        )
        fallback = {
            "kind": ldt_decision["fallback"],
            "executed": False,
        }
        mesh_receipt = seal_mesh_receipt(
            envelope={
                "task_card": task_card,
                "workflow_state": workflow_state,
                "tool_contracts": tool_contracts,
            },
            proposal=candidate,
            route=route,
            ldt_decision=ldt_decision,
            fallback=fallback,
        )
        result = {
            "role": role,
            "ldt_decision": ldt_decision,
            "mesh_receipt": mesh_receipt,
            "candidate_schema": CANDIDATE_SCHEMA,
        }
        authority = "no_effect"
        if operation == "materialize_capability_request":
            capability_material = materialize_capability_request(
                protocol,
                candidate,
                ldt_decision,
            )
            result["capability_request"] = capability_material
            result["authorization"] = False
            result["execution"] = False
            authority = "candidate_only_or_capability_material_no_execution"

    return {
        "schema": RESPONSE_SCHEMA,
        "request_id": request_id,
        "operation": operation,
        "ok": True,
        "authority": authority,
        "result": result,
        "response_sha256": canonical_hash(
            {
                "request_id": request_id,
                "operation": operation,
                "result": result,
            }
        ),
    }


def error_response(request_id: str, operation: str, code: str) -> dict[str, Any]:
    return {
        "schema": RESPONSE_SCHEMA,
        "request_id": request_id,
        "operation": operation,
        "ok": False,
        "authority": "no_effect",
        "error": {
            "code": code,
            "detail": "Request rejected by the Hermes Lite typed boundary.",
        },
    }


def process_line(
    raw: bytes,
    *,
    role_config: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    request_id = ""
    operation = ""
    try:
        value = json.loads(raw.decode("utf-8"))
        if isinstance(value, dict):
            request_id = str(value.get("request_id", ""))[:128]
            operation = str(value.get("operation", ""))[:64]
        request_hash = canonical_hash(value)
        return (
            process_request(
                value,
                role_config=role_config,
                protocol=protocol,
            ),
            request_hash,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        return error_response(request_id, operation, "invalid_json"), ""
    except ContractError:
        return error_response(request_id, operation, "contract_rejected"), ""
    except Exception:
        return error_response(request_id, operation, "internal_error"), ""


def _drain_oversized_line(input_stream: BinaryIO, max_line_bytes: int) -> None:
    while True:
        remainder = input_stream.readline(max_line_bytes + 1)
        if not remainder or remainder.endswith(b"\n"):
            return


def serve_stdio(
    *,
    role_config: dict[str, Any],
    protocol: dict[str, Any],
    max_line_bytes: int,
    idempotency_entries: int = 256,
    input_stream: BinaryIO = sys.stdin.buffer,
    output_stream: BinaryIO = sys.stdout.buffer,
) -> None:
    cache: OrderedDict[str, tuple[str, dict[str, Any]]] = OrderedDict()
    while True:
        raw = input_stream.readline(max_line_bytes + 1)
        if not raw:
            return
        if len(raw) > max_line_bytes:
            if not raw.endswith(b"\n"):
                _drain_oversized_line(input_stream, max_line_bytes)
            response = error_response("", "", "request_too_large")
        else:
            response, request_hash = process_line(
                raw,
                role_config=role_config,
                protocol=protocol,
            )
            request_id = response["request_id"]
            cached = cache.get(request_id) if request_id else None
            if cached is not None:
                cached_hash, cached_response = cached
                response = (
                    cached_response
                    if request_hash and request_hash == cached_hash
                    else error_response(
                        request_id,
                        response["operation"],
                        "request_id_conflict",
                    )
                )
                cache.move_to_end(request_id)
            elif request_id and request_hash:
                cache[request_id] = (request_hash, response)
                if len(cache) > idempotency_entries:
                    cache.popitem(last=False)
        output_stream.write(canonical_bytes(response) + b"\n")
        output_stream.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/bitagent_hermes_sidecar_v1.json"),
    )
    parser.add_argument("--role-config", default="")
    parser.add_argument("--protocol", default="")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("serve")
    args = parser.parse_args()
    sidecar, role_path, protocol_path, _ = _resolve_config_paths(
        args.config,
        role_config_override=args.role_config,
        protocol_override=args.protocol,
    )
    if args.command == "preflight":
        print(
            json.dumps(
                validate_preflight(
                    args.config,
                    role_config_override=args.role_config,
                    protocol_override=args.protocol,
                ),
                indent=2,
            )
        )
    else:
        serve_stdio(
            role_config=load_role_config(role_path),
            protocol=load_protocol(protocol_path),
            max_line_bytes=int(sidecar["transport"]["max_line_bytes"]),
            idempotency_entries=int(
                sidecar["transport"]["idempotency_entries"]
            ),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
