"""Candidate-only llama.cpp client for host-routed BitAgent role adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import request

from agent.bitagent_role_adapters import (
    ContractError,
    build_packet,
    load_config,
    route_role,
    validate_candidate,
)


RUNTIME_SCHEMA = "hermes.bitagent_bonsai_runtime.v1"


def load_runtime(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict) or value.get("schema") != RUNTIME_SCHEMA:
        raise ContractError("invalid BitAgent Bonsai runtime manifest")
    activation = value.get("activation", {})
    if activation.get("owner") != "deterministic_host":
        raise ContractError("adapter activation must be host-owned")
    if activation.get("per_request") is not True or int(activation.get("active_adapter_count", 0)) != 1:
        raise ContractError("exactly one role adapter must be activated per request")
    return value


def map_adapter_ids(
    runtime: dict[str, Any],
    server_adapters: list[dict[str, Any]],
) -> dict[str, int]:
    result: dict[str, int] = {}
    for role, filename in runtime["adapter_files"].items():
        matches = [
            row
            for row in server_adapters
            if Path(str(row.get("path", ""))).name.lower() == str(filename).lower()
        ]
        if len(matches) != 1:
            raise ContractError(f"expected exactly one loaded llama.cpp adapter for {role}: {filename}")
        result[role] = int(matches[0]["id"])
    if len(set(result.values())) != len(result):
        raise ContractError("role adapters must map to distinct llama.cpp IDs")
    return result


def build_request(
    config: dict[str, Any],
    runtime: dict[str, Any],
    adapter_ids: dict[str, int],
    *,
    task_card: dict[str, Any],
    workflow_state: dict[str, Any],
    tool_contracts: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    replay_candidates: list[dict[str, Any]] | None = None,
    max_completion_tokens: int = 384,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    role = route_role(task_card)
    if role not in adapter_ids:
        raise ContractError(f"no server adapter ID registered for role {role}")
    packet, receipt = build_packet(
        config,
        task_card=task_card,
        workflow_state=workflow_state,
        tool_contracts=tool_contracts,
        evidence=evidence,
        replay_candidates=replay_candidates,
        role=role,
    )
    payload = {
        "model": runtime["model_alias"],
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are the BitAgent {role} role. Return one JSON candidate only. "
                    "The host owns financial truth and all effects."
                ),
            },
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False, separators=(",", ":"))},
        ],
        "temperature": 0,
        "max_tokens": max_completion_tokens,
        "response_format": {"type": "json_object"},
        "lora": [{"id": adapter_ids[role], "scale": float(runtime["activation"]["scale"])}],
    }
    return role, payload, receipt


class BitAgentRoleClient:
    def __init__(
        self,
        *,
        config_path: str | Path,
        runtime_path: str | Path,
        timeout_seconds: int = 120,
    ):
        self.config = load_config(config_path)
        self.runtime = load_runtime(runtime_path)
        self.timeout_seconds = timeout_seconds
        self.base_url = str(self.runtime["endpoint"]).rstrip("/")
        self.adapter_ids: dict[str, int] = {}

    def discover(self) -> dict[str, int]:
        if self.runtime.get("status") != "ready":
            raise ContractError("BitAgent role runtime is not operator-promoted to ready")
        endpoint = self.base_url.removesuffix("/v1") + "/lora-adapters"
        with request.urlopen(endpoint, timeout=10) as response:
            value = json.loads(response.read().decode("utf-8"))
        if not isinstance(value, list):
            raise ContractError("llama.cpp /lora-adapters did not return a list")
        self.adapter_ids = map_adapter_ids(self.runtime, value)
        return dict(self.adapter_ids)

    def propose(
        self,
        *,
        task_card: dict[str, Any],
        workflow_state: dict[str, Any],
        tool_contracts: list[dict[str, Any]] | None = None,
        evidence: list[dict[str, Any]] | None = None,
        replay_candidates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not self.adapter_ids:
            self.discover()
        role, payload, receipt = build_request(
            self.config,
            self.runtime,
            self.adapter_ids,
            task_card=task_card,
            workflow_state=workflow_state,
            tool_contracts=tool_contracts,
            evidence=evidence,
            replay_candidates=replay_candidates,
        )
        req = request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read().decode("utf-8"))
        try:
            content = value["choices"][0]["message"]["content"]
            candidate = json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ContractError("Bonsai role adapter returned malformed JSON") from exc
        if not isinstance(candidate, dict):
            raise ContractError("Bonsai role adapter candidate must be an object")
        validate_candidate(self.config, role, candidate)
        return {
            "schema": "hermes.bitagent_role_candidate.v1",
            "role": role,
            "candidate": candidate,
            "packet_receipt": receipt,
            "authority": "candidate_only_no_effect",
        }
