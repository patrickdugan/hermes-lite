"""Role-scoped Bonsai adapter contracts for BitAgent and Hermes Lite.

This module does not load model weights. It validates the deterministic seed
corpus, routes a host-owned workflow phase to one role, constructs a bounded
Hermes packet, and rejects candidate actions that cross financial authority
boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA = "hermes.bitagent_bonsai_role_adapters.v1"
EXAMPLE_SCHEMA = "hermes.bitagent_role_example.v1"
PACKET_SCHEMA = "hermes.bitagent_role_packet.v1"
ROLES = {
    "intent_planner",
    "utxo_tradelayer_specialist",
    "risk_approval_guard",
    "recovery_operator",
}
SECRET_VALUE_PATTERNS = (
    re.compile(r"\b[KL5][1-9A-HJ-NP-Za-km-z]{50,51}\b"),
    re.compile(r"\b(?:sk|xoxb|ghp)_[A-Za-z0-9_-]{16,}\b"),
)
SECRET_KEY_PATTERN = re.compile(
    r"(?:private.?key|seed.?phrase|mnemonic|wif|api.?key|signing.?secret)",
    re.IGNORECASE,
)
SECRET_ACTION_PATTERN = re.compile(
    r"(?:request|collect|store|paste|provide|send|use)[_\s-]{0,3}"
    r"(?:a[_\s-]+)?(?:private[_\s-]*key|seed[_\s-]*phrase|mnemonic|wif|api[_\s-]*key)",
    re.IGNORECASE,
)
FORBIDDEN_AUTHORITY_VALUES = {
    "approved",
    "executed",
    "signed",
    "broadcast",
    "broadcasted",
}
FORBIDDEN_TRUE_KEYS = {
    "approve",
    "approved",
    "execute",
    "executed",
    "sign",
    "signed",
    "broadcast",
    "broadcasted",
}
EXECUTION_TOOLS = {
    "bitagent.wallet.request_approval",
    "bitagent.wallet.resolve_approval",
    "bitagent.action.execute",
    "bitagent.signal.request_approval",
    "bitagent.signal.resolve_approval",
    "bitagent.signal.execute",
}


class ContractError(ValueError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain one JSON object")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(Path(path))
    if config.get("schema") != SCHEMA:
        raise ContractError(f"unsupported config schema: {config.get('schema')}")
    roles = config.get("roles")
    if not isinstance(roles, dict) or set(roles) != ROLES:
        raise ContractError(f"roles must be exactly {sorted(ROLES)}")
    if config.get("routing", {}).get("model_may_select_role") is not False:
        raise ContractError("the model must not select its own financial role")
    if config.get("authority", {}).get("model_output_is_candidate_only") is not True:
        raise ContractError("model output must remain candidate-only")
    context = config.get("hermes_context", {})
    hard = int(context.get("hard_context_tokens", 0))
    working = int(context.get("working_packet_max_tokens", 0))
    summary = int(context.get("summary_lane_tokens", 0))
    if hard > 12000 or working > 6000 or working + summary > 12000:
        raise ContractError("Hermes packet exceeds the 6k working / 6k summary or 12k hard ceiling")
    for role, contract in roles.items():
        tools = contract.get("allowed_tools", [])
        if not isinstance(tools, list) or any(tool in EXECUTION_TOOLS for tool in tools):
            raise ContractError(f"{role} contains an approval or execution tool")
    return config


def _walk(value: Any, path: str = "$"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{path}.{key}", key, item
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def validate_candidate(
    config: dict[str, Any],
    role: str,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    if role not in ROLES:
        raise ContractError(f"unknown role: {role}")
    serialized = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
    if any(pattern.search(serialized) for pattern in SECRET_VALUE_PATTERNS):
        raise ContractError("candidate contains possible secret value")
    if SECRET_ACTION_PATTERN.search(serialized):
        raise ContractError("candidate may not request, collect, store, or use secret material")
    allowed_tools = set(config["roles"][role]["allowed_tools"])
    for path, key, value in _walk(candidate):
        normalized = key.lower().replace("-", "_")
        if SECRET_KEY_PATTERN.search(normalized) and value not in (None, "", False, "[REDACTED]"):
            raise ContractError(f"secret-bearing field prohibited at {path}")
        if normalized in FORBIDDEN_TRUE_KEYS and value is True:
            raise ContractError(f"model may not assert {key}=true at {path}")
        if isinstance(value, str) and value.lower() in FORBIDDEN_AUTHORITY_VALUES:
            raise ContractError(f"model may not assert authority state {value!r} at {path}")
        is_tool_field = normalized in {"tool", "toolname", "tool_name"} or (
            normalized == "name" and path.endswith(".tool.name")
        )
        if is_tool_field and isinstance(value, str):
            if value not in allowed_tools:
                raise ContractError(f"{role} may not propose tool {value}")
            if value in EXECUTION_TOOLS:
                raise ContractError(f"model may not propose approval or execution tool {value}")
    return candidate


def _load_examples(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def validate_corpus(
    config: dict[str, Any],
    examples_path: str | Path,
    *,
    promotion: bool = False,
) -> dict[str, Any]:
    path = Path(examples_path)
    rows = _load_examples(path)
    ids: set[str] = set()
    role_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    role_split_counts: Counter[tuple[str, str]] = Counter()
    content_splits: dict[str, str] = {}
    token_estimates = []
    for index, row in enumerate(rows, 1):
        if row.get("schema") != EXAMPLE_SCHEMA:
            raise ContractError(f"row {index}: unsupported example schema")
        example_id = str(row.get("id", ""))
        if not example_id or example_id in ids:
            raise ContractError(f"row {index}: missing or duplicate id {example_id!r}")
        ids.add(example_id)
        role = str(row.get("role", ""))
        split = str(row.get("split", ""))
        if role not in ROLES:
            raise ContractError(f"row {index}: invalid role {role!r}")
        if split not in {"train", "validation", "test"}:
            raise ContractError(f"row {index}: invalid split {split!r}")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise ContractError(f"row {index}: exactly system, user, assistant messages are required")
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise ContractError(f"row {index}: invalid message order")
        try:
            candidate = json.loads(str(messages[-1]["content"]))
        except json.JSONDecodeError as exc:
            raise ContractError(f"row {index}: assistant content must be strict JSON") from exc
        if not isinstance(candidate, dict):
            raise ContractError(f"row {index}: candidate must be an object")
        validate_candidate(config, role, candidate)
        authority = row.get("authority", {})
        if authority.get("proposeOnly") is not True:
            raise ContractError(f"row {index}: proposeOnly must be true")
        if sorted(authority.get("allowedTools", [])) != sorted(config["roles"][role]["allowed_tools"]):
            raise ContractError(f"row {index}: role tool authority drift")
        serialized_messages = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if any(pattern.search(serialized_messages) for pattern in SECRET_VALUE_PATTERNS):
            raise ContractError(f"row {index}: possible secret value")
        content_hash = hashlib.sha256(serialized_messages.encode("utf-8")).hexdigest()
        previous_split = content_splits.setdefault(content_hash, split)
        if previous_split != split:
            raise ContractError(f"row {index}: exact content appears across splits")
        role_counts[role] += 1
        split_counts[split] += 1
        role_split_counts[(role, split)] += 1
        token_estimates.append(math.ceil(len(serialized_messages) / 3.2))

    minimum_key = "minimum_promotion_examples_per_role" if promotion else "minimum_seed_examples_per_role"
    minimum = int(config["training"][minimum_key])
    under_minimum = {role: role_counts[role] for role in sorted(ROLES) if role_counts[role] < minimum}
    if under_minimum:
        raise ContractError(f"corpus below {minimum_key}={minimum}: {under_minimum}")
    if promotion:
        for split in ("validation", "test"):
            split_minimum = int(config["training"][f"minimum_promotion_{split}_examples_per_role"])
            under_split_minimum = {
                role: role_split_counts[(role, split)]
                for role in sorted(ROLES)
                if role_split_counts[(role, split)] < split_minimum
            }
            if under_split_minimum:
                raise ContractError(
                    f"promotion {split} split below per-role minimum={split_minimum}: {under_split_minimum}"
                )
    if not split_counts["validation"] or not split_counts["test"]:
        raise ContractError("validation and test splits must both be non-empty")
    corpus_bytes = path.read_bytes()
    return {
        "schema": "hermes.bitagent_role_corpus_validation.v1",
        "status": "passed",
        "promotion_mode": promotion,
        "examples": len(rows),
        "counts_by_role": dict(sorted(role_counts.items())),
        "counts_by_split": dict(sorted(split_counts.items())),
        "counts_by_role_and_split": {
            role: {
                split: role_split_counts[(role, split)]
                for split in ("train", "validation", "test")
            }
            for role in sorted(ROLES)
        },
        "max_estimated_tokens": max(token_estimates, default=0),
        "mean_estimated_tokens": round(sum(token_estimates) / max(1, len(token_estimates)), 2),
        "examples_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "raw_transcripts_included": False,
        "secret_values_detected": False,
        "unauthorized_effects_detected": False,
    }


def route_role(task_card: dict[str, Any]) -> str:
    """Route from host-supplied state; the model never invokes this policy."""
    status = str(task_card.get("status", "")).lower()
    phase = str(task_card.get("phase", "")).lower()
    operation = str(task_card.get("operation", "")).lower()
    error_code = str(task_card.get("error_code", "")).lower()
    if error_code or status in {
        "interrupted",
        "rejected",
        "cancelled",
        "stale",
        "submitted",
        "verification_failed",
    }:
        return "recovery_operator"
    if phase in {"pre_approval", "approval", "pre_execute"} or operation in {
        "review_exact_effects",
        "check_approval",
        "check_risk",
    }:
        return "risk_approval_guard"
    if phase in {"simulation", "utxo_mapping", "tradelayer_build"} or operation in {
        "simulate",
        "map_utxo",
        "build_tradelayer",
    }:
        return "utxo_tradelayer_specialist"
    return "intent_planner"


def _compact(value: Any, *, max_string: int = 1600) -> Any:
    if isinstance(value, dict):
        return {
            key: _compact(item, max_string=max_string)
            for key, item in sorted(value.items())
            if item not in (None, "", [], {})
        }
    if isinstance(value, list):
        return [_compact(item, max_string=max_string) for item in value[:12]]
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= max_string else text[: max_string - 15] + "...[external]"
    return value


def build_packet(
    config: dict[str, Any],
    *,
    task_card: dict[str, Any],
    workflow_state: dict[str, Any],
    tool_contracts: list[dict[str, Any]] | None = None,
    evidence: list[dict[str, Any]] | None = None,
    replay_candidates: list[dict[str, Any]] | None = None,
    role: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_role = role or route_role(task_card)
    if selected_role not in ROLES:
        raise ContractError(f"unknown role: {selected_role}")
    allowed = set(config["roles"][selected_role]["allowed_tools"])
    selected_tools = []
    for contract in tool_contracts or []:
        name = str(contract.get("name", ""))
        if name in allowed:
            selected_tools.append(_compact(contract, max_string=1200))
    replay = []
    for candidate in replay_candidates or []:
        if candidate.get("role") == selected_role or candidate.get("error_code") == task_card.get("error_code"):
            replay = [_compact(candidate, max_string=800)]
            break
    packet = {
        "schema": PACKET_SCHEMA,
        "task_card": _compact(task_card, max_string=2400),
        "role_contract": {
            "role": selected_role,
            "purpose": config["roles"][selected_role]["purpose"],
            "candidate_only": True,
            "allowed_tools": sorted(allowed),
            "forbidden_effects": config["authority"]["forbidden_model_effects"],
        },
        "compact_workflow_state": _compact(workflow_state, max_string=1200),
        "typed_tool_contracts": selected_tools,
        "current_evidence": _compact(evidence or [], max_string=1200),
        "single_matching_failure_replay": replay,
        "output_contract": {
            "format": "one_json_object",
            "must_name_role": selected_role,
            "may_approve_sign_broadcast_or_execute": False,
            "truth_source": "supplied_host_state_and_tool_results_only",
        },
    }
    text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    tokens = math.ceil(len(text) / 3.2)
    max_tokens = int(config["hermes_context"]["working_packet_max_tokens"])
    if tokens > max_tokens:
        packet["single_matching_failure_replay"] = []
        packet["current_evidence"] = packet["current_evidence"][:1]
        text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        tokens = math.ceil(len(text) / 3.2)
    if tokens > max_tokens:
        raise ContractError(f"required BitAgent packet is approximately {tokens} tokens; limit is {max_tokens}")
    receipt = {
        "schema": "hermes.bitagent_role_packet_receipt.v1",
        "role": selected_role,
        "estimated_tokens": tokens,
        "working_packet_max_tokens": max_tokens,
        "hard_context_tokens": int(config["hermes_context"]["hard_context_tokens"]),
        "replay_hints": len(packet["single_matching_failure_replay"]),
        "tool_contracts": len(packet["typed_tool_contracts"]),
        "packet_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return packet, receipt


def _parse_json_arg(value: str) -> dict[str, Any]:
    path = Path(value)
    data = _read_json(path) if path.exists() else json.loads(value)
    if not isinstance(data, dict):
        raise ContractError("expected a JSON object")
    return data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate and packetize BitAgent Bonsai role adapters")
    parser.add_argument("--config", default="configs/bitagent_bonsai_role_adapters_v1.json")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-corpus")
    validate.add_argument("--examples", required=True)
    validate.add_argument("--promotion", action="store_true")
    route = sub.add_parser("route")
    route.add_argument("--task-card", required=True)
    packet = sub.add_parser("packet")
    packet.add_argument("--task-card", required=True)
    packet.add_argument("--workflow-state", required=True)
    packet.add_argument("--role", choices=sorted(ROLES))
    candidate = sub.add_parser("validate-candidate")
    candidate.add_argument("--role", required=True, choices=sorted(ROLES))
    candidate.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate-corpus":
        result = validate_corpus(config, args.examples, promotion=args.promotion)
    elif args.command == "route":
        result = {"role": route_role(_parse_json_arg(args.task_card))}
    elif args.command == "packet":
        built, receipt = build_packet(
            config,
            task_card=_parse_json_arg(args.task_card),
            workflow_state=_parse_json_arg(args.workflow_state),
            role=args.role,
        )
        result = {"packet": built, "receipt": receipt}
    else:
        result = validate_candidate(config, args.role, _parse_json_arg(args.candidate))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
