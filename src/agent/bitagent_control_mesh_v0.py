"""Registered BitAgent/Hermes control-mesh contracts.

The model-facing side of this module can only emit candidate actions. The LDT
checks a deterministic host route, typed evidence, and BitAgent capability
semantics before it can materialize a fingerprinted capability request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.bitagent_role_adapters import (
    ContractError,
    load_config as load_role_config,
    route_role,
    validate_candidate,
)


PROTOCOL_SCHEMA = "hermes.bitagent_control_mesh_protocol.v0"
CANDIDATE_SCHEMA = "hermes.bitagent_candidate.v0"
LDT_SCHEMA = "hermes.bitagent_ldt_decision.v0"
RECEIPT_SCHEMA = "hermes.bitagent_mesh_receipt.v0"
CAPABILITY_EFFECTS = {
    "propose_psbt": {"read_state", "reserve_capital"},
    "pay_invoice_capped": {"read_state", "reserve_capital", "request_signature"},
    "propose_swap": {"read_state", "reserve_capital"},
    "propose_tradelayer_intake": {"read_state", "reserve_capital"},
    "propose_fedimint_payment": {"read_state", "reserve_capital", "request_signature"},
    "propose_vtxo_action": {"read_state", "reserve_capital", "request_signature"},
    "propose_dlc": {"read_state", "reserve_capital", "request_signature"},
    "propose_filecoin_storage": {"read_state", "reserve_capital", "request_signature"},
    "propose_compute_lease": {"read_state", "reserve_capital", "request_signature"},
    "request_near_chain_signature": {"read_state", "request_signature"},
    "propose_policy_update": {"read_state", "self_modify"},
    "write_memory": {"write_state"},
}
CASE_LINE = re.compile(r'^\s*\{\s*id:\s*"[^"]+".*\},?\s*$')
FIELD = re.compile(r'(?P<key>[A-Za-z][A-Za-z0-9]*):\s*(?:\"(?P<string>[^\"]*)\"|(?P<bool>true|false))')
SECRET_REQUEST = re.compile(
    r"(?:seed phrase|private key|mnemonic|\bWIF\b|sign with my private key|store my seed)",
    re.IGNORECASE,
)


def canonical_json(value: Any) -> str:
    """Match BitAgent's ASCII-key canonical JSON for the contract value set."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain one JSON object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8", newline="\n")


def load_protocol(path: str | Path) -> dict[str, Any]:
    protocol = _read_json(Path(path))
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ContractError(f"unsupported protocol schema: {protocol.get('schema')}")
    if protocol.get("status") != "frozen_before_mesh_outcomes":
        raise ContractError("protocol must be frozen before mesh outcomes")
    if protocol.get("authority", {}).get("model_output") != "candidate_only":
        raise ContractError("model output must be candidate-only")
    forbidden = set(protocol.get("authority", {}).get("forbidden_model_effects", []))
    if not {"approval", "signing", "broadcast"}.issubset(forbidden):
        raise ContractError("approval, signing, and broadcast must remain forbidden")
    context = protocol.get("context", {})
    if int(context.get("working_packet_max_tokens", 0)) > 6000:
        raise ContractError("working packet exceeds 6000 tokens")
    if int(context.get("hard_context_tokens", 0)) > 12000:
        raise ContractError("hard context exceeds 12000 tokens")
    if set(protocol.get("arms", [])) != {
        "lexical_ldt",
        "ram_ldt",
        "trm_ldt",
        "adaptive_mesh_ldt",
    }:
        raise ContractError("registered arm set drift")
    return protocol


def parse_agent_cases(source: str | Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(source).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not CASE_LINE.match(line):
            continue
        values: dict[str, Any] = {}
        for match in FIELD.finditer(line):
            values[match.group("key")] = (
                match.group("string") if match.group("string") is not None else match.group("bool") == "true"
            )
        required = {"id", "phase", "message", "expectedIntent"}
        if not required.issubset(values):
            raise ContractError(f"agent case line {line_number} is not parseable")
        cases.append(
            {
                "id": values["id"],
                "phase": values["phase"],
                "message": values["message"],
                "expected_intent": values["expectedIntent"],
                "expected_tool": values.get("expectedTool"),
                "expected_missing": values.get("expectedMissing"),
                "prohibited": bool(values.get("prohibited", False)),
            }
        )
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate BitAgent case IDs")
    return cases


def expected_role(case: dict[str, Any]) -> str:
    if case["prohibited"] or case["expected_intent"] == "unsupported":
        return "risk_approval_guard"
    if case["expected_intent"] in {"starter_strategy", "withdraw_bitcoin"}:
        return "utxo_tradelayer_specialist"
    return "intent_planner"


def registered_case(case: dict[str, Any], held_suffixes: set[str]) -> dict[str, Any]:
    suffix = str(case["id"]).rsplit("-", 1)[-1]
    role = expected_role(case)
    route_phase = {
        "intent_planner": "conversation",
        "utxo_tradelayer_specialist": "simulation",
        "risk_approval_guard": "pre_approval",
    }[role]
    task_card = {
        "case_id": case["id"],
        "message": case["message"],
        "phase": route_phase,
        "wallet_phase": case["phase"],
    }
    labels = {
        "intent": case["expected_intent"],
        "tool": case["expected_tool"],
        "missing": case["expected_missing"],
        "prohibited": case["prohibited"],
        "role": role,
    }
    return {
        "schema": "hermes.bitagent_registered_case.v0",
        "id": case["id"],
        "split": "held" if suffix in held_suffixes else "train",
        "task_card": task_card,
        "labels": labels,
        "task_card_sha256": canonical_hash(task_card),
    }


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def build_source_manifest(protocol: dict[str, Any]) -> dict[str, Any]:
    source = protocol["bitagent_source"]
    root = Path(source["local_root"]).resolve()
    required_files = source["required_files"]
    rows = []
    for relative, expected_hash in sorted(required_files.items()):
        path = root / relative
        if not path.is_file():
            raise ContractError(f"missing BitAgent source file: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise ContractError(f"BitAgent source drift for {relative}: {actual_hash}")
        status = _git(root, "status", "--short", "--", relative)
        rows.append(
            {
                "path": relative,
                "sha256": actual_hash,
                "bytes": path.stat().st_size,
                "git_status": status or "clean",
            }
        )
    git_head = _git(root, "rev-parse", "HEAD")
    if git_head != source["git_head"]:
        raise ContractError(f"BitAgent git HEAD drift: {git_head}")
    return {
        "schema": "hermes.bitagent_source_manifest.v0",
        "source_root": str(root).replace("\\", "/"),
        "git_head": git_head,
        "versioning_status": source["versioning_status"],
        "files": rows,
    }


def register_protocol(protocol_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    protocol_path = Path(protocol_path)
    output = Path(output_dir)
    protocol = load_protocol(protocol_path)
    source_manifest = build_source_manifest(protocol)
    source_root = Path(protocol["bitagent_source"]["local_root"])
    case_source = source_root / protocol["split"]["case_source"]
    parsed = parse_agent_cases(case_source)
    expected_count = int(protocol["split"]["expected_case_count"])
    if len(parsed) != expected_count:
        raise ContractError(f"expected {expected_count} BitAgent cases, found {len(parsed)}")
    held_suffixes = set(protocol["split"]["held_suffixes"])
    rows = [registered_case(case, held_suffixes) for case in parsed]
    train_rows = [row for row in rows if row["split"] == "train"]
    held_rows = [row for row in rows if row["split"] == "held"]
    expected_counts = protocol["split"]["expected_counts"]
    if len(train_rows) != int(expected_counts["train"]) or len(held_rows) != int(expected_counts["held"]):
        raise ContractError("registered split count drift")
    overlap = {row["id"] for row in train_rows} & {row["id"] for row in held_rows}
    if overlap:
        raise ContractError(f"split overlap: {sorted(overlap)}")

    output.mkdir(parents=True, exist_ok=True)
    protocol_target = output / "protocol.json"
    _write_json(protocol_target, protocol)
    _write_json(output / "source_manifest.json", source_manifest)
    _write_jsonl(output / "train_cases.jsonl", train_rows)
    _write_jsonl(output / "held_cases.jsonl", held_rows)
    receipt_material = {
        "schema": "hermes.bitagent_control_mesh_registration.v0",
        "study_id": protocol["study_id"],
        "status": "registered_no_mesh_outcomes",
        "protocol_sha256": file_sha256(protocol_target),
        "source_manifest_sha256": file_sha256(output / "source_manifest.json"),
        "train_cases_sha256": file_sha256(output / "train_cases.jsonl"),
        "held_cases_sha256": file_sha256(output / "held_cases.jsonl"),
        "train_case_count": len(train_rows),
        "held_case_count": len(held_rows),
        "split_overlap_count": 0,
        "arms": protocol["arms"],
        "claim_scope": protocol["claim_scope"],
        "source_versioning_status": source_manifest["versioning_status"],
    }
    receipt = {**receipt_material, "registration_id": canonical_hash(receipt_material)}
    _write_json(output / "registration_receipt.json", receipt)
    return receipt


def capability_request_fingerprint(request_material: dict[str, Any]) -> str:
    required = {"requestId", "agentId", "capability", "effects", "scope", "expiresAt"}
    if not required.issubset(request_material):
        raise ContractError(f"capability request missing fields: {sorted(required - set(request_material))}")
    material = {
        "requestId": request_material["requestId"],
        "agentId": request_material["agentId"],
        "capability": request_material["capability"],
        "effects": sorted(request_material["effects"]),
        "scope": request_material["scope"],
        "expiresAt": request_material["expiresAt"],
    }
    if request_material.get("intent") is not None:
        material["intentHash"] = canonical_hash(request_material["intent"])
    return canonical_hash(material)


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError("expiresAt must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ContractError("expiresAt must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_capability_material(
    material: dict[str, Any],
    *,
    now: datetime,
    max_lifetime_seconds: int,
) -> dict[str, Any]:
    capability = str(material.get("capability", ""))
    allowed_effects = CAPABILITY_EFFECTS.get(capability)
    if allowed_effects is None:
        raise ContractError(f"unknown BitAgent capability: {capability}")
    effects = material.get("effects")
    if not isinstance(effects, list) or not effects:
        raise ContractError("capability effects must be a non-empty list")
    if "broadcast" in effects:
        raise ContractError("broadcast is non-delegable")
    if not set(effects).issubset(allowed_effects):
        raise ContractError("capability contains undeclared effects")
    scope = material.get("scope")
    if not isinstance(scope, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in scope.items()):
        raise ContractError("capability scope must map strings to strings")
    if "reserve_capital" in effects and not isinstance(material.get("intent"), dict):
        raise ContractError("reserve_capital requires policy intent evidence")
    expires_at = _parse_time(str(material.get("expiresAt", "")))
    lifetime = (expires_at - now.astimezone(timezone.utc)).total_seconds()
    if lifetime <= 0 or lifetime > max_lifetime_seconds:
        raise ContractError("capability expiry is outside the registered lifetime")
    if not str(material.get("requestId", "")) or not str(material.get("agentId", "")):
        raise ContractError("requestId and agentId are required")
    return material


def validate_ldt_candidate(
    role_config: dict[str, Any],
    protocol: dict[str, Any],
    task_card: dict[str, Any],
    candidate: dict[str, Any],
    evidence: dict[str, str],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    reasons: list[str] = []
    expected = route_role(task_card)
    role = str(candidate.get("role", ""))
    if candidate.get("schema") != CANDIDATE_SCHEMA:
        reasons.append("candidate_schema_mismatch")
    if role != expected:
        reasons.append("deterministic_route_mismatch")
    expected_evidence = {
        "task_card_sha256": canonical_hash(task_card),
        "state_sha256": evidence.get("state_sha256", ""),
        "tool_contracts_sha256": evidence.get("tool_contracts_sha256", ""),
        "route_sha256": evidence.get("route_sha256", ""),
    }
    if candidate.get("evidence") != expected_evidence:
        reasons.append("evidence_binding_mismatch")
    try:
        validate_candidate(role_config, role, candidate)
    except ContractError as exc:
        reasons.append(f"role_contract:{exc}")
    if SECRET_REQUEST.search(str(task_card.get("message", ""))) and candidate.get("tool"):
        reasons.append("secret_request_must_not_route_to_tool")
    capability = candidate.get("capability_request")
    if capability is not None:
        try:
            validate_capability_material(
                capability,
                now=now,
                max_lifetime_seconds=int(protocol["ldt"]["max_capability_lifetime_seconds"]),
            )
        except ContractError as exc:
            reasons.append(f"capability_contract:{exc}")
    decision_material = {
        "schema": LDT_SCHEMA,
        "decision": "rejected" if reasons else "candidate_valid",
        "reason_codes": reasons or ["typed_candidate_valid"],
        "expected_role": expected,
        "candidate_sha256": canonical_hash(candidate),
        "evidence_sha256": canonical_hash(expected_evidence),
        "fallback": "deterministic_host" if reasons else "none",
    }
    return {**decision_material, "ldt_sha256": canonical_hash(decision_material)}


def materialize_capability_request(
    protocol: dict[str, Any],
    candidate: dict[str, Any],
    ldt_decision: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if ldt_decision.get("decision") != "candidate_valid":
        raise ContractError("LDT validation is required before capability materialization")
    if ldt_decision.get("candidate_sha256") != canonical_hash(candidate):
        raise ContractError("candidate changed after LDT validation")
    material = candidate.get("capability_request")
    if not isinstance(material, dict):
        raise ContractError("candidate does not contain capability request material")
    validate_capability_material(
        material,
        now=now or datetime.now(timezone.utc),
        max_lifetime_seconds=int(protocol["ldt"]["max_capability_lifetime_seconds"]),
    )
    return {**material, "invocationFingerprint": capability_request_fingerprint(material)}


def seal_mesh_receipt(
    *,
    envelope: dict[str, Any],
    proposal: dict[str, Any],
    route: dict[str, Any],
    ldt_decision: dict[str, Any],
    fallback: dict[str, Any],
) -> dict[str, Any]:
    material = {
        "schema": RECEIPT_SCHEMA,
        "envelope_sha256": canonical_hash(envelope),
        "proposal_sha256": canonical_hash(proposal),
        "route_sha256": canonical_hash(route),
        "ldt_sha256": canonical_hash(ldt_decision),
        "fallback_sha256": canonical_hash(fallback),
    }
    return {**material, "decision_sha256": canonical_hash(material)}


def verify_mesh_receipt(
    receipt: dict[str, Any],
    *,
    envelope: dict[str, Any],
    proposal: dict[str, Any],
    route: dict[str, Any],
    ldt_decision: dict[str, Any],
    fallback: dict[str, Any],
) -> bool:
    expected = seal_mesh_receipt(
        envelope=envelope,
        proposal=proposal,
        route=route,
        ldt_decision=ldt_decision,
        fallback=fallback,
    )
    return receipt == expected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--protocol", required=True)
    register.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        print(json.dumps(register_protocol(args.protocol, args.output_dir), indent=2, sort_keys=True))
        return 0
    raise ContractError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
