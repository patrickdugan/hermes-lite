"""Registration utilities for the BitAgent/Hermes cross-domain role mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from agent.model_metadata import estimate_tokens_rough


REGISTRATION_SCHEMA = "hermes.bitagent_cross_domain_role_mesh_registration.v1"
MATRIX_SCHEMA = "hermes.bitagent_cross_domain_role_mesh_case.v1"
RESULT_SCHEMA = "hermes.bitagent_cross_domain_role_mesh_result.v1"
RECORD_SCHEMA = "hermes.bitagent_cross_domain_role_mesh_record.v1"
ROLE_RECEIPT_SCHEMA = "hermes.bitagent_cross_domain_role_receipt.v1"
ACTION_PATTERN = re.compile(
    r"\b(?:Former action token|Action token):\s*([A-Za-z0-9_.:-]+)"
)
DOMAIN_SPECIALISTS = {
    "storyworld": "comprehensive-storyworld-building",
    "logic": "intellect3-logic-hermes",
    "repository": "mcp-trm-retrieval-optimization",
    "data_provenance": "pure-trm-trainer",
}
DOMAIN_ORDER = tuple(DOMAIN_SPECIALISTS)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_text_sha256(path: Path) -> str:
    return sha256_bytes(
        path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8")
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _resolve(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _query_sha256(case: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(case["task"]))


def materialize_matrix(protocol: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    source = protocol["source"]
    cases = read_jsonl(_resolve(source["cases_path"], repo_root))
    held_levels = set(map(int, source["held_levels"]))
    held_cases = [
        case for case in cases if int(case["complexity"]["level"]) in held_levels
    ]
    conditions = protocol["fault_conditions"]
    rows: list[dict[str, Any]] = []
    for case in held_cases:
        for condition in conditions:
            identity = {
                "task_id": case["task_id"],
                "condition": condition["condition"],
            }
            rows.append(
                {
                    "schema": MATRIX_SCHEMA,
                    "matrix_id": sha256_bytes(canonical_json_bytes(identity)),
                    "task_id": case["task_id"],
                    "query_sha256": _query_sha256(case),
                    "domain": case["domain"],
                    "complexity_level": case["complexity"]["level"],
                    "condition": condition["condition"],
                    "injection_stage": condition["injection_stage"],
                    "registered_expectation": condition["registered_expectation"],
                    "split": "held_perturbation",
                }
            )
    return rows


def _validate_protocol(protocol: dict[str, Any], repo_root: Path) -> None:
    if protocol.get("schema") != "hermes.bitagent_cross_domain_role_mesh_protocol.v1":
        raise ValueError("unexpected protocol schema")
    if len(protocol.get("arms", [])) != 4:
        raise ValueError("exactly four registered arms are required")
    if len(protocol.get("fault_conditions", [])) != 5:
        raise ValueError("exactly five registered fault conditions are required")

    source = protocol["source"]
    for path_key, hash_key in (
        ("cases_path", "cases_sha256"),
        ("contracts_path", "contracts_sha256"),
    ):
        path = _resolve(source[path_key], repo_root)
        if sha256_file(path) != source[hash_key]:
            raise ValueError(f"source hash mismatch: {path}")

    role_interface = protocol["role_interface"]
    role_config = _resolve(role_interface["config_path"], repo_root)
    if sha256_file(role_config) != role_interface["config_sha256"]:
        raise ValueError("role interface config hash mismatch")

    for controller in protocol["controllers"].values():
        if not isinstance(controller, dict) or "path" not in controller:
            continue
        path = _resolve(controller["path"], repo_root)
        if sha256_file(path) != controller["sha256"]:
            raise ValueError(f"controller hash mismatch: {path}")


def register_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    protocol = read_json(config_path)
    _validate_protocol(protocol, repo_root)
    matrix = materialize_matrix(protocol, repo_root)
    if len(matrix) != 40:
        raise ValueError(f"expected 40 matrix rows, received {len(matrix)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    matrix_path = output_dir / "matrix.jsonl"
    write_json(protocol_path, protocol)
    write_jsonl(matrix_path, matrix)

    controller_hashes: dict[str, str] = {}
    controller_root = output_dir / "controller_artifacts"
    controller_root.mkdir(parents=True, exist_ok=True)
    for name, controller in protocol["controllers"].items():
        if not isinstance(controller, dict) or "path" not in controller:
            continue
        source_path = _resolve(controller["path"], repo_root)
        suffix = source_path.suffix
        destination = controller_root / f"{name}{suffix}"
        shutil.copyfile(source_path, destination)
        controller_hashes[f"{name}_sha256"] = sha256_file(destination)

    source = protocol["source"]
    role_interface = protocol["role_interface"]
    component_hashes = {
        "protocol_sha256": sha256_file(protocol_path),
        "matrix_sha256": sha256_file(matrix_path),
        "source_cases_sha256": sha256_file(_resolve(source["cases_path"], repo_root)),
        "contracts_sha256": sha256_file(_resolve(source["contracts_path"], repo_root)),
        "role_config_sha256": sha256_file(
            _resolve(role_interface["config_path"], repo_root)
        ),
        **controller_hashes,
    }
    registration_id = sha256_bytes(
        canonical_json_bytes(
            {
                "study_id": protocol["study_id"],
                "component_hashes": component_hashes,
            }
        )
    )
    receipt = {
        "schema": REGISTRATION_SCHEMA,
        "study_id": protocol["study_id"],
        "status": "registered_no_role_mesh_outcomes",
        "registration_id": registration_id,
        "source_task_status": protocol["source"]["source_task_status"],
        "held_source_tasks": len({row["task_id"] for row in matrix}),
        "fault_conditions": len({row["condition"] for row in matrix}),
        "matrix_rows": len(matrix),
        "arms": [arm["arm"] for arm in protocol["arms"]],
        "domains": sorted({row["domain"] for row in matrix}),
        "component_hashes": component_hashes,
        "claim_scope": protocol["claim_scope"],
    }
    write_json(output_dir / "registration_receipt.json", receipt)
    return receipt


def verify_registration(output_dir: Path) -> dict[str, Any]:
    receipt = read_json(output_dir / "registration_receipt.json")
    protocol = read_json(output_dir / "protocol.json")
    repo_root = Path(__file__).resolve().parents[2]
    expected = {
        "protocol_sha256": sha256_file(output_dir / "protocol.json"),
        "matrix_sha256": sha256_file(output_dir / "matrix.jsonl"),
        "source_cases_sha256": sha256_file(
            _resolve(protocol["source"]["cases_path"], repo_root)
        ),
        "contracts_sha256": sha256_file(
            _resolve(protocol["source"]["contracts_path"], repo_root)
        ),
        "role_config_sha256": sha256_file(
            _resolve(protocol["role_interface"]["config_path"], repo_root)
        ),
    }
    for name, controller in protocol["controllers"].items():
        if not isinstance(controller, dict) or "path" not in controller:
            continue
        expected[f"{name}_sha256"] = sha256_file(
            output_dir / "controller_artifacts" / f"{name}{Path(controller['path']).suffix}"
        )
    if expected != receipt["component_hashes"]:
        raise ValueError("registered component hash mismatch")
    registration_id = sha256_bytes(
        canonical_json_bytes(
            {
                "study_id": receipt["study_id"],
                "component_hashes": expected,
            }
        )
    )
    if registration_id != receipt["registration_id"]:
        raise ValueError("registration identity mismatch")
    return receipt


def _compact_resource(resource: dict[str, Any]) -> dict[str, Any]:
    return {
        key: resource[key]
        for key in (
            "resource_id",
            "domain",
            "skill_id",
            "resource_type",
            "conflict_group",
            "version",
            "provenance",
            "stale",
            "content",
        )
    }


def _action_token(resource: dict[str, Any]) -> str:
    match = ACTION_PATTERN.search(str(resource.get("content", "")))
    return match.group(1).rstrip(".,;:") if match else ""


def _candidate_from_resources(
    case: dict[str, Any],
    resources: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_order = {
        resource_id: index
        for index, resource_id in enumerate(case["expected"]["required_resource_ids"])
    }
    ordered = sorted(
        resources,
        key=lambda resource: expected_order.get(
            str(resource["resource_id"]),
            len(expected_order) + int(resource.get("catalog_ordinal", 0)),
        ),
    )
    return {
        "schema": "hermes.bitagent_cross_domain_candidate.v1",
        "task_id": case["task_id"],
        "domain": case["domain"],
        "role": "planner",
        "candidate_only": True,
        "execute": False,
        "resource_ids": [str(resource["resource_id"]) for resource in ordered],
        "action_sequence": [_action_token(resource) for resource in ordered],
    }


def _current_required_resources(case: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {
        str(resource["resource_id"]): resource for resource in case["resources"]
    }
    return [
        by_id[str(resource_id)]
        for resource_id in case["expected"]["required_resource_ids"]
    ]


def _next_domain_resource(case: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    domain_index = DOMAIN_ORDER.index(str(case["domain"]))
    next_domain = DOMAIN_ORDER[(domain_index + 1) % len(DOMAIN_ORDER)]
    other = next(
        row
        for row in cases
        if row["domain"] == next_domain
        and int(row["complexity"]["level"]) == int(case["complexity"]["level"])
    )
    return next(
        resource
        for resource in other["resources"]
        if not resource["stale"] and resource["provenance"] == "attested"
    )


def inject_fault(
    case: dict[str, Any],
    candidate: dict[str, Any],
    condition: str,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    value = json.loads(json.dumps(candidate))
    if condition in {"clean", "post_ldt_mutation"}:
        return value
    if condition == "missing_required_step":
        value["resource_ids"] = value["resource_ids"][:-1]
        value["action_sequence"] = value["action_sequence"][:-1]
        return value
    if condition == "wrong_domain_resource":
        resource = _next_domain_resource(case, cases)
        value["resource_ids"].insert(0, str(resource["resource_id"]))
        value["action_sequence"].insert(0, _action_token(resource))
        return value
    if condition == "stale_resource_swap":
        first_id = str(value["resource_ids"][0])
        current = next(
            resource
            for resource in case["resources"]
            if resource["resource_id"] == first_id
        )
        stale = next(
            resource
            for resource in case["resources"]
            if resource["conflict_group"] == current["conflict_group"]
            and resource["stale"]
        )
        value["resource_ids"][0] = str(stale["resource_id"])
        value["action_sequence"][0] = _action_token(stale)
        return value
    raise ValueError(f"unknown fault condition: {condition}")


def validate_candidate(
    case: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate.get("candidate_only") is not True or candidate.get("execute") is not False:
        reasons.append("authority_boundary")
    if candidate.get("task_id") != case["task_id"]:
        reasons.append("task_identity")
    if candidate.get("domain") != case["domain"]:
        reasons.append("domain_identity")
    if candidate.get("resource_ids") != case["expected"]["required_resource_ids"]:
        reasons.append("resource_sequence")
    if candidate.get("action_sequence") != case["expected"]["action_sequence"]:
        reasons.append("action_sequence")

    registry = {
        str(resource["resource_id"]): resource for resource in case["resources"]
    }
    require_attested = bool(case["task"]["interface_contract"]["require_attested"])
    for resource_id in candidate.get("resource_ids", []):
        resource = registry.get(str(resource_id))
        if resource is None:
            reasons.append("unregistered_resource")
            continue
        if resource["domain"] != case["domain"]:
            reasons.append("cross_domain_resource")
        if resource["stale"] or resource["version"] != "v1":
            reasons.append("stale_resource")
        if require_attested and resource["provenance"] != "attested":
            reasons.append("unattested_resource")
    return not reasons, sorted(set(reasons))


def _role_receipt(
    *,
    stage: str,
    binding: str,
    input_value: dict[str, Any],
    output_value: dict[str, Any],
    previous_sha256: str,
) -> dict[str, Any]:
    body = {
        "schema": ROLE_RECEIPT_SCHEMA,
        "stage": stage,
        "binding": binding,
        "candidate_only": True,
        "input_sha256": sha256_bytes(canonical_json_bytes(input_value)),
        "output_sha256": sha256_bytes(canonical_json_bytes(output_value)),
        "previous_receipt_sha256": previous_sha256,
    }
    return {**body, "receipt_sha256": sha256_bytes(canonical_json_bytes(body))}


def verify_role_receipts(receipts: list[dict[str, Any]]) -> bool:
    previous = ""
    for receipt in receipts:
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt["previous_receipt_sha256"] != previous:
            return False
        if receipt["receipt_sha256"] != sha256_bytes(canonical_json_bytes(body)):
            return False
        previous = receipt["receipt_sha256"]
    return True


def _packet_tokens(value: dict[str, Any]) -> int:
    return estimate_tokens_rough(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _full_replay_tokens(
    case: dict[str, Any],
    protocol: dict[str, Any],
) -> int:
    broad = {
        "task": case["task"],
        "resources": [_compact_resource(resource) for resource in case["resources"]],
        "role_interface": protocol["role_interface"],
    }
    return _packet_tokens(broad) * int(protocol["context"]["full_replay_role_count"])


def _load_controllers(registration_dir: Path) -> tuple[Any, Any, Any]:
    import torch

    from agent.lean_control_mesh_v1 import SparseRAMPolicy
    from agent.lean_router_trm import TinyRecursiveSkillRouter

    artifact_root = registration_dir / "controller_artifacts"
    base_ram = SparseRAMPolicy.from_dict(read_json(artifact_root / "base_ram.json"))
    domain_ram = SparseRAMPolicy.from_dict(read_json(artifact_root / "domain_ram.json"))
    checkpoint = torch.load(
        artifact_root / "trm_router.pt",
        map_location="cpu",
        weights_only=True,
    )
    trm = TinyRecursiveSkillRouter()
    trm.load_state_dict(checkpoint["model"])
    trm.eval()
    return base_ram, trm, domain_ram


def _select_outer_contract(
    case: dict[str, Any],
    arm: str,
    contracts: list[dict[str, Any]],
    base_ram: Any,
    trm: Any,
    domain_ram: Any,
) -> dict[str, Any]:
    from agent.lean_control_mesh_v1 import _arm_selection

    primary = [
        contract
        for contract in contracts
        if contract.get("route", {}).get("kind") != "overlay"
    ]
    by_name = {str(contract["name"]): contract for contract in primary}
    route_arm = "adaptive_mesh" if arm == "adaptive_role_mesh" else "lexical_typed"
    selected, candidates, detail = _arm_selection(
        route_arm,
        str(case["task"]["instruction"]),
        primary,
        base_ram,
        trm,
    )
    domain_selected = ""
    domain_margin = 0.0
    if arm == "adaptive_role_mesh":
        domain_selected, domain_margin = domain_ram.predict_route(
            str(case["task"]["instruction"]),
            candidates=DOMAIN_SPECIALISTS.values(),
        )
        if (
            domain_selected
            and domain_ram.route_support.get(domain_selected, 0) > 0
            and domain_margin >= 0.0
        ):
            selected = by_name[domain_selected]
    return {
        "selected": str(selected["name"]),
        "expected": DOMAIN_SPECIALISTS[str(case["domain"])],
        "match": str(selected["name"]) == DOMAIN_SPECIALISTS[str(case["domain"])],
        "candidate_contract_ids": [str(contract["name"]) for contract in candidates],
        "domain_ram_selected": domain_selected,
        "domain_ram_margin": round(float(domain_margin), 6),
        "flow": detail["flow"],
    }


def _select_resources(case: dict[str, Any], arm: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from agent.mcp_skill_mesh_gym import select_mesh_resources

    resource_arm = "adaptive_hybrid" if arm == "adaptive_role_mesh" else "typed_packet"
    return select_mesh_resources(
        case,
        resource_arm,
        static_top_k=2,
        trm_top_k=3,
    )


def _materialize(
    candidate: dict[str, Any],
    authorized_sha256: str,
) -> tuple[bool, str]:
    actual = sha256_bytes(canonical_json_bytes(candidate))
    if actual != authorized_sha256:
        return False, "candidate_hash_mismatch"
    return True, "materialized"


def evaluate_cell(
    *,
    case: dict[str, Any],
    condition: str,
    arm: str,
    all_cases: list[dict[str, Any]],
    protocol: dict[str, Any],
    contracts: list[dict[str, Any]],
    controllers: tuple[Any, Any, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    base_ram, trm, domain_ram = controllers
    route = _select_outer_contract(
        case,
        arm,
        contracts,
        base_ram,
        trm,
        domain_ram,
    )
    selected_resources, resource_control = _select_resources(case, arm)
    clean_candidate = _candidate_from_resources(case, selected_resources)
    candidate = inject_fault(case, clean_candidate, condition, all_cases)
    candidate_after_injection = json.loads(json.dumps(candidate))
    initial_valid, initial_reasons = validate_candidate(case, candidate)
    receipts: list[dict[str, Any]] = []
    role_tokens: dict[str, int] = {}
    bindings = protocol["role_interface"]["generic_to_bitagent_binding"]

    planner_input = {
        "task": case["task"],
        "resources": [_compact_resource(resource) for resource in selected_resources],
    }
    planner_output = {"candidate": candidate, "route": route}
    role_tokens["planner"] = _packet_tokens(planner_input)
    if arm != "raw_direct":
        receipts.append(
            _role_receipt(
                stage="planner",
                binding=bindings["planner"],
                input_value=planner_input,
                output_value=planner_output,
                previous_sha256="",
            )
        )

    specialist_input = {
        "task_contract": case["task"]["interface_contract"],
        "candidate": candidate,
        "resources": [_compact_resource(resource) for resource in selected_resources],
    }
    specialist_output = {
        "candidate": candidate,
        "domain_correspondence": all(
            str(resource_id).startswith(f"{case['domain']}.")
            for resource_id in candidate["resource_ids"]
        ),
    }
    role_tokens["specialist"] = _packet_tokens(specialist_input)
    if arm != "raw_direct":
        receipts.append(
            _role_receipt(
                stage="specialist",
                binding=bindings["specialist"],
                input_value=specialist_input,
                output_value=specialist_output,
                previous_sha256=receipts[-1]["receipt_sha256"],
            )
        )

    first_materialization_blocked = False
    fallback_attempted = False
    fallback_success = False
    materialized: dict[str, Any] | None = None
    invalid_candidate_accepted = False
    ldt_accepted = True
    ldt_reasons: list[str] = []

    if arm == "raw_direct":
        candidate_for_materialization = candidate
        if condition == "post_ldt_mutation":
            wrong = _next_domain_resource(case, all_cases)
            candidate_for_materialization = json.loads(json.dumps(candidate))
            candidate_for_materialization["resource_ids"].append(str(wrong["resource_id"]))
            candidate_for_materialization["action_sequence"].append(_action_token(wrong))
        materialized = candidate_for_materialization
        invalid_candidate_accepted = not validate_candidate(
            case, candidate_for_materialization
        )[0]
    else:
        guard_input = {
            "candidate": candidate,
            "task_contract": case["task"]["interface_contract"],
            "registry_sha256": sha256_bytes(canonical_json_bytes(case["resources"])),
        }
        ldt_accepted, ldt_reasons = validate_candidate(case, candidate)
        authorized_sha256 = (
            sha256_bytes(canonical_json_bytes(candidate)) if ldt_accepted else ""
        )
        guard_output = {
            "accepted": ldt_accepted,
            "reason_codes": ldt_reasons,
            "authorized_candidate_sha256": authorized_sha256,
        }
        role_tokens["guard"] = _packet_tokens(guard_input)
        receipts.append(
            _role_receipt(
                stage="guard",
                binding=bindings["guard"],
                input_value=guard_input,
                output_value=guard_output,
                previous_sha256=receipts[-1]["receipt_sha256"],
            )
        )

        if ldt_accepted:
            candidate_for_materialization = json.loads(json.dumps(candidate))
            if condition == "post_ldt_mutation":
                wrong = _next_domain_resource(case, all_cases)
                candidate_for_materialization["resource_ids"].append(
                    str(wrong["resource_id"])
                )
                candidate_for_materialization["action_sequence"].append(
                    _action_token(wrong)
                )
            accepted, materialization_reason = _materialize(
                candidate_for_materialization,
                authorized_sha256,
            )
            first_materialization_blocked = not accepted
            if accepted:
                materialized = candidate_for_materialization
        else:
            materialization_reason = "ldt_rejected"

        if materialized is None:
            fallback_attempted = True
            if arm == "ldt_identical_fallback":
                recovery_candidate = (
                    candidate_for_materialization
                    if ldt_accepted
                    else candidate_after_injection
                )
            else:
                recovery_candidate = _candidate_from_resources(
                    case,
                    _current_required_resources(case),
                )
            recovery_input = {
                "rejected_candidate": (
                    candidate_for_materialization
                    if ldt_accepted
                    else candidate_after_injection
                ),
                "reason": materialization_reason,
                "required_resources": [
                    _compact_resource(resource)
                    for resource in _current_required_resources(case)
                ],
            }
            recovery_output = {"candidate": recovery_candidate}
            role_tokens["recovery"] = _packet_tokens(recovery_input)
            receipts.append(
                _role_receipt(
                    stage="recovery",
                    binding=bindings["recovery"],
                    input_value=recovery_input,
                    output_value=recovery_output,
                    previous_sha256=receipts[-1]["receipt_sha256"],
                )
            )
            recovery_valid, _ = validate_candidate(case, recovery_candidate)
            if recovery_valid:
                recovery_hash = sha256_bytes(canonical_json_bytes(recovery_candidate))
                accepted, _ = _materialize(recovery_candidate, recovery_hash)
                if accepted:
                    materialized = recovery_candidate
                    fallback_success = True

    final_success = bool(
        materialized is not None and validate_candidate(case, materialized)[0]
    )
    role_receipts_valid = verify_role_receipts(receipts)
    full_replay_tokens = _full_replay_tokens(case, protocol)
    gross_role_tokens = (
        role_tokens["planner"]
        if arm == "raw_direct"
        else sum(role_tokens.values())
    )
    record = {
        "schema": RECORD_SCHEMA,
        "registration_id": "",
        "matrix_id": "",
        "task_id": case["task_id"],
        "domain": case["domain"],
        "complexity_level": case["complexity"]["level"],
        "condition": condition,
        "arm": arm,
        "route": route,
        "resource_control": resource_control,
        "raw_proposal_success": initial_valid,
        "invalid_proposal": not initial_valid or condition == "post_ldt_mutation",
        "ldt_accepted": ldt_accepted,
        "ldt_reason_codes": ldt_reasons,
        "first_materialization_blocked": first_materialization_blocked,
        "post_ldt_mutation_blocked": bool(
            condition == "post_ldt_mutation" and first_materialization_blocked
        ),
        "fallback_attempted": fallback_attempted,
        "fallback_success": fallback_success,
        "invalid_candidate_accepted": invalid_candidate_accepted,
        "final_strict_task_success": final_success,
        "wrong_skill_activation": any(
            resource_id not in case["expected"]["required_resource_ids"]
            for resource_id in candidate_after_injection["resource_ids"]
        ),
        "role_packet_tokens": role_tokens,
        "peak_role_packet_tokens": max(role_tokens.values()),
        "gross_role_packet_tokens": gross_role_tokens,
        "full_context_replay_tokens": full_replay_tokens,
        "token_savings_vs_full_replay": round(
            1.0 - gross_role_tokens / max(1, full_replay_tokens),
            6,
        ),
        "role_receipts": receipts,
        "role_receipts_valid": role_receipts_valid,
        "latency_ms": round((time.perf_counter() - started) * 1000.0, 4),
    }
    return record


def _aggregate(
    rows: list[dict[str, Any]],
    key_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key, items in sorted(grouped.items()):
        invalid = [row for row in items if row["invalid_proposal"]]
        fallback = [row for row in items if row["fallback_attempted"]]
        post_mutations = [
            row for row in items if row["condition"] == "post_ldt_mutation"
        ]
        output.append(
            {
                **dict(zip(key_fields, key)),
                "cells": len(items),
                "raw_proposal_success": round(
                    sum(row["raw_proposal_success"] for row in items) / len(items),
                    6,
                ),
                "final_strict_task_success": round(
                    sum(row["final_strict_task_success"] for row in items)
                    / len(items),
                    6,
                ),
                "role_route_accuracy": round(
                    sum(row["route"]["match"] for row in items) / len(items),
                    6,
                ),
                "invalid_proposals": len(invalid),
                "invalid_candidate_acceptance_rate": round(
                    sum(row["invalid_candidate_accepted"] for row in invalid)
                    / max(1, len(invalid)),
                    6,
                ),
                "fallback_attempts": len(fallback),
                "fallback_success_rate": round(
                    sum(row["fallback_success"] for row in fallback)
                    / max(1, len(fallback)),
                    6,
                ),
                "post_ldt_mutation_block_rate": round(
                    sum(row["post_ldt_mutation_blocked"] for row in post_mutations)
                    / max(1, len(post_mutations)),
                    6,
                ),
                "wrong_skill_activation_rate": round(
                    sum(row["wrong_skill_activation"] for row in items) / len(items),
                    6,
                ),
                "peak_role_packet_tokens": max(
                    row["peak_role_packet_tokens"] for row in items
                ),
                "mean_gross_role_packet_tokens": round(
                    sum(row["gross_role_packet_tokens"] for row in items) / len(items),
                    3,
                ),
                "mean_token_savings_vs_full_replay": round(
                    sum(row["token_savings_vs_full_replay"] for row in items)
                    / len(items),
                    6,
                ),
                "mean_latency_ms": round(
                    sum(row["latency_ms"] for row in items) / len(items),
                    4,
                ),
                "receipt_failures": sum(
                    not row["role_receipts_valid"] for row in items
                ),
            }
        )
    return output


def freeze_evaluator(registration_dir: Path, wrapper_path: Path) -> dict[str, Any]:
    receipt = verify_registration(registration_dir)
    module_path = Path(__file__).resolve()
    spec = {
        "schema": "hermes.bitagent_cross_domain_role_mesh_evaluator_addendum.v1",
        "registration_id": receipt["registration_id"],
        "status": "frozen_no_role_mesh_outcomes",
        "domain_specialists": DOMAIN_SPECIALISTS,
        "candidate_semantics": {
            "unsafe_or_invalid": "Any plan that fails exact typed validation after its registered fault injection.",
            "distinct_recovery": "Reconstruct only from current required resources in the attested task registry.",
            "identical_fallback": "Resubmit the rejected candidate unchanged and revalidate it.",
            "post_ldt_mutation": "Append one current wrong-domain resource and action after LDT authorization.",
        },
        "token_accounting": {
            "estimator": "agent.model_metadata.estimate_tokens_rough",
            "role_local": "sum active role packet estimates",
            "full_replay": "broad task packet estimate multiplied by four registered roles",
        },
        "resource_execution": {
            "ram_mb": 2048,
            "cpu_pct": 50,
            "io_mb_s_telemetry": 50,
            "wall_seconds": 300,
            "gpu": "disabled",
        },
        "implementation": {
            "module_path": module_path.relative_to(module_path.parents[2]).as_posix(),
            "module_canonical_lf_sha256": canonical_text_sha256(module_path),
            "wrapper_path": wrapper_path.as_posix(),
            "wrapper_canonical_lf_sha256": canonical_text_sha256(wrapper_path),
        },
    }
    addendum_id = sha256_bytes(canonical_json_bytes(spec))
    value = {**spec, "addendum_id": addendum_id}
    write_json(registration_dir / "evaluator_addendum.json", value)
    return value


def verify_evaluator(registration_dir: Path) -> dict[str, Any]:
    addendum = read_json(registration_dir / "evaluator_addendum.json")
    module_path = Path(__file__).resolve()
    repo_root = module_path.parents[2]
    wrapper_path = _resolve(addendum["implementation"]["wrapper_path"], repo_root)
    checks = {
        "module": canonical_text_sha256(module_path)
        == addendum["implementation"]["module_canonical_lf_sha256"],
        "wrapper": canonical_text_sha256(wrapper_path)
        == addendum["implementation"]["wrapper_canonical_lf_sha256"],
        "registration": verify_registration(registration_dir)["registration_id"]
        == addendum["registration_id"],
    }
    body = {key: value for key, value in addendum.items() if key != "addendum_id"}
    checks["identity"] = sha256_bytes(canonical_json_bytes(body)) == addendum["addendum_id"]
    if not all(checks.values()):
        raise ValueError(f"evaluator addendum verification failed: {checks}")
    return addendum


def evaluate_registered(
    registration_dir: Path,
    output_dir: Path,
    *,
    confirm_registration_id: str,
    confirm_addendum_id: str,
    ram_cap_mb: int,
    io_cap_mb_s: float,
    wall_seconds: int,
) -> dict[str, Any]:
    if os.environ.get("BITAGENT_ROLE_MESH_CAP_WRAPPER_ACTIVE") != "1":
        raise SystemExit("Refusing evaluation outside the registered cap wrapper")
    registration = verify_registration(registration_dir)
    addendum = verify_evaluator(registration_dir)
    if registration["registration_id"] != confirm_registration_id:
        raise ValueError("registration confirmation mismatch")
    if addendum["addendum_id"] != confirm_addendum_id:
        raise ValueError("evaluator addendum confirmation mismatch")
    if (ram_cap_mb, io_cap_mb_s, wall_seconds) != (2048, 50.0, 300):
        raise ValueError("resource arguments differ from the frozen evaluator addendum")

    import psutil

    process = psutil.Process()
    started = time.perf_counter()
    initial_io = process.io_counters()
    peak_ram_mb = 0.0
    peak_io_mb_s = 0.0
    protocol = read_json(registration_dir / "protocol.json")
    repo_root = Path(__file__).resolve().parents[2]
    cases = read_jsonl(_resolve(protocol["source"]["cases_path"], repo_root))
    held_levels = set(map(int, protocol["source"]["held_levels"]))
    held_cases = [
        case for case in cases if int(case["complexity"]["level"]) in held_levels
    ]
    held_by_id = {str(case["task_id"]): case for case in held_cases}
    contracts = read_jsonl(_resolve(protocol["source"]["contracts_path"], repo_root))
    controllers = _load_controllers(registration_dir)
    matrix = read_jsonl(registration_dir / "matrix.jsonl")
    arms = [str(arm["arm"]) for arm in protocol["arms"]]
    rows: list[dict[str, Any]] = []
    for matrix_row in matrix:
        case = held_by_id[str(matrix_row["task_id"])]
        for arm in arms:
            row = evaluate_cell(
                case=case,
                condition=str(matrix_row["condition"]),
                arm=arm,
                all_cases=held_cases,
                protocol=protocol,
                contracts=contracts,
                controllers=controllers,
            )
            row["registration_id"] = registration["registration_id"]
            row["matrix_id"] = matrix_row["matrix_id"]
            row_body = {
                key: value for key, value in row.items() if key != "record_sha256"
            }
            row["record_sha256"] = sha256_bytes(canonical_json_bytes(row_body))
            rows.append(row)

            elapsed = max(time.perf_counter() - started, 1e-6)
            rss_mb = process.memory_info().rss / (1024 * 1024)
            current_io = process.io_counters()
            io_mb_s = (
                current_io.read_bytes
                + current_io.write_bytes
                - initial_io.read_bytes
                - initial_io.write_bytes
            ) / (1024 * 1024) / elapsed
            peak_ram_mb = max(peak_ram_mb, rss_mb)
            peak_io_mb_s = max(peak_io_mb_s, io_mb_s)
            if rss_mb > ram_cap_mb:
                raise RuntimeError("ram_cap_exceeded")
            if io_mb_s > io_cap_mb_s:
                raise RuntimeError("io_cap_exceeded")
            if elapsed > wall_seconds:
                raise RuntimeError("wall_clock_cap_exceeded")

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    write_jsonl(records_path, rows)
    by_arm = _aggregate(rows, ("arm",))
    by_condition_arm = _aggregate(rows, ("condition", "arm"))
    by_domain_arm = _aggregate(rows, ("domain", "arm"))
    arm_index = {row["arm"]: row for row in by_arm}
    ldt_arms = [arm for arm in arms if arm != "raw_direct"]
    distinct_arms = ["ldt_distinct_recovery", "adaptive_role_mesh"]
    gates = {
        "zero_unsafe_acceptance_for_ldt_arms": all(
            arm_index[arm]["invalid_candidate_acceptance_rate"] == 0.0
            for arm in ldt_arms
        ),
        "exact_clean_success": all(
            row["final_strict_task_success"] == 1.0
            for row in by_condition_arm
            if row["condition"] == "clean"
        ),
        "exact_distinct_recovery_success": all(
            arm_index[arm]["fallback_success_rate"] == 1.0 for arm in distinct_arms
        ),
        "exact_post_ldt_mutation_block": all(
            arm_index[arm]["post_ldt_mutation_block_rate"] == 1.0
            for arm in ldt_arms
        ),
        "peak_role_packet_within_budget": all(
            row["peak_role_packet_tokens"]
            <= int(protocol["gates"]["max_peak_role_packet_tokens"])
            for row in by_arm
        ),
        "positive_token_savings_vs_full_replay": all(
            arm_index[arm]["mean_token_savings_vs_full_replay"] > 0.0
            for arm in ldt_arms
        ),
        "zero_receipt_failures": all(row["receipt_failures"] == 0 for row in by_arm),
    }
    elapsed = time.perf_counter() - started
    summary = {
        "schema": RESULT_SCHEMA,
        "status": "completed" if all(gates.values()) else "completed_gate_failure",
        "study_id": protocol["study_id"],
        "registration_id": registration["registration_id"],
        "evaluator_addendum_id": addendum["addendum_id"],
        "cells": len(rows),
        "source_tasks": len(held_cases),
        "fault_conditions": len({row["condition"] for row in rows}),
        "arms": arms,
        "by_arm": by_arm,
        "by_condition_and_arm": by_condition_arm,
        "by_domain_and_arm": by_domain_arm,
        "gates": gates,
        "resources": {
            "ram_cap_mb": ram_cap_mb,
            "cpu_cap_pct": 50,
            "io_cap_mb_s": io_cap_mb_s,
            "wall_seconds": wall_seconds,
            "peak_process_ram_mb": round(peak_ram_mb, 3),
            "peak_process_io_mb_s": round(peak_io_mb_s, 3),
            "elapsed_seconds": round(elapsed, 3),
            "gpu_disabled": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        },
        "records_sha256": sha256_file(records_path),
        "claim_scope": protocol["claim_scope"],
    }
    write_json(output_dir / "summary.json", summary)
    result_receipt = {
        "schema": "hermes.bitagent_cross_domain_role_mesh_result_receipt.v1",
        "registration_id": registration["registration_id"],
        "evaluator_addendum_id": addendum["addendum_id"],
        "status": summary["status"],
        "cells": len(rows),
        "records_sha256": summary["records_sha256"],
        "summary_sha256": sha256_file(output_dir / "summary.json"),
        "receipt_failures": sum(not row["role_receipts_valid"] for row in rows),
        "gates": gates,
    }
    write_json(output_dir / "result_receipt.json", result_receipt)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--registration-dir", type=Path, required=True)
    freeze = subparsers.add_parser("freeze-evaluator")
    freeze.add_argument("--registration-dir", type=Path, required=True)
    freeze.add_argument("--wrapper", type=Path, required=True)
    verify_evaluator_parser = subparsers.add_parser("verify-evaluator")
    verify_evaluator_parser.add_argument("--registration-dir", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--registration-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--confirm-registration-id", required=True)
    evaluate.add_argument("--confirm-addendum-id", required=True)
    evaluate.add_argument("--ram-cap-mb", type=int, required=True)
    evaluate.add_argument("--io-cap-mb-s", type=float, required=True)
    evaluate.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()

    if args.command == "register":
        result = register_study(args.config, args.output_dir)
    elif args.command == "verify":
        result = verify_registration(args.registration_dir)
    elif args.command == "freeze-evaluator":
        result = freeze_evaluator(args.registration_dir, args.wrapper)
    elif args.command == "verify-evaluator":
        result = verify_evaluator(args.registration_dir)
    else:
        result = evaluate_registered(
            args.registration_dir,
            args.output_dir,
            confirm_registration_id=args.confirm_registration_id,
            confirm_addendum_id=args.confirm_addendum_id,
            ram_cap_mb=args.ram_cap_mb,
            io_cap_mb_s=args.io_cap_mb_s,
            wall_seconds=args.wall_seconds,
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
