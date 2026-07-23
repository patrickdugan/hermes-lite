"""Sealed MCP retrieval-mesh benchmark for compact local models.

The benchmark separates construction, deterministic packet calibration, and
live model evaluation. A live run is accepted only when it names the exact
registration ID produced before outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from agent.mcp_trm_ldt_lab import (
    DEFAULT_LOCAL_BONSAI_BASE_URL,
    _post_json,
    gpu_pressure_snapshot,
    run_live_smoke,
)
from agent.model_metadata import estimate_tokens_rough


SCHEMA_VERSION = "hermes.mcp_skill_mesh_gym.v0"
CASE_SCHEMA = "hermes.mcp_skill_mesh_case.v0"
PACKET_SCHEMA = "hermes.mcp_skill_mesh_packet.v0"
MESH_ARMS = (
    "full_context",
    "static_topk",
    "typed_packet",
    "trm_rerank",
    "ldt_verified",
    "adaptive_hybrid",
)
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
        handle.flush()


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"config schema must be {SCHEMA_VERSION}")
    domains = config.get("domains")
    levels = config.get("complexity_levels")
    if not isinstance(domains, list) or len(domains) < 3:
        raise ValueError("at least three domains are required")
    if not isinstance(levels, list) or len(levels) < 4:
        raise ValueError("at least four complexity levels are required")
    if tuple(config.get("arms", [])) != MESH_ARMS:
        raise ValueError(f"arms must be exactly {list(MESH_ARMS)}")
    if int(config.get("hard_context_tokens", 0)) > 12000:
        raise ValueError("hard context exceeds the registered 12k ceiling")

    domain_ids: set[str] = set()
    for domain in domains:
        if not isinstance(domain, dict):
            raise ValueError("domain entries must be objects")
        domain_id = str(domain.get("domain_id") or "")
        if not domain_id or domain_id in domain_ids:
            raise ValueError(f"invalid or duplicate domain_id: {domain_id}")
        domain_ids.add(domain_id)
        skills = domain.get("skills")
        goals = domain.get("goals")
        if not isinstance(skills, list) or len(skills) < 4:
            raise ValueError(f"{domain_id} requires at least four skills")
        if not isinstance(goals, list) or len(goals) != len(levels):
            raise ValueError(f"{domain_id} must define one goal per complexity level")
        skill_ids = [str(skill.get("skill_id") or "") for skill in skills if isinstance(skill, dict)]
        if len(skill_ids) != len(set(skill_ids)) or any(not skill_id for skill_id in skill_ids):
            raise ValueError(f"{domain_id} has invalid skill identifiers")
        for goal in goals:
            required = goal.get("required_skills") if isinstance(goal, dict) else None
            if not isinstance(required, list) or not required:
                raise ValueError(f"{domain_id} goal has no required_skills")
            unknown = set(map(str, required)) - set(skill_ids)
            if unknown:
                raise ValueError(f"{domain_id} goal references unknown skills: {sorted(unknown)}")


def _resource(domain: dict[str, Any], skill: dict[str, Any], *, stale: bool, ordinal: int) -> dict[str, Any]:
    domain_id = str(domain["domain_id"])
    skill_id = str(skill["skill_id"])
    action = str(skill["action"])
    suffix = "legacy" if stale else "current"
    if stale:
        content = (
            f"Superseded legacy guidance for {skill['title']}. "
            f"Former action token: legacy_{action}. {skill['legacy_note']}"
        )
    else:
        content = (
            f"Current attested guidance for {skill['title']}. "
            f"Action token: {action}. {skill['procedure']}"
        )
    return {
        "resource_id": f"{domain_id}.{skill_id}.{suffix}",
        "catalog_ordinal": ordinal,
        "domain": domain_id,
        "skill_id": skill_id,
        "resource_type": str(skill["resource_type"]),
        "title": str(skill["title"]),
        "content": content,
        "version": "v0" if stale else "v1",
        "provenance": "legacy_unattested" if stale else "attested",
        "stale": stale,
        "conflict_group": f"{domain_id}.{skill_id}",
    }


def materialize_cases(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen domain grammar into canonical benchmark cases."""
    _validate_config(config)
    levels = {int(level["level"]): level for level in config["complexity_levels"]}
    cases: list[dict[str, Any]] = []

    for domain in config["domains"]:
        skills = {str(skill["skill_id"]): skill for skill in domain["skills"]}
        resources: list[dict[str, Any]] = []
        ordinal = 0
        for skill in domain["skills"]:
            # Legacy-first catalog order prevents freshness from being encoded by order.
            resources.append(_resource(domain, skill, stale=True, ordinal=ordinal))
            ordinal += 1
            resources.append(_resource(domain, skill, stale=False, ordinal=ordinal))
            ordinal += 1

        for goal in domain["goals"]:
            level_number = int(goal["level"])
            level = levels[level_number]
            required_skills = [str(value) for value in goal["required_skills"]]
            required_resources = [f"{domain['domain_id']}.{skill_id}.current" for skill_id in required_skills]
            forbidden_resources = [
                resource["resource_id"]
                for resource in resources
                if resource["stale"] or resource["skill_id"] not in required_skills
            ]
            action_sequence = [str(skills[skill_id]["action"]) for skill_id in required_skills]
            required_types = [str(skills[skill_id]["resource_type"]) for skill_id in required_skills]
            task_id = f"{domain['domain_id']}.l{level_number:02d}"
            case = {
                "schema": CASE_SCHEMA,
                "task_id": task_id,
                "domain": str(domain["domain_id"]),
                "split": str(domain["split"]),
                "bench_family": "domain_general_skill_retrieval",
                "complexity": {
                    "level": level_number,
                    "name": str(level["name"]),
                    "required_skill_count": len(required_skills),
                    "has_distractors": bool(level["has_distractors"]),
                    "has_conflict": bool(level["has_conflict"]),
                    "requires_provenance": bool(level["requires_provenance"]),
                    "stateful": bool(level["stateful"]),
                },
                "task": {
                    "task_id": task_id,
                    "instruction": str(goal["instruction"]),
                    "domain": str(domain["domain_id"]),
                    "allowed_tools": list(domain["allowed_tools"]),
                    "interface_contract": {
                        "required_resource_types": required_types,
                        "ordered_steps": len(required_skills),
                        "require_current": True,
                        "require_attested": bool(level["requires_provenance"] or level["has_conflict"]),
                    },
                },
                "resources": resources,
                "replay_candidates": [
                    {
                        "task_id": task_id,
                        "bench_family": "domain_general_skill_retrieval",
                        "passed": False,
                        "score": 0,
                        "failure": "wrong_skill_activation",
                        "resource_id": required_resources[0],
                        "action": action_sequence[0],
                        "rationale": "Prefer a task-matched current resource over a broad or stale lookup.",
                    }
                ],
                "expected": {
                    "required_skill_ids": required_skills,
                    "required_resource_ids": required_resources,
                    "forbidden_resource_ids": forbidden_resources,
                    "action_sequence": action_sequence,
                    "solvable": True,
                },
            }
            cases.append(case)

    task_ids = [case["task_id"] for case in cases]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("materialized task IDs are not unique")
    return cases


def _resource_text(resource: dict[str, Any]) -> str:
    return " ".join(
        str(resource.get(key) or "")
        for key in ("resource_id", "domain", "skill_id", "resource_type", "title", "content")
    )


def _lexical_score(task: dict[str, Any], resource: dict[str, Any]) -> float:
    query = _tokens(str(task.get("instruction") or ""))
    text = _tokens(_resource_text(resource))
    return float(len(query & text))


def _rank_static(case: dict[str, Any]) -> list[dict[str, Any]]:
    task = case["task"]
    return sorted(
        case["resources"],
        key=lambda resource: (-_lexical_score(task, resource), int(resource["catalog_ordinal"])),
    )


def _replay_resource_ids(case: dict[str, Any]) -> set[str]:
    return {
        str(row.get("resource_id"))
        for row in case.get("replay_candidates", [])
        if isinstance(row, dict) and row.get("resource_id")
    }


def _rank_trm(case: dict[str, Any]) -> list[dict[str, Any]]:
    task = case["task"]
    required_types = set(task["interface_contract"]["required_resource_types"])
    replay_ids = _replay_resource_ids(case)

    def score(resource: dict[str, Any]) -> tuple[float, int]:
        value = _lexical_score(task, resource)
        value += 4.0 if resource["resource_id"] in replay_ids else 0.0
        value += 1.5 if resource["resource_type"] in required_types else 0.0
        value += 1.0 if resource["provenance"] == "attested" else 0.0
        value += 1.0 if not resource["stale"] else 0.0
        return -value, int(resource["catalog_ordinal"])

    return sorted(case["resources"], key=score)


def _valid_resource(resource: dict[str, Any], contract: dict[str, Any]) -> bool:
    if contract.get("require_current") and resource.get("stale"):
        return False
    if contract.get("require_attested") and resource.get("provenance") != "attested":
        return False
    return True


def _contract_coverage(selected: list[dict[str, Any]], contract: dict[str, Any]) -> float:
    required = list(dict.fromkeys(map(str, contract.get("required_resource_types", []))))
    if not required:
        return 1.0
    present = {
        str(resource["resource_type"])
        for resource in selected
        if _valid_resource(resource, contract)
    }
    return sum(resource_type in present for resource_type in required) / len(required)


def _typed_selection(case: dict[str, Any]) -> list[dict[str, Any]]:
    contract = case["task"]["interface_contract"]
    ranked = _rank_static(case)
    selected: list[dict[str, Any]] = []
    for resource_type in dict.fromkeys(contract["required_resource_types"]):
        match = next((row for row in ranked if row["resource_type"] == resource_type), None)
        if match is not None:
            selected.append(match)
    return selected


def select_mesh_resources(
    case: dict[str, Any],
    arm: str,
    *,
    static_top_k: int,
    trm_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if arm not in MESH_ARMS:
        raise ValueError(f"unknown mesh arm: {arm}")
    contract = case["task"]["interface_contract"]
    selected: list[dict[str, Any]]
    examined = 0
    rejected: list[str] = []
    expanded = False
    abstain = False

    if arm == "full_context":
        selected = list(case["resources"])
        examined = len(selected)
    elif arm == "static_topk":
        selected = _rank_static(case)[:static_top_k]
        examined = len(selected)
    elif arm == "typed_packet":
        selected = _typed_selection(case)
        examined = len(selected)
    elif arm == "trm_rerank":
        selected = _rank_trm(case)[:trm_top_k]
        examined = len(selected)
    elif arm == "ldt_verified":
        initial = _typed_selection(case)
        examined = len(initial)
        selected = []
        for resource in initial:
            if _valid_resource(resource, contract):
                selected.append(resource)
            else:
                rejected.append(str(resource["resource_id"]))
        abstain = _contract_coverage(selected, contract) < 1.0
    else:
        selected = []
        seen_types: set[str] = set()
        for resource in _rank_trm(case):
            examined += 1
            if not _valid_resource(resource, contract):
                rejected.append(str(resource["resource_id"]))
                continue
            resource_type = str(resource["resource_type"])
            if resource_type not in contract["required_resource_types"] or resource_type in seen_types:
                continue
            selected.append(resource)
            seen_types.add(resource_type)
            if _contract_coverage(selected, contract) >= 1.0:
                break
        expanded = examined > min(trm_top_k, len(case["resources"]))
        abstain = _contract_coverage(selected, contract) < 1.0

    control = {
        "arm": arm,
        "examined_resources": examined,
        "selected_resources": len(selected),
        "mcp_calls_est": examined,
        "rejected_resource_ids": rejected,
        "expanded": expanded,
        "abstain": abstain,
        "contract_coverage": _contract_coverage(selected, contract),
    }
    return selected, control


def build_mesh_packet(
    case: dict[str, Any],
    arm: str,
    *,
    working_budget_tokens: int,
    hard_context_tokens: int,
    static_top_k: int,
    trm_top_k: int,
) -> tuple[str, dict[str, Any]]:
    selected, control = select_mesh_resources(
        case,
        arm,
        static_top_k=static_top_k,
        trm_top_k=trm_top_k,
    )
    packet: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "arm": arm,
        "task": case["task"],
        "resources": selected,
        "control": control,
        "budget": {
            "working_tokens": working_budget_tokens,
            "hard_context_tokens": hard_context_tokens,
            "rule": "Use only current evidence that changes the next action.",
        },
    }
    if arm in {"trm_rerank", "adaptive_hybrid"}:
        packet["replay_hints"] = case.get("replay_candidates", [])[:1]
    text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "# MCP Skill Retrieval Mesh Packet\n" + text, packet


def score_packet(case: dict[str, Any], packet: str, packet_data: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    selected = packet_data["resources"]
    selected_ids = {str(resource["resource_id"]) for resource in selected}
    required_ids = set(map(str, expected["required_resource_ids"]))
    forbidden_ids = set(map(str, expected["forbidden_resource_ids"]))
    required_skills = set(map(str, expected["required_skill_ids"]))
    stale_count = sum(bool(resource.get("stale")) for resource in selected)
    wrong_count = sum(
        bool(resource.get("stale")) or str(resource.get("skill_id")) not in required_skills
        for resource in selected
    )
    conflict_groups: dict[str, int] = defaultdict(int)
    for resource in selected:
        conflict_groups[str(resource["conflict_group"])] += 1
    conflict_count = sum(count > 1 for count in conflict_groups.values())
    recall = len(selected_ids & required_ids) / max(1, len(required_ids))
    wrong_rate = wrong_count / max(1, len(selected))
    forbidden_rate = len(selected_ids & forbidden_ids) / max(1, len(selected))
    tokens = estimate_tokens_rough(packet)
    hard_context = int(packet_data["budget"]["hard_context_tokens"])
    working_budget = int(packet_data["budget"]["working_tokens"])
    abstain = bool(packet_data["control"]["abstain"])
    packet_valid = bool(packet_data) and tokens <= working_budget and tokens <= hard_context
    proxy_success = bool(
        packet_valid
        and recall == 1.0
        and wrong_count == 0
        and conflict_count == 0
        and not abstain
    )
    routing_utility = (
        recall
        - 0.35 * wrong_rate
        - 0.15 * forbidden_rate
        - 0.20 * float(abstain)
        - 0.10 * float(not packet_valid)
    )
    return {
        "task_id": case["task_id"],
        "domain": case["domain"],
        "split": case["split"],
        "complexity_level": int(case["complexity"]["level"]),
        "arm": packet_data["arm"],
        "packet_tokens_est": tokens,
        "packet_valid": packet_valid,
        "retrieval_recall": round(recall, 6),
        "wrong_skill_activation_rate": round(wrong_rate, 6),
        "forbidden_resource_rate": round(forbidden_rate, 6),
        "stale_resource_count": stale_count,
        "conflict_count": conflict_count,
        "abstain": abstain,
        "proxy_success": proxy_success,
        "routing_utility": round(routing_utility, 6),
        "utility_per_1k_tokens": round(routing_utility * 1000.0 / max(1, tokens), 6),
        "mcp_calls_est": int(packet_data["control"]["mcp_calls_est"]),
        "contract_coverage": float(packet_data["control"]["contract_coverage"]),
        "selected_resource_ids": sorted(selected_ids),
    }


def _arm_manifest(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "hermes.mcp_skill_mesh_arms.v0",
        "arms": [
            {
                "arm": "full_context",
                "initial_selection": "all candidate resources",
                "verification": "none",
                "adaptive_expansion": False,
            },
            {
                "arm": "static_topk",
                "initial_selection": f"lexical top-{config['static_top_k']}",
                "verification": "none",
                "adaptive_expansion": False,
            },
            {
                "arm": "typed_packet",
                "initial_selection": "one lexical candidate per typed interface requirement",
                "verification": "none",
                "adaptive_expansion": False,
            },
            {
                "arm": "trm_rerank",
                "initial_selection": f"task/replay/provenance reranked top-{config['trm_top_k']}",
                "verification": "none",
                "adaptive_expansion": False,
            },
            {
                "arm": "ldt_verified",
                "initial_selection": "typed candidates",
                "verification": "freshness, provenance, and contract coverage",
                "adaptive_expansion": False,
            },
            {
                "arm": "adaptive_hybrid",
                "initial_selection": "TRM-ranked typed candidates",
                "verification": "freshness, provenance, and contract coverage",
                "adaptive_expansion": True,
            },
        ],
    }


def register_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    _validate_config(config)
    cases = materialize_cases(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    arms_path = output_dir / "arms.json"
    protocol_path = output_dir / "protocol.json"
    receipt_path = output_dir / "registration_receipt.json"

    write_jsonl(cases_path, cases)
    write_json(arms_path, _arm_manifest(config))
    write_json(protocol_path, config)
    component_hashes = {
        "source_config_sha256": sha256_file(config_path),
        "protocol_sha256": sha256_file(protocol_path),
        "cases_sha256": sha256_file(cases_path),
        "arms_sha256": sha256_file(arms_path),
    }
    registration_id = sha256_bytes(canonical_json_bytes(component_hashes))
    split_tasks: dict[str, list[str]] = defaultdict(list)
    for case in cases:
        split_tasks[str(case["split"])].append(str(case["task_id"]))
    overlap_count = 0
    split_names = sorted(split_tasks)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap_count += len(set(split_tasks[left]) & set(split_tasks[right]))

    receipt = {
        "schema": "hermes.mcp_skill_mesh_registration.v0",
        "registered_at": utc_now(),
        "status": "registered_no_outcomes",
        "study_id": config["study_id"],
        "registration_id": registration_id,
        "component_hashes": component_hashes,
        "case_count": len(cases),
        "domain_count": len(config["domains"]),
        "complexity_level_count": len(config["complexity_levels"]),
        "arm_count": len(MESH_ARMS),
        "seeds": list(config["seeds"]),
        "splits": {key: sorted(value) for key, value in sorted(split_tasks.items())},
        "split_overlap_count": overlap_count,
        "claim_scope": config["claim_scope"],
        "live_outcomes_present": False,
    }
    write_json(receipt_path, receipt)
    (output_dir / "registration_receipt.sha256").write_text(
        f"{sha256_file(receipt_path)}  registration_receipt.json\n",
        encoding="ascii",
    )
    return receipt


def verify_registration(registration_dir: Path, expected_registration_id: str = "") -> dict[str, Any]:
    receipt = read_json(registration_dir / "registration_receipt.json")
    if expected_registration_id and receipt.get("registration_id") != expected_registration_id:
        raise ValueError("registration ID does not match --confirm-registration-id")
    expected = receipt["component_hashes"]
    actual = {
        "protocol_sha256": sha256_file(registration_dir / "protocol.json"),
        "cases_sha256": sha256_file(registration_dir / "cases.jsonl"),
        "arms_sha256": sha256_file(registration_dir / "arms.json"),
    }
    mismatches = {
        key: {"expected": expected[key], "actual": value}
        for key, value in actual.items()
        if expected.get(key) != value
    }
    if mismatches:
        raise ValueError(f"registration component hash mismatch: {mismatches}")
    computed_id = sha256_bytes(canonical_json_bytes(expected))
    if computed_id != receipt.get("registration_id"):
        raise ValueError("registration ID is inconsistent with component hashes")
    return receipt


def _aggregate(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in key_fields)].append(row)
    result: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        summary = {field: value for field, value in zip(key_fields, key)}
        summary.update(
            {
                "cells": len(items),
                "mean_retrieval_recall": round(sum(row["retrieval_recall"] for row in items) / len(items), 6),
                "mean_wrong_skill_activation_rate": round(
                    sum(row["wrong_skill_activation_rate"] for row in items) / len(items), 6
                ),
                "abstention_rate": round(sum(bool(row["abstain"]) for row in items) / len(items), 6),
                "proxy_success_rate": round(sum(bool(row["proxy_success"]) for row in items) / len(items), 6),
                "mean_packet_tokens_est": round(sum(row["packet_tokens_est"] for row in items) / len(items), 3),
                "mean_mcp_calls_est": round(sum(row["mcp_calls_est"] for row in items) / len(items), 3),
                "mean_routing_utility": round(sum(row["routing_utility"] for row in items) / len(items), 6),
                "mean_utility_per_1k_tokens": round(
                    sum(row["utility_per_1k_tokens"] for row in items) / len(items), 6
                ),
            }
        )
        result.append(summary)
    return result


def calibrate_registered(registration_dir: Path, output_dir: Path) -> dict[str, Any]:
    receipt = verify_registration(registration_dir)
    config = read_json(registration_dir / "protocol.json")
    cases = read_jsonl(registration_dir / "cases.jsonl")
    rows: list[dict[str, Any]] = []

    for case in cases:
        for arm in MESH_ARMS:
            packet, packet_data = build_mesh_packet(
                case,
                arm,
                working_budget_tokens=int(config["working_budget_tokens"]),
                hard_context_tokens=int(config["hard_context_tokens"]),
                static_top_k=int(config["static_top_k"]),
                trm_top_k=int(config["trm_top_k"]),
            )
            rows.append(score_packet(case, packet, packet_data))

    output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = output_dir / "deterministic_cells.jsonl"
    write_jsonl(cells_path, rows)
    by_arm = _aggregate(rows, ("arm",))
    by_level_arm = _aggregate(rows, ("complexity_level", "arm"))
    full_tokens = next(row["mean_packet_tokens_est"] for row in by_arm if row["arm"] == "full_context")
    for row in by_arm:
        row["token_savings_vs_full_context"] = round(
            1.0 - row["mean_packet_tokens_est"] / max(1.0, full_tokens),
            6,
        )
    summary = {
        "schema": "hermes.mcp_skill_mesh_calibration.v0",
        "created_at": utc_now(),
        "study_id": receipt["study_id"],
        "registration_id": receipt["registration_id"],
        "status": "deterministic_calibration_only",
        "cell_count": len(rows),
        "cases": len(cases),
        "arms": list(MESH_ARMS),
        "cells_sha256": sha256_file(cells_path),
        "by_arm": by_arm,
        "by_complexity_and_arm": by_level_arm,
        "claim_boundary": (
            "Packet calibration measures deterministic retrieval/control properties only; "
            "it is not a Bonsai task-performance result."
        ),
    }
    summary_path = output_dir / "calibration_summary.json"
    write_json(summary_path, summary)
    result_receipt = {
        "schema": "hermes.mcp_skill_mesh_result_receipt.v0",
        "registration_id": receipt["registration_id"],
        "calibration_summary_sha256": sha256_file(summary_path),
        "deterministic_cells_sha256": sha256_file(cells_path),
        "cell_count": len(rows),
        "integrity_failures": 0,
    }
    write_json(output_dir / "result_receipt.json", result_receipt)
    return summary


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _score_live_response(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    expected = case["expected"]
    resource_ids = response.get("resource_ids", [])
    actions = response.get("action_sequence", [])
    if not isinstance(resource_ids, list):
        resource_ids = []
    if not isinstance(actions, list):
        actions = []
    selected = set(map(str, resource_ids))
    required = set(map(str, expected["required_resource_ids"]))
    forbidden = set(map(str, expected["forbidden_resource_ids"]))
    abstain = bool(response.get("abstain"))
    retrieval_recall = len(selected & required) / max(1, len(required))
    wrong_rate = len(selected & forbidden) / max(1, len(selected))
    action_exact = list(map(str, actions)) == list(map(str, expected["action_sequence"]))
    success = bool(not abstain and retrieval_recall == 1.0 and wrong_rate == 0.0 and action_exact)
    return {
        "task_success": success,
        "model_retrieval_recall": round(retrieval_recall, 6),
        "model_wrong_skill_activation_rate": round(wrong_rate, 6),
        "action_sequence_exact": action_exact,
        "model_abstain": abstain,
    }


def _confirmation_arms(output_dir: Path, config: dict[str, Any], registration_id: str) -> list[str]:
    screening_path = output_dir / "live_screening_summary.json"
    if not screening_path.exists():
        raise ValueError("confirmation requires a completed screening summary")
    screening = read_json(screening_path)
    if screening.get("registration_id") != registration_id:
        raise ValueError("screening summary registration ID mismatch")
    if screening.get("status") != "completed":
        raise ValueError("confirmation requires a non-aborted screening run")
    attestation_path = output_dir / "live_screening_resource_attestation.json"
    if not attestation_path.exists():
        raise ValueError("confirmation requires an all-cell resource attestation")
    attestation = read_json(attestation_path)
    if (
        attestation.get("registration_id") != registration_id
        or attestation.get("all_completed_cells_cap_valid") is not True
        or int(attestation.get("completed_cells", 0)) != int(screening.get("expected_cells", 0))
    ):
        raise ValueError("screening resource attestation does not cover every registered cell")
    thresholds = config["promotion_policy"]["confirmation_requires"]
    promoted: list[str] = []
    for row in screening.get("by_arm", []):
        if not isinstance(row, dict):
            continue
        if (
            float(row.get("task_success_rate", 0)) >= float(thresholds["min_screening_task_success_rate"])
            and float(row.get("mean_model_wrong_skill_activation_rate", 1))
            <= float(thresholds["max_wrong_skill_activation_rate"])
            and float(row.get("token_savings_vs_full_context", 0))
            >= float(thresholds["min_token_savings_vs_full_context"])
        ):
            promoted.append(str(row["arm"]))
    if not promoted:
        raise ValueError("no screening arm passed the frozen confirmation gates")
    return promoted


def run_live_registered(
    registration_dir: Path,
    output_dir: Path,
    *,
    confirm_registration_id: str,
    base_url: str,
    model: str,
    stage: str,
    timeout_s: int,
    max_tokens: int,
    min_free_mb: int,
    max_temp_c: int,
) -> dict[str, Any]:
    receipt = verify_registration(registration_dir, confirm_registration_id)
    config = read_json(registration_dir / "protocol.json")
    cases = read_jsonl(registration_dir / "cases.jsonl")
    if stage not in {"screening", "confirmation"}:
        raise ValueError("stage must be screening or confirmation")
    seeds = list(config["seeds"][:1] if stage == "screening" else config["seeds"])
    arms = list(MESH_ARMS) if stage == "screening" else _confirmation_arms(output_dir, config, receipt["registration_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = output_dir / f"live_{stage}_cells.jsonl"
    existing = read_jsonl(cells_path) if cells_path.exists() else []
    completed = {
        (str(row["task_id"]), str(row["arm"]), int(row["seed"]))
        for row in existing
        if row.get("status") == "completed"
    }
    rows = list(existing)
    url = base_url.rstrip("/") + "/chat/completions"
    start = time.perf_counter()
    abort_reason = ""
    smoke = run_live_smoke(
        base_url=base_url,
        model=model,
        max_tokens=8,
        timeout_s=timeout_s,
        min_free_mb=min_free_mb,
        max_temp_c=max_temp_c,
    )
    write_json(output_dir / f"live_{stage}_smoke.json", smoke)
    if not smoke.get("passed", False):
        abort_reason = "local_model_smoke_failure"

    for seed in seeds if not abort_reason else []:
        for case in cases:
            for arm in arms:
                cell_key = (str(case["task_id"]), arm, int(seed))
                if cell_key in completed:
                    continue
                pressure_before = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=max_temp_c)
                if not pressure_before.get("passed", False):
                    abort_reason = "resource_pressure_gate"
                    break
                packet, packet_data = build_mesh_packet(
                    case,
                    arm,
                    working_budget_tokens=int(config["working_budget_tokens"]),
                    hard_context_tokens=int(config["hard_context_tokens"]),
                    static_top_k=int(config["static_top_k"]),
                    trm_top_k=int(config["trm_top_k"]),
                )
                prompt = (
                    "Solve the task using only the packet below. Return one JSON object with exactly "
                    "task_id, resource_ids, action_sequence, and abstain. resource_ids and action_sequence "
                    "must be arrays of exact tokens from the packet. Do not add prose.\n\n"
                    f"{packet}"
                )
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0,
                    "seed": int(seed),
                    "max_tokens": max(32, min(int(max_tokens), 256)),
                }
                cell_start = time.perf_counter()
                error = ""
                raw_response: dict[str, Any] = {}
                try:
                    raw_response = _post_json(url, payload, timeout_s=timeout_s)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                latency_ms = int((time.perf_counter() - cell_start) * 1000)
                content = ""
                choices = raw_response.get("choices")
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    message = choices[0].get("message")
                    if isinstance(message, dict):
                        content = str(message.get("content") or "")
                parsed = _extract_json_object(content)
                live_score = _score_live_response(case, parsed)
                pressure_after = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=max_temp_c)
                deterministic = score_packet(case, packet, packet_data)
                row = {
                    **deterministic,
                    **live_score,
                    "schema": "hermes.mcp_skill_mesh_live_cell.v0",
                    "registration_id": receipt["registration_id"],
                    "stage": stage,
                    "seed": int(seed),
                    "status": "completed" if not error else "api_error",
                    "error": error,
                    "latency_ms": latency_ms,
                    "usage": raw_response.get("usage", {}),
                    "response": parsed,
                    "raw_content": content[:2000],
                    "pressure_before": pressure_before,
                    "pressure_after": pressure_after,
                }
                rows.append(row)
                append_jsonl(cells_path, [row])
                if error:
                    abort_reason = "api_error"
                    break
                if not pressure_after.get("passed", False):
                    abort_reason = "resource_pressure_gate"
                    break
            if abort_reason:
                break
        if abort_reason:
            break

    completed_rows = [row for row in rows if row.get("status") == "completed"]
    live_by_arm: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in completed_rows:
        grouped[str(row["arm"])].append(row)
    for arm, items in sorted(grouped.items()):
        live_by_arm.append(
            {
                "arm": arm,
                "cells": len(items),
                "task_success_rate": round(sum(bool(row["task_success"]) for row in items) / len(items), 6),
                "mean_model_retrieval_recall": round(
                    sum(row["model_retrieval_recall"] for row in items) / len(items),
                    6,
                ),
                "mean_model_wrong_skill_activation_rate": round(
                    sum(row["model_wrong_skill_activation_rate"] for row in items) / len(items),
                    6,
                ),
                "mean_packet_tokens_est": round(sum(row["packet_tokens_est"] for row in items) / len(items), 3),
                "mean_latency_ms": round(sum(row["latency_ms"] for row in items) / len(items), 3),
            }
        )
    full_context = next(
        (row["mean_packet_tokens_est"] for row in live_by_arm if row["arm"] == "full_context"),
        0,
    )
    if stage == "confirmation":
        screening = read_json(output_dir / "live_screening_summary.json")
        full_context = next(
            (
                row["mean_packet_tokens_est"]
                for row in screening.get("by_arm", [])
                if isinstance(row, dict) and row.get("arm") == "full_context"
            ),
            0,
        )
    for row in live_by_arm:
        row["token_savings_vs_full_context"] = round(
            1.0 - row["mean_packet_tokens_est"] / max(1.0, full_context),
            6,
        )
    summary = {
        "schema": "hermes.mcp_skill_mesh_live_summary.v0",
        "created_at": utc_now(),
        "study_id": receipt["study_id"],
        "registration_id": receipt["registration_id"],
        "stage": stage,
        "status": "aborted" if abort_reason else "completed",
        "abort_reason": abort_reason,
        "model": model,
        "base_url": base_url,
        "seeds": seeds,
        "arms": arms,
        "expected_cells": len(cases) * len(arms) * len(seeds),
        "completed_cells": len(completed_rows),
        "api_error_cells": sum(row.get("status") == "api_error" for row in rows),
        "duration_ms": int((time.perf_counter() - start) * 1000),
        "cells_sha256": sha256_file(cells_path) if cells_path.exists() else "",
        "by_arm": live_by_arm,
        "promotion_policy": config["promotion_policy"],
        "smoke_receipt": str(output_dir / f"live_{stage}_smoke.json"),
        "claim_boundary": config["claim_scope"],
    }
    write_json(output_dir / f"live_{stage}_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sealed domain-general MCP skill retrieval mesh gym")
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="Materialize and seal cases without running a model")
    register.add_argument("--config", required=True)
    register.add_argument("--output-dir", required=True)

    verify = sub.add_parser("verify", help="Verify a sealed registration")
    verify.add_argument("--registration-dir", required=True)
    verify.add_argument("--registration-id", default="")

    calibrate = sub.add_parser("calibrate", help="Run deterministic packet calibration only")
    calibrate.add_argument("--registration-dir", required=True)
    calibrate.add_argument("--output-dir", required=True)

    live = sub.add_parser("live", help="Run a registered Bonsai screening or confirmation stage")
    live.add_argument("--registration-dir", required=True)
    live.add_argument("--output-dir", required=True)
    live.add_argument("--confirm-registration-id", required=True)
    live.add_argument("--base-url", default=DEFAULT_LOCAL_BONSAI_BASE_URL)
    live.add_argument("--model", default="local/bonsai-8b")
    live.add_argument("--stage", choices=["screening", "confirmation"], default="screening")
    live.add_argument("--timeout-s", type=int, default=90)
    live.add_argument("--max-tokens", type=int, default=128)
    live.add_argument("--min-free-mb", type=int, default=512)
    live.add_argument("--max-temp-c", type=int, default=86)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        result = register_study(Path(args.config), Path(args.output_dir))
    elif args.command == "verify":
        result = verify_registration(Path(args.registration_dir), args.registration_id)
    elif args.command == "calibrate":
        result = calibrate_registered(Path(args.registration_dir), Path(args.output_dir))
    else:
        result = run_live_registered(
            Path(args.registration_dir),
            Path(args.output_dir),
            confirm_registration_id=args.confirm_registration_id,
            base_url=args.base_url,
            model=args.model,
            stage=args.stage,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            min_free_mb=args.min_free_mb,
            max_temp_c=args.max_temp_c,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"aborted", "construction_failure"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
