"""Executable bridge from the 12k control mesh to the Bonsai MCP task gym."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from math import log
from pathlib import Path
from typing import Any, Iterable

from agent.lean_control_mesh_v1 import (
    SparseRAMPolicy,
    _arm_selection,
    expected_plan,
    read_json,
    read_jsonl,
    sha256_file,
    sparse_features,
    validate_typed_plan,
    write_json,
    write_jsonl,
)
from agent.mcp_skill_mesh_gym import (
    _extract_json_object,
    _post_json,
    _score_live_response,
    estimate_tokens_rough,
    select_mesh_resources,
)


SCHEMA = "hermes.executable_control_mesh_protocol.v1"
REGISTRATION_SCHEMA = "hermes.executable_control_mesh_registration.v1"
TRAINING_SCHEMA = "hermes.executable_domain_ram_training.v1"
LOCAL_CELL_SCHEMA = "hermes.executable_control_mesh_local_cell.v1"
LOCAL_SUMMARY_SCHEMA = "hermes.executable_control_mesh_local_summary.v1"
FULL_SUMMARY_SCHEMA = "hermes.executable_control_mesh_full_hermes_summary.v1"
ARMS = ("lexical_typed", "adaptive_mesh")
ACTION_PATTERN = re.compile(r"\bAction token:\s*([A-Za-z0-9_.:-]+)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _resolve_repo_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (_repo_root() / path).resolve()


def _query_sha256(case: dict[str, Any]) -> str:
    return hashlib.sha256(str(case["task"]["instruction"]).encode("utf-8")).hexdigest()


def _load_protocol_cases(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve_repo_path(str(protocol["source_cases_path"]))
    if sha256_file(path) != str(protocol["source_cases_sha256"]):
        raise ValueError("source case hash mismatch")
    return read_jsonl(path)


def _load_contracts(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    path = _resolve_repo_path(str(protocol["contracts_path"]))
    if sha256_file(path) != str(protocol["contracts_file_sha256"]):
        raise ValueError("contract file hash mismatch")
    return read_jsonl(path)


def _split_cases(
    protocol: dict[str, Any],
    cases: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_levels = set(map(int, protocol["split"]["domain_ram_train_levels"]))
    held_levels = set(map(int, protocol["split"]["held_executable_levels"]))
    if train_levels & held_levels:
        raise ValueError("train and held levels overlap")
    train = [case for case in cases if int(case["complexity"]["level"]) in train_levels]
    held = [case for case in cases if int(case["complexity"]["level"]) in held_levels]
    if not train or not held:
        raise ValueError("both train and held case sets are required")
    if {case["task_id"] for case in train} & {case["task_id"] for case in held}:
        raise ValueError("task split overlap")
    if {_query_sha256(case) for case in train} & {_query_sha256(case) for case in held}:
        raise ValueError("query split overlap")
    return train, held


def _validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("schema") != SCHEMA:
        raise ValueError(f"protocol schema must be {SCHEMA}")
    if tuple(protocol.get("arms", [])) != ARMS:
        raise ValueError(f"arms must be exactly {list(ARMS)}")
    inference = protocol.get("inference", {})
    if int(inference.get("bonsai_context_tokens", 0)) != 12000:
        raise ValueError("Bonsai context must be exactly 12000")
    if int(inference.get("full_hermes_context_tokens", 0)) != 160000:
        raise ValueError("full Hermes context must be exactly 160000")
    if int(inference.get("bonsai_working_tokens", 0)) >= 12000:
        raise ValueError("Bonsai working packet must leave a positive reserve")
    if set(protocol.get("domain_specialists", {})) != {
        "storyworld",
        "logic",
        "repository",
        "data_provenance",
    }:
        raise ValueError("the four registered domains require explicit specialists")


def register_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    protocol = read_json(config_path)
    _validate_protocol(protocol)
    cases = _load_protocol_cases(protocol)
    contracts = _load_contracts(protocol)
    train, held = _split_cases(protocol, cases)
    contract_names = {str(contract["name"]) for contract in contracts}
    missing = sorted(set(protocol["domain_specialists"].values()) - contract_names)
    if missing:
        raise ValueError(f"domain specialist contracts missing: {missing}")

    split_manifest = {
        "schema": "hermes.executable_control_mesh_split.v1",
        "train_task_ids": [str(case["task_id"]) for case in train],
        "held_task_ids": [str(case["task_id"]) for case in held],
        "train_query_sha256": [_query_sha256(case) for case in train],
        "held_query_sha256": [_query_sha256(case) for case in held],
        "task_overlap_count": 0,
        "query_overlap_count": 0,
    }
    registration_id = _sha256_value(
        {
            "protocol": protocol,
            "split_manifest": split_manifest,
            "source_cases_sha256": protocol["source_cases_sha256"],
            "contracts_file_sha256": protocol["contracts_file_sha256"],
        }
    )
    receipt = {
        "schema": REGISTRATION_SCHEMA,
        "study_id": protocol["study_id"],
        "registration_id": registration_id,
        "status": "registered_no_executable_outcomes",
        "train_case_count": len(train),
        "held_case_count": len(held),
        "domains": sorted(protocol["domain_specialists"]),
        "arms": list(ARMS),
        "source_cases_sha256": protocol["source_cases_sha256"],
        "contracts_file_sha256": protocol["contracts_file_sha256"],
        "split_manifest_sha256": _sha256_value(split_manifest),
        "claim_scope": protocol["claim_scope"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "protocol.json", protocol)
    write_json(output_dir / "split_manifest.json", split_manifest)
    write_json(output_dir / "registration_receipt.json", receipt)
    return receipt


def verify_registration(registration_dir: Path, expected_id: str = "") -> dict[str, Any]:
    protocol = read_json(registration_dir / "protocol.json")
    receipt = read_json(registration_dir / "registration_receipt.json")
    split_manifest = read_json(registration_dir / "split_manifest.json")
    _validate_protocol(protocol)
    cases = _load_protocol_cases(protocol)
    train, held = _split_cases(protocol, cases)
    registration_id = _sha256_value(
        {
            "protocol": protocol,
            "split_manifest": split_manifest,
            "source_cases_sha256": protocol["source_cases_sha256"],
            "contracts_file_sha256": protocol["contracts_file_sha256"],
        }
    )
    checks = {
        "schema": receipt.get("schema") == REGISTRATION_SCHEMA,
        "registration_id": receipt.get("registration_id") == registration_id,
        "expected_id": not expected_id or registration_id == expected_id,
        "split_manifest": receipt.get("split_manifest_sha256") == _sha256_value(split_manifest),
        "train_count": int(receipt.get("train_case_count", -1)) == len(train),
        "held_count": int(receipt.get("held_case_count", -1)) == len(held),
    }
    if not all(checks.values()):
        raise ValueError(f"registration verification failed: {checks}")
    return {**receipt, "verification": checks}


def _training_rows(
    protocol: dict[str, Any],
    train_cases: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    specialists = protocol["domain_specialists"]
    rows: list[dict[str, Any]] = []
    for case in train_cases:
        target = str(specialists[str(case["domain"])])
        query = str(case["task"]["instruction"])
        rows.append(
            {
                "kind": "positive",
                "query": query,
                "expected_contract_id": target,
            }
        )
        for other in sorted(set(map(str, specialists.values())) - {target}):
            rows.append(
                {
                    "kind": "negative",
                    "query": query,
                    "forbidden_contract_id": other,
                }
            )
    return rows


def _fit_domain_ram(
    protocol: dict[str, Any],
    train_cases: list[dict[str, Any]],
    *,
    registration_id: str,
    epochs: int,
) -> tuple[SparseRAMPolicy, dict[str, Any]]:
    labels = sorted(set(map(str, protocol["domain_specialists"].values())))
    policy = SparseRAMPolicy.empty(labels, ["abstain"], registration_id)
    documents: dict[str, list[set[str]]] = defaultdict(list)
    for case in train_cases:
        label = str(protocol["domain_specialists"][str(case["domain"])])
        documents[label].append(set(sparse_features(str(case["task"]["instruction"]))))
        policy.route_support[label] += 1
    vocabulary = set().union(
        *(features for label_documents in documents.values() for features in label_documents)
    )
    scale = max(1, epochs)
    for label in labels:
        positive = documents[label]
        negative = [
            features
            for other_label, label_documents in documents.items()
            if other_label != label
            for features in label_documents
        ]
        positive_counts = Counter(feature for features in positive for feature in features)
        negative_counts = Counter(feature for features in negative for feature in features)
        policy.route_weights[label] = {
            feature: scale
            * (
                log((positive_counts[feature] + 1) / (len(positive) + 2))
                - log((negative_counts[feature] + 1) / (len(negative) + 2))
            )
            for feature in vocabulary
        }
    correct = 0
    for case in train_cases:
        target = str(protocol["domain_specialists"][str(case["domain"])])
        predicted, _ = policy.predict_route(
            str(case["task"]["instruction"]),
            candidates=labels,
        )
        correct += predicted == target
    return policy, {
        "method": "sparse_bernoulli_log_odds_v1",
        "route_train_accuracy": correct / len(train_cases),
        "route_train_rows": len(train_cases),
        "contrast_rows": len(_training_rows(protocol, train_cases)) - len(train_cases),
        "supported_labels": len(labels),
        "unsupported_labels": 0,
        "vocabulary_features": len(vocabulary),
        "epochs": epochs,
    }


def train_domain_ram(
    registration_dir: Path,
    output_dir: Path,
    *,
    epochs: int,
    seed: int,
    ram_cap_mb: int,
    io_cap_mb_s: float,
) -> dict[str, Any]:
    if os.getenv("LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE") != "1":
        raise SystemExit("Refusing uncapped domain RAM training")
    receipt = verify_registration(registration_dir)
    protocol = read_json(registration_dir / "protocol.json")
    contracts = _load_contracts(protocol)
    train_cases, _ = _split_cases(protocol, _load_protocol_cases(protocol))
    start = time.perf_counter()

    import psutil

    process = psutil.Process()
    io_before = process.io_counters()
    policy, metrics = _fit_domain_ram(
        protocol,
        train_cases,
        registration_id=str(receipt["registration_id"]),
        epochs=epochs,
    )
    elapsed = max(1e-6, time.perf_counter() - start)
    io_after = process.io_counters()
    io_bytes = (
        io_after.read_bytes
        + io_after.write_bytes
        - io_before.read_bytes
        - io_before.write_bytes
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "domain_ram_policy.json", policy.to_dict())
    summary = {
        "schema": TRAINING_SCHEMA,
        "status": "completed",
        "registration_id": receipt["registration_id"],
        "held_outcomes_consumed": False,
        "seed": seed,
        "epochs": epochs,
        "train_case_count": len(train_cases),
        "metrics": metrics,
        "resources": {
            "ram_cap_mb": ram_cap_mb,
            "io_cap_mb_s": io_cap_mb_s,
            "process_rss_mb": round(process.memory_info().rss / (1024 * 1024), 3),
            "mean_io_mb_s": round(io_bytes / (1024 * 1024) / elapsed, 6),
            "elapsed_seconds": round(elapsed, 6),
        },
        "domain_ram_policy_sha256": sha256_file(output_dir / "domain_ram_policy.json"),
    }
    write_json(output_dir / "training_receipt.json", summary)
    return summary


def _load_router_models(base_model_dir: Path) -> tuple[SparseRAMPolicy, Any]:
    import torch

    from agent.lean_router_trm import TinyRecursiveSkillRouter

    ram = SparseRAMPolicy.from_dict(read_json(base_model_dir / "ram_policy.json"))
    checkpoint = torch.load(base_model_dir / "trm_router.pt", map_location="cpu", weights_only=True)
    trm = TinyRecursiveSkillRouter()
    trm.load_state_dict(checkpoint["model"])
    trm.eval()
    return ram, trm


def _canonical_action_tokens(resources: Iterable[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for resource in resources:
        if resource.get("stale"):
            continue
        match = ACTION_PATTERN.search(str(resource.get("content", "")))
        token = match.group(1).rstrip(".,;:") if match else ""
        if token and token not in result:
            result.append(token)
    return result


def _compact_contract(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": contract["name"],
        "purpose": contract.get("purpose", ""),
        "canonical_plan": expected_plan(contract),
        "gates": contract.get("gates", [])[:4],
        "output_contract": contract.get("output_contract", ""),
    }


def _task_without_aliases(case: dict[str, Any]) -> dict[str, Any]:
    task = case["task"]
    return {
        "task_id": task["task_id"],
        "domain": task["domain"],
        "instruction": task["instruction"],
        "interface_contract": task["interface_contract"],
    }


def _build_local_packet(
    case: dict[str, Any],
    arm: str,
    protocol: dict[str, Any],
    contracts: list[dict[str, Any]],
    base_ram: SparseRAMPolicy,
    trm: Any,
    domain_ram: SparseRAMPolicy,
) -> tuple[str, dict[str, Any]]:
    primary = [item for item in contracts if item.get("route", {}).get("kind") != "overlay"]
    by_name = {str(item["name"]): item for item in primary}
    raw_selected, candidates, detail = _arm_selection(
        arm,
        str(case["task"]["instruction"]),
        primary,
        base_ram,
        trm,
    )
    domain_selected = ""
    domain_margin = 0.0
    selected = raw_selected
    if arm == "adaptive_mesh":
        domain_selected, domain_margin = domain_ram.predict_route(
            str(case["task"]["instruction"]),
            candidates=protocol["domain_specialists"].values(),
        )
        if (
            domain_selected
            and domain_ram.route_support.get(domain_selected, 0) > 0
            and domain_margin >= float(protocol["control_flow"]["domain_ram_min_margin"])
        ):
            selected = by_name[domain_selected]

    repaired_plan, ldt = validate_typed_plan(
        selected,
        expected_plan(selected),
        repair=True,
    )
    resources, resource_control = select_mesh_resources(
        case,
        str(protocol["control_flow"]["inner_resource_arm"]),
        static_top_k=2,
        trm_top_k=3,
    )
    action_vocabulary = _canonical_action_tokens(resources)
    packet_data = {
        "schema": "hermes.executable_control_mesh_packet.v1",
        "arm": arm,
        "task": _task_without_aliases(case),
        "skill_contract": _compact_contract(selected),
        "resources": resources,
        "output_action_vocabulary": action_vocabulary,
        "control": {
            "outer_flow": detail["flow"],
            "raw_selected_contract": raw_selected["name"],
            "selected_contract": selected["name"],
            "domain_ram_selected_contract": domain_selected,
            "domain_ram_margin": round(domain_margin, 6),
            "domain_ram_override": selected["name"] != raw_selected["name"],
            "candidate_contract_ids": [item["name"] for item in candidates],
            "typed_plan": repaired_plan,
            "typed_ldt": ldt,
            "resource_mesh": resource_control,
        },
        "budget": {
            "working_tokens": int(protocol["inference"]["bonsai_working_tokens"]),
            "hard_context_tokens": int(protocol["inference"]["bonsai_context_tokens"]),
        },
    }
    text = json.dumps(packet_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "# Hermes Lite Executable Skill Packet\n" + text, packet_data


def _build_full_packet(
    case: dict[str, Any],
    protocol: dict[str, Any],
    contracts: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    primary = [item for item in contracts if item.get("route", {}).get("kind") != "overlay"]
    packet_data = {
        "schema": "hermes.full_context_executable_packet.v1",
        "arm": "full_hermes_160k",
        "task": _task_without_aliases(case),
        "skill_catalog": [_compact_contract(contract) for contract in primary],
        "resources": case["resources"],
        "output_action_vocabulary": _canonical_action_tokens(case["resources"]),
        "budget": {
            "hard_context_tokens": int(protocol["inference"]["full_hermes_context_tokens"]),
        },
    }
    text = json.dumps(packet_data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "# Full Hermes 160k Skill and MCP Packet\n" + text, packet_data


def _prompt(packet: str) -> str:
    return (
        "Solve the task using only the packet. Return one JSON object with exactly "
        "task_id, resource_ids, action_sequence, and abstain. resource_ids must use "
        "exact resource_id values. action_sequence must use only exact tokens from "
        "output_action_vocabulary in the required order. Do not use allowed-tool "
        "aliases and do not add prose.\n\n"
        + packet
    )


def _held_cases(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    _, held = _split_cases(protocol, _load_protocol_cases(protocol))
    return held


def _cell_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return str(row["task_id"]), str(row["arm"]), int(row["seed"])


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def run_local(
    registration_dir: Path,
    output_dir: Path,
    *,
    confirm_registration_id: str,
    base_model_dir: Path,
    domain_model_dir: Path,
    base_url: str,
    model: str,
    stage: str,
    timeout_s: int,
    max_tokens: int,
    resource_run_id: str,
    inter_cell_delay_s: float,
) -> dict[str, Any]:
    if stage != "screening":
        raise ValueError("the executable bridge has one frozen screening stage")
    receipt = verify_registration(registration_dir, confirm_registration_id)
    protocol = read_json(registration_dir / "protocol.json")
    contracts = _load_contracts(protocol)
    cases = _held_cases(protocol)
    base_ram, trm = _load_router_models(base_model_dir)
    domain_ram = SparseRAMPolicy.from_dict(read_json(domain_model_dir / "domain_ram_policy.json"))
    if domain_ram.registration_id != receipt["registration_id"]:
        raise ValueError("domain RAM registration mismatch")

    output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = output_dir / "live_screening_cells.jsonl"
    existing = read_jsonl(cells_path) if cells_path.exists() else []
    if any(row.get("status") != "completed" for row in existing):
        raise ValueError("non-completed checkpoint requires a fresh output lane")
    completed = {_cell_key(row) for row in existing}
    rows = list(existing)
    seed = int(protocol["inference"]["seed"])
    url = base_url.rstrip("/") + "/chat/completions"
    start = time.perf_counter()
    abort_reason = ""
    for case in cases:
        for arm in ARMS:
            key = (str(case["task_id"]), arm, seed)
            if key in completed:
                continue
            packet, packet_data = _build_local_packet(
                case,
                arm,
                protocol,
                contracts,
                base_ram,
                trm,
                domain_ram,
            )
            packet_tokens = estimate_tokens_rough(packet)
            error = ""
            raw_response: dict[str, Any] = {}
            cell_start = time.perf_counter()
            try:
                raw_response = _post_json(
                    url,
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": _prompt(packet)}],
                        "temperature": float(protocol["inference"]["temperature"]),
                        "seed": seed,
                        "max_tokens": max(32, min(max_tokens, 256)),
                    },
                    timeout_s=timeout_s,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            latency_ms = int((time.perf_counter() - cell_start) * 1000)
            choices = raw_response.get("choices")
            content = ""
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                message = choices[0].get("message")
                if isinstance(message, dict):
                    content = str(message.get("content") or "")
            parsed = _extract_json_object(content)
            score = _score_live_response(case, parsed)
            expected_specialist = str(protocol["domain_specialists"][str(case["domain"])])
            selected_specialist = str(packet_data["control"]["selected_contract"])
            row = {
                **score,
                "schema": LOCAL_CELL_SCHEMA,
                "registration_id": receipt["registration_id"],
                "task_id": case["task_id"],
                "domain": case["domain"],
                "complexity_level": case["complexity"]["level"],
                "arm": arm,
                "seed": seed,
                "status": "completed" if not error else "api_error",
                "error": error,
                "latency_ms": latency_ms,
                "usage": raw_response.get("usage", {}),
                "raw_content": content[:2000],
                "packet_tokens_est": packet_tokens,
                "context_compliant": packet_tokens
                <= int(protocol["inference"]["bonsai_working_tokens"]),
                "selected_specialist": selected_specialist,
                "expected_specialist": expected_specialist,
                "domain_specialist_match": selected_specialist == expected_specialist,
                "domain_ram_override": packet_data["control"]["domain_ram_override"],
                "typed_ldt_valid": packet_data["control"]["typed_ldt"]["valid"],
                "resource_control": packet_data["control"]["resource_mesh"],
                "resource_run_id": resource_run_id,
            }
            rows.append(row)
            _append_jsonl(cells_path, row)
            if error:
                abort_reason = "api_error"
                break
            if inter_cell_delay_s > 0:
                time.sleep(inter_cell_delay_s)
        if abort_reason:
            break

    completed_rows = [row for row in rows if row.get("status") == "completed"]
    by_arm: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_rows = [row for row in completed_rows if row["arm"] == arm]
        if not arm_rows:
            continue
        by_arm.append(
            {
                "arm": arm,
                "cells": len(arm_rows),
                "task_success_rate": sum(row["task_success"] for row in arm_rows) / len(arm_rows),
                "domain_specialist_accuracy": sum(
                    row["domain_specialist_match"] for row in arm_rows
                )
                / len(arm_rows),
                "mean_model_retrieval_recall": sum(
                    row["model_retrieval_recall"] for row in arm_rows
                )
                / len(arm_rows),
                "mean_packet_tokens_est": sum(row["packet_tokens_est"] for row in arm_rows)
                / len(arm_rows),
                "max_packet_tokens_est": max(row["packet_tokens_est"] for row in arm_rows),
                "mean_latency_ms": sum(row["latency_ms"] for row in arm_rows) / len(arm_rows),
            }
        )
    summary = {
        "schema": LOCAL_SUMMARY_SCHEMA,
        "status": "aborted" if abort_reason else "completed",
        "abort_reason": abort_reason,
        "study_id": protocol["study_id"],
        "registration_id": receipt["registration_id"],
        "stage": stage,
        "model": model,
        "arms": list(ARMS),
        "seed": seed,
        "expected_cells": len(cases) * len(ARMS),
        "completed_cells": len(completed_rows),
        "api_error_cells": sum(row.get("status") == "api_error" for row in rows),
        "duration_ms": int((time.perf_counter() - start) * 1000),
        "cells_sha256": sha256_file(cells_path) if cells_path.exists() else "",
        "by_arm": by_arm,
        "promotion": protocol["promotion"],
        "claim_scope": protocol["claim_scope"],
    }
    write_json(output_dir / "live_screening_summary.json", summary)
    return summary


def prepare_full_hermes(
    registration_dir: Path,
    output_path: Path,
    *,
    confirm_registration_id: str,
) -> dict[str, Any]:
    receipt = verify_registration(registration_dir, confirm_registration_id)
    protocol = read_json(registration_dir / "protocol.json")
    contracts = _load_contracts(protocol)
    rows = []
    for case in _held_cases(protocol):
        packet, _ = _build_full_packet(case, protocol, contracts)
        tokens = estimate_tokens_rough(packet)
        if tokens > int(protocol["inference"]["full_hermes_context_tokens"]):
            raise ValueError(f"full Hermes packet exceeds 160k for {case['task_id']}")
        rows.append(
            {
                "schema": "hermes.full_hermes_prompt.v1",
                "registration_id": receipt["registration_id"],
                "task_id": case["task_id"],
                "prompt": _prompt(packet),
                "packet_tokens_est": tokens,
            }
        )
    write_jsonl(output_path, rows)
    return {
        "prompts": len(rows),
        "prompts_sha256": sha256_file(output_path),
        "max_packet_tokens_est": max(row["packet_tokens_est"] for row in rows),
    }


def score_full_hermes(
    registration_dir: Path,
    results_path: Path,
    output_dir: Path,
    *,
    confirm_registration_id: str,
) -> dict[str, Any]:
    receipt = verify_registration(registration_dir, confirm_registration_id)
    protocol = read_json(registration_dir / "protocol.json")
    cases = {str(case["task_id"]): case for case in _held_cases(protocol)}
    results = read_jsonl(results_path)
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for result in results:
        task_id = str(result["task_id"])
        if task_id in seen or task_id not in cases:
            raise ValueError(f"invalid or duplicate full Hermes task: {task_id}")
        seen.add(task_id)
        parsed = _extract_json_object(str(result.get("raw_content", "")))
        score = _score_live_response(cases[task_id], parsed)
        rows.append(
            {
                **score,
                "schema": "hermes.executable_control_mesh_full_hermes_cell.v1",
                "registration_id": receipt["registration_id"],
                "task_id": task_id,
                "status": result.get("status", "api_error"),
                "error": result.get("error", ""),
                "latency_ms": result.get("latency_ms", 0),
                "usage": result.get("usage", {}),
                "raw_content": str(result.get("raw_content", ""))[:2000],
                "packet_tokens_est": int(result.get("packet_tokens_est", 0)),
            }
        )
    expected_ids = set(cases)
    missing = sorted(expected_ids - seen)
    output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = output_dir / "live_full_hermes_cells.jsonl"
    write_jsonl(cells_path, rows)
    completed = [row for row in rows if row["status"] == "completed"]
    summary = {
        "schema": FULL_SUMMARY_SCHEMA,
        "status": "completed" if not missing and len(completed) == len(cases) else "incomplete",
        "study_id": protocol["study_id"],
        "registration_id": receipt["registration_id"],
        "model": protocol["inference"]["full_hermes_model"],
        "context_tokens": protocol["inference"]["full_hermes_context_tokens"],
        "expected_cells": len(cases),
        "completed_cells": len(completed),
        "missing_task_ids": missing,
        "api_error_cells": sum(row["status"] != "completed" for row in rows),
        "task_success_rate": (
            sum(row["task_success"] for row in completed) / len(completed) if completed else 0.0
        ),
        "mean_model_retrieval_recall": (
            sum(row["model_retrieval_recall"] for row in completed) / len(completed)
            if completed
            else 0.0
        ),
        "cells_sha256": sha256_file(cells_path),
        "source_results_sha256": sha256_file(results_path),
        "claim_scope": protocol["claim_scope"],
    }
    write_json(output_dir / "live_full_hermes_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes Lite executable control mesh v1")
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register")
    register.add_argument("--config", required=True)
    register.add_argument("--output-dir", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--registration-dir", required=True)
    verify.add_argument("--registration-id", default="")
    train = sub.add_parser("train")
    train.add_argument("--registration-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--seed", type=int, default=8819)
    train.add_argument("--ram-cap-mb", type=int, default=2048)
    train.add_argument("--io-cap-mb-s", type=float, default=50.0)
    live = sub.add_parser("live")
    live.add_argument("--registration-dir", required=True)
    live.add_argument("--output-dir", required=True)
    live.add_argument("--confirm-registration-id", required=True)
    live.add_argument("--base-model-dir", required=True)
    live.add_argument("--domain-model-dir", required=True)
    live.add_argument("--base-url", required=True)
    live.add_argument("--model", default="local/bonsai-8b")
    live.add_argument("--stage", default="screening")
    live.add_argument("--timeout-s", type=int, default=180)
    live.add_argument("--max-tokens", type=int, default=128)
    live.add_argument("--min-free-mb", type=int, default=0)
    live.add_argument("--max-temp-c", type=int, default=100)
    live.add_argument("--external-pressure-monitor", action="store_true")
    live.add_argument("--resource-run-id", default="")
    live.add_argument("--inter-cell-delay-s", type=float, default=0.0)
    prepare = sub.add_parser("prepare-full-hermes")
    prepare.add_argument("--registration-dir", required=True)
    prepare.add_argument("--registration-id", required=True)
    prepare.add_argument("--output-path", required=True)
    score = sub.add_parser("score-full-hermes")
    score.add_argument("--registration-dir", required=True)
    score.add_argument("--registration-id", required=True)
    score.add_argument("--results-path", required=True)
    score.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "register":
        result = register_study(Path(args.config), Path(args.output_dir))
    elif args.command == "verify":
        result = verify_registration(Path(args.registration_dir), args.registration_id)
    elif args.command == "train":
        result = train_domain_ram(
            Path(args.registration_dir),
            Path(args.output_dir),
            epochs=args.epochs,
            seed=args.seed,
            ram_cap_mb=args.ram_cap_mb,
            io_cap_mb_s=args.io_cap_mb_s,
        )
    elif args.command == "live":
        result = run_local(
            Path(args.registration_dir),
            Path(args.output_dir),
            confirm_registration_id=args.confirm_registration_id,
            base_model_dir=Path(args.base_model_dir),
            domain_model_dir=Path(args.domain_model_dir),
            base_url=args.base_url,
            model=args.model,
            stage=args.stage,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            resource_run_id=args.resource_run_id,
            inter_cell_delay_s=args.inter_cell_delay_s,
        )
    elif args.command == "prepare-full-hermes":
        result = prepare_full_hermes(
            Path(args.registration_dir),
            Path(args.output_path),
            confirm_registration_id=args.registration_id,
        )
    else:
        result = score_full_hermes(
            Path(args.registration_dir),
            Path(args.results_path),
            Path(args.output_dir),
            confirm_registration_id=args.registration_id,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
