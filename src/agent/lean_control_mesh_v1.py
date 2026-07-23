"""Registered 12k-context control-mesh gym for Hermes Lite.

The gym snapshots ultra-lean skill contracts, trains a sparse in-RAM
retrieval/action memory and a tiny recursive router, then evaluates several
orders of those modules behind one typed schedule validator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agent.lean_contracts import contract_phases, deterministic_route_score, route_feature_vector


SCHEMA = "hermes.lean_control_mesh.v1"
REGISTRATION_SCHEMA = "hermes.lean_control_mesh_registration.v1"
RAM_SCHEMA = "hermes.sparse_retrieval_action_memory.v1_1"
IMPLEMENTATION_VERSION = "v1.1"
ARMS = (
    "lexical_typed",
    "trm_typed",
    "ram_typed",
    "trm_then_ram_typed",
    "ram_then_trm_typed",
    "adaptive_mesh",
)
TOKEN_RE = re.compile(r"[a-z0-9_]+")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _tokens(value: str) -> list[str]:
    return TOKEN_RE.findall(value.lower())


def sparse_features(value: str) -> tuple[str, ...]:
    words = _tokens(value)
    features = {"__bias__"}
    features.update(f"u:{word}" for word in words)
    features.update(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
    return tuple(sorted(features))


def typed_current_task(value: str) -> tuple[str, str]:
    """Return the authoritative route text and its compartment label."""
    for line in value.splitlines():
        if line.startswith("CURRENT_TASK="):
            return line.removeprefix("CURRENT_TASK=").strip(), "current_task"
    return value.strip(), "untyped"


def discover_contracts(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contracts: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, followlinks=False):
        names[:] = sorted(
            name for name in names if name not in {".git", "artifacts", "__pycache__", "node_modules"}
        )
        if "ULTRA_LEAN.json" not in files:
            continue
        path = Path(directory) / "ULTRA_LEAN.json"
        try:
            raw = path.read_bytes()
            value = json.loads(raw.decode("utf-8-sig"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict) or value.get("schema") not in {
            "hermes.ultra_lean_skill.v1",
            "hermes.ultra_lean_skill.v2",
        }:
            continue
        if value.get("name") != path.parent.name:
            continue
        contracts.append(value)
        manifest.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "raw_sha256": sha256_bytes(raw),
                "normalized_sha256": sha256_bytes(canonical_json_bytes(value)),
                "bytes": len(raw),
                "name": value["name"],
            }
        )
    contracts.sort(key=lambda item: str(item["name"]))
    manifest.sort(key=lambda item: str(item["relative_path"]))
    return contracts, manifest


def contracts_digest(contracts: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(contracts))


def canonical_operation(phase: dict[str, Any]) -> str:
    allowed = [str(value) for value in phase.get("allowed_operations", [])]
    if not allowed:
        return "abstain"
    module = str(phase.get("module", "")).lower()
    phase_id = str(phase.get("id", "")).lower()
    preferred = {
        "artifact_retrieval": "retrieve",
        "model_generation": "execute",
        "verifier": "verify",
        "finalizer": "commit",
    }.get(module)
    if module == "coordinator":
        if any(token in phase_id for token in ("route", "rerank", "index")):
            preferred = "route"
        elif "verify" in phase_id or "audit" in phase_id:
            preferred = "verify"
        elif "repair" in phase_id:
            preferred = "repair"
        else:
            preferred = "execute"
    if preferred in allowed:
        return str(preferred)
    for fallback in ("execute", "select", "route", "verify", "retrieve", "commit", "repair", "abstain"):
        if fallback in allowed:
            return fallback
    return allowed[0]


def expected_plan(contract: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "phase": str(phase["id"]),
            "module": str(phase.get("module", "coordinator")),
            "operation": canonical_operation(phase),
        }
        for phase in contract_phases(contract)
    ]


def validate_typed_plan(
    contract: dict[str, Any],
    plan: list[dict[str, str]],
    *,
    repair: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    phases = contract_phases(contract)
    expected = expected_plan(contract)
    repaired: list[dict[str, str]] = []
    errors: list[str] = []
    repair_count = 0
    by_phase = {str(item.get("phase")): item for item in plan if isinstance(item, dict)}
    for index, phase in enumerate(phases):
        phase_id = str(phase["id"])
        candidate = by_phase.get(phase_id)
        if candidate is None:
            errors.append(f"missing_phase:{phase_id}")
            if repair:
                repaired.append(expected[index])
                repair_count += 1
                continue
            candidate = {}
        operation = str(candidate.get("operation", ""))
        allowed = [str(value) for value in phase.get("allowed_operations", [])]
        canonical = expected[index]["operation"]
        if operation not in allowed:
            errors.append(f"invalid_operation:{phase_id}:{operation}")
            if repair:
                operation = canonical
                repair_count += 1
        elif operation != canonical:
            errors.append(f"noncanonical_operation:{phase_id}:{operation}")
            if repair:
                operation = canonical
                repair_count += 1
        repaired.append(
            {
                "phase": phase_id,
                "module": str(phase.get("module", "coordinator")),
                "operation": operation,
            }
        )
    if [str(item.get("phase")) for item in plan] != [str(item["id"]) for item in phases]:
        errors.append("phase_order_mismatch")
        if repair:
            repair_count += 1
    valid = repaired == expected and (repair or not errors)
    return repaired, {
        "valid": valid,
        "errors": errors,
        "repair_count": repair_count,
        "changed": repair_count > 0,
    }


def _context_variants(
    query: str,
    *,
    target: dict[str, Any],
    contracts: list[dict[str, Any]],
    seed: int,
) -> list[tuple[str, str]]:
    others = [item for item in contracts if item["name"] != target["name"]]
    digest = hashlib.sha256(f"{seed}:{target['name']}:{query}".encode("utf-8")).digest()
    distractor = others[int.from_bytes(digest[:4], "big") % len(others)]
    stale_examples = distractor.get("route", {}).get("positive_examples", [])
    stale = str(stale_examples[0]) if stale_examples else str(distractor.get("purpose", ""))
    return [
        ("plain", query),
        (
            "checkpoint_state",
            "CURRENT_TASK="
            + query
            + "\nSTATE=Resume from a verified checkpoint and preserve the exact typed skill contract.",
        ),
        (
            "stale_hint",
            "STALE_HINT="
            + stale
            + "\nCURRENT_TASK="
            + query
            + "\nRULE=The current task overrides the stale hint.",
        ),
    ]


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise ValueError(f"config schema must be {SCHEMA}")
    if tuple(config.get("arms", [])) != ARMS:
        raise ValueError(f"arms must be exactly {list(ARMS)}")
    hard_context = int(config.get("context", {}).get("hard_window_tokens", 0))
    if hard_context != 12000:
        raise ValueError("v1 hard context window must be exactly 12000 tokens")
    if int(config.get("context", {}).get("active_working_set_tokens", 0)) >= hard_context:
        raise ValueError("active working set must leave a positive reserve")
    if not config.get("held_contract_ids"):
        raise ValueError("held_contract_ids must be explicit")


def register_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json(config_path)
    _validate_config(config)
    source_root = Path(str(config["source_contract_root"])).expanduser().resolve()
    contracts, source_manifest = discover_contracts(source_root)
    if len(contracts) < int(config.get("minimum_contract_count", 3)):
        raise ValueError("too few valid ultra-lean contracts")
    primary = [item for item in contracts if item.get("route", {}).get("kind") != "overlay"]
    by_name = {str(item["name"]): item for item in primary}
    held_ids = {str(value) for value in config["held_contract_ids"]}
    missing = sorted(held_ids - set(by_name))
    if missing:
        raise ValueError(f"held contracts missing from source: {missing}")

    train_rows: list[dict[str, Any]] = []
    held_cases: list[dict[str, Any]] = []
    perturb_seed = int(config["perturbation_seed"])
    for contract in primary:
        name = str(contract["name"])
        positives = [str(value) for value in contract.get("route", {}).get("positive_examples", [])]
        negatives = [str(value) for value in contract.get("route", {}).get("hard_negatives", [])]
        if len(positives) < 3 or len(negatives) < 2:
            raise ValueError(f"{name} requires at least three positives and two hard negatives")
        if name in held_ids:
            held_positive = list(enumerate(positives))
        else:
            for index, query in enumerate(positives[:2]):
                train_rows.append(
                    {
                        "row_id": f"train.{name}.positive.{index}",
                        "kind": "positive",
                        "query": query,
                        "expected_contract_id": name,
                        "forbidden_contract_id": "",
                    }
                )
            held_positive = [(2, positives[2])]
        train_rows.append(
            {
                "row_id": f"train.{name}.negative.1",
                "kind": "negative",
                "query": negatives[1],
                "expected_contract_id": "",
                "forbidden_contract_id": name,
            }
        )
        for example_index, query in held_positive:
            lane = "contract_transfer" if name in held_ids else "query_holdout"
            for perturbation, perturbed in _context_variants(
                query,
                target=contract,
                contracts=primary,
                seed=perturb_seed,
            ):
                held_cases.append(
                    {
                        "case_id": f"held.{name}.{example_index}.{perturbation}",
                        "kind": "positive",
                        "lane": lane,
                        "perturbation": perturbation,
                        "query": perturbed,
                        "base_query": query,
                        "expected_contract_id": name,
                        "forbidden_contract_id": "",
                        "expected_plan": expected_plan(contract),
                    }
                )
        held_cases.append(
            {
                "case_id": f"held.{name}.negative.0",
                "kind": "negative",
                "lane": "negative_control",
                "perturbation": "plain",
                "query": negatives[0],
                "base_query": negatives[0],
                "expected_contract_id": "",
                "forbidden_contract_id": name,
                "expected_plan": [],
            }
        )

    train_queries = {sha256_bytes(str(row["query"]).strip().lower().encode("utf-8")) for row in train_rows}
    held_queries = {
        sha256_bytes(str(row["base_query"]).strip().lower().encode("utf-8")) for row in held_cases
    }
    overlap = sorted(train_queries & held_queries)
    if overlap:
        raise ValueError(f"train/evaluation query overlap: {len(overlap)}")

    registration_basis = {
        "schema": REGISTRATION_SCHEMA,
        "study_id": config["study_id"],
        "config_sha256": sha256_bytes(canonical_json_bytes(config)),
        "contracts_sha256": contracts_digest(contracts),
        "train_rows_sha256": sha256_bytes(b"".join(canonical_json_bytes(row) for row in train_rows)),
        "held_cases_sha256": sha256_bytes(b"".join(canonical_json_bytes(row) for row in held_cases)),
    }
    registration_id = sha256_bytes(canonical_json_bytes(registration_basis))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "protocol.json", config)
    write_jsonl(output_dir / "contracts.jsonl", contracts)
    write_jsonl(output_dir / "train_rows.jsonl", train_rows)
    write_jsonl(output_dir / "held_cases.jsonl", held_cases)
    write_json(
        output_dir / "source_manifest.json",
        {
            "schema": "hermes.lean_control_mesh_source_manifest.v1",
            "source_root": str(source_root),
            "source_git_head": _git_head(source_root),
            "source_worktree_clean": _git_clean(source_root),
            "contracts": source_manifest,
        },
    )
    receipt = {
        **registration_basis,
        "registration_id": registration_id,
        "status": "registered_no_outcomes",
        "contract_count": len(contracts),
        "routable_contract_count": len(primary),
        "overlay_count": len(contracts) - len(primary),
        "train_row_count": len(train_rows),
        "held_case_count": len(held_cases),
        "held_contract_ids": sorted(held_ids),
        "split_overlap_count": 0,
        "arms": list(ARMS),
        "hard_context_tokens": 12000,
        "full_hermes_baseline_required": True,
    }
    write_json(output_dir / "registration_receipt.json", receipt)
    return receipt


def _git_head(path: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _git_clean(path: Path) -> bool | None:
    import subprocess

    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return not bool(output.strip())
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class SparseRAMPolicy:
    labels: list[str]
    route_weights: dict[str, dict[str, float]]
    route_support: dict[str, int]
    action_labels: list[str]
    action_weights: dict[str, dict[str, float]]
    registration_id: str = ""

    @classmethod
    def empty(cls, labels: Iterable[str], action_labels: Iterable[str], registration_id: str) -> "SparseRAMPolicy":
        label_list = sorted(set(map(str, labels)))
        action_list = sorted(set(map(str, action_labels)))
        return cls(
            labels=label_list,
            route_weights={label: {} for label in label_list},
            route_support={label: 0 for label in label_list},
            action_labels=action_list,
            action_weights={label: {} for label in action_list},
            registration_id=registration_id,
        )

    @staticmethod
    def _score(weights: dict[str, float], features: Iterable[str]) -> float:
        return sum(float(weights.get(feature, 0.0)) for feature in features)

    def route_scores(self, query: str) -> dict[str, float]:
        features = sparse_features(query)
        return {label: self._score(self.route_weights[label], features) for label in self.labels}

    def predict_route(self, query: str, candidates: Iterable[str] | None = None) -> tuple[str, float]:
        allowed = sorted(set(candidates or self.labels))
        scores = self.route_scores(query)
        ranked = sorted(((scores.get(label, float("-inf")), label) for label in allowed), reverse=True)
        if not ranked:
            return "", 0.0
        margin = ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 0.0)
        return ranked[0][1], float(margin)

    def predict_action(self, phase: dict[str, Any]) -> tuple[str, float]:
        descriptor = " ".join(
            [
                str(phase.get("id", "")),
                str(phase.get("module", "")),
                *map(str, phase.get("allowed_operations", [])),
            ]
        )
        features = sparse_features(descriptor)
        ranked = sorted(
            (
                (self._score(self.action_weights[label], features), label)
                for label in self.action_labels
            ),
            reverse=True,
        )
        if not ranked:
            return "abstain", 0.0
        margin = ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 0.0)
        return ranked[0][1], float(margin)

    def to_dict(self) -> dict[str, Any]:
        def compact(weights: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
            return {
                label: {
                    feature: round(value, 6)
                    for feature, value in sorted(values.items())
                    if abs(value) > 1e-12
                }
                for label, values in sorted(weights.items())
            }

        return {
            "schema": RAM_SCHEMA,
            "registration_id": self.registration_id,
            "labels": self.labels,
            "route_weights": compact(self.route_weights),
            "route_support": self.route_support,
            "action_labels": self.action_labels,
            "action_weights": compact(self.action_weights),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SparseRAMPolicy":
        if value.get("schema") != RAM_SCHEMA:
            raise ValueError("invalid RAM policy schema")
        labels = list(map(str, value["labels"]))
        support = value.get("route_support", {})
        return cls(
            labels=labels,
            route_weights={
                str(label): {str(key): float(score) for key, score in weights.items()}
                for label, weights in value["route_weights"].items()
            },
            route_support={
                label: int(support.get(label, 0))
                for label in labels
            },
            action_labels=list(map(str, value["action_labels"])),
            action_weights={
                str(label): {str(key): float(score) for key, score in weights.items()}
                for label, weights in value["action_weights"].items()
            },
            registration_id=str(value.get("registration_id", "")),
        )


def _update(weights: dict[str, float], features: Iterable[str], amount: float) -> None:
    for feature in features:
        weights[feature] = float(weights.get(feature, 0.0)) + amount
        if abs(weights[feature]) < 1e-12:
            del weights[feature]


def train_ram_policy(
    contracts: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    registration_id: str,
    epochs: int,
    seed: int,
) -> tuple[SparseRAMPolicy, dict[str, Any]]:
    primary = [item for item in contracts if item.get("route", {}).get("kind") != "overlay"]
    action_labels = {canonical_operation(phase) for item in primary for phase in contract_phases(item)}
    policy = SparseRAMPolicy.empty(
        (str(item["name"]) for item in primary),
        action_labels,
        registration_id,
    )
    positives = [row for row in train_rows if row["kind"] == "positive"]
    negatives = [row for row in train_rows if row["kind"] == "negative"]
    for row in positives:
        label = str(row["expected_contract_id"])
        policy.route_support[label] = policy.route_support.get(label, 0) + 1
    randomizer = random.Random(seed)
    for _ in range(epochs):
        shuffled = list(positives)
        randomizer.shuffle(shuffled)
        for row in shuffled:
            query = str(row["query"])
            target = str(row["expected_contract_id"])
            prediction, _ = policy.predict_route(query)
            if prediction != target:
                features = sparse_features(query)
                _update(policy.route_weights[target], features, 1.0)
                if prediction:
                    _update(policy.route_weights[prediction], features, -1.0)
        for row in negatives:
            target = str(row["forbidden_contract_id"])
            _update(policy.route_weights[target], sparse_features(str(row["query"])), -0.25)

        action_rows: list[tuple[dict[str, Any], str]] = []
        for contract in primary:
            if contract["name"] not in {row["expected_contract_id"] for row in positives}:
                continue
            for phase in contract_phases(contract):
                action_rows.append((phase, canonical_operation(phase)))
        randomizer.shuffle(action_rows)
        for phase, target in action_rows:
            prediction, _ = policy.predict_action(phase)
            if prediction != target:
                descriptor = " ".join(
                    [
                        str(phase.get("id", "")),
                        str(phase.get("module", "")),
                        *map(str, phase.get("allowed_operations", [])),
                    ]
                )
                features = sparse_features(descriptor)
                _update(policy.action_weights[target], features, 1.0)
                if prediction:
                    _update(policy.action_weights[prediction], features, -1.0)

    route_correct = sum(
        policy.predict_route(str(row["query"]))[0] == row["expected_contract_id"]
        for row in positives
    )
    action_total = action_correct = 0
    trained_labels = {row["expected_contract_id"] for row in positives}
    for contract in primary:
        if contract["name"] not in trained_labels:
            continue
        for phase in contract_phases(contract):
            action_total += 1
            action_correct += policy.predict_action(phase)[0] == canonical_operation(phase)
    nonzero = sum(
        len(values) for values in policy.route_weights.values()
    ) + sum(len(values) for values in policy.action_weights.values())
    return policy, {
        "route_train_accuracy": route_correct / max(1, len(positives)),
        "route_train_rows": len(positives),
        "negative_train_rows": len(negatives),
        "action_train_accuracy": action_correct / max(1, action_total),
        "action_train_rows": action_total,
        "nonzero_weights": nonzero,
        "supported_labels": sum(count > 0 for count in policy.route_support.values()),
        "unsupported_labels": sum(count == 0 for count in policy.route_support.values()),
        "epochs": epochs,
    }


def _trm_rows(
    contracts: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    *,
    seed: int,
) -> list[tuple[list[list[float]], int]]:
    primary = [item for item in contracts if item.get("route", {}).get("kind") != "overlay"]
    by_name = {str(item["name"]): item for item in primary}
    result: list[tuple[list[list[float]], int]] = []
    randomizer = random.Random(seed)
    for row in train_rows:
        if row["kind"] != "positive":
            continue
        query = str(row["query"])
        target = by_name[str(row["expected_contract_id"])]
        ranked = sorted(
            primary,
            key=lambda item: deterministic_route_score(query, item),
            reverse=True,
        )[:5]
        if target not in ranked:
            ranked[-1] = target
        randomizer.shuffle(ranked)
        features = [
            route_feature_vector(query, item, deterministic_route_score(query, item))
            for item in ranked
        ]
        result.append((features, ranked.index(target)))
    return result


def train_registered(
    registration_dir: Path,
    output_dir: Path,
    *,
    steps: int,
    ram_epochs: int,
    seed: int,
    ram_cap_mb: int,
    io_cap_mb_s: float,
) -> dict[str, Any]:
    if os.getenv("LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE") != "1":
        raise SystemExit("Refusing uncapped training; use scripts/run_capped_lean_control_mesh_v1.ps1")
    receipt = read_json(registration_dir / "registration_receipt.json")
    contracts = read_jsonl(registration_dir / "contracts.jsonl")
    train_rows = read_jsonl(registration_dir / "train_rows.jsonl")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ram, ram_summary = train_ram_policy(
        contracts,
        train_rows,
        registration_id=str(receipt["registration_id"]),
        epochs=ram_epochs,
        seed=seed,
    )
    write_json(output_dir / "ram_policy.json", ram.to_dict())

    import psutil
    import torch
    from torch import nn

    from agent.lean_router_trm import TinyRecursiveSkillRouter

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(seed)
    torch.manual_seed(seed)
    rows = _trm_rows(contracts, train_rows, seed=seed)
    features = torch.tensor([row[0] for row in rows], dtype=torch.float32)
    labels = torch.tensor([row[1] for row in rows], dtype=torch.long)
    model = TinyRecursiveSkillRouter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    batch_size = min(16, len(rows))
    last_loss = 0.0
    process = psutil.Process()
    peak_ram_mb = peak_io_mb_s = 0.0
    last_io = process.io_counters().read_bytes + process.io_counters().write_bytes
    last_sample = time.monotonic()
    io_excess_streak = 0
    status = "completed"
    abort_reason = ""

    def sample_resources() -> None:
        nonlocal peak_ram_mb, peak_io_mb_s, last_io, last_sample, io_excess_streak
        now = time.monotonic()
        rss_mb = process.memory_info().rss / 1024 / 1024
        current_io = process.io_counters().read_bytes + process.io_counters().write_bytes
        io_mb_s = max(0, current_io - last_io) / 1024 / 1024 / max(0.001, now - last_sample)
        peak_ram_mb = max(peak_ram_mb, rss_mb)
        peak_io_mb_s = max(peak_io_mb_s, io_mb_s)
        io_excess_streak = io_excess_streak + 1 if io_mb_s > io_cap_mb_s else 0
        last_io, last_sample = current_io, now
        if rss_mb > ram_cap_mb:
            raise RuntimeError("ram_cap_exceeded")
        if io_excess_streak >= 3:
            raise RuntimeError("sustained_io_cap_exceeded")

    completed_steps = 0
    try:
        sample_resources()
        for completed_steps in range(1, steps + 1):
            indexes = torch.randint(0, len(rows), (batch_size,), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features[indexes])
            loss = loss_fn(logits, labels[indexes])
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
            if completed_steps % 10 == 0 or completed_steps == steps:
                sample_resources()
    except RuntimeError as exc:
        status = "aborted"
        abort_reason = str(exc)

    model.eval()
    with torch.no_grad():
        logits, _ = model(features)
        train_accuracy = float((logits.argmax(-1) == labels).float().mean())
    trm_sha256 = ""
    if status == "completed":
        checkpoint = {
            "model": model.state_dict(),
            "manifest": model.manifest(),
            "steps": completed_steps,
            "registration_id": receipt["registration_id"],
            "train_rows_sha256": receipt["train_rows_sha256"],
        }
        torch.save(checkpoint, output_dir / "trm_router.pt")
        trm_sha256 = sha256_file(output_dir / "trm_router.pt")
    summary = {
        "schema": "hermes.lean_control_mesh_training_receipt.v1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": status,
        "abort_reason": abort_reason,
        "registration_id": receipt["registration_id"],
        "seed": seed,
        "steps_completed": completed_steps,
        "expected_steps": steps,
        "ram": ram_summary,
        "trm": {
            "train_rows": len(rows),
            "train_accuracy": train_accuracy,
            "last_loss": last_loss,
            "manifest": model.manifest(),
        },
        "resources": {
            "peak_ram_mb": round(peak_ram_mb, 3),
            "peak_io_mb_s": round(peak_io_mb_s, 3),
            "ram_cap_mb": ram_cap_mb,
            "io_cap_mb_s": io_cap_mb_s,
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "ram_policy_sha256": sha256_file(output_dir / "ram_policy.json"),
        "trm_router_sha256": trm_sha256,
        "held_outcomes_consumed": False,
    }
    write_json(output_dir / "training_receipt.json", summary)
    del model, optimizer, features, labels, logits
    return summary


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    exponentials = {key: math.exp(min(40.0, value - maximum)) for key, value in scores.items()}
    denominator = sum(exponentials.values()) or 1.0
    return {key: value / denominator for key, value in exponentials.items()}


def _score_margin(scores: dict[str, float]) -> tuple[str, float]:
    ranked = sorted(((value, key) for key, value in scores.items()), reverse=True)
    if not ranked:
        return "", 0.0
    return ranked[0][1], ranked[0][0] - (ranked[1][0] if len(ranked) > 1 else 0.0)


def _unique_candidates(*groups: Iterable[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            name = str(item["name"])
            if name not in seen:
                result.append(item)
                seen.add(name)
            if len(result) >= limit:
                return result
    return result


def _trm_probabilities(query: str, candidates: list[dict[str, Any]], model: Any) -> dict[str, float]:
    import torch

    if not candidates:
        return {}
    features = torch.tensor(
        [
            [
                route_feature_vector(query, item, deterministic_route_score(query, item))
                for item in candidates
            ]
        ],
        dtype=torch.float32,
    )
    with torch.no_grad():
        logits, _ = model(features)
    probabilities = torch.softmax(logits[0, : len(candidates)], dim=-1).tolist()
    return {str(item["name"]): float(value) for item, value in zip(candidates, probabilities)}


def _top_contracts(
    scores: dict[str, float],
    by_name: dict[str, dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    names = sorted(scores, key=lambda name: (scores[name], name), reverse=True)[:limit]
    return [by_name[name] for name in names]


def _arm_selection(
    arm: str,
    query: str,
    primary: list[dict[str, Any]],
    ram: SparseRAMPolicy,
    trm: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    by_name = {str(item["name"]): item for item in primary}
    route_query, compartment = typed_current_task(query)
    lexical_scores = {
        str(item["name"]): deterministic_route_score(route_query, item) for item in primary
    }
    ram_scores = ram.route_scores(route_query)
    lexical_top = _top_contracts(lexical_scores, by_name)
    ram_top = _top_contracts(ram_scores, by_name)
    supported_ram_top = [
        item for item in ram_top if ram.route_support.get(str(item["name"]), 0) > 0
    ]
    lexical_probability = _softmax(lexical_scores)
    lexical_leader, lexical_margin = _score_margin(lexical_probability)
    ram_probability = _softmax(
        {
            name: score
            for name, score in ram_scores.items()
            if ram.route_support.get(name, 0) > 0
        }
    )
    ram_leader, _ = _score_margin(ram_probability)
    ram_enabled = ram.route_support.get(lexical_leader, 0) > 0
    guarded_ram_leader = ram_leader if ram_enabled else ""
    if arm == "lexical_typed":
        candidates = lexical_top
        selected = candidates[0]
        detail = {"flow": ["lexical", "typed_ldt"]}
    elif arm == "trm_typed":
        candidates = lexical_top
        trm_scores = _trm_probabilities(route_query, candidates, trm)
        selected = by_name[max(trm_scores, key=trm_scores.get)]
        detail = {"flow": ["lexical_shortlist", "trm", "typed_ldt"], "trm": trm_scores}
    elif arm == "ram_typed":
        candidates = _unique_candidates(lexical_top[:1], supported_ram_top, lexical_top)
        if (
            guarded_ram_leader
            and guarded_ram_leader in {str(item["name"]) for item in lexical_top}
            and lexical_margin < 0.10
        ):
            selected = by_name[guarded_ram_leader]
        else:
            selected = by_name[lexical_leader]
        detail = {"flow": ["lexical_support_gate", "ram_residual", "typed_ldt"]}
    elif arm == "trm_then_ram_typed":
        candidates = lexical_top
        trm_scores = _trm_probabilities(route_query, candidates, trm)
        ram_prob = (
            _softmax(
                {
                    str(item["name"]): ram_scores[str(item["name"])]
                    for item in candidates
                    if ram.route_support.get(str(item["name"]), 0) > 0
                }
            )
            if ram_enabled
            else {}
        )
        combined = {
            str(item["name"]): 0.80 * trm_scores[str(item["name"])]
            + 0.20 * ram_prob.get(str(item["name"]), 0.0)
            for item in candidates
        }
        selected = by_name[max(combined, key=combined.get)]
        detail = {
            "flow": ["lexical_shortlist", "trm", "ram", "typed_ldt"],
            "combined": combined,
        }
    elif arm == "ram_then_trm_typed":
        candidates = _unique_candidates(lexical_top, supported_ram_top[:2], limit=7)
        trm_scores = _trm_probabilities(route_query, candidates, trm)
        selected = by_name[max(trm_scores, key=trm_scores.get)]
        detail = {"flow": ["ram_shortlist", "trm", "typed_ldt"], "trm": trm_scores}
    elif arm == "adaptive_mesh":
        candidates = _unique_candidates(lexical_top, supported_ram_top[:2], limit=7)
        trm_prob = _trm_probabilities(route_query, candidates, trm)
        trm_leader, trm_margin = _score_margin(trm_prob)
        if trm_leader == lexical_leader:
            selected_name = trm_leader
        elif guarded_ram_leader in {trm_leader, lexical_leader}:
            selected_name = guarded_ram_leader
        elif trm_margin >= 0.20:
            selected_name = trm_leader
        else:
            selected_name = lexical_leader
        selected = by_name[selected_name]
        detail = {
            "flow": ["lexical_ram_union", "trm", "agreement_arbiter", "typed_ldt"],
            "trm_ram_agree": bool(guarded_ram_leader and trm_leader == guarded_ram_leader),
            "ram_support_gate": ram_enabled,
        }
    else:
        raise ValueError(f"unknown arm: {arm}")
    detail["routing_compartment"] = compartment
    return selected, candidates, detail


def _ram_plan(contract: dict[str, Any], ram: SparseRAMPolicy) -> list[dict[str, str]]:
    result = []
    for phase in contract_phases(contract):
        operation, _ = ram.predict_action(phase)
        result.append(
            {
                "phase": str(phase["id"]),
                "module": str(phase.get("module", "coordinator")),
                "operation": operation,
            }
        )
    return result


def _packet_tokens(query: str, selected: dict[str, Any], candidates: list[dict[str, Any]], detail: dict[str, Any]) -> int:
    packet = {
        "task": query,
        "selected_contract": {
            "name": selected["name"],
            "purpose": selected.get("purpose", ""),
            "phases": expected_plan(selected),
            "gates": selected.get("gates", [])[:4],
            "output_contract": selected.get("output_contract", ""),
        },
        "candidate_cards": [
            {
                "name": item["name"],
                "family": item.get("route", {}).get("family", ""),
                "purpose": str(item.get("purpose", ""))[:180],
            }
            for item in candidates
        ],
        "flow": detail.get("flow", []),
    }
    return max(1, len(json.dumps(packet, ensure_ascii=False, separators=(",", ":"))) // 4)


def calibrate_registered(
    registration_dir: Path,
    model_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    receipt = read_json(registration_dir / "registration_receipt.json")
    protocol = read_json(registration_dir / "protocol.json")
    contracts = read_jsonl(registration_dir / "contracts.jsonl")
    cases = read_jsonl(registration_dir / "held_cases.jsonl")
    ram = SparseRAMPolicy.from_dict(read_json(model_dir / "ram_policy.json"))
    if ram.registration_id != receipt["registration_id"]:
        raise ValueError("RAM policy registration mismatch")

    import torch

    from agent.lean_router_trm import TinyRecursiveSkillRouter

    checkpoint = torch.load(model_dir / "trm_router.pt", map_location="cpu", weights_only=True)
    if checkpoint.get("registration_id") != receipt["registration_id"]:
        raise ValueError("TRM checkpoint registration mismatch")
    trm = TinyRecursiveSkillRouter()
    trm.load_state_dict(checkpoint["model"])
    trm.eval()
    primary = [item for item in contracts if item.get("route", {}).get("kind") != "overlay"]
    rows: list[dict[str, Any]] = []
    hard_context = int(protocol["context"]["hard_window_tokens"])
    for case in cases:
        for arm in ARMS:
            selected, candidates, detail = _arm_selection(
                arm,
                str(case["query"]),
                primary,
                ram,
                trm,
            )
            use_ram_plan = arm not in {"lexical_typed", "trm_typed"}
            raw_plan = _ram_plan(selected, ram) if use_ram_plan else expected_plan(selected)
            repaired_plan, ldt = validate_typed_plan(selected, raw_plan, repair=True)
            expected_contract = str(case.get("expected_contract_id", ""))
            forbidden_contract = str(case.get("forbidden_contract_id", ""))
            if case["kind"] == "positive":
                route_pass = str(selected["name"]) == expected_contract
                plan_exact = repaired_plan == case["expected_plan"]
            else:
                route_pass = str(selected["name"]) != forbidden_contract
                plan_exact = True
            tokens = _packet_tokens(str(case["query"]), selected, candidates, detail)
            rows.append(
                {
                    "schema": "hermes.lean_control_mesh_calibration_cell.v1",
                    "registration_id": receipt["registration_id"],
                    "case_id": case["case_id"],
                    "kind": case["kind"],
                    "lane": case["lane"],
                    "perturbation": case["perturbation"],
                    "arm": arm,
                    "selected_contract_id": selected["name"],
                    "expected_contract_id": expected_contract,
                    "forbidden_contract_id": forbidden_contract,
                    "route_pass": route_pass,
                    "pre_ldt_plan_exact": raw_plan == case["expected_plan"] if case["kind"] == "positive" else True,
                    "post_ldt_plan_exact": plan_exact,
                    "ldt_valid": ldt["valid"],
                    "ldt_repair_count": ldt["repair_count"],
                    "strict_success": bool(route_pass and plan_exact and ldt["valid"] and tokens <= hard_context),
                    "packet_tokens_est": tokens,
                    "context_compliant": tokens <= hard_context,
                    "flow": detail["flow"],
                    "routing_compartment": detail["routing_compartment"],
                    "trm_ram_agree": detail.get("trm_ram_agree"),
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "calibration_cells.jsonl", rows)
    by_arm = []
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        positives = [row for row in arm_rows if row["kind"] == "positive"]
        negatives = [row for row in arm_rows if row["kind"] == "negative"]
        transfer = [row for row in positives if row["lane"] == "contract_transfer"]
        by_arm.append(
            {
                "arm": arm,
                "cells": len(arm_rows),
                "strict_success_rate": sum(row["strict_success"] for row in arm_rows) / len(arm_rows),
                "positive_route_accuracy": sum(row["route_pass"] for row in positives) / len(positives),
                "contract_transfer_route_accuracy": sum(row["route_pass"] for row in transfer) / max(1, len(transfer)),
                "negative_control_pass_rate": sum(row["route_pass"] for row in negatives) / max(1, len(negatives)),
                "pre_ldt_plan_exact_rate": sum(row["pre_ldt_plan_exact"] for row in positives) / len(positives),
                "post_ldt_plan_exact_rate": sum(row["post_ldt_plan_exact"] for row in positives) / len(positives),
                "mean_ldt_repairs": sum(row["ldt_repair_count"] for row in arm_rows) / len(arm_rows),
                "mean_packet_tokens_est": sum(row["packet_tokens_est"] for row in arm_rows) / len(arm_rows),
                "max_packet_tokens_est": max(row["packet_tokens_est"] for row in arm_rows),
                "context_compliance_rate": sum(row["context_compliant"] for row in arm_rows) / len(arm_rows),
            }
        )
    summary = {
        "schema": "hermes.lean_control_mesh_calibration_summary.v1",
        "implementation_version": IMPLEMENTATION_VERSION,
        "status": "completed",
        "registration_id": receipt["registration_id"],
        "cells": len(rows),
        "case_count": len(cases),
        "arms": list(ARMS),
        "by_arm": by_arm,
        "cells_sha256": sha256_file(output_dir / "calibration_cells.jsonl"),
        "ram_policy_sha256": sha256_file(model_dir / "ram_policy.json"),
        "trm_router_sha256": sha256_file(model_dir / "trm_router.pt"),
        "full_hermes_baseline_present": False,
        "claim_boundary": protocol["claim_scope"],
    }
    write_json(output_dir / "calibration_summary.json", summary)
    return summary


def verify_registration(registration_dir: Path, expected_id: str = "") -> dict[str, Any]:
    receipt = read_json(registration_dir / "registration_receipt.json")
    checks = {
        "config": sha256_bytes(canonical_json_bytes(read_json(registration_dir / "protocol.json")))
        == receipt["config_sha256"],
        "contracts": contracts_digest(read_jsonl(registration_dir / "contracts.jsonl"))
        == receipt["contracts_sha256"],
        "train_rows": sha256_file(registration_dir / "train_rows.jsonl")
        == receipt["train_rows_sha256"],
        "held_cases": sha256_file(registration_dir / "held_cases.jsonl")
        == receipt["held_cases_sha256"],
        "registration_id": not expected_id or receipt["registration_id"] == expected_id,
    }
    if not all(checks.values()):
        raise ValueError(f"registration verification failed: {checks}")
    return {**receipt, "verification": checks}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes Lite 12k control-mesh v1")
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
    train.add_argument("--steps", type=int, default=400)
    train.add_argument("--ram-epochs", type=int, default=40)
    train.add_argument("--seed", type=int, default=73011)
    train.add_argument("--ram-cap-mb", type=int, default=2048)
    train.add_argument("--io-cap-mb-s", type=float, default=50.0)
    calibrate = sub.add_parser("calibrate")
    calibrate.add_argument("--registration-dir", required=True)
    calibrate.add_argument("--model-dir", required=True)
    calibrate.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register":
        result = register_study(Path(args.config), Path(args.output_dir))
    elif args.command == "verify":
        result = verify_registration(Path(args.registration_dir), args.registration_id)
    elif args.command == "train":
        result = train_registered(
            Path(args.registration_dir),
            Path(args.output_dir),
            steps=args.steps,
            ram_epochs=args.ram_epochs,
            seed=args.seed,
            ram_cap_mb=args.ram_cap_mb,
            io_cap_mb_s=args.io_cap_mb_s,
        )
    else:
        result = calibrate_registered(
            Path(args.registration_dir),
            Path(args.model_dir),
            Path(args.output_dir),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"failed", "aborted"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
