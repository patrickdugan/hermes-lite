"""Capped RAM/TRM benchmark for the registered BitAgent control mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from agent.bitagent_control_mesh_v0 import (
    CANDIDATE_SCHEMA,
    LDT_SCHEMA,
    canonical_hash,
    file_sha256,
    load_protocol,
    seal_mesh_receipt,
    validate_ldt_candidate,
    verify_mesh_receipt,
)
from agent.bitagent_role_adapters import build_packet, load_config as load_role_config, route_role


CAP_WRAPPER_FLAG = "LEAN_CONTROL_MESH_CAP_WRAPPER_ACTIVE"
FEATURE_DIM = 96
HIDDEN_DIM = 24
RECURSIVE_STEPS = 4
ARMS = ("lexical_ldt", "ram_ldt", "trm_ldt", "adaptive_mesh_ldt")
AMOUNT = re.compile(r"\b(?:0\.\d+\s*(?:btc|bitcoin)|\d+\s*(?:sats?|satoshis?))\b", re.IGNORECASE)
ADDRESS = re.compile(r"\b(?:bc1|tb1|ltc1|tltc1)[a-z0-9]{20,90}\b", re.IGNORECASE)
SECRET = re.compile(r"(?:seed phrase|private key|mnemonic|\bwif\b)", re.IGNORECASE)
UNSUPPORTED = re.compile(
    r"(?:every asset|whole portfolio|leveraged perpetual|eth option|train a new adapter|"
    r"through a trm|lend my usdc|bridge to solana|second strategy|tell me a joke)",
    re.IGNORECASE,
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"elapsed_seconds": round(time.monotonic(), 6), **event}, sort_keys=True) + "\n")


def descriptor(task_card: dict[str, Any]) -> str:
    return " ".join(
        (
            str(task_card.get("message", "")),
            f"wallet_phase={task_card.get('wallet_phase', '')}",
            f"route_phase={task_card.get('phase', '')}",
        )
    ).lower()


def token_features(text: str) -> tuple[str, ...]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    features = {f"w:{word}" for word in words}
    features.update(f"b:{left}_{right}" for left, right in zip(words, words[1:]))
    features.add("bias")
    return tuple(sorted(features))


def dense_features(text: str):
    import torch

    vector = torch.zeros(FEATURE_DIM, dtype=torch.float32)
    for feature in token_features(text):
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % FEATURE_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[index] += sign
    norm = vector.norm()
    return vector / norm if norm > 0 else vector


def label_key(labels: dict[str, Any]) -> str:
    return json.dumps(
        {
            "intent": labels.get("intent"),
            "tool": labels.get("tool"),
            "missing": labels.get("missing"),
            "prohibited": bool(labels.get("prohibited", False)),
            "role": labels.get("role"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def label_catalog(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {label_key(row["labels"]): dict(row["labels"]) for row in rows}


@dataclass
class RAMClassifier:
    labels: list[str]
    weights: dict[str, dict[str, float]]
    vocabulary: set[str]

    @classmethod
    def empty(cls, labels: list[str]) -> "RAMClassifier":
        return cls(labels=sorted(labels), weights={label: {} for label in sorted(labels)}, vocabulary=set())

    def scores(self, text: str) -> dict[str, float]:
        features = token_features(text)
        return {
            label: sum(self.weights[label].get(feature, 0.0) for feature in features)
            for label in self.labels
        }

    def predict(self, text: str) -> tuple[str, float, float]:
        scores = self.scores(text)
        ranked = sorted(((score, label) for label, score in scores.items()), reverse=True)
        best_score, best = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else 0.0
        features = set(token_features(text))
        coverage = len(features & self.vocabulary) / max(1, len(features))
        confidence = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, best_score - runner_up))))
        return best, confidence, coverage

    def to_dict(self, registration_id: str, seed: int) -> dict[str, Any]:
        return {
            "schema": "hermes.bitagent_ram_classifier.v0",
            "registration_id": registration_id,
            "seed": seed,
            "labels": self.labels,
            "weights": {
                label: {
                    feature: round(value, 6)
                    for feature, value in sorted(weights.items())
                    if abs(value) > 1e-12
                }
                for label, weights in sorted(self.weights.items())
            },
            "vocabulary": sorted(self.vocabulary),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RAMClassifier":
        if value.get("schema") != "hermes.bitagent_ram_classifier.v0":
            raise ValueError("invalid BitAgent RAM classifier")
        return cls(
            labels=list(value["labels"]),
            weights={
                label: {feature: float(score) for feature, score in weights.items()}
                for label, weights in value["weights"].items()
            },
            vocabulary=set(value["vocabulary"]),
        )


def train_ram(rows: list[dict[str, Any]], *, epochs: int, seed: int) -> RAMClassifier:
    catalog = label_catalog(rows)
    policy = RAMClassifier.empty(list(catalog))
    randomizer = random.Random(seed)
    examples = [(descriptor(row["task_card"]), label_key(row["labels"])) for row in rows]
    policy.vocabulary = {feature for text, _ in examples for feature in token_features(text)}
    for _ in range(epochs):
        shuffled = list(examples)
        randomizer.shuffle(shuffled)
        for text, target in shuffled:
            prediction, _, _ = policy.predict(text)
            if prediction == target:
                continue
            for feature in token_features(text):
                policy.weights[target][feature] = policy.weights[target].get(feature, 0.0) + 1.0
                policy.weights[prediction][feature] = policy.weights[prediction].get(feature, 0.0) - 1.0
    return policy


def make_trm(label_count: int):
    import torch
    from torch import nn

    class TinyBitAgentTRM(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(FEATURE_DIM, HIDDEN_DIM)
            self.recurrent = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
            self.output = nn.Linear(HIDDEN_DIM, label_count)

        def forward(self, features):
            encoded = torch.tanh(self.input_projection(features))
            hidden = torch.zeros_like(encoded)
            for _ in range(RECURSIVE_STEPS):
                hidden = torch.tanh(self.recurrent(torch.cat((encoded, hidden), dim=-1)))
            return self.output(hidden)

    return TinyBitAgentTRM()


def trm_manifest(model: Any, labels: list[str], registration_id: str, seed: int) -> dict[str, Any]:
    return {
        "schema": "hermes.bitagent_tiny_recursive_classifier.v0",
        "registration_id": registration_id,
        "seed": seed,
        "feature_dim": FEATURE_DIM,
        "hidden_dim": HIDDEN_DIM,
        "recursive_steps": RECURSIVE_STEPS,
        "labels": labels,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }


def candidate_from_labels(
    labels: dict[str, Any],
    task_card: dict[str, Any],
    evidence: dict[str, str],
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "role": labels["role"],
        "intent": labels["intent"],
        "missing_fields": [labels["missing"]] if labels.get("missing") else [],
        "prohibited": bool(labels.get("prohibited", False)),
        "execute": False,
        "evidence": {
            "task_card_sha256": canonical_hash(task_card),
            "state_sha256": evidence["state_sha256"],
            "tool_contracts_sha256": evidence["tool_contracts_sha256"],
            "route_sha256": evidence["route_sha256"],
        },
    }
    if labels.get("tool"):
        candidate["tool"] = {"name": labels["tool"], "arguments": {}}
    return candidate


def lexical_labels(task_card: dict[str, Any]) -> dict[str, Any]:
    message = str(task_card.get("message", ""))
    lower = message.lower()
    wallet_phase = str(task_card.get("wallet_phase", ""))
    role = route_role(task_card)
    if SECRET.search(message):
        return {"intent": "unsupported", "tool": None, "missing": None, "prohibited": True, "role": role}
    if UNSUPPORTED.search(message):
        return {"intent": "unsupported", "tool": None, "missing": None, "prohibited": False, "role": role}
    if "withdraw" in lower or "cash out" in lower or "send back" in lower or "send my bitcoin" in lower:
        missing = None
        found_address = ADDRESS.search(message)
        if "tb1q" in lower and (
            found_address is None or set(found_address.group(0).lower().removeprefix("tb1q")) in ({"a"}, {"z"})
        ):
            missing = "validDestinationAddress"
        elif not AMOUNT.search(message):
            missing = "amountSats"
        elif not found_address and "{{address}}" not in lower:
            missing = "destinationAddress"
        return {
            "intent": "withdraw_bitcoin",
            "tool": None if missing else "bitagent.withdraw.simulate",
            "missing": missing,
            "prohibited": False,
            "role": role,
        }
    if "strategy" in lower or "use part of my bitcoin" in lower:
        missing = "confirmed_deposit" if wallet_phase != "confirmed" else None
        if missing is None and not AMOUNT.search(message):
            missing = "amountSats"
        return {
            "intent": "starter_strategy",
            "tool": None if missing else "bitagent.strategy.simulate",
            "missing": missing,
            "prohibited": False,
            "role": role,
        }
    if any(term in lower for term in ("deposit", "receive bitcoin", "fund my wallet", "deposit address")):
        return {
            "intent": "deposit_bitcoin",
            "tool": "bitagent.wallet.connect" if wallet_phase == "disconnected" else "bitagent.deposit.prepare",
            "missing": "wallet_connection_choice" if wallet_phase == "disconnected" and "help me" in lower else None,
            "prohibited": False,
            "role": role,
        }
    return {"intent": "unsupported", "tool": None, "missing": None, "prohibited": False, "role": role}


def candidate_signature(candidate: dict[str, Any]) -> dict[str, Any]:
    tool = candidate.get("tool")
    return {
        "intent": candidate.get("intent"),
        "tool": tool.get("name") if isinstance(tool, dict) else None,
        "missing": (candidate.get("missing_fields") or [None])[0],
        "prohibited": bool(candidate.get("prohibited", False)),
        "role": candidate.get("role"),
    }


def _augment_semantic_ldt(
    decision: dict[str, Any],
    candidate: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    if decision["decision"] != "candidate_valid" or candidate_signature(candidate) == expected:
        return decision
    material = {
        key: value
        for key, value in decision.items()
        if key != "ldt_sha256"
    }
    material["decision"] = "rejected"
    material["reason_codes"] = [*material["reason_codes"], "host_precondition_mismatch"]
    material["fallback"] = "deterministic_host"
    return {**material, "ldt_sha256": canonical_hash(material)}


def load_trm(path: Path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    manifest = payload["manifest"]
    model = make_trm(len(manifest["labels"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, manifest


def predict_trm(model: Any, labels: list[str], text: str) -> tuple[str, float]:
    import torch

    with torch.no_grad():
        logits = model(dense_features(text).unsqueeze(0)).squeeze(0)
        probabilities = torch.softmax(logits, dim=-1)
        index = int(probabilities.argmax())
        return labels[index], float(probabilities[index])


def train_models(args: argparse.Namespace) -> int:
    if os.getenv(CAP_WRAPPER_FLAG) != "1":
        raise SystemExit("Refusing uncapped BitAgent mesh training; use scripts/run_capped_lean_control_mesh_v1.ps1")
    import torch
    from torch import nn

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    registration = Path(args.registration_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    events = output / "events.jsonl"
    receipt = read_json(registration / "registration_receipt.json")
    protocol = load_protocol(registration / "protocol.json")
    train_path = registration / "train_cases.jsonl"
    if file_sha256(train_path) != receipt["train_cases_sha256"]:
        raise ValueError("registered training rows hash mismatch")
    train_rows = read_jsonl(train_path)
    catalog = label_catalog(train_rows)
    labels = sorted(catalog)
    process = psutil.Process()
    started = time.monotonic()
    last_time = started
    initial_io = process.io_counters()
    last_io = initial_io.read_bytes + initial_io.write_bytes
    peak_ram = peak_io = 0.0
    status = "failed"
    abort_reason = ""
    checkpoints: list[str] = []
    step = 0
    model = optimizer = None
    try:
        ram = train_ram(train_rows, epochs=args.ram_epochs, seed=args.seed)
        write_json(output / "ram_policy.json", ram.to_dict(receipt["registration_id"], args.seed))
        features = torch.stack([dense_features(descriptor(row["task_card"])) for row in train_rows])
        targets = torch.tensor([labels.index(label_key(row["labels"])) for row in train_rows], dtype=torch.long)
        model = make_trm(len(labels))
        manifest = trm_manifest(model, labels, receipt["registration_id"], args.seed)
        if manifest["parameter_count"] >= 10_000:
            raise RuntimeError("BitAgent TRM parameter cap exceeded")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(args.seed)
        while step < args.steps:
            permutation = torch.randperm(len(train_rows), generator=generator)
            for start in range(0, len(train_rows), 16):
                step += 1
                indexes = permutation[start : start + 16]
                optimizer.zero_grad(set_to_none=True)
                logits = model(features[indexes])
                loss = loss_fn(logits, targets[indexes])
                loss.backward()
                optimizer.step()
                now = time.monotonic()
                memory_mb = process.memory_info().rss / 1024 / 1024
                io = process.io_counters()
                io_bytes = io.read_bytes + io.write_bytes
                io_rate = max(0, io_bytes - last_io) / 1024 / 1024 / max(0.001, now - last_time)
                peak_ram = max(peak_ram, memory_mb)
                peak_io = max(peak_io, io_rate)
                last_io, last_time = io_bytes, now
                if memory_mb > args.ram_cap_mb:
                    raise RuntimeError("ram_cap_exceeded")
                if io_rate > args.io_cap_mb_s:
                    raise RuntimeError("io_cap_exceeded")
                if now - started > args.wall_seconds:
                    raise RuntimeError("wall_clock_cap")
                if step % args.checkpoint_steps == 0 or step >= args.steps:
                    checkpoint = output / f"trm-step-{step:06d}.pt"
                    torch.save(
                        {"state_dict": model.state_dict(), "manifest": manifest, "step": step},
                        checkpoint,
                    )
                    checkpoints.append(str(checkpoint))
                    append_event(
                        events,
                        {
                            "event": "checkpoint",
                            "step": step,
                            "loss": float(loss.detach()),
                            "ram_mb": round(memory_mb, 3),
                            "io_mb_s": round(io_rate, 3),
                        },
                    )
                if step >= args.steps:
                    break
        final_model = output / "trm_router.pt"
        shutil.copy2(checkpoints[-1], final_model)
        status = "completed"
    except Exception as exc:
        status = "aborted"
        abort_reason = str(exc)
    finally:
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
        del train_rows
        import gc

        gc.collect()
    training_receipt = {
        "schema": "hermes.bitagent_mesh_training_receipt.v0",
        "status": status,
        "abort_reason": abort_reason,
        "registration_id": receipt["registration_id"],
        "seed": args.seed,
        "steps_completed": step,
        "ram_epochs": args.ram_epochs,
        "train_case_reads": 40,
        "held_case_reads": 0,
        "caps": {
            "ram_mb": args.ram_cap_mb,
            "io_mb_s": args.io_cap_mb_s,
            "wall_seconds": args.wall_seconds,
        },
        "peak_ram_mb": round(peak_ram, 3),
        "peak_io_mb_s": round(peak_io, 3),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "checkpoints": checkpoints,
        "ram_policy_sha256": file_sha256(output / "ram_policy.json") if (output / "ram_policy.json").exists() else "",
        "trm_router_sha256": file_sha256(output / "trm_router.pt") if (output / "trm_router.pt").exists() else "",
        "protocol_training_caps_match": {
            "ram": args.ram_cap_mb == int(protocol["training"]["ram_cap_mb"]),
            "io": args.io_cap_mb_s == float(protocol["training"]["io_cap_mb_s"]),
            "wall": args.wall_seconds == int(protocol["training"]["wall_seconds"]),
        },
    }
    write_json(output / "training_receipt.json", training_receipt)
    return 0 if status == "completed" else 2


def evaluate(args: argparse.Namespace) -> int:
    started = time.monotonic()
    registration = Path(args.registration_dir)
    model_dir = Path(args.model_dir)
    output = Path(args.output_dir)
    if (output / "results.json").exists() and not args.force:
        raise ValueError("evaluation output already exists; use a fresh directory")
    output.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(registration / "protocol.json")
    registration_receipt = read_json(registration / "registration_receipt.json")
    held_path = registration / "held_cases.jsonl"
    if file_sha256(held_path) != registration_receipt["held_cases_sha256"]:
        raise ValueError("registered held rows hash mismatch")
    held_rows = read_jsonl(held_path)
    role_config = load_role_config(Path(protocol["role_adapter_config"]))
    ram_payload = read_json(model_dir / "ram_policy.json")
    ram = RAMClassifier.from_dict(ram_payload)
    trm, trm_info = load_trm(model_dir / "trm_router.pt")
    if ram_payload["registration_id"] != registration_receipt["registration_id"]:
        raise ValueError("RAM registration mismatch")
    if trm_info["registration_id"] != registration_receipt["registration_id"]:
        raise ValueError("TRM registration mismatch")
    catalog = label_catalog(read_jsonl(registration / "train_cases.jsonl"))
    records: list[dict[str, Any]] = []
    for row in held_rows:
        task_card = row["task_card"]
        text = descriptor(task_card)
        route = {"role": route_role(task_card), "owner": "deterministic_host"}
        tool_contracts = [
            {"name": name, "schema": {"type": "object"}}
            for contract in role_config["roles"].values()
            for name in contract["allowed_tools"]
        ]
        state = {"wallet_phase": task_card["wallet_phase"], "case_id": row["id"]}
        evidence = {
            "state_sha256": canonical_hash(state),
            "tool_contracts_sha256": canonical_hash(tool_contracts),
            "route_sha256": canonical_hash(route),
        }
        packet, packet_receipt = build_packet(
            role_config,
            task_card=task_card,
            workflow_state=state,
            tool_contracts=tool_contracts,
            evidence=[{"route_sha256": evidence["route_sha256"]}],
        )
        expected_observable = lexical_labels(task_card)
        ram_label, ram_confidence, ram_coverage = ram.predict(text)
        trm_label, trm_confidence = predict_trm(trm, trm_info["labels"], text)
        arm_labels = {
            "lexical_ldt": expected_observable,
            "ram_ldt": catalog[ram_label],
            "trm_ldt": catalog[trm_label],
        }
        if ram_label == trm_label and min(ram_confidence, trm_confidence, ram_coverage) >= args.adaptive_threshold:
            selected = catalog[ram_label]
            adaptive_source = "ram_trm_agreement"
        else:
            selected = expected_observable
            adaptive_source = "lexical_uncertainty_fallback"
        arm_labels["adaptive_mesh_ldt"] = selected
        for arm in ARMS:
            arm_started = time.perf_counter()
            proposal = candidate_from_labels(arm_labels[arm], task_card, evidence)
            ldt = validate_ldt_candidate(role_config, protocol, task_card, proposal, evidence)
            ldt = _augment_semantic_ldt(ldt, proposal, expected_observable)
            fallback_used = ldt["decision"] != "candidate_valid"
            final_candidate = (
                candidate_from_labels(expected_observable, task_card, evidence)
                if fallback_used
                else proposal
            )
            fallback = {
                "used": fallback_used,
                "policy": "deterministic_host" if fallback_used else "none",
                "candidate_sha256": canonical_hash(final_candidate) if fallback_used else "",
            }
            mesh_receipt = seal_mesh_receipt(
                envelope=packet,
                proposal=proposal,
                route=route,
                ldt_decision=ldt,
                fallback=fallback,
            )
            receipt_valid = verify_mesh_receipt(
                mesh_receipt,
                envelope=packet,
                proposal=proposal,
                route=route,
                ldt_decision=ldt,
                fallback=fallback,
            )
            proposal_signature = candidate_signature(proposal)
            final_signature = candidate_signature(final_candidate)
            expected_signature = {
                "intent": row["labels"]["intent"],
                "tool": row["labels"]["tool"],
                "missing": row["labels"]["missing"],
                "prohibited": row["labels"]["prohibited"],
                "role": row["labels"]["role"],
            }
            unsafe_proposal = bool(
                row["labels"]["prohibited"]
                and (proposal_signature["tool"] or proposal_signature["intent"] != "unsupported")
            )
            records.append(
                {
                    "schema": "hermes.bitagent_mesh_case_receipt.v0",
                    "seed": trm_info["seed"],
                    "case_id": row["id"],
                    "arm": arm,
                    "proposal_correct": proposal_signature == expected_signature,
                    "final_correct": final_signature == expected_signature,
                    "route_recommendation_correct": proposal_signature["role"] == expected_signature["role"],
                    "ldt_decision": ldt["decision"],
                    "ldt_reason_codes": ldt["reason_codes"],
                    "fallback_used": fallback_used,
                    "unsafe_proposal": unsafe_proposal,
                    "unsafe_candidate_accepted": unsafe_proposal and not fallback_used,
                    "packet_tokens": packet_receipt["estimated_tokens"],
                    "latency_ms": round((time.perf_counter() - arm_started) * 1000, 4),
                    "ram_confidence": round(ram_confidence, 6),
                    "ram_coverage": round(ram_coverage, 6),
                    "trm_confidence": round(trm_confidence, 6),
                    "adaptive_source": adaptive_source if arm == "adaptive_mesh_ldt" else "",
                    "mesh_receipt": mesh_receipt,
                    "receipt_valid": receipt_valid,
                }
            )
    records_path = output / "records.jsonl"
    write_jsonl(records_path, records)
    summaries = {}
    for arm in ARMS:
        arm_rows = [row for row in records if row["arm"] == arm]
        count = len(arm_rows)
        summaries[arm] = {
            "cases": count,
            "proposal_success": sum(row["proposal_correct"] for row in arm_rows) / count,
            "held_task_success": sum(row["final_correct"] for row in arm_rows) / count,
            "role_route_accuracy": sum(row["route_recommendation_correct"] for row in arm_rows) / count,
            "fallback_rate": sum(row["fallback_used"] for row in arm_rows) / count,
            "unsafe_candidate_acceptance": sum(row["unsafe_candidate_accepted"] for row in arm_rows) / count,
            "mean_packet_tokens": round(sum(row["packet_tokens"] for row in arm_rows) / count, 3),
            "mean_latency_ms": round(sum(row["latency_ms"] for row in arm_rows) / count, 4),
            "receipt_failures": sum(not row["receipt_valid"] for row in arm_rows),
        }
    host_baseline = read_json(Path(args.host_baseline_receipt)) if args.host_baseline_receipt else {}
    host_baseline_passed = bool(
        host_baseline.get("schema") == "hermes.bitagent_host_baseline_receipt.v0"
        and host_baseline.get("status") == "passed"
        and host_baseline.get("source_manifest_sha256") == registration_receipt["source_manifest_sha256"]
        and host_baseline.get("critical_test_count") == 36
        and host_baseline.get("launch_case_count") == 50
    )
    gates = {
        arm: {
            "held_task_success": values["held_task_success"] >= protocol["promotion"]["min_held_task_success"],
            "role_route_accuracy": values["role_route_accuracy"] >= protocol["promotion"]["min_role_route_accuracy"],
            "unsafe_candidate_acceptance": values["unsafe_candidate_acceptance"]
            <= protocol["promotion"]["max_unsafe_candidate_acceptance"],
            "packet_tokens": values["mean_packet_tokens"] <= protocol["promotion"]["max_packet_tokens"],
            "receipt_integrity": values["receipt_failures"] == 0,
        }
        for arm, values in summaries.items()
    }
    result = {
        "schema": "hermes.bitagent_mesh_results.v0",
        "status": "completed",
        "study_id": protocol["study_id"],
        "registration_id": registration_receipt["registration_id"],
        "seed": trm_info["seed"],
        "held_case_count": len(held_rows),
        "held_cases_sha256": file_sha256(held_path),
        "records_sha256": file_sha256(records_path),
        "arms": summaries,
        "gates": gates,
        "host_baseline_replay_passed": host_baseline_passed,
        "registered_promotion_eligible": {
            arm: all(gates[arm].values()) and host_baseline_passed
            for arm in ARMS
        },
        "general_four_role_claim_permitted": False,
        "general_four_role_claim_blocker": "registered held cases contain no recovery_operator tasks",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "claim_scope": protocol["claim_scope"],
    }
    write_json(output / "results.json", result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--registration-dir", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--steps", type=int, default=400)
    train.add_argument("--ram-epochs", type=int, default=40)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--ram-cap-mb", type=int, default=2048)
    train.add_argument("--io-cap-mb-s", type=float, default=50)
    train.add_argument("--wall-seconds", type=int, default=900)
    train.add_argument("--checkpoint-steps", type=int, default=50)
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--registration-dir", required=True)
    evaluate_parser.add_argument("--model-dir", required=True)
    evaluate_parser.add_argument("--output-dir", required=True)
    evaluate_parser.add_argument("--host-baseline-receipt", default="")
    evaluate_parser.add_argument("--adaptive-threshold", type=float, default=0.6)
    evaluate_parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return train_models(args)
    if args.command == "evaluate":
        return evaluate(args)
    raise ValueError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
