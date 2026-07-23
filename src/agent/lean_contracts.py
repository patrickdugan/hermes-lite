"""Contract projections and hybrid routing for the Bonsai lean runtime."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from agent.skill_catalog import iter_skill_files, load_ultra_lean_contract


GENERAL_LEAN_CONTRACT: dict[str, Any] = {
    "schema": "hermes.ultra_lean_skill.v2",
    "name": "general-lean",
    "purpose": "Fallback context-light execution for tasks without a matching specialist skill.",
    "route": {
        "kind": "primary",
        "family": "general",
        "aliases": ["general-lean"],
        "positive_examples": ["Handle a general local task.", "Inspect and act on a local coding request.", "Answer a task without a specialist skill."],
        "hard_negatives": ["Use a named specialist benchmark skill.", "Run a domain-specific verifier."],
        "compatible_overlays": [],
        "conflicts": [],
    },
    "context": {"hard_window_tokens": 12000, "active_working_set_tokens": 8000, "reserve_tokens": 4000},
    "conveyor": {
        "phases": [
            {"id": "INSPECT", "module": "artifact_retrieval", "allowed_operations": ["retrieve", "execute", "verify"], "retrieval_slots": 1, "tool_profile": ["search", "read"], "max_output_tokens": 512},
            {"id": "ACT", "module": "model_generation", "allowed_operations": ["execute", "verify", "repair"], "retrieval_slots": 1, "tool_profile": ["run", "write"], "max_output_tokens": 2048},
            {"id": "VERIFY", "module": "verifier", "allowed_operations": ["verify", "repair", "abstain"], "retrieval_slots": 0, "tool_profile": [], "max_output_tokens": 256},
            {"id": "FINAL", "module": "finalizer", "allowed_operations": ["commit", "repair", "abstain"], "retrieval_slots": 0, "tool_profile": [], "max_output_tokens": 512},
        ],
        "model_action_fields": ["phase", "operation", "gate"],
        "harness_owned_fields": ["task_id", "contract_id", "candidate_ref", "summary_delta"],
    },
    "modules": ["artifact_retrieval", "model_generation", "verifier", "finalizer"],
    "gates": ["Treat generated output as a candidate until verification passes.", "Use the smallest available tool profile."],
    "retrieval": {"load_order": ["task_card", "current_phase", "state", "one_evidence_item"], "reference_handles": [], "tool_hints": [], "forbidden": ["full_reference_tree", "raw_transcript"]},
    "output_contract": "Return the verified result directly and concisely.",
    "repair": "Record one failure code and retry the smallest failed phase once.",
}


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokens(value: str) -> set[str]:
    return set(TOKEN_RE.findall(value.lower()))


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def contract_phases(contract: dict[str, Any]) -> list[dict[str, Any]]:
    raw = contract.get("conveyor", {}).get("phases", [])
    phases = []
    for item in raw:
        if isinstance(item, str):
            phases.append({"id": item, "module": "coordinator", "allowed_operations": ["select", "execute", "verify"], "retrieval_slots": 1, "tool_profile": [], "max_output_tokens": 256})
        elif isinstance(item, dict) and item.get("id"):
            phases.append(item)
    return phases


def phase_projection(contract: dict[str, Any], phase_index: int, overlays: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    phases = contract_phases(contract)
    phase = phases[min(max(phase_index, 0), max(0, len(phases) - 1))] if phases else {"id": "ACT"}
    overlay_gates = []
    for overlay in overlays:
        overlay_gates.extend(overlay.get("gates", [])[:2])
    return {
        "contract_id": contract.get("name"),
        "purpose": str(contract.get("purpose", ""))[:300],
        "phase": phase,
        "gates": list(contract.get("gates", []))[:4] + overlay_gates[:2],
        "output_contract": str(contract.get("output_contract", ""))[:300],
        "repair": str(contract.get("repair", ""))[:220],
    }


def load_contracts() -> list[dict[str, Any]]:
    result = []
    for _, skill_file in iter_skill_files():
        contract = load_ultra_lean_contract(skill_file)
        if contract:
            result.append(contract)
    return sorted(result, key=lambda item: str(item.get("name", "")))


def route_feature_vector(query: str, contract: dict[str, Any], deterministic_score: float = 0.0) -> list[float]:
    route = contract.get("route", {})
    q = tokens(query)
    aliases = [str(value).lower() for value in route.get("aliases", [])]
    positives = [tokens(str(value)) for value in route.get("positive_examples", [])]
    negatives = [tokens(str(value)) for value in route.get("hard_negatives", [])]
    purpose = tokens(str(contract.get("purpose", "")))
    family = str(route.get("family", "")).lower()
    base = [
        float(any(alias == query.lower().strip() for alias in aliases)),
        float(any(alias and alias in query.lower() for alias in aliases)),
        float(family in q),
        max((_jaccard(q, value) for value in positives), default=0.0),
        max((_jaccard(q, value) for value in negatives), default=0.0),
        _jaccard(q, purpose),
        min(1.0, deterministic_score / 12.0),
        min(1.0, len(q) / 40.0),
    ]
    hashed = [0.0] * 24
    for token in q:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=2).digest()
        index = int.from_bytes(digest, "big") % len(hashed)
        hashed[index] = min(1.0, hashed[index] + 0.25)
    return base + hashed


def deterministic_route_score(query: str, contract: dict[str, Any]) -> float:
    route = contract.get("route", {})
    q_text = query.lower()
    q = tokens(query)
    score = 0.0
    for alias in route.get("aliases", []):
        alias_text = str(alias).lower().strip()
        if alias_text and alias_text in q_text:
            score += 8.0 if alias_text == q_text.strip() else 5.0
    family = str(route.get("family", "")).lower()
    if family and family in q:
        score += 3.0
    positives = [tokens(str(value)) for value in route.get("positive_examples", [])]
    score += 10.0 * max((_jaccard(q, value) for value in positives), default=0.0)
    purpose_overlap = len(q & tokens(str(contract.get("purpose", ""))))
    score += min(4.0, purpose_overlap * 0.5)
    negatives = [tokens(str(value)) for value in route.get("hard_negatives", [])]
    score -= 4.0 * max((_jaccard(q, value) for value in negatives), default=0.0)
    return score


@dataclass
class RouteCandidate:
    contract: dict[str, Any]
    deterministic_score: float
    probability: float = 0.0

    def card(self) -> dict[str, Any]:
        route = self.contract.get("route", {})
        return {
            "contract_id": self.contract.get("name"),
            "family": route.get("family"),
            "purpose": str(self.contract.get("purpose", ""))[:180],
            "score": round(self.deterministic_score, 4),
            "probability": round(self.probability, 4),
        }


class HybridSkillRouter:
    def __init__(self, contracts: list[dict[str, Any]], checkpoint: str | Path | None = None):
        self.contracts = [item for item in contracts if item.get("route", {}).get("kind") != "overlay"]
        self.overlays = {str(item.get("name")): item for item in contracts if item.get("route", {}).get("kind") == "overlay"}
        self.checkpoint = Path(checkpoint).expanduser() if checkpoint else None
        self._model = None

    def _load_model(self):
        if self._model is not None or not self.checkpoint or not self.checkpoint.exists():
            return self._model
        try:
            import torch
            from agent.lean_router_trm import TinyRecursiveSkillRouter

            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
            model = TinyRecursiveSkillRouter()
            model.load_state_dict(payload["model"])
            model.eval()
            self._model = model
        except Exception:
            self._model = None
        return self._model

    def candidates(self, query: str, limit: int = 5) -> list[RouteCandidate]:
        ranked = sorted(
            (RouteCandidate(contract, deterministic_route_score(query, contract)) for contract in self.contracts),
            key=lambda item: item.deterministic_score,
            reverse=True,
        )[:limit]
        model = self._load_model()
        if model and ranked:
            try:
                import torch

                features = torch.tensor([[route_feature_vector(query, item.contract, item.deterministic_score) for item in ranked]], dtype=torch.float32)
                with torch.no_grad():
                    logits, _ = model(features)
                probabilities = torch.softmax(logits[0, : len(ranked)], dim=-1).tolist()
                for item, probability in zip(ranked, probabilities):
                    item.probability = float(probability)
                ranked.sort(key=lambda item: item.probability, reverse=True)
                return ranked
            except Exception:
                pass
        positive = [max(0.0, item.deterministic_score) for item in ranked]
        denominator = sum(math.exp(min(12.0, value)) for value in positive) or 1.0
        for item, value in zip(ranked, positive):
            item.probability = math.exp(min(12.0, value)) / denominator
        return ranked

    def compatible_overlays(self, primary: dict[str, Any], query: str, limit: int = 2) -> list[dict[str, Any]]:
        allowed = primary.get("route", {}).get("compatible_overlays", [])
        ranked = []
        for name in allowed:
            overlay = self.overlays.get(str(name))
            if overlay:
                ranked.append((deterministic_route_score(query, overlay), overlay))
        return [item for score, item in sorted(ranked, key=lambda pair: pair[0], reverse=True) if score > 0.5][:limit]


def contracts_digest(contracts: list[dict[str, Any]]) -> str:
    payload = json.dumps(contracts, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
