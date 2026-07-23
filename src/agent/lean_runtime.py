"""Stateless ultra-lean execution profile for Bonsai-class local models."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import request

from agent.lean_contracts import (
    GENERAL_LEAN_CONTRACT,
    HybridSkillRouter,
    contract_phases,
    load_contracts,
    phase_projection,
)
from agent.lean_packet import ContextOverflow, LEAN_KERNEL, StepPacketBuilder, TokenCounter
from agent.lean_state import ArtifactBroker, LeanStateStore, utc_now
from agent.skill_catalog import find_skill


ACTIONS = {"select", "route", "retrieve", "execute", "verify", "commit", "repair", "abstain"}


def should_use_lean_runtime(config: dict[str, Any], model: str, context_length: int) -> bool:
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    mode = os.getenv("HERMES_RUNTIME_MODE", str(runtime.get("mode", "auto"))).strip().lower()
    if mode == "standard":
        return False
    if mode == "lean":
        return True
    skills = config.get("skills", {}) if isinstance(config, dict) else {}
    models = skills.get("ultra_lean_models", ["local/bonsai-8b", "digitsflow/bonsai-8b"])
    return model in models and int(context_length) <= 12600


def _parse_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:].lstrip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(stripped[start : end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
    return None


class LeanSkillRuntime:
    def __init__(self, *, agent, config: dict[str, Any]):
        self.agent = agent
        self.config = config
        lean = config.get("runtime", {}).get("lean", {}) if isinstance(config, dict) else {}
        self.max_input_tokens = int(lean.get("max_input_tokens", 8000))
        self.max_repair_attempts = int(lean.get("max_repair_attempts", 1))
        self.router_confidence = float(lean.get("router_confidence", 0.80))
        self.router_margin = float(lean.get("router_margin", 0.15))
        self.state_root = Path(str(lean.get("state_dir", ".hermes/runtime")))
        checkpoint = str(lean.get("router_checkpoint", "~/.hermes-lite/models/skill-router.pt"))
        self.contracts = load_contracts()
        self.router = HybridSkillRouter(self.contracts, checkpoint=checkpoint)
        self.counter = TokenCounter(agent.base_url)
        self.packet_builder = StepPacketBuilder(counter=self.counter, max_input_tokens=self.max_input_tokens)

    def _post(self, messages: list[dict[str, str]], *, max_tokens: int, json_mode: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.agent.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        req = request.Request(
            self.agent.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.agent.api_key or 'local'}"},
            method="POST",
        )
        with request.urlopen(req, timeout=180) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _content(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                return str(message.get("content") or "")
        return ""

    def _bonsai_route(self, query: str, candidates) -> str | None:
        cards = [item.card() for item in candidates[:3]]
        prompt = (
            "REQUEST=" + query[:4000] + "\nCANDIDATES=" + json.dumps(cards, separators=(",", ":")) +
            "\nReturn JSON only: {\"contract_id\": one candidate id or \"general-lean\"}."
        )
        try:
            response = self._post(
                [{"role": "system", "content": "Route one request among at most three compact skill cards."}, {"role": "user", "content": prompt}],
                max_tokens=64,
                json_mode=True,
            )
            value = _parse_object(self._content(response))
            chosen = str(value.get("contract_id")) if value else ""
            if chosen == "general-lean" or any(chosen == item.contract.get("name") for item in candidates[:3]):
                return chosen
        except Exception:
            pass
        return None

    def route(self, query: str) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        candidates = self.router.candidates(query, limit=5)
        if not candidates or candidates[0].deterministic_score <= 0.5:
            return GENERAL_LEAN_CONTRACT, [], {"mode": "general_fallback", "candidates": [item.card() for item in candidates[:3]]}
        top = candidates[0]
        margin = top.probability - (candidates[1].probability if len(candidates) > 1 else 0.0)
        chosen = top.contract
        mode = "auto"
        if top.probability < self.router_confidence or margin < self.router_margin:
            chosen_id = self._bonsai_route(query, candidates)
            mode = "bonsai_shortlist"
            if chosen_id == "general-lean":
                chosen = GENERAL_LEAN_CONTRACT
            elif chosen_id:
                chosen = next((item.contract for item in candidates if item.contract.get("name") == chosen_id), top.contract)
        overlays = [] if chosen is GENERAL_LEAN_CONTRACT else self.router.compatible_overlays(chosen, query, limit=2)
        return chosen, overlays, {
            "mode": mode,
            "api_calls": 1 if mode == "bonsai_shortlist" else 0,
            "confidence": top.probability,
            "margin": margin,
            "candidates": [item.card() for item in candidates[:3]],
            "selected": chosen.get("name"),
            "overlays": [item.get("name") for item in overlays],
        }

    def _coordinator_action(self, packet: str, phase: dict[str, Any], store: LeanStateStore) -> tuple[dict[str, str], dict[str, Any]]:
        expected_phase = str(phase.get("phase", {}).get("id") or phase.get("phase") or "ACT")
        allowed = list(phase.get("phase", {}).get("allowed_operations", [])) or ["execute"]
        action_footer = (
            f"\nACTION_REQUIRED=Return exactly {{\"phase\":\"{expected_phase}\","
            f"\"operation\":\"{allowed[0]}\",\"gate\":\"pending\"}}."
        )
        messages = [{"role": "system", "content": LEAN_KERNEL}, {"role": "user", "content": packet + action_footer}]
        attempts = []
        suggested = allowed[0]
        normalized = False
        normalized_phase = False
        for attempt in range(self.max_repair_attempts + 1):
            started = time.perf_counter()
            try:
                response = self._post(messages, max_tokens=64, json_mode=True)
                content = self._content(response)
                value = _parse_object(content)
                error = ""
            except Exception as exc:
                response, content, value, error = {}, "", None, f"{type(exc).__name__}: {exc}"
            errors = []
            if error:
                errors.append("request_error")
            elif value is None:
                errors.append("not_json_object")
            else:
                if set(value) != {"phase", "operation", "gate"}:
                    errors.append("wrong_keys")
                if value.get("phase") != expected_phase:
                    value["phase"] = expected_phase
                    normalized_phase = True
                operation = value.get("operation")
                if operation not in allowed:
                    value["operation"] = suggested
                    normalized = True
                if not isinstance(value.get("gate"), str):
                    errors.append("invalid_gate")
            attempts.append({"attempt": attempt + 1, "errors": errors, "content": content, "elapsed_ms": int((time.perf_counter() - started) * 1000), "usage": response.get("usage", {})})
            if not errors:
                return {key: str(value[key]) for key in ("phase", "operation", "gate")}, {"attempts": attempts, "first_pass": attempt == 0, "normalized_operation": normalized, "normalized_phase": normalized_phase}
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"REPAIR. Return exactly {{\"phase\":\"{expected_phase}\",\"operation\":\"{suggested}\",\"gate\":\"pending\"}}."})
        store.append("action_failed", phase=expected_phase, failure_code="schema_repair_exhausted", attempts=attempts)
        return {"phase": expected_phase, "operation": "abstain", "gate": "schema_repair_exhausted"}, {"attempts": attempts, "first_pass": False}

    def _retrieve_evidence(self, contract: dict[str, Any], broker: ArtifactBroker) -> str | None:
        handles = contract.get("retrieval", {}).get("reference_handles", [])
        found = find_skill(str(contract.get("name")))
        if not found or not handles:
            return None
        skill_path = found[1].parent
        for handle in handles:
            path = skill_path / str(handle)
            if path.exists() and path.is_file():
                try:
                    return broker.import_file(path, kind="evidence", max_chars=9000)
                except OSError:
                    continue
        return None

    def _generate_candidate(self, task_card: dict[str, Any], projection: dict[str, Any], broker: ArtifactBroker, evidence_refs: list[str], max_tokens: int) -> str:
        evidence = [broker.read(ref, max_chars=1800) for ref in evidence_refs[-1:]]
        prompt = {
            "task": task_card,
            "phase": projection,
            "evidence": evidence,
            "instruction": "Produce only the candidate required by output_contract in at most 160 words. Do not echo this packet, the task card, or the phase schema.",
        }
        user = "GENERATION_PACKET=" + json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
        token_count = self.counter.count("Generate one candidate.\n" + user)
        if token_count > self.max_input_tokens:
            prompt["evidence"] = [evidence[0][:1500] + "...[trimmed]"] if evidence else []
            user = "GENERATION_PACKET=" + json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
            token_count = self.counter.count("Generate one candidate.\n" + user)
        if token_count > self.max_input_tokens:
            raise ContextOverflow(f"generation packet is {token_count} tokens")
        response = self._post(
            [{"role": "system", "content": "Generate one candidate under the supplied contract. No analysis or Markdown fences."}, {"role": "user", "content": user}],
            max_tokens=max(64, min(max_tokens, 2048)),
            json_mode=False,
        )
        content = self._content(response).strip()
        try:
            unwrapped = json.loads(content)
            if isinstance(unwrapped, str):
                content = unwrapped.strip()
            elif isinstance(unwrapped, dict) and len(unwrapped) == 1:
                only_value = next(iter(unwrapped.values()))
                if isinstance(only_value, str):
                    content = only_value.strip()
        except json.JSONDecodeError:
            pass
        usage = response.get("usage", {}) if isinstance(response.get("usage"), dict) else {}
        choices = response.get("choices", [])
        finish_reason = choices[0].get("finish_reason", "") if choices and isinstance(choices[0], dict) else ""
        truncated = finish_reason == "length" or int(usage.get("completion_tokens", 0) or 0) >= max(64, min(max_tokens, 2048))
        return broker.put_text("candidate", content, metadata={"input_tokens": token_count, "usage": usage, "finish_reason": finish_reason, "truncated": truncated})

    def run(self, user_message: str, *, task_id: str | None = None, conversation_history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        task_id = task_id or f"lean-{uuid.uuid4().hex[:12]}"
        store = LeanStateStore(self.state_root, task_id)
        broker = ArtifactBroker(store)
        snapshot = store.load_snapshot()
        if snapshot.get("status") in {"running", "paused"} and snapshot.get("contract_id"):
            contract = next((item for item in self.contracts if item.get("name") == snapshot.get("contract_id")), GENERAL_LEAN_CONTRACT)
            overlays = [item for item in self.contracts if item.get("name") in snapshot.get("overlays", [])]
            route_receipt = {"mode": "resume", "selected": contract.get("name"), "overlays": snapshot.get("overlays", [])}
            phase_index = int(snapshot.get("phase_index", 0))
            task_card = snapshot.get("task_card") or {"instruction": user_message}
            if user_message.strip() and user_message.strip() != str(task_card.get("instruction", "")).strip():
                task_card["follow_up"] = user_message
        else:
            contract, overlays, route_receipt = self.route(user_message)
            phase_index = 0
            task_card = {"instruction": user_message, "pinned_literals": []}
            snapshot = {
                "status": "running",
                "contract_id": contract.get("name"),
                "overlays": [item.get("name") for item in overlays],
                "phase_index": 0,
                "task_card": task_card,
                "evidence_refs": [],
                "gate_status": "unknown",
            }
            store.append("routed", contract_id=contract.get("name"), overlays=snapshot["overlays"], route=route_receipt, phase_index=0, task_card=task_card, status="running")
            store.save_snapshot(snapshot)

        phases = contract_phases(contract)
        evidence_refs = list(snapshot.get("evidence_refs", []))
        candidate_ref = snapshot.get("candidate_ref")
        api_calls = int(route_receipt.get("api_calls", 0))
        final_response = ""
        latest_result: dict[str, Any] = {}
        for index in range(phase_index, min(len(phases), phase_index + 8)):
            projection = phase_projection(contract, index, overlays)
            phase = phases[index]
            evidence_cards = [{"handle": ref, "preview": broker.read(ref, max_chars=1200)} for ref in evidence_refs[-1:]]
            state_projection = {
                "phase_index": index,
                "candidate_ref": candidate_ref,
                "verified_candidate_ref": snapshot.get("verified_candidate_ref"),
                "gate_status": snapshot.get("gate_status", "unknown"),
                "artifact_index": broker.descriptors()[-4:],
            }
            try:
                packet, budget = self.packet_builder.build(
                    task_card=task_card,
                    phase=projection,
                    state=state_projection,
                    evidence=evidence_cards,
                    latest_result=latest_result,
                    route=route_receipt if index == phase_index else {},
                )
            except ContextOverflow as exc:
                store.append("packet_rejected", phase=phase["id"], phase_index=index, failure_code="context_overflow", error=str(exc), status="failed")
                final_response = f"Lean runtime stopped: {exc}"
                snapshot.update({"status": "failed", "failure_code": "context_overflow", "phase_index": index})
                break
            action, action_receipt = self._coordinator_action(packet, projection, store)
            api_calls += len(action_receipt["attempts"])
            store.append("model_action", contract_id=contract.get("name"), phase=phase["id"], phase_index=index, action=action, budget=budget, action_receipt=action_receipt, status="running")
            if action["operation"] == "abstain":
                final_response = f"Lean runtime abstained at {phase['id']}: {action['gate']}"
                snapshot.update({"status": "abstained", "phase_index": index, "failure_code": action["gate"]})
                break

            module = str(phase.get("module", "coordinator"))
            latest_result = {"status": "passed", "module": module}
            if module == "artifact_retrieval":
                evidence_ref = self._retrieve_evidence(contract, broker)
                if evidence_ref:
                    evidence_refs.append(evidence_ref)
                    latest_result["artifact_ref"] = evidence_ref
            elif module == "model_generation":
                try:
                    candidate_ref = self._generate_candidate(task_card, projection, broker, evidence_refs, int(phase.get("max_output_tokens", 2048)))
                    api_calls += 1
                    latest_result["candidate_ref"] = candidate_ref
                    snapshot["gate_status"] = "unknown"
                    snapshot.pop("verified_candidate_ref", None)
                except Exception as exc:
                    latest_result = {"status": "failed", "failure_code": "generation_failed", "error": str(exc)}
            elif module == "verifier":
                candidate_meta = broker.metadata(candidate_ref) if candidate_ref else {}
                if not candidate_ref or not broker.read(candidate_ref, max_chars=20).strip():
                    latest_result = {"status": "failed", "failure_code": "empty_candidate"}
                elif candidate_meta.get("truncated"):
                    latest_result = {"status": "failed", "failure_code": "truncated_candidate"}
                else:
                    snapshot["gate_status"] = "pass"
                    snapshot["verified_candidate_ref"] = candidate_ref
                    latest_result["verified_candidate_ref"] = candidate_ref
            elif module == "finalizer":
                candidate_meta = broker.metadata(candidate_ref) if candidate_ref else {}
                verified_ref = snapshot.get("verified_candidate_ref")
                if (
                    candidate_ref
                    and verified_ref == candidate_ref
                    and snapshot.get("gate_status") == "pass"
                    and not candidate_meta.get("truncated")
                ):
                    final_response = broker.read(candidate_ref, max_chars=24_000).strip()
                else:
                    latest_result = {"status": "failed", "failure_code": "candidate_not_verified"}
            store.append("phase_result", contract_id=contract.get("name"), phase=phase["id"], phase_index=index, result=latest_result, candidate_ref=candidate_ref, gate_status=snapshot.get("gate_status", "unknown"), status="running")
            snapshot.update({"phase_index": index + 1, "phase": phase["id"], "candidate_ref": candidate_ref, "evidence_refs": evidence_refs, "last_result": latest_result})
            store.save_snapshot(snapshot)
            if final_response:
                snapshot["status"] = "completed"
                break

        if not final_response and snapshot.get("status") not in {"failed", "abstained"}:
            if (
                candidate_ref
                and snapshot.get("verified_candidate_ref") == candidate_ref
                and snapshot.get("gate_status") == "pass"
                and not broker.metadata(candidate_ref).get("truncated")
            ):
                final_response = broker.read(candidate_ref, max_chars=24_000).strip()
                snapshot["status"] = "completed"
            else:
                final_response = f"Lean runtime stopped: {contract.get('name')} has no verifier-passed final candidate."
                snapshot["status"] = "failed"
                snapshot["failure_code"] = "candidate_not_verified"
        store.append("run_complete", contract_id=contract.get("name"), status=snapshot.get("status"), candidate_ref=candidate_ref, api_calls=api_calls, final_chars=len(final_response))
        store.save_snapshot(snapshot)
        messages = [{"role": "user", "content": user_message}, {"role": "assistant", "content": final_response}]
        self.agent._session_messages = messages
        return {
            "final_response": final_response,
            "messages": messages,
            "api_calls": api_calls,
            "completed": snapshot.get("status") == "completed",
            "partial": snapshot.get("status") not in {"completed", "failed", "abstained"},
            "interrupted": False,
            "lean_runtime": {"task_id": task_id, "contract_id": contract.get("name"), "overlays": snapshot.get("overlays", []), "state_dir": str(store.root), "route": route_receipt},
        }
