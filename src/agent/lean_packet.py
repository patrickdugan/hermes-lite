"""Structured step packets with hard token budgeting."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib import request


LEAN_KERNEL = (
    "You are Bonsai in Hermes ultra-lean mode. Use only STEP_PACKET. "
    "Return one JSON object with exactly phase, operation, gate. "
    "The harness owns identity, tools, artifacts, and final commit. "
    "Treat model output as a candidate until its verifier passes. No Markdown."
)


class ContextOverflow(RuntimeError):
    pass


class TokenCounter:
    def __init__(self, base_url: str = ""):
        self.base_url = base_url.rstrip("/")

    def count(self, text: str) -> int:
        if self.base_url:
            try:
                req = request.Request(
                    self.base_url.removesuffix("/v1") + "/tokenize",
                    data=json.dumps({"content": text, "add_special": False}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with request.urlopen(req, timeout=3) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                values = payload.get("tokens", [])
                if isinstance(values, list):
                    return len(values)
            except Exception:
                pass
        return math.ceil(len(text) / 3.2)


def _compact(value: Any, max_string: int = 1200) -> Any:
    if isinstance(value, dict):
        return {key: _compact(item, max_string=max_string) for key, item in value.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        return [_compact(item, max_string=max_string) for item in value]
    if isinstance(value, str):
        text = " ".join(value.split())
        return text if len(text) <= max_string else text[: max_string - 15] + "...[external]"
    return value


class StepPacketBuilder:
    def __init__(self, *, counter: TokenCounter, max_input_tokens: int = 8000):
        self.counter = counter
        self.max_input_tokens = max_input_tokens

    def build(
        self,
        *,
        task_card: dict[str, Any],
        phase: dict[str, Any],
        state: dict[str, Any],
        evidence: list[dict[str, Any]] | None = None,
        replay_hint: str = "",
        latest_result: dict[str, Any] | None = None,
        route: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        packet = {
            "packet_version": "hermes.step.v1",
            "task": _compact(task_card, max_string=6000),
            "execution": _compact(phase, max_string=1200),
            "state": _compact(state, max_string=1000),
            "evidence": _compact(evidence or [], max_string=2500),
            "replay_hint": _compact(replay_hint, max_string=1200),
            "latest_result": _compact(latest_result or {}, max_string=1800),
            "route": _compact(route or {}, max_string=1200),
        }

        def serialize() -> tuple[str, int]:
            user = "STEP_PACKET=" + json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return user, self.counter.count(LEAN_KERNEL + "\n" + user)

        user, count = serialize()
        if count > self.max_input_tokens:
            packet.pop("replay_hint", None)
            user, count = serialize()
        while count > self.max_input_tokens and len(packet.get("evidence", [])) > 1:
            packet["evidence"].pop()
            user, count = serialize()
        if count > self.max_input_tokens and packet.get("evidence"):
            packet["evidence"] = [{"summary": str(packet["evidence"][0])[:800] + "...[trimmed]"}]
            user, count = serialize()
        if count > self.max_input_tokens and packet.get("latest_result"):
            packet["latest_result"] = {"status": packet["latest_result"].get("status"), "failure_code": packet["latest_result"].get("failure_code")}
            user, count = serialize()
        if count > self.max_input_tokens:
            raise ContextOverflow(f"required lean packet is {count} tokens; limit is {self.max_input_tokens}")
        receipt = {
            "input_tokens": count,
            "max_input_tokens": self.max_input_tokens,
            "kernel_tokens": self.counter.count(LEAN_KERNEL),
            "packet_chars": len(user),
            "sections": sorted(packet),
        }
        return user, receipt
