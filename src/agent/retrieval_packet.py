"""Compact TRM/LDT retrieval packets for small-context local agents.

This module turns local run artifacts into a short, deterministic prompt
packet. It is intentionally conservative: no raw transcripts, at most one
replay hint by default, and a hard character budget derived from token budget.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_RESOURCE_DIRS = [".hermes/trm", ".hermes/ldt", ".hermes"]
SMALL_LOCAL_MODEL_IDS = {"digitsflow/bonsai-8b"}


def _truthy(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _read_jsonl_tail(path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

    rows: List[Dict[str, Any]] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _compact(value: Any, *, max_chars: int = 900) -> Any:
    if isinstance(value, dict):
        result = {}
        for key in sorted(value.keys()):
            if key.startswith("_"):
                continue
            item = _compact(value[key], max_chars=max_chars)
            if item not in (None, "", [], {}):
                result[key] = item
        return result
    if isinstance(value, list):
        return [_compact(item, max_chars=max_chars) for item in value[:8]]
    if isinstance(value, str):
        text = " ".join(value.split())
        if len(text) > max_chars:
            return text[: max_chars - 32] + "...[truncated]"
        return text
    return value


def _first_present(resources: Iterable[Path], names: Iterable[str]) -> tuple[Optional[Path], Optional[Dict[str, Any]]]:
    for root in resources:
        for name in names:
            path = root / name
            if path.exists():
                data = _read_json(path)
                if data is not None:
                    return path, data
    return None, None


def _jsonl_candidates(resources: Iterable[Path], names: Iterable[str]) -> List[tuple[Path, Dict[str, Any]]]:
    candidates: List[tuple[Path, Dict[str, Any]]] = []
    for root in resources:
        for name in names:
            path = root / name
            if path.exists():
                candidates.extend((path, row) for row in _read_jsonl_tail(path))
    return candidates


class RetrievalPacketBuilder:
    """Builds a compact prompt packet from local TRM/LDT artifacts."""

    def __init__(
        self,
        *,
        cwd: str | Path | None = None,
        enabled: bool = False,
        budget_tokens: int = 1200,
        hard_context_tokens: int = 12000,
        max_replay_hints: int = 1,
        resource_dirs: Optional[List[str]] = None,
    ):
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.enabled = enabled
        self.budget_tokens = max(200, int(budget_tokens))
        self.hard_context_tokens = max(1000, int(hard_context_tokens))
        self.max_replay_hints = max(0, int(max_replay_hints))
        self.resource_dirs = resource_dirs or list(DEFAULT_RESOURCE_DIRS)
        self.active_skill_contract: Optional[Dict[str, Any]] = None

    def set_skill_contract(self, contract: Optional[Dict[str, Any]]) -> None:
        """Activate one session-local lean skill contract."""
        if isinstance(contract, dict) and contract.get("schema") in {"hermes.ultra_lean_skill.v1", "hermes.ultra_lean_skill.v2"}:
            self.active_skill_contract = _compact(contract, max_chars=420)
        else:
            self.active_skill_contract = None

    @classmethod
    def from_config(cls, config: Dict[str, Any], *, model: str, context_length: int, cwd: str | Path | None = None):
        retrieval = config.get("retrieval", {}) if isinstance(config, dict) else {}
        if not isinstance(retrieval, dict):
            retrieval = {}

        env_enabled = os.getenv("HERMES_RETRIEVAL_ENABLED", "").strip()
        auto_small_local = bool(retrieval.get("auto_for_small_local_models", True))
        enabled = bool(retrieval.get("enabled", False))
        is_small_local_model = model.startswith("local/") or model in SMALL_LOCAL_MODEL_IDS
        hard_context = int(retrieval.get("hard_context_tokens", 12000))
        implementation_ceiling = hard_context + max(512, int(hard_context * 0.05))
        if env_enabled:
            enabled = _truthy(env_enabled)
        elif auto_small_local and is_small_local_model and context_length <= implementation_ceiling:
            enabled = True

        return cls(
            cwd=cwd,
            enabled=enabled,
            budget_tokens=int(os.getenv("HERMES_RETRIEVAL_BUDGET_TOKENS", retrieval.get("budget_tokens", 1200))),
            hard_context_tokens=int(retrieval.get("hard_context_tokens", 12000)),
            max_replay_hints=int(retrieval.get("max_replay_hints", 1)),
            resource_dirs=list(retrieval.get("resource_dirs", DEFAULT_RESOURCE_DIRS)),
        )

    def _resource_roots(self) -> List[Path]:
        roots = []
        for entry in self.resource_dirs:
            path = Path(entry)
            if not path.is_absolute():
                path = self.cwd / path
            if path.exists() and path.is_dir():
                roots.append(path.resolve())
        return roots

    def build(self, *, user_message: str = "") -> str:
        if not self.enabled:
            return ""

        roots = self._resource_roots()
        if not roots and not self.active_skill_contract:
            return ""
        task_path, task = _first_present(
            roots,
            ["task_card.json", "current_task.json", "curriculum_state.json", "run_manifest.json"],
        )
        _, self_model = _first_present(roots, ["self_model.json", "agent_self_model.json"])

        replay_rows = _jsonl_candidates(
            roots,
            ["replay_candidates.jsonl", "task_attempts.jsonl", "ldt.jsonl", "decision_traces.jsonl"],
        )
        score_rows = _jsonl_candidates(roots, ["scores.jsonl", "score_history.jsonl"])

        task_id = self._task_id(task)
        replay_hint = self._select_replay_hint(replay_rows, task_id=task_id, user_message=user_message)
        failure_note = self._select_failure_note(score_rows + replay_rows, task_id=task_id)

        packet: Dict[str, Any] = {
            "skill": self.active_skill_contract or {},
            "task": _compact(task or {}, max_chars=700),
            "self_model": _compact(self_model or {}, max_chars=500),
            "replay_hints": [replay_hint] if replay_hint else [],
            "failure_note": failure_note,
            "budget": {
                "target_tokens": self.budget_tokens,
                "hard_context_tokens": self.hard_context_tokens,
                "rule": "Use this packet for the next action; fetch raw artifacts only if necessary.",
            },
            "sources": [str(p.relative_to(self.cwd)) if p.is_relative_to(self.cwd) else str(p) for p in roots],
        }
        if task_path:
            try:
                packet["task_source"] = str(task_path.relative_to(self.cwd))
            except ValueError:
                packet["task_source"] = str(task_path)

        packet = {k: v for k, v in packet.items() if v not in (None, "", [], {})}
        if not packet:
            return ""

        text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        max_chars = self.budget_tokens * 4
        if len(text) > max_chars:
            packet.pop("sources", None)
            packet.pop("task_source", None)
            text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(text) > max_chars:
            packet.pop("self_model", None)
            text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(text) > max_chars:
            packet.pop("replay_hints", None)
            packet.pop("failure_note", None)
            text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(text) > max_chars:
            text = text[: max_chars - 32] + "...[packet truncated]"

        return "# TRM/LDT Retrieval Packet\n" + text

    def _task_id(self, task: Optional[Dict[str, Any]]) -> str:
        if not task:
            return ""
        for key in ("task_id", "id", "current_task_id"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        current = task.get("current_task")
        if isinstance(current, dict):
            value = current.get("task_id") or current.get("id")
            if isinstance(value, str):
                return value.strip()
        return ""

    def _select_replay_hint(
        self,
        rows: List[tuple[Path, Dict[str, Any]]],
        *,
        task_id: str,
        user_message: str,
    ) -> Optional[str]:
        if self.max_replay_hints <= 0:
            return None
        if not rows:
            return None

        def score(item: tuple[Path, Dict[str, Any]]) -> tuple[int, int]:
            _, row = item
            row_task = str(row.get("task_id") or row.get("id") or "")
            failed = row.get("passed") is False or row.get("score") == 0 or bool(row.get("failure") or row.get("error"))
            same_task = bool(task_id and row_task == task_id)
            mentions = bool(user_message and row_task and row_task.lower() in user_message.lower())
            return (3 if same_task else 0) + (2 if failed else 0) + (1 if mentions else 0), len(str(row))

        _, row = max(rows, key=score)
        fields = []
        for key in ("task_id", "output", "action", "score", "passed", "failure", "error", "rationale"):
            if key in row and row[key] not in (None, "", [], {}):
                fields.append(f"{key}={_compact(row[key], max_chars=160)}")
        return "; ".join(fields[:7]) if fields else json.dumps(_compact(row, max_chars=160), ensure_ascii=False)

    def _select_failure_note(self, rows: List[tuple[Path, Dict[str, Any]]], *, task_id: str) -> str:
        failures = []
        for _, row in rows:
            row_task = str(row.get("task_id") or row.get("id") or "")
            is_failure = row.get("passed") is False or row.get("score") == 0 or bool(row.get("failure") or row.get("error"))
            if is_failure and (not task_id or not row_task or row_task == task_id):
                failures.append(row)
        if not failures:
            return ""
        row = failures[-1]
        reason = row.get("failure") or row.get("error") or row.get("failure_mode") or "recent failed attempt"
        return str(_compact(reason, max_chars=240))
