"""File-backed state and artifact handles for stateless lean-agent steps."""

from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip(".-")
    return cleaned[:96] or uuid.uuid4().hex


class LeanStateStore:
    def __init__(self, root: str | Path, task_id: str):
        self.root = Path(root).resolve() / safe_id(task_id)
        self.task_id = task_id
        self.events_path = self.root / "events.jsonl"
        self.snapshot_path = self.root / "snapshot.json"
        self.artifacts_dir = self.root / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **payload: Any) -> dict[str, Any]:
        event = {"seq": self.event_count() + 1, "ts": utc_now(), "task_id": self.task_id, "kind": kind, **payload}
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        return event

    def event_count(self) -> int:
        try:
            return sum(1 for line in self.events_path.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            return 0

    def load_snapshot(self) -> dict[str, Any]:
        try:
            value = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save_snapshot(self, snapshot: dict[str, Any]) -> None:
        snapshot = {**snapshot, "task_id": self.task_id, "updated_at": utc_now(), "event_count": self.event_count()}
        tmp = self.snapshot_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.snapshot_path)

    def rebuild_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {"task_id": self.task_id, "status": "new", "phase_index": 0}
        try:
            lines = self.events_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            lines = []
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("contract_id", "overlays", "phase", "phase_index", "status", "candidate_ref", "gate_status", "failure_code", "next_action", "task_card"):
                if key in event:
                    snapshot[key] = event[key]
        self.save_snapshot(snapshot)
        return snapshot


class ArtifactBroker:
    def __init__(self, store: LeanStateStore):
        self.store = store

    def put_text(self, kind: str, content: str, suffix: str = ".txt", metadata: dict[str, Any] | None = None) -> str:
        artifact_id = safe_id(f"{kind}-{uuid.uuid4().hex[:10]}")
        path = self.store.artifacts_dir / f"{artifact_id}{suffix}"
        path.write_text(content, encoding="utf-8")
        meta = {"artifact_id": artifact_id, "kind": kind, "path": str(path), "chars": len(content), **(metadata or {})}
        (self.store.artifacts_dir / f"{artifact_id}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self.store.append("artifact_created", artifact_ref=f"artifact://{self.store.task_id}/{artifact_id}", artifact=meta)
        return f"artifact://{self.store.task_id}/{artifact_id}"

    def put_json(self, kind: str, value: Any, metadata: dict[str, Any] | None = None) -> str:
        return self.put_text(kind, json.dumps(value, ensure_ascii=False, indent=2) + "\n", suffix=".json", metadata=metadata)

    def resolve(self, handle: str) -> Path:
        prefix = f"artifact://{self.store.task_id}/"
        if not handle.startswith(prefix):
            raise ValueError("artifact handle belongs to another task")
        artifact_id = safe_id(handle[len(prefix):])
        matches = [path for path in self.store.artifacts_dir.glob(f"{artifact_id}.*") if not path.name.endswith(".meta.json")]
        if not matches:
            raise FileNotFoundError(handle)
        return matches[0]

    def read(self, handle: str, max_chars: int = 10_000) -> str:
        path = self.resolve(handle)
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]

    def metadata(self, handle: str) -> dict[str, Any]:
        path = self.resolve(handle)
        meta_path = path.with_name(path.stem + ".meta.json")
        try:
            value = json.loads(meta_path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def import_file(self, path: str | Path, kind: str = "evidence", max_chars: int = 10_000) -> str:
        source = Path(path).resolve()
        content = source.read_text(encoding="utf-8", errors="replace")[:max_chars]
        return self.put_text(kind, content, suffix=source.suffix or ".txt", metadata={"source": str(source)})

    def descriptors(self) -> list[dict[str, Any]]:
        result = []
        for path in sorted(self.store.artifacts_dir.glob("*.meta.json")):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            result.append({key: value.get(key) for key in ("artifact_id", "kind", "chars", "source") if value.get(key) is not None})
        return result
