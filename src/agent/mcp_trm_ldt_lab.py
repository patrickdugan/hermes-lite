"""Deterministic MCP TRM/LDT experiment harness.

This module evaluates compact retrieval packets without calling a live model.
It is the eval-first layer before any RTX 3050 training or adapter work.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib import request
from urllib.error import HTTPError, URLError

from agent.model_metadata import estimate_tokens_rough
from agent.retrieval_packet import RetrievalPacketBuilder


DEFAULT_ABORT_CONDITIONS = [
    "ram_cap_exceeded",
    "gpu_oom",
    "sustained_io_over_cap",
    "nan_or_exploding_loss",
    "no_eval_progress",
    "timeout",
    "checkpoint_missing",
]

DEFAULT_PROMOTION_GATES = [
    "schema_gate",
    "tool_gate",
    "eval_gate",
    "packet_gate",
    "training_gate",
    "no_baseline_regression",
]

DEFAULT_LOCAL_BONSAI_BASE_URL = "http://127.0.0.1:8801/v1"
PACKET_VARIANTS = ["baseline", "trm", "ldt", "hybrid", "hybrid_reranked"]
DEFAULT_POLICY_WEIGHTS = {
    "same_task": 5.0,
    "same_bench_family": 2.0,
    "recent_failure": 3.0,
    "borderline_score": 1.5,
    "forbidden_term_failure": 2.0,
    "packet_too_large": 2.0,
    "live_model_wrong_tool": 2.5,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
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


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json_dumps(row) + "\n")


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def create_run_manifest(
    *,
    mcp: str,
    task_id: str,
    out: Optional[Path] = None,
    budget_tokens: int = 1200,
    max_runtime_minutes: int = 60,
    checkpoint_interval: str = "500 steps or 15 minutes",
    ram_mb: int = 2048,
    cpu_pct: int = 50,
    io_mb_s: int = 50,
    chunk_strategy: str = "stream JSONL in bounded batches",
) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"mcp-trm-ldt-{task_id}-{now}".replace("/", "-").replace("\\", "-")
    run_dir = Path("experiments") / run_id
    manifest_path = out or run_dir / "run_manifest.json"
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "mcp": mcp,
        "task_id": task_id,
        "packet_budget_tokens": budget_tokens,
        "caps": {
            "ram_mb": ram_mb,
            "cpu_pct": cpu_pct,
            "io_mb_s": io_mb_s,
            "gpu_processes": 1,
        },
        "runtime": {
            "max_runtime_minutes": max_runtime_minutes,
            "checkpoint_interval": checkpoint_interval,
            "chunk_strategy": chunk_strategy,
        },
        "artifacts": {
            "manifest": str(manifest_path),
            "run_dir": str(run_dir),
            "events": str(run_dir / "events.jsonl"),
            "summary": str(run_dir / "summary.json"),
            "checkpoints": str(run_dir / "checkpoints"),
            "current_task": ".hermes/trm/current_task.json",
            "self_model": ".hermes/trm/self_model.json",
            "replay_candidates": ".hermes/trm/replay_candidates.jsonl",
            "scores": ".hermes/trm/scores.jsonl",
            "decision_traces": ".hermes/ldt/decision_traces.jsonl",
        },
        "abort_conditions": list(DEFAULT_ABORT_CONDITIONS),
        "promotion_gates": list(DEFAULT_PROMOTION_GATES),
        "owned_pids": [],
        "status": "planned",
    }
    return manifest


def gpu_preflight(*, min_free_mb: int = 1024, max_temp_c: int = 86) -> Dict[str, Any]:
    query = "name,memory.total,memory.free,temperature.gpu,utilization.gpu,driver_version"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "passed": False,
            "error": str(exc),
            "min_free_mb": min_free_mb,
            "max_temp_c": max_temp_c,
        }

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        return {"available": True, "passed": False, "error": "unexpected nvidia-smi output", "raw": line}

    name, total, free, temp, util, driver = parts[:6]
    total_mb = _safe_int(total)
    free_mb = _safe_int(free)
    temp_c = _safe_int(temp)
    util_pct = _safe_int(util)
    passed = free_mb >= min_free_mb and temp_c <= max_temp_c
    return {
        "available": True,
        "passed": passed,
        "name": name,
        "driver_version": driver,
        "memory_total_mb": total_mb,
        "memory_free_mb": free_mb,
        "temperature_c": temp_c,
        "utilization_pct": util_pct,
        "min_free_mb": min_free_mb,
        "max_temp_c": max_temp_c,
    }


def gpu_pressure_snapshot(*, min_free_mb: int = 512, max_temp_c: int = 86) -> Dict[str, Any]:
    """Return a lightweight GPU pressure snapshot suitable for eval logs."""
    query = "name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": False,
            "passed": False,
            "error": str(exc),
            "min_free_mb": min_free_mb,
            "max_temp_c": max_temp_c,
        }

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 6:
        return {
            "available": True,
            "passed": False,
            "error": "unexpected nvidia-smi output",
            "raw": line,
            "min_free_mb": min_free_mb,
            "max_temp_c": max_temp_c,
        }

    name, total, used, free, temp, util = parts[:6]
    total_mb = _safe_int(total)
    used_mb = _safe_int(used)
    free_mb = _safe_int(free)
    temp_c = _safe_int(temp)
    util_pct = _safe_int(util)
    passed = free_mb >= min_free_mb and temp_c <= max_temp_c
    return {
        "available": True,
        "passed": passed,
        "name": name,
        "memory_total_mb": total_mb,
        "memory_used_mb": used_mb,
        "memory_free_mb": free_mb,
        "memory_used_pct": round((used_mb / total_mb) * 100, 2) if total_mb else 0.0,
        "temperature_c": temp_c,
        "utilization_pct": util_pct,
        "min_free_mb": min_free_mb,
        "max_temp_c": max_temp_c,
    }


def _safe_int(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _case_task(case: Dict[str, Any], mcp: str) -> Dict[str, Any]:
    task = case.get("task")
    if isinstance(task, dict):
        result = dict(task)
    else:
        result = {}
    for key in ("task_id", "instruction", "difficulty", "allowed_tools"):
        if key in case and key not in result:
            result[key] = case[key]
    if "task_id" not in result:
        result["task_id"] = str(case.get("id") or case.get("name") or "case")
    result.setdefault("mcp", mcp)
    return result


def _write_case_artifacts(case_root: Path, case: Dict[str, Any], *, mcp: str) -> Dict[str, Any]:
    trm = case_root / ".hermes" / "trm"
    ldt = case_root / ".hermes" / "ldt"
    task = _case_task(case, mcp)
    write_json(trm / "current_task.json", task)

    self_model = case.get("self_model")
    if isinstance(self_model, dict):
        write_json(trm / "self_model.json", self_model)

    replay = case.get("replay_candidates")
    if isinstance(replay, list):
        append_jsonl(trm / "replay_candidates.jsonl", [row for row in replay if isinstance(row, dict)])

    scores = case.get("scores")
    if isinstance(scores, list):
        append_jsonl(trm / "scores.jsonl", [row for row in scores if isinstance(row, dict)])

    traces = case.get("decision_traces")
    if isinstance(traces, list):
        append_jsonl(ldt / "decision_traces.jsonl", [row for row in traces if isinstance(row, dict)])

    return task


def _extract_packet_json(packet: str) -> Dict[str, Any]:
    if "\n" not in packet:
        return {}
    _, _, body = packet.partition("\n")
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _packet_text(packet_data: Dict[str, Any], *, budget_tokens: int) -> str:
    packet = {k: v for k, v in packet_data.items() if v not in (None, "", [], {})}
    text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    max_chars = max(200, int(budget_tokens) * 4)
    if len(text) > max_chars:
        packet.pop("sources", None)
        packet.pop("task_source", None)
        text = json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if len(text) > max_chars:
        text = text[: max_chars - 32] + "...[packet truncated]"
    return "# TRM/LDT Retrieval Packet\n" + text


def _default_budget(budget_tokens: int, hard_context_tokens: int) -> Dict[str, Any]:
    return {
        "target_tokens": budget_tokens,
        "hard_context_tokens": hard_context_tokens,
        "rule": "Use this packet for the next action; fetch raw artifacts only if necessary.",
    }


def _load_policy(policy_path: Optional[Path]) -> Dict[str, Any]:
    if not policy_path or not policy_path.exists():
        return {"weights": dict(DEFAULT_POLICY_WEIGHTS)}
    try:
        data = json.loads(policy_path.read_text(encoding="utf-8"))
    except Exception:
        return {"weights": dict(DEFAULT_POLICY_WEIGHTS)}
    if not isinstance(data, dict):
        return {"weights": dict(DEFAULT_POLICY_WEIGHTS)}
    weights = dict(DEFAULT_POLICY_WEIGHTS)
    raw_weights = data.get("weights")
    if isinstance(raw_weights, dict):
        for key, value in raw_weights.items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                pass
    data["weights"] = weights
    return data


def _policy_replay_hint(case: Dict[str, Any], policy: Dict[str, Any]) -> str:
    replay = case.get("replay_candidates")
    if not isinstance(replay, list):
        return ""
    rows = [row for row in replay if isinstance(row, dict)]
    if not rows:
        return ""

    weights = policy.get("weights") if isinstance(policy.get("weights"), dict) else DEFAULT_POLICY_WEIGHTS
    task_id = str(case.get("task_id") or case.get("id") or "")
    bench_family = str(case.get("bench_family") or "")

    def row_score(index_and_row: tuple[int, Dict[str, Any]]) -> tuple[float, int]:
        index, row = index_and_row
        score = 0.0
        if task_id and str(row.get("task_id") or row.get("id") or "") == task_id:
            score += float(weights.get("same_task", 0))
        if bench_family and str(row.get("bench_family") or "") == bench_family:
            score += float(weights.get("same_bench_family", 0))
        if row.get("passed") is False or row.get("score") == 0 or row.get("failure") or row.get("error"):
            score += float(weights.get("recent_failure", 0))
        try:
            numeric = float(row.get("score"))
            if 0 < numeric < 0.8:
                score += float(weights.get("borderline_score", 0))
        except (TypeError, ValueError):
            pass
        failure = str(row.get("failure") or row.get("error") or "")
        if "forbidden" in failure or "over_context" in failure:
            score += float(weights.get("forbidden_term_failure", 0))
        if "packet" in failure or "budget" in failure:
            score += float(weights.get("packet_too_large", 0))
        if "wrong_tool" in failure or "tool" in failure:
            score += float(weights.get("live_model_wrong_tool", 0))
        return score, index

    _, row = max(enumerate(rows), key=row_score)
    fields = []
    for key in ("task_id", "output", "action", "bench_family", "score", "passed", "failure", "error", "rationale"):
        if key in row and row[key] not in (None, "", [], {}):
            value = str(row[key])
            fields.append(f"{key}={value[:180]}")
    return "; ".join(fields[:8])


def build_variant_packet(
    *,
    case_root: Path,
    case: Dict[str, Any],
    task: Dict[str, Any],
    variant: str,
    budget_tokens: int,
    hard_context_tokens: int,
    max_replay_hints: int = 1,
    policy_path: Optional[Path] = None,
) -> str:
    if variant not in PACKET_VARIANTS:
        raise ValueError(f"unknown packet variant: {variant}")

    user_message = str(case.get("user_message") or case.get("instruction") or task.get("instruction") or "")
    builder = RetrievalPacketBuilder(
        cwd=case_root,
        enabled=True,
        budget_tokens=budget_tokens,
        hard_context_tokens=hard_context_tokens,
        max_replay_hints=max_replay_hints,
    )
    full_packet = builder.build(user_message=user_message)
    full_data = _extract_packet_json(full_packet)
    base = {
        "task": full_data.get("task") or task,
        "budget": full_data.get("budget") or _default_budget(budget_tokens, hard_context_tokens),
    }

    if variant == "baseline":
        data = base
    elif variant == "trm":
        data = {
            **base,
            "self_model": full_data.get("self_model"),
            "failure_note": full_data.get("failure_note"),
        }
    elif variant == "ldt":
        data = {
            **base,
            "replay_hints": full_data.get("replay_hints"),
            "failure_note": full_data.get("failure_note"),
        }
    elif variant == "hybrid_reranked":
        data = dict(full_data)
        hint = _policy_replay_hint(case, _load_policy(policy_path))
        if hint:
            data["replay_hints"] = [hint]
        data.setdefault("task", base["task"])
        data.setdefault("budget", base["budget"])
        data["retrieval_policy"] = str(policy_path) if policy_path else "default"
    else:
        data = full_data or base

    return _packet_text(data, budget_tokens=budget_tokens)


def _expected_terms(expected: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    required = expected.get("required_terms", [])
    forbidden = expected.get("forbidden_terms", [])
    if isinstance(required, str):
        required = [required]
    if isinstance(forbidden, str):
        forbidden = [forbidden]
    return [str(v) for v in required], [str(v) for v in forbidden]


def _subset_matches(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        for key, value in expected.items():
            if key not in actual or not _subset_matches(value, actual[key]):
                return False
        return True
    if isinstance(expected, list):
        return expected == actual
    return expected == actual


def score_case(
    case: Dict[str, Any],
    *,
    packet: str,
    packet_data: Dict[str, Any],
    packet_tokens_est: int,
    budget_tokens: int,
) -> Dict[str, Any]:
    task = case.get("_task", {})
    expected = case.get("expected", {})
    actual = case.get("actual")
    if not isinstance(expected, dict):
        expected = {}

    gates: Dict[str, Dict[str, Any]] = {}
    gates["schema_gate"] = {
        "passed": bool(task.get("task_id") and task.get("instruction") and expected),
        "required": True,
    }
    gates["packet_gate"] = {
        "passed": bool(packet_data) and packet_tokens_est <= budget_tokens and str(task.get("task_id", "")) in packet,
        "required": True,
    }

    required_terms, forbidden_terms = _expected_terms(expected)
    term_passed = all(term in packet for term in required_terms) and all(term not in packet for term in forbidden_terms)
    gates["eval_gate"] = {
        "passed": term_passed,
        "required": bool(required_terms or forbidden_terms),
    }

    if isinstance(actual, dict):
        expected_call = {key: expected[key] for key in ("tool", "resource", "args") if key in expected}
        gates["tool_gate"] = {
            "passed": _subset_matches(expected_call, actual) if expected_call else True,
            "required": bool(expected_call),
        }
    else:
        gates["tool_gate"] = {"passed": True, "required": False}

    required_gates = [gate for gate in gates.values() if gate["required"]]
    if not required_gates:
        required_gates = list(gates.values())
    passed = all(gate["passed"] for gate in required_gates)
    score = sum(1 for gate in required_gates if gate["passed"]) / max(1, len(required_gates))
    failure = ""
    if not passed:
        failure = next(name for name, gate in gates.items() if gate["required"] and not gate["passed"])

    return {
        "passed": passed,
        "score": score,
        "failure": failure,
        "gates": gates,
    }


def _post_json(url: str, payload: Dict[str, Any], *, timeout_s: int) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8", errors="replace")
    value = json.loads(raw)
    return value if isinstance(value, dict) else {"raw": value}


def run_live_smoke(
    *,
    base_url: str = DEFAULT_LOCAL_BONSAI_BASE_URL,
    model: str = "local/bonsai-8b",
    prompt: str = "Reply with exactly: ok",
    max_tokens: int = 8,
    timeout_s: int = 90,
    min_free_mb: int = 512,
    max_temp_c: int = 86,
) -> Dict[str, Any]:
    """Run one bounded local-model smoke request and record pressure/timing."""
    url = base_url.rstrip("/") + "/chat/completions"
    pressure_before = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=max_temp_c)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(1, min(int(max_tokens), 64)),
        "temperature": 0,
    }
    start = time.perf_counter()
    try:
        response = _post_json(url, payload, timeout_s=timeout_s)
        error = ""
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        response = {}
        error = str(exc)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    pressure_after = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=max_temp_c)

    content = ""
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        if isinstance(message, dict):
            content = str(message.get("content") or "")

    timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}
    passed = not error and bool(content.strip()) and bool(pressure_after.get("passed", False))
    return {
        "created_at": utc_now(),
        "base_url": base_url,
        "model": model,
        "passed": passed,
        "error": error,
        "content": content,
        "elapsed_ms": elapsed_ms,
        "usage": response.get("usage", {}),
        "timings": timings,
        "pressure_before": pressure_before,
        "pressure_after": pressure_after,
        "limits": {
            "max_tokens": max_tokens,
            "timeout_s": timeout_s,
            "min_free_mb": min_free_mb,
            "max_temp_c": max_temp_c,
        },
    }


def run_live_case_probe(
    *,
    case: Dict[str, Any],
    packet: str,
    base_url: str = DEFAULT_LOCAL_BONSAI_BASE_URL,
    model: str = "local/bonsai-8b",
    max_tokens: int = 40,
    timeout_s: int = 90,
    min_free_mb: int = 512,
    max_temp_c: int = 86,
) -> Dict[str, Any]:
    """Run one bounded live probe for a promoted packet/case pair."""
    expected = case.get("expected", {})
    if not isinstance(expected, dict):
        expected = {}
    required_terms, forbidden_terms = _expected_terms(expected)
    instruction = str(case.get("instruction") or case.get("user_message") or "")
    task_id = str(case.get("task_id") or case.get("id") or "")
    prompt = (
        "Choose the next MCP action for this task.\n"
        "Return one plain-text line only, no JSON and no prose.\n"
        f"Start the line with task_id={task_id}.\n"
        "Then copy the exact MCP action/resource terms from the packet or replay hint that match the task.\n"
        "Do not invent alternate tool names or broad retrieval steps.\n\n"
        f"{packet}\n\nTask: {instruction}"
    )
    url = base_url.rstrip("/") + "/chat/completions"
    pressure_before = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=max_temp_c)
    start = time.perf_counter()
    try:
        response = _post_json(
            url,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max(1, min(int(max_tokens), 128)),
                "temperature": 0,
                "stop": ["\n", "\r\n"],
            },
            timeout_s=timeout_s,
        )
        error = ""
    except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        response = {}
        error = str(exc)
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    pressure_after = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=max_temp_c)

    content = ""
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else {}
        if isinstance(message, dict):
            content = str(message.get("content") or "")

    required_ok = all(term in content for term in required_terms)
    forbidden_ok = all(term not in content for term in forbidden_terms)
    passed = not error and required_ok and forbidden_ok and bool(pressure_after.get("passed", False))
    return {
        "ts": utc_now(),
        "event": "live_case_probe",
        "case_id": str(case.get("task_id") or case.get("id") or ""),
        "bench_family": case.get("bench_family"),
        "passed": passed,
        "error": error,
        "content": content,
        "required_terms": required_terms,
        "forbidden_terms": forbidden_terms,
        "elapsed_ms": elapsed_ms,
        "usage": response.get("usage", {}),
        "timings": response.get("timings", {}),
        "pressure_before": pressure_before,
        "pressure_after": pressure_after,
    }


def train_retrieval_policy(events_paths: List[Path], *, out: Optional[Path] = None) -> Dict[str, Any]:
    """Create a deterministic lightweight replay-selection policy from event logs."""
    weights = dict(DEFAULT_POLICY_WEIGHTS)
    failure_counts: Dict[str, int] = {}
    event_count = 0
    live_failures = 0

    for path in events_paths:
        for event in read_jsonl(path):
            event_count += 1
            failure = str(event.get("failure") or "")
            if failure:
                failure_counts[failure] = failure_counts.get(failure, 0) + 1
            if event.get("event") == "live_case_probe" and not event.get("passed", False):
                live_failures += 1
                weights["live_model_wrong_tool"] += 0.25
            gates = event.get("gates") if isinstance(event.get("gates"), dict) else {}
            if gates.get("eval_gate", {}).get("passed") is False:
                weights["same_bench_family"] += 0.05
                weights["recent_failure"] += 0.05
            if gates.get("packet_gate", {}).get("passed") is False:
                weights["packet_too_large"] += 0.10
            if failure in {"eval_gate", "tool_gate"}:
                weights["live_model_wrong_tool"] += 0.10
            if failure == "packet_gate":
                weights["packet_too_large"] += 0.15

    policy = {
        "created_at": utc_now(),
        "version": 1,
        "training_type": "deterministic_event_weight_update",
        "events": [str(path) for path in events_paths],
        "event_count": event_count,
        "failure_counts": failure_counts,
        "live_failures": live_failures,
        "weights": {key: round(value, 4) for key, value in sorted(weights.items())},
        "selection_rule": "Score replay rows with fixed weighted features; choose the highest score, latest row wins ties.",
    }
    if out:
        write_json(out, policy)
    return policy


def _write_leaderboard(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "block",
        "variant",
        "pass_rate",
        "passed_cases",
        "total_cases",
        "avg_packet_tokens_est",
        "max_packet_tokens_est",
        "max_packet_context_ratio",
        "pressure_failures",
        "promotion_ready",
        "live_pass_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_marathon_report(path: Path, summary: Dict[str, Any], leaderboard: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Storyworld MCP Marathon Report: {summary['run_id']}",
        "",
        f"- Status: {summary['status']}",
        f"- Abort reason: {summary.get('abort_reason') or 'none'}",
        f"- Best variant: {summary.get('best_variant') or 'none'}",
        f"- Duration ms: {summary.get('duration_ms', 0)}",
        f"- Blocks completed: {summary.get('blocks_completed', 0)}",
        f"- Live failures: {summary.get('live_failures', 0)}",
        f"- Pressure failures: {summary.get('pressure_failures', 0)}",
        "",
        "## Leaderboard",
        "",
        "| block | variant | pass_rate | max_tokens | pressure_failures | promotion_ready | live_pass_rate |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in leaderboard:
        lines.append(
            f"| {row.get('block')} | {row.get('variant')} | {row.get('pass_rate', 0):.3f} | "
            f"{row.get('max_packet_tokens_est', 0)} | {row.get('pressure_failures', 0)} | "
            f"{row.get('promotion_ready')} | {row.get('live_pass_rate', '')} |"
        )
    lines.extend([
        "",
        "## Recommendation",
        "",
        summary.get("recommendation", "Run another short pilot before increasing duration."),
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _hard_pressure_failed(pressure: Dict[str, Any], *, hard_stop_free_mb: int, hard_stop_temp_c: int) -> str:
    if not pressure.get("available", False):
        return str(pressure.get("error") or "gpu_pressure_unavailable")
    if int(pressure.get("memory_free_mb", 0)) < hard_stop_free_mb:
        return "hard_stop_low_vram"
    if int(pressure.get("temperature_c", 0)) > hard_stop_temp_c:
        return "hard_stop_temperature"
    return ""


def run_marathon(
    *,
    suite: str = "storyworld",
    cases_path: Optional[Path] = None,
    output_dir: Path = Path("experiments") / "marathons",
    run_id: Optional[str] = None,
    duration_minutes: int = 240,
    block_minutes: int = 20,
    cooldown_minutes: int = 5,
    base_url: str = DEFAULT_LOCAL_BONSAI_BASE_URL,
    model: str = "local/bonsai-8b",
    mcp: str = "storyworld-encounter",
    variants: Optional[List[str]] = None,
    budget_tokens: int = 1200,
    hard_context_tokens: int = 12000,
    min_free_mb: int = 512,
    hard_stop_free_mb: int = 256,
    warn_temp_c: int = 87,
    hard_stop_temp_c: int = 87,
    max_consecutive_pressure_failures: int = 3,
    max_live_failures_per_block: int = 2,
    max_live_cases: int = 3,
    live_probe_max_tokens: int = 40,
    live_probe_timeout_s: int = 90,
    dry_run: bool = False,
) -> Dict[str, Any]:
    if suite != "storyworld":
        raise ValueError("only the storyworld suite is implemented")
    variants = variants or list(PACKET_VARIANTS)
    for variant in variants:
        if variant not in PACKET_VARIANTS:
            raise ValueError(f"unknown packet variant: {variant}")

    cases_path = cases_path or Path("evals") / "storyworld_mcp_cases.example.jsonl"
    if run_id is None:
        run_id = f"storyworld-marathon-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = output_dir / run_id
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "run_manifest.json"
    leaderboard_path = run_dir / "leaderboard.csv"
    policy_path = run_dir / "retrieval_policy.json"
    report_path = run_dir / "report.md"

    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "suite": suite,
        "mcp": mcp,
        "cases_path": str(cases_path),
        "base_url": base_url,
        "model": model,
        "variants": variants,
        "dry_run": dry_run,
        "recursion_caps": {"max_cycles": 2, "max_nested_depth": 1},
        "budgets": {
            "duration_minutes": duration_minutes,
            "block_minutes": block_minutes,
            "cooldown_minutes": cooldown_minutes,
            "packet_tokens": budget_tokens,
            "hard_context_tokens": hard_context_tokens,
            "max_live_cases": max_live_cases,
            "live_probe_max_tokens": live_probe_max_tokens,
            "live_probe_timeout_s": live_probe_timeout_s,
        },
        "pressure_limits": {
            "min_free_mb": min_free_mb,
            "hard_stop_free_mb": hard_stop_free_mb,
            "warn_temp_c": warn_temp_c,
            "hard_stop_temp_c": hard_stop_temp_c,
            "max_consecutive_pressure_failures": max_consecutive_pressure_failures,
            "max_live_failures_per_block": max_live_failures_per_block,
        },
        "artifacts": {
            "events": str(events_path),
            "summary": str(summary_path),
            "leaderboard": str(leaderboard_path),
            "retrieval_policy": str(policy_path),
            "report": str(report_path),
        },
    }
    write_json(manifest_path, manifest)

    start = time.perf_counter()
    block_count = 1 if dry_run else max(1, math.ceil(max(1, duration_minutes) / max(1, block_minutes)))
    block_limit = min(block_count, 1) if dry_run else block_count
    leaderboard: List[Dict[str, Any]] = []
    event_paths: List[Path] = []
    abort_reason = ""
    consecutive_pressure_failures = 0
    total_live_failures = 0
    cases = read_jsonl(cases_path)

    append_jsonl(events_path, [{
        "ts": utc_now(),
        "event": "marathon_start",
        "run_id": run_id,
        "manifest": str(manifest_path),
        "dry_run": dry_run,
    }])

    for block in range(block_limit):
        pressure = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=warn_temp_c)
        hard_failure = _hard_pressure_failed(
            pressure,
            hard_stop_free_mb=hard_stop_free_mb,
            hard_stop_temp_c=hard_stop_temp_c,
        )
        if hard_failure and not dry_run:
            abort_reason = hard_failure
            append_jsonl(events_path, [{"ts": utc_now(), "event": "abort", "run_id": run_id, "reason": abort_reason, "pressure": pressure}])
            break
        if not pressure.get("passed", True):
            consecutive_pressure_failures += 1
        else:
            consecutive_pressure_failures = 0
        if consecutive_pressure_failures >= max_consecutive_pressure_failures and not dry_run:
            abort_reason = "consecutive_pressure_failures"
            append_jsonl(events_path, [{"ts": utc_now(), "event": "abort", "run_id": run_id, "reason": abort_reason, "pressure": pressure}])
            break

        smoke = {"passed": True, "dry_run": True}
        if not dry_run:
            smoke = run_live_smoke(
                base_url=base_url,
                model=model,
                min_free_mb=hard_stop_free_mb,
                max_temp_c=hard_stop_temp_c,
            )
            if not smoke.get("passed", False):
                abort_reason = "live_smoke_failed"
                append_jsonl(events_path, [{"ts": utc_now(), "event": "abort", "run_id": run_id, "reason": abort_reason, "smoke": smoke}])
                break

        block_events = [{"ts": utc_now(), "event": "block_start", "run_id": run_id, "block": block, "pressure": pressure, "smoke": smoke}]
        append_jsonl(events_path, block_events)

        block_summaries: Dict[str, Dict[str, Any]] = {}
        non_reranked = [variant for variant in variants if variant != "hybrid_reranked"]
        for variant in non_reranked:
            eval_summary = run_packet_eval(
                cases_path=cases_path,
                output_dir=run_dir / "packet_evals",
                mcp=mcp,
                budget_tokens=budget_tokens,
                hard_context_tokens=hard_context_tokens,
                run_id=f"block_{block:03d}_{variant}",
                sample_pressure=not dry_run,
                min_free_mb=hard_stop_free_mb,
                max_temp_c=hard_stop_temp_c,
                variant=variant,
                policy_path=policy_path if policy_path.exists() else None,
            )
            event_path = Path(eval_summary["events"])
            event_paths.append(event_path)
            block_summaries[variant] = eval_summary
            leaderboard.append({
                "block": block,
                "variant": variant,
                "pass_rate": eval_summary["pass_rate"],
                "passed_cases": eval_summary["passed_cases"],
                "total_cases": eval_summary["total_cases"],
                "avg_packet_tokens_est": round(eval_summary["avg_packet_tokens_est"], 2),
                "max_packet_tokens_est": eval_summary["max_packet_tokens_est"],
                "max_packet_context_ratio": round(eval_summary["max_packet_context_ratio"], 4),
                "pressure_failures": eval_summary["pressure_failures"],
                "promotion_ready": eval_summary["promotion_ready"],
                "live_pass_rate": "",
            })

        train_retrieval_policy(event_paths, out=policy_path)

        if "hybrid_reranked" in variants:
            eval_summary = run_packet_eval(
                cases_path=cases_path,
                output_dir=run_dir / "packet_evals",
                mcp=mcp,
                budget_tokens=budget_tokens,
                hard_context_tokens=hard_context_tokens,
                run_id=f"block_{block:03d}_hybrid_reranked",
                sample_pressure=not dry_run,
                min_free_mb=hard_stop_free_mb,
                max_temp_c=hard_stop_temp_c,
                variant="hybrid_reranked",
                policy_path=policy_path,
            )
            event_paths.append(Path(eval_summary["events"]))
            block_summaries["hybrid_reranked"] = eval_summary
            leaderboard.append({
                "block": block,
                "variant": "hybrid_reranked",
                "pass_rate": eval_summary["pass_rate"],
                "passed_cases": eval_summary["passed_cases"],
                "total_cases": eval_summary["total_cases"],
                "avg_packet_tokens_est": round(eval_summary["avg_packet_tokens_est"], 2),
                "max_packet_tokens_est": eval_summary["max_packet_tokens_est"],
                "max_packet_context_ratio": round(eval_summary["max_packet_context_ratio"], 4),
                "pressure_failures": eval_summary["pressure_failures"],
                "promotion_ready": eval_summary["promotion_ready"],
                "live_pass_rate": "",
            })

        live_passes = 0
        live_total = 0
        live_probe_error = ""
        if not dry_run and cases:
            promoted = [item for item in block_summaries.items() if item[1].get("promotion_ready")]
            if promoted:
                best_variant = max(promoted, key=lambda item: (item[1]["pass_rate"], -item[1]["max_packet_tokens_est"]))[0]
                for idx, case in enumerate(cases[:max_live_cases]):
                    case_id = str(case.get("task_id") or case.get("id") or f"case_{idx:04d}")
                    case_root = run_dir / "live_probe_artifacts" / f"block_{block:03d}" / best_variant / case_id
                    task = _write_case_artifacts(case_root, case, mcp=mcp)
                    packet = build_variant_packet(
                        case_root=case_root,
                        case=case,
                        task=task,
                        variant=best_variant,
                        budget_tokens=budget_tokens,
                        hard_context_tokens=hard_context_tokens,
                        policy_path=policy_path,
                    )
                    probe = run_live_case_probe(
                        case=case,
                        packet=packet,
                        base_url=base_url,
                        model=model,
                        max_tokens=live_probe_max_tokens,
                        timeout_s=live_probe_timeout_s,
                        min_free_mb=hard_stop_free_mb,
                        max_temp_c=hard_stop_temp_c,
                    )
                    probe["run_id"] = run_id
                    probe["block"] = block
                    probe["variant"] = best_variant
                    append_jsonl(events_path, [probe])
                    live_total += 1
                    live_passes += 1 if probe.get("passed") else 0
                    if probe.get("error"):
                        live_probe_error = str(probe.get("error"))
                        append_jsonl(events_path, [{
                            "ts": utc_now(),
                            "event": "abort",
                            "run_id": run_id,
                            "reason": "live_probe_error",
                            "block": block,
                            "case_id": case_id,
                            "error": live_probe_error,
                        }])
                        break
                live_failures = live_total - live_passes
                total_live_failures += live_failures
                for row in reversed(leaderboard):
                    if row["block"] == block and row["variant"] == best_variant:
                        row["live_pass_rate"] = round(live_passes / live_total, 4) if live_total else ""
                        break
                if live_probe_error:
                    abort_reason = "live_probe_error"
                    break
                if live_failures > max_live_failures_per_block:
                    abort_reason = "too_many_live_failures"
                    append_jsonl(events_path, [{"ts": utc_now(), "event": "abort", "run_id": run_id, "reason": abort_reason, "block": block}])
                    break

        append_jsonl(events_path, [{
            "ts": utc_now(),
            "event": "block_complete",
            "run_id": run_id,
            "block": block,
            "variants": {name: {"pass_rate": data["pass_rate"], "promotion_ready": data["promotion_ready"]} for name, data in block_summaries.items()},
            "live_passes": live_passes,
            "live_total": live_total,
        }])

        if not dry_run and block < block_limit - 1:
            time.sleep(max(0, cooldown_minutes) * 60)

    _write_leaderboard(leaderboard_path, leaderboard)
    best_row = None
    if leaderboard:
        best_row = max(
            leaderboard,
            key=lambda row: (
                float(row.get("pass_rate") or 0),
                float(row.get("live_pass_rate") or 0),
                -int(row.get("max_packet_tokens_est") or 0),
            ),
        )
    pressure_failures = sum(int(row.get("pressure_failures") or 0) for row in leaderboard)
    summary = {
        "run_id": run_id,
        "created_at": utc_now(),
        "status": "aborted" if abort_reason else "completed",
        "abort_reason": abort_reason,
        "suite": suite,
        "cases_path": str(cases_path),
        "manifest": str(manifest_path),
        "events": str(events_path),
        "leaderboard": str(leaderboard_path),
        "retrieval_policy": str(policy_path),
        "report": str(report_path),
        "blocks_completed": len({row["block"] for row in leaderboard}),
        "duration_ms": int((time.perf_counter() - start) * 1000),
        "best_variant": best_row.get("variant") if best_row else "",
        "best_pass_rate": best_row.get("pass_rate") if best_row else 0.0,
        "live_failures": total_live_failures,
        "pressure_failures": pressure_failures,
        "promotion_ready": bool(best_row and best_row.get("promotion_ready") and not abort_reason),
        "recommendation": (
            f"Use {best_row.get('variant')} for the next storyworld MCP run; "
            "increase duration only after another clean pilot."
            if best_row else "No variant produced a usable result; inspect events before rerunning."
        ),
    }
    write_json(summary_path, summary)
    _write_marathon_report(report_path, summary, leaderboard)
    append_jsonl(events_path, [{"ts": utc_now(), "event": "marathon_complete", "run_id": run_id, "summary": summary}])
    return summary


def run_packet_eval(
    *,
    cases_path: Path,
    output_dir: Path,
    mcp: str,
    budget_tokens: int = 1200,
    hard_context_tokens: int = 12000,
    max_replay_hints: int = 1,
    run_id: Optional[str] = None,
    sample_pressure: bool = False,
    min_free_mb: int = 512,
    max_temp_c: int = 86,
    max_packet_context_ratio: float = 0.20,
    variant: str = "hybrid",
    policy_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if variant not in PACKET_VARIANTS:
        raise ValueError(f"unknown packet variant: {variant}")
    cases = read_jsonl(cases_path)
    if run_id is None:
        run_id = f"mcp-trm-ldt-eval-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = output_dir / run_id
    events_path = run_dir / "events.jsonl"
    summary_path = run_dir / "summary.json"
    artifacts_dir = run_dir / "case_artifacts"

    start = time.perf_counter()
    events: List[Dict[str, Any]] = []
    pass_count = 0
    token_counts: List[int] = []
    failures: Dict[str, int] = {}
    pressure_failures = 0
    pressure_start = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=max_temp_c) if sample_pressure else {}

    for idx, case in enumerate(cases):
        case_id = str(case.get("task_id") or case.get("id") or f"case_{idx:04d}")
        case_root = artifacts_dir / case_id
        task = _write_case_artifacts(case_root, case, mcp=mcp)
        case["_task"] = task

        case_start = time.perf_counter()
        packet = build_variant_packet(
            case_root=case_root,
            case=case,
            task=task,
            variant=variant,
            budget_tokens=budget_tokens,
            hard_context_tokens=hard_context_tokens,
            max_replay_hints=max_replay_hints,
            policy_path=policy_path,
        )
        elapsed_ms = int((time.perf_counter() - case_start) * 1000)
        packet_data = _extract_packet_json(packet)
        packet_tokens_est = estimate_tokens_rough(packet)
        token_counts.append(packet_tokens_est)
        packet_context_ratio = (packet_tokens_est / hard_context_tokens) if hard_context_tokens else 0.0
        pressure = gpu_pressure_snapshot(min_free_mb=min_free_mb, max_temp_c=max_temp_c) if sample_pressure else {}
        pressure_passed = pressure.get("passed", True) if pressure else True
        if not pressure_passed:
            pressure_failures += 1

        result = score_case(
            case,
            packet=packet,
            packet_data=packet_data,
            packet_tokens_est=packet_tokens_est,
            budget_tokens=budget_tokens,
        )
        pass_count += 1 if result["passed"] else 0
        if result["failure"]:
            failures[result["failure"]] = failures.get(result["failure"], 0) + 1

        events.append({
            "ts": utc_now(),
            "event": "case_eval",
            "run_id": run_id,
            "case_id": case_id,
            "task_id": task.get("task_id"),
            "bench_family": case.get("bench_family"),
            "variant": variant,
            "packet_tokens_est": packet_tokens_est,
            "packet_context_ratio": packet_context_ratio,
            "latency_ms": elapsed_ms,
            "pressure": pressure,
            "passed": result["passed"],
            "score": result["score"],
            "failure": result["failure"],
            "gates": result["gates"],
            "artifact_root": str(case_root),
        })

    append_jsonl(events_path, events)
    total = len(cases)
    max_packet_tokens = max(token_counts) if token_counts else 0
    max_packet_ratio = (max_packet_tokens / hard_context_tokens) if hard_context_tokens else 0.0
    packet_pressure_ready = max_packet_ratio <= max_packet_context_ratio
    resource_ready = pressure_failures == 0
    summary = {
        "run_id": run_id,
        "created_at": utc_now(),
        "mcp": mcp,
        "variant": variant,
        "policy_path": str(policy_path) if policy_path else "",
        "cases_path": str(cases_path),
        "events": str(events_path),
        "status": "completed",
        "total_cases": total,
        "passed_cases": pass_count,
        "pass_rate": (pass_count / total) if total else 0.0,
        "avg_packet_tokens_est": (sum(token_counts) / len(token_counts)) if token_counts else 0.0,
        "max_packet_tokens_est": max_packet_tokens,
        "max_packet_context_ratio": max_packet_ratio,
        "max_packet_context_ratio_allowed": max_packet_context_ratio,
        "packet_pressure_ready": packet_pressure_ready,
        "packet_budget_tokens": budget_tokens,
        "hard_context_tokens": hard_context_tokens,
        "pressure_start": pressure_start,
        "pressure_failures": pressure_failures,
        "resource_ready": resource_ready,
        "min_free_mb": min_free_mb,
        "max_temp_c": max_temp_c,
        "failure_counts": failures,
        "duration_ms": int((time.perf_counter() - start) * 1000),
        "promotion_ready": bool(
            total
            and pass_count / total >= 0.80
            and max_packet_tokens <= budget_tokens
            and packet_pressure_ready
            and resource_ready
        ),
    }
    write_json(summary_path, summary)
    return summary


def _cmd_manifest(args: argparse.Namespace) -> int:
    manifest = create_run_manifest(
        mcp=args.mcp,
        task_id=args.task_id,
        out=Path(args.out) if args.out else None,
        budget_tokens=args.budget_tokens,
        max_runtime_minutes=args.max_runtime_minutes,
        checkpoint_interval=args.checkpoint_interval,
        ram_mb=args.ram_mb,
        cpu_pct=args.cpu_pct,
        io_mb_s=args.io_mb_s,
        chunk_strategy=args.chunk_strategy,
    )
    out = Path(args.out) if args.out else Path(manifest["artifacts"]["manifest"])
    write_json(out, manifest)
    print(out)
    return 0


def _cmd_preflight(args: argparse.Namespace) -> int:
    result = gpu_preflight(min_free_mb=args.min_free_mb, max_temp_c=args.max_temp_c)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 1


def _cmd_pressure(args: argparse.Namespace) -> int:
    result = gpu_pressure_snapshot(min_free_mb=args.min_free_mb, max_temp_c=args.max_temp_c)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 1


def _cmd_eval(args: argparse.Namespace) -> int:
    summary = run_packet_eval(
        cases_path=Path(args.cases),
        output_dir=Path(args.output_dir),
        mcp=args.mcp,
        budget_tokens=args.budget_tokens,
        hard_context_tokens=args.hard_context_tokens,
        max_replay_hints=args.max_replay_hints,
        run_id=args.run_id or None,
        sample_pressure=args.sample_pressure,
        min_free_mb=args.min_free_mb,
        max_temp_c=args.max_temp_c,
        max_packet_context_ratio=args.max_packet_context_ratio,
        variant=args.variant,
        policy_path=Path(args.policy) if args.policy else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["total_cases"] else 1


def _cmd_live_smoke(args: argparse.Namespace) -> int:
    result = run_live_smoke(
        base_url=args.base_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        timeout_s=args.timeout_s,
        min_free_mb=args.min_free_mb,
        max_temp_c=args.max_temp_c,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("passed") else 1


def _cmd_train_retrieval_policy(args: argparse.Namespace) -> int:
    policy = train_retrieval_policy(
        [Path(path) for path in args.events],
        out=Path(args.out) if args.out else None,
    )
    print(json.dumps(policy, indent=2, sort_keys=True))
    return 0


def _cmd_marathon(args: argparse.Namespace) -> int:
    summary = run_marathon(
        suite=args.suite,
        cases_path=Path(args.cases) if args.cases else None,
        output_dir=Path(args.output_dir),
        run_id=args.run_id or None,
        duration_minutes=args.duration_minutes,
        block_minutes=args.block_minutes,
        cooldown_minutes=args.cooldown_minutes,
        base_url=args.base_url,
        model=args.model,
        mcp=args.mcp,
        variants=list(args.variants) if args.variants else None,
        budget_tokens=args.budget_tokens,
        hard_context_tokens=args.hard_context_tokens,
        min_free_mb=args.min_free_mb,
        hard_stop_free_mb=args.hard_stop_free_mb,
        warn_temp_c=args.warn_temp_c,
        hard_stop_temp_c=args.hard_stop_temp_c,
        max_consecutive_pressure_failures=args.max_consecutive_pressure_failures,
        max_live_failures_per_block=args.max_live_failures_per_block,
        max_live_cases=args.max_live_cases,
        live_probe_max_tokens=args.live_probe_max_tokens,
        live_probe_timeout_s=args.live_probe_timeout_s,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("status") == "completed" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCP TRM/LDT local experiment harness")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="Create a run manifest")
    manifest.add_argument("--mcp", required=True)
    manifest.add_argument("--task-id", required=True)
    manifest.add_argument("--out", default="")
    manifest.add_argument("--budget-tokens", type=int, default=1200)
    manifest.add_argument("--max-runtime-minutes", type=int, default=60)
    manifest.add_argument("--checkpoint-interval", default="500 steps or 15 minutes")
    manifest.add_argument("--ram-mb", type=int, default=2048)
    manifest.add_argument("--cpu-pct", type=int, default=50)
    manifest.add_argument("--io-mb-s", type=int, default=50)
    manifest.add_argument("--chunk-strategy", default="stream JSONL in bounded batches")
    manifest.set_defaults(func=_cmd_manifest)

    preflight = sub.add_parser("preflight", help="Check local NVIDIA GPU readiness")
    preflight.add_argument("--min-free-mb", type=int, default=1024)
    preflight.add_argument("--max-temp-c", type=int, default=86)
    preflight.set_defaults(func=_cmd_preflight)

    pressure = sub.add_parser("pressure", help="Record a lightweight GPU pressure snapshot")
    pressure.add_argument("--min-free-mb", type=int, default=512)
    pressure.add_argument("--max-temp-c", type=int, default=86)
    pressure.set_defaults(func=_cmd_pressure)

    eval_parser = sub.add_parser("eval", help="Run deterministic packet evals")
    eval_parser.add_argument("--cases", required=True)
    eval_parser.add_argument("--mcp", required=True)
    eval_parser.add_argument("--output-dir", default="experiments")
    eval_parser.add_argument("--run-id", default="")
    eval_parser.add_argument("--budget-tokens", type=int, default=1200)
    eval_parser.add_argument("--hard-context-tokens", type=int, default=12000)
    eval_parser.add_argument("--max-replay-hints", type=int, default=1)
    eval_parser.add_argument("--sample-pressure", action="store_true")
    eval_parser.add_argument("--min-free-mb", type=int, default=512)
    eval_parser.add_argument("--max-temp-c", type=int, default=86)
    eval_parser.add_argument("--max-packet-context-ratio", type=float, default=0.20)
    eval_parser.add_argument("--variant", choices=PACKET_VARIANTS, default="hybrid")
    eval_parser.add_argument("--policy", default="")
    eval_parser.set_defaults(func=_cmd_eval)

    smoke = sub.add_parser("live-smoke", help="Run one bounded OpenAI-compatible local model smoke request")
    smoke.add_argument("--base-url", default=DEFAULT_LOCAL_BONSAI_BASE_URL)
    smoke.add_argument("--model", default="local/bonsai-8b")
    smoke.add_argument("--prompt", default="Reply with exactly: ok")
    smoke.add_argument("--max-tokens", type=int, default=8)
    smoke.add_argument("--timeout-s", type=int, default=90)
    smoke.add_argument("--min-free-mb", type=int, default=512)
    smoke.add_argument("--max-temp-c", type=int, default=86)
    smoke.set_defaults(func=_cmd_live_smoke)

    train = sub.add_parser("train-retrieval-policy", help="Train a deterministic retrieval policy from event logs")
    train.add_argument("--events", nargs="+", required=True)
    train.add_argument("--out", default="")
    train.set_defaults(func=_cmd_train_retrieval_policy)

    marathon = sub.add_parser("marathon", help="Run a conservative storyworld MCP packet marathon")
    marathon.add_argument("--suite", default="storyworld")
    marathon.add_argument("--cases", default="")
    marathon.add_argument("--output-dir", default=str(Path("experiments") / "marathons"))
    marathon.add_argument("--run-id", default="")
    marathon.add_argument("--duration-minutes", type=int, default=240)
    marathon.add_argument("--block-minutes", type=int, default=20)
    marathon.add_argument("--cooldown-minutes", type=int, default=5)
    marathon.add_argument("--base-url", default=DEFAULT_LOCAL_BONSAI_BASE_URL)
    marathon.add_argument("--model", default="local/bonsai-8b")
    marathon.add_argument("--mcp", default="storyworld-encounter")
    marathon.add_argument("--variants", nargs="+", choices=PACKET_VARIANTS, default=PACKET_VARIANTS)
    marathon.add_argument("--budget-tokens", type=int, default=1200)
    marathon.add_argument("--hard-context-tokens", type=int, default=12000)
    marathon.add_argument("--min-free-mb", type=int, default=512)
    marathon.add_argument("--hard-stop-free-mb", type=int, default=256)
    marathon.add_argument("--warn-temp-c", type=int, default=87)
    marathon.add_argument("--hard-stop-temp-c", type=int, default=87)
    marathon.add_argument("--max-consecutive-pressure-failures", type=int, default=3)
    marathon.add_argument("--max-live-failures-per-block", type=int, default=2)
    marathon.add_argument("--max-live-cases", type=int, default=3)
    marathon.add_argument("--live-probe-max-tokens", type=int, default=40)
    marathon.add_argument("--live-probe-timeout-s", type=int, default=90)
    marathon.add_argument("--dry-run", action="store_true")
    marathon.set_defaults(func=_cmd_marathon)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
