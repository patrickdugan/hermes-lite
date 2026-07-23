"""Live schema bench for Bonsai ultra-lean TRM/LDT skill contracts."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import request

from agent.skill_catalog import iter_skill_files, load_ultra_lean_contract
from agent.lean_contracts import contract_phases


ACTIONS = {"select", "route", "retrieve", "execute", "verify", "commit", "repair", "abstain"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_probe(contract: dict) -> tuple[dict, str]:
    name = str(contract["name"])
    phases = contract_phases(contract) or [{"id": "ROUTE"}]
    task = {
        "task_id": f"lean-probe-{name}",
        "instruction": "Select the first conveyor phase. Do not perform the domain task.",
        "gate_status": "unknown",
    }
    expected = {
        "task_id": task["task_id"],
        "contract_id": name,
        "phase": str(phases[0]["id"]),
    }
    prompt = (
        "ULTRA_LEAN_CONTRACT=" + compact_json(contract) + "\n"
        "TASK=" + task["instruction"] + "\n"
        "Return one JSON object with exactly three keys: phase, operation, gate. "
        "phase must be the first contract conveyor phase. operation must be one of "
        "select,route,retrieve,execute,verify,commit,repair,abstain. gate must be pending. "
        "The harness owns all identifiers. No Markdown or extra text."
    )
    return expected, prompt


def parse_object(text: str) -> dict | None:
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
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(stripped[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None


def validate_response(value: dict | None, expected: dict) -> list[str]:
    if value is None:
        return ["not_json_object"]
    errors = []
    required = {"phase", "operation", "gate"}
    if set(value) != required:
        errors.append("wrong_keys")
    if value.get("phase") != expected["phase"]:
        errors.append("wrong_phase")
    if value.get("operation") not in ACTIONS:
        errors.append("invalid_operation")
    if value.get("gate") != "pending":
        errors.append("wrong_gate")
    return errors


def gpu_temperature() -> int | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return int(proc.stdout.splitlines()[0].strip())
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None


def wait_for_thermal_gate(max_temp_c: int, timeout_s: int = 180) -> int | None:
    deadline = time.monotonic() + timeout_s
    while True:
        temp = gpu_temperature()
        if temp is None or temp <= max_temp_c:
            return temp
        if time.monotonic() >= deadline:
            raise RuntimeError(f"GPU remained above {max_temp_c} C (last={temp} C)")
        time.sleep(5)


def post_completion(base_url: str, model: str, prompt: str, max_tokens: int, timeout_s: int) -> dict:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are Bonsai acting only as a TRM coordinator. Return strict JSON, never domain content.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    req = request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_s) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def response_content(response: dict) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    return str(message.get("content") or "") if isinstance(message, dict) else ""


def run_probe(
    contract: dict,
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_s: int,
    max_temp_c: int,
) -> dict:
    expected, prompt = build_probe(contract)
    attempts = []
    repair = ""
    for attempt_index in range(2):
        temp_before = wait_for_thermal_gate(max_temp_c)
        attempt_prompt = prompt + repair
        started = time.perf_counter()
        try:
            response = post_completion(base_url, model, attempt_prompt, max_tokens, timeout_s)
            content = response_content(response)
            error = ""
        except Exception as exc:
            response, content, error = {}, "", f"{type(exc).__name__}: {exc}"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        parsed = parse_object(content)
        errors = ["request_error"] if error else validate_response(parsed, expected)
        attempts.append({
            "attempt": attempt_index + 1,
            "passed": not errors,
            "errors": errors,
            "error": error,
            "content": content,
            "parsed": parsed,
            "elapsed_ms": elapsed_ms,
            "temperature_before_c": temp_before,
            "temperature_after_c": gpu_temperature(),
            "usage": response.get("usage", {}),
        })
        if not errors:
            break
        repair = (
            "\nREPAIR: The prior output failed with " + ",".join(errors) + ". "
            f"Discard it. Return exactly {{\"phase\":\"{expected['phase']}\","
            "\"operation\":\"select\",\"gate\":\"pending\"}}."
        )
    final_value = attempts[-1].get("parsed") if attempts[-1]["passed"] else None
    envelope = None
    if isinstance(final_value, dict):
        envelope = {
            "task_id": expected["task_id"],
            "contract_id": expected["contract_id"],
            **final_value,
        }
    return {
        "ts": utc_now(),
        "skill": contract["name"],
        "contract_estimated_tokens": (len(compact_json(contract)) + 3) // 4,
        "prompt_estimated_tokens": (len(prompt) + 3) // 4,
        "first_pass": bool(attempts[0]["passed"]),
        "passed": bool(attempts[-1]["passed"]),
        "harness_envelope": envelope,
        "attempts": attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8801/v1")
    parser.add_argument("--model", default="local/bonsai-8b")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout-s", type=int, default=90)
    parser.add_argument("--max-temp-c", type=int, default=87)
    parser.add_argument("--skill", action="append", dest="skills")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/ultra-lean-skills"))
    args = parser.parse_args()

    contracts = []
    for _, skill_file in iter_skill_files():
        contract = load_ultra_lean_contract(skill_file)
        if contract and (not args.skills or contract["name"] in args.skills):
            contracts.append(contract)
    contracts.sort(key=lambda item: item["name"])
    if args.limit > 0:
        contracts = contracts[: args.limit]
    if not contracts:
        raise SystemExit("No ultra-lean contracts found.")

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    transcript = run_dir / "probes.jsonl"
    for contract in contracts:
        row = run_probe(
            contract,
            base_url=args.base_url,
            model=args.model,
            max_tokens=max(32, min(args.max_tokens, 256)),
            timeout_s=args.timeout_s,
            max_temp_c=args.max_temp_c,
        )
        rows.append(row)
        with transcript.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{row['skill']}: first={row['first_pass']} final={row['passed']}")

    summary = {
        "created_at": utc_now(),
        "scope": "schema coordination only; not domain correctness",
        "model": args.model,
        "base_url": args.base_url,
        "hard_context_tokens": 12000,
        "skills": len(rows),
        "first_pass": sum(row["first_pass"] for row in rows),
        "passed": sum(row["passed"] for row in rows),
        "first_pass_rate": sum(row["first_pass"] for row in rows) / len(rows),
        "final_pass_rate": sum(row["passed"] for row in rows) / len(rows),
        "max_prompt_estimated_tokens": max(row["prompt_estimated_tokens"] for row in rows),
        "max_contract_estimated_tokens": max(row["contract_estimated_tokens"] for row in rows),
        "transcript": str(transcript.resolve()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["final_pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
