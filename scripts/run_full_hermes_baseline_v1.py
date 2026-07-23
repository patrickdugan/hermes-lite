"""Run frozen benchmark prompts through the full Hermes agent without tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bare_key(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    if not value.startswith("sk-"):
        raise ValueError("GPT API credential is not a bare OpenAI key")
    return value


def run(
    prompts_path: Path,
    output_path: Path,
    *,
    hermes_root: Path,
    api_key_path: Path,
    model: str,
    context_tokens: int,
    max_tokens: int,
    expected_prompts_sha256: str,
) -> dict[str, Any]:
    sys.path.insert(0, str(hermes_root.resolve()))
    from run_agent import AIAgent

    prompts_sha256 = _sha256_file(prompts_path)
    if prompts_sha256 != expected_prompts_sha256:
        raise ValueError("full Hermes prompt batch hash mismatch")
    prompts = _read_jsonl(prompts_path)
    existing = _read_jsonl(output_path) if output_path.exists() else []
    if any(row.get("status") != "completed" for row in existing):
        raise ValueError("non-completed full Hermes checkpoint requires a fresh output lane")
    completed = {str(row["task_id"]) for row in existing}
    key = os.getenv("OPENAI_API_KEY", "").strip() or _load_bare_key(api_key_path)
    started = time.perf_counter()
    for prompt in prompts:
        task_id = str(prompt["task_id"])
        if task_id in completed:
            continue
        error = ""
        content = ""
        usage: dict[str, Any] = {}
        cell_start = time.perf_counter()
        try:
            agent = AIAgent(
                api_key=key,
                base_url="https://api.openai.com/v1",
                provider="openai",
                model=model,
                max_iterations=1,
                enabled_toolsets=["__no_tools__"],
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                max_tokens=max_tokens,
                context_length_override=context_tokens,
                minimum_context_length=context_tokens,
            )
            result = agent.run_conversation(
                str(prompt["prompt"]),
                system_message=(
                    "You are the full Hermes 160k baseline. Follow the frozen output "
                    "contract exactly and do not call tools."
                ),
                task_id=task_id,
                sync_honcho=False,
            )
            content = str(result.get("final_response") or "")
            usage = result.get("usage", {}) if isinstance(result, dict) else {}
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = {
            "schema": "hermes.full_hermes_raw_result.v1",
            "task_id": task_id,
            "status": "completed" if not error else "api_error",
            "error": error,
            "latency_ms": int((time.perf_counter() - cell_start) * 1000),
            "packet_tokens_est": int(prompt["packet_tokens_est"]),
            "usage": usage,
            "raw_content": content[:4000],
        }
        _append_jsonl(output_path, row)
        if error:
            break
    rows = _read_jsonl(output_path)
    return {
        "status": "completed"
        if len(rows) == len(prompts) and all(row["status"] == "completed" for row in rows)
        else "incomplete",
        "expected": len(prompts),
        "completed": sum(row["status"] == "completed" for row in rows),
        "api_errors": sum(row["status"] != "completed" for row in rows),
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "model": model,
        "context_tokens": context_tokens,
        "prompts_sha256": prompts_sha256,
        "results_sha256": _sha256_file(output_path) if output_path.exists() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--hermes-root", required=True)
    parser.add_argument("--api-key-path", required=True)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--context-tokens", type=int, default=160000)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--expected-prompts-sha256", required=True)
    parser.add_argument("--summary-path", required=True)
    args = parser.parse_args()
    result = run(
        Path(args.prompts_path),
        Path(args.output_path),
        hermes_root=Path(args.hermes_root),
        api_key_path=Path(args.api_key_path),
        model=args.model,
        context_tokens=args.context_tokens,
        max_tokens=args.max_tokens,
        expected_prompts_sha256=args.expected_prompts_sha256,
    )
    Path(args.summary_path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
