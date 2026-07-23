"""Run frozen benchmark prompts through the full Hermes agent without tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
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


def _load_api_key(path: Path, variable: str = "") -> str:
    text = path.read_text(encoding="utf-8")
    if variable:
        value = ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, candidate = line.split("=", 1)
            if name.strip().lstrip("\ufeff") == variable:
                value = candidate.strip().strip("\"'")
                break
        if not value:
            raise ValueError(f"credential variable is absent: {variable}")
        return value
    value = text.strip()
    if not value:
        raise ValueError("credential file is empty")
    return value


def _load_codex_cli_credentials(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    tokens = payload.get("tokens", {}) if isinstance(payload, dict) else {}
    if not isinstance(tokens, dict):
        raise ValueError("Codex CLI credential token object is absent")
    access_token = str(tokens.get("access_token") or "").strip()
    account_id = str(tokens.get("account_id") or "").strip()
    if not access_token or not account_id:
        raise ValueError("Codex CLI access token or account id is absent")
    return access_token, account_id


class AgentResultError(RuntimeError):
    """A controlled failure returned as an AIAgent result dictionary."""


def _extract_agent_response(result: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(result, dict):
        raise AgentResultError("invalid_agent_result")
    if result.get("failed") or result.get("error"):
        raise AgentResultError("agent_result_failed")
    if result.get("interrupted"):
        raise AgentResultError("agent_result_interrupted")
    if result.get("partial"):
        raise AgentResultError("agent_result_partial")
    content = str(result.get("final_response") or "").strip()
    if not content:
        raise AgentResultError("empty_final_response")
    usage = result.get("usage")
    if not isinstance(usage, dict):
        usage = {
            name: result.get(name, 0)
            for name in (
                "input_tokens",
                "output_tokens",
                "cache_read_tokens",
                "cache_write_tokens",
                "reasoning_tokens",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
        }
    return content, usage


def _synthesize_codex_response(response: Any, text: str) -> Any:
    if getattr(response, "output", None) or not text:
        return response
    message = SimpleNamespace(
        type="message",
        status="completed",
        phase="final_answer",
        content=[SimpleNamespace(type="output_text", text=text)],
    )
    return SimpleNamespace(
        output=[message],
        status=getattr(response, "status", "completed"),
        model=getattr(response, "model", None),
        usage=getattr(response, "usage", None),
        incomplete_details=getattr(response, "incomplete_details", None),
        error=getattr(response, "error", None),
    )


def _install_codex_cli_compatibility(agent: Any, account_id: str) -> None:
    client_kwargs = dict(agent._client_kwargs)
    headers = dict(client_kwargs.get("default_headers") or {})
    headers["ChatGPT-Account-Id"] = account_id
    client_kwargs["default_headers"] = headers
    agent._client_kwargs = client_kwargs
    if not agent._replace_primary_openai_client(reason="codex_cli_account_binding"):
        raise AgentResultError("codex_cli_client_rebuild_failed")

    original_preflight = agent._preflight_codex_api_kwargs

    def compatible_preflight(
        api_kwargs: Any,
        *,
        allow_stream: bool = False,
    ) -> dict[str, Any]:
        normalized = original_preflight(api_kwargs, allow_stream=allow_stream)
        if normalized.get("tools") is None:
            normalized.pop("tools")
        normalized.pop("max_output_tokens", None)
        return normalized

    def compatible_stream(
        api_kwargs: dict[str, Any],
        client: Any = None,
        on_first_delta: Any = None,
    ) -> Any:
        active_client = client or agent._ensure_primary_openai_client(
            reason="codex_cli_compatible_stream"
        )
        text_parts: list[str] = []
        first_delta_fired = False
        with active_client.responses.stream(**api_kwargs) as stream:
            for event in stream:
                if agent._interrupt_requested:
                    break
                event_type = str(getattr(event, "type", ""))
                if "output_text.delta" in event_type:
                    delta = getattr(event, "delta", "")
                    if isinstance(delta, str) and delta:
                        text_parts.append(delta)
                        if not first_delta_fired:
                            first_delta_fired = True
                            if on_first_delta:
                                on_first_delta()
                        agent._fire_stream_delta(delta)
                elif "reasoning" in event_type and "delta" in event_type:
                    reasoning = getattr(event, "delta", "")
                    if isinstance(reasoning, str) and reasoning:
                        agent._fire_reasoning_delta(reasoning)
            response = stream.get_final_response()
        return _synthesize_codex_response(response, "".join(text_parts))

    agent._preflight_codex_api_kwargs = compatible_preflight
    agent._run_codex_stream = compatible_stream


def run(
    prompts_path: Path,
    output_path: Path,
    *,
    hermes_root: Path,
    api_key_path: Path,
    api_key_variable: str,
    credential_mode: str,
    provider: str,
    base_url: str,
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
    codex_account_id = ""
    if credential_mode == "codex_cli":
        key, codex_account_id = _load_codex_cli_credentials(api_key_path)
        if provider != "openai-codex":
            raise ValueError("codex_cli credential mode requires openai-codex provider")
    else:
        key = _load_api_key(api_key_path, api_key_variable)
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
                base_url=base_url,
                provider=provider,
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
            if credential_mode == "codex_cli":
                _install_codex_cli_compatibility(agent, codex_account_id)
            result = agent.run_conversation(
                str(prompt["prompt"]),
                system_message=(
                    "You are the full Hermes 160k baseline. Follow the frozen output "
                    "contract exactly and do not call tools."
                ),
                task_id=task_id,
                sync_honcho=False,
            )
            content, usage = _extract_agent_response(result)
            if len(content) > 4000:
                raise AgentResultError("response_exceeds_4000_character_receipt_cap")
        except Exception as exc:
            error = (
                f"{type(exc).__name__}: {exc}"
                if isinstance(exc, AgentResultError)
                else type(exc).__name__
            )
        row = {
            "schema": "hermes.full_hermes_raw_result.v1",
            "task_id": task_id,
            "status": "completed" if not error else "api_error",
            "error": error,
            "latency_ms": int((time.perf_counter() - cell_start) * 1000),
            "packet_tokens_est": int(prompt["packet_tokens_est"]),
            "usage": usage,
            "raw_content": content[:4000],
            "provider": provider,
            "model": model,
            "context_tokens": context_tokens,
            "output_limit_mode": (
                "prompt_contract_and_4000_character_receipt_cap"
                if credential_mode == "codex_cli"
                else "api_max_tokens"
            ),
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
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "context_tokens": context_tokens,
        "credential_mode": credential_mode,
        "output_limit_mode": (
            "prompt_contract_and_4000_character_receipt_cap"
            if credential_mode == "codex_cli"
            else "api_max_tokens"
        ),
        "prompts_sha256": prompts_sha256,
        "results_sha256": _sha256_file(output_path) if output_path.exists() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--hermes-root", required=True)
    parser.add_argument("--api-key-path", required=True)
    parser.add_argument("--api-key-variable", default="")
    parser.add_argument(
        "--credential-mode",
        choices=("dotenv_or_bare", "codex_cli"),
        default="dotenv_or_bare",
    )
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
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
        api_key_variable=args.api_key_variable,
        credential_mode=args.credential_mode,
        provider=args.provider,
        base_url=args.base_url,
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
