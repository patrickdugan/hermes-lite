"""Live low-context Bonsai benchmark over the Intellect-3 Campsite MCP."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter, defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.intellect3_logic import (
    ACTION_KEYS,
    CampsiteTask,
    build_coordinator_packet,
    expected_action,
    failure_class,
    generate_unseen_tasks,
    parse_normalized_campsite_rows,
    stable_holdout,
    validate_action,
    verify_candidate,
)
from agent.mcp_trm_ldt_lab import gpu_pressure_snapshot


DEFAULT_SOURCE = Path(
    os.environ.get(
        "HERMES_INTELLECT3_SOURCE",
        "data/intellect_3_logic.jsonl",
    )
)
DEFAULT_ENDPOINT = "http://127.0.0.1:8801/v1"
DEFAULT_MODEL = "Bonsai-8B.gguf"
DEFAULT_OUTPUT = Path("experiments") / "intellect3-campsite"
VARIANTS = ("baseline", "ldt", "trm", "hybrid")
PROFILES = ("normal", "pressure", "no_pass")


SYSTEM_PROMPT = """You are the Bonsai coordinator for an Intellect-3 Campsite MCP.
The packet contains opaque candidate summaries, never final grids.
Return one JSON object only. Use exactly the listed keys in exactly that order.
If any candidate has official_pass=true, commit the first such candidate.
If none pass, repair a known candidate with module_id=repair and the matching failure class.
Use action=commit_candidate with module_id=commit, or action=repair_candidate with module_id=repair.
Copy repair_class exactly from output_contract.repair_classes; use none when committing.
Never output a grid, markdown, prose, or a tool-call wrapper.
visible_output_allowed must be false."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _tool_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        return structured
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", "")
        if text:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
    raise ValueError("MCP tool returned no object payload")


def _resource_text(result: Any) -> str:
    for item in getattr(result, "contents", []) or []:
        text = getattr(item, "text", "")
        if text:
            return text
    return ""


@asynccontextmanager
async def mcp_session() -> AsyncIterator[ClientSession]:
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent.intellect3_logic_mcp"],
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
    )
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            yield session


def _post_chat(endpoint: str, model: str, messages: list[dict[str, str]], timeout: float, max_tokens: int) -> dict[str, Any]:
    response = requests.post(
        endpoint.rstrip("/") + "/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("chat endpoint returned non-object JSON")
    return value


def _content(response: dict[str, Any]) -> str:
    try:
        return str(response["choices"][0]["message"]["content"] or "")
    except Exception:
        return ""


def _extract_action(text: str) -> tuple[dict[str, Any] | None, str | None]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None, "no_json_object"
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"json_decode:{exc.msg}"
    return (value, None) if isinstance(value, dict) else (None, "json_not_object")


def _semantic_action_correct(action: dict[str, Any] | None, candidates: list[dict[str, Any]]) -> bool:
    if not action:
        return False
    by_id = {item["candidate_id"]: item for item in candidates}
    passing = [item for item in candidates if item["official_pass"]]
    if passing:
        return action.get("action") == "commit_candidate" and bool(
            by_id.get(str(action.get("candidate_id")), {}).get("official_pass")
        )
    return action.get("action") == "repair_candidate" and action.get("candidate_id") in by_id


def _trm_proposal(task_id: str, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    action = expected_action(task_id, candidates)
    return {
        "action": action["action"],
        "candidate_id": action["candidate_id"],
        "repair_class": action["repair_class"],
        "confidence": 1.0 if any(item["official_pass"] for item in candidates) else 0.75,
    }


def _model_action(
    packet: dict[str, Any],
    *,
    endpoint: str,
    model: str,
    timeout: float,
    max_tokens: int,
    no_model: bool,
) -> dict[str, Any]:
    candidates = list(packet["state"]["candidates"])
    task_id = str(packet["task"]["task_id"])
    if no_model:
        action = expected_action(task_id, candidates)
        return {
            "action": action,
            "raw_text": json.dumps(action, separators=(",", ":")),
            "parse_error": None,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "attempts": 0,
        }

    packet_text = json.dumps(packet, separators=(",", ":"), ensure_ascii=False)
    base = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Choose the next coordinator action.\n" + packet_text},
    ]
    last: dict[str, Any] = {}
    previous_raw = ""
    for attempt in (1, 2):
        messages = list(base)
        if attempt == 2:
            messages.append({"role": "assistant", "content": previous_raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous response failed validation. Return JSON with exactly these keys in order: "
                        + ",".join(ACTION_KEYS)
                        + ". schema must be intellect3_campsite_mcp_action_v2. "
                        + "action must be invoke_module, commit_candidate, repair_candidate, or abstain. "
                        + "module_id must be propose, verify, repair, or commit. "
                        + "repair_class must be copied exactly from output_contract.repair_classes. No prose."
                    ),
                }
            )
        response = _post_chat(endpoint, model, messages, timeout, max_tokens)
        raw = _content(response)
        previous_raw = raw
        action, parse_error = _extract_action(raw)
        check = validate_action(action, task_id=task_id, candidate_ids={item["candidate_id"] for item in candidates})
        last = {
            "action": action,
            "raw_text": raw,
            "parse_error": parse_error,
            "schema_check": check,
            "usage": response.get("usage") or {},
            "timings": response.get("timings") or {},
            "attempts": attempt,
        }
        if parse_error is None and check["valid"]:
            break
    return last


async def _read_hint(session: ClientSession, candidates: list[dict[str, Any]]) -> str:
    replay_class = "none" if any(item["official_pass"] for item in candidates) else failure_class(candidates)
    result = await session.read_resource(f"int3://replay/{replay_class}")
    return _resource_text(result)


async def run_episode(
    session: ClientSession,
    task: CampsiteTask,
    target: list[list[str]] | None,
    *,
    variant: str,
    profile: str,
    order_seed: int,
    endpoint: str,
    model: str,
    timeout: float,
    max_tokens: int,
    no_model: bool,
    production_fallback: bool,
) -> dict[str, Any]:
    episode_task_id = f"{task.task_id}__{variant}__{profile}__{order_seed}"
    runtime = task.runtime_payload()
    runtime["task_id"] = episode_task_id
    await session.call_tool("open_task", {"task": runtime})
    proposed = _tool_payload(
        await session.call_tool(
            "propose_candidates",
            {
                "task_id": episode_task_id,
                "max_candidates": 3,
                "candidate_profile": profile,
                "order_seed": order_seed,
            },
        )
    )
    candidates = list(proposed["candidates"])
    replay = await _read_hint(session, candidates) if variant in {"trm", "hybrid"} else ""
    proposal = _trm_proposal(episode_task_id, candidates) if variant in {"trm", "hybrid"} else None
    packet = build_coordinator_packet(
        CampsiteTask(
            task_id=episode_task_id,
            grid=task.grid,
            row_constraints=task.row_constraints,
            col_constraints=task.col_constraints,
        ),
        candidates,
        variant=variant,
        replay_hint=replay,
        trm_proposal=proposal,
    )
    first = await asyncio.to_thread(
        _model_action,
        packet,
        endpoint=endpoint,
        model=model,
        timeout=timeout,
        max_tokens=max_tokens,
        no_model=no_model,
    )
    first_action = first.get("action")
    schema_valid = bool(first.get("schema_check", {"valid": True}).get("valid"))
    semantic_correct = _semantic_action_correct(first_action, candidates)
    fallback_used = False
    if (not schema_valid or not semantic_correct) and production_fallback:
        first_action = expected_action(episode_task_id, candidates)
        fallback_used = True
    elif not schema_valid or not semantic_correct:
        first_action = None

    committed: dict[str, Any] = {"committed": False, "reason": "no_valid_action"}
    second: dict[str, Any] | None = None
    if isinstance(first_action, dict) and first_action.get("action") == "commit_candidate":
        committed = _tool_payload(
            await session.call_tool(
                "commit_candidate",
                {"task_id": episode_task_id, "candidate_id": first_action["candidate_id"]},
            )
        )
    elif isinstance(first_action, dict) and first_action.get("action") == "repair_candidate":
        repaired = _tool_payload(
            await session.call_tool(
                "repair_candidate",
                {
                    "task_id": episode_task_id,
                    "candidate_id": first_action["candidate_id"],
                    "repair_class": first_action["repair_class"],
                },
            )
        )
        state = _tool_payload(await session.call_tool("verify_candidates", {"task_id": episode_task_id}))
        candidates2 = list(state["candidates"])
        repaired_candidate = repaired.get("candidate") if isinstance(repaired, dict) else None
        if isinstance(repaired_candidate, dict) and repaired_candidate.get("official_pass") is True:
            # LDT monotonic refinement: rejected candidates do not return once
            # an environment-sound repaired candidate has been certified.
            candidates2 = [repaired_candidate]
        replay2 = await _read_hint(session, candidates2) if variant in {"trm", "hybrid"} else ""
        proposal2 = _trm_proposal(episode_task_id, candidates2) if variant in {"trm", "hybrid"} else None
        packet2 = build_coordinator_packet(
            CampsiteTask(
                task_id=episode_task_id,
                grid=task.grid,
                row_constraints=task.row_constraints,
                col_constraints=task.col_constraints,
            ),
            candidates2,
            variant=variant,
            replay_hint=replay2,
            trm_proposal=proposal2,
        )
        second = await asyncio.to_thread(
            _model_action,
            packet2,
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            no_model=no_model,
        )
        second_action = second.get("action")
        second_valid = bool(second.get("schema_check", {"valid": True}).get("valid"))
        second_correct = _semantic_action_correct(second_action, candidates2)
        if (not second_valid or not second_correct) and production_fallback:
            second_action = expected_action(episode_task_id, candidates2)
            fallback_used = True
        elif not second_valid or not second_correct:
            second_action = None
        if isinstance(second_action, dict) and second_action.get("action") == "commit_candidate":
            committed = _tool_payload(
                await session.call_tool(
                    "commit_candidate",
                    {"task_id": episode_task_id, "candidate_id": second_action["candidate_id"]},
                )
            )
        committed["repair_tool_result"] = repaired

    await session.call_tool("reset_task", {"task_id": episode_task_id})
    grid = committed.get("grid")
    final_report = verify_candidate(task, grid) if grid else {"official_pass": False, "failed_gates": ["no_grid"]}
    usages = [first.get("usage") or {}] + ([second.get("usage") or {}] if second else [])
    prompt_tokens = sum(int(item.get("prompt_tokens") or 0) for item in usages)
    completion_tokens = sum(int(item.get("completion_tokens") or 0) for item in usages)
    raw_texts = [str(first.get("raw_text") or "")] + ([str(second.get("raw_text") or "")] if second else [])
    return {
        "ts": utc_now(),
        "row_id": task.task_id,
        "episode_task_id": episode_task_id,
        "variant": variant,
        "profile": profile,
        "order_seed": order_seed,
        "packet_chars": len(json.dumps(packet, separators=(",", ":"))),
        "packet_tokens_est": len(json.dumps(packet, separators=(",", ":"))) // 4,
        "candidate_count": len(candidates),
        "first_schema_valid": schema_valid,
        "first_semantic_action_correct": semantic_correct,
        "first_action": first_action,
        "first_raw_text": first.get("raw_text"),
        "second": second,
        "fallback_used": fallback_used,
        "final_grid_leak": any("[[" in text or '"grid"' in text for text in raw_texts),
        "committed": bool(committed.get("committed")),
        "official_semantic_pass": bool(final_report.get("official_pass")),
        "failed_gates": final_report.get("failed_gates") or [],
        "target_exact_diagnostic": bool(target is not None and grid == target),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_calls": 0 if no_model else 1 + int(second is not None),
    }


def summarize(rows: list[dict[str, Any]], *, pressure: dict[str, Any]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    variants = {}
    for variant, items in by_variant.items():
        count = len(items)
        rate = lambda key: round(sum(bool(item.get(key)) for item in items) / count, 4) if count else 0.0
        variants[variant] = {
            "episodes": count,
            "schema_valid_rate": rate("first_schema_valid"),
            "action_accuracy": rate("first_semantic_action_correct"),
            "official_semantic_pass_rate": rate("official_semantic_pass"),
            "invalid_commit_rate": round(
                sum(bool(item.get("committed")) and not bool(item.get("official_semantic_pass")) for item in items) / count,
                4,
            ),
            "final_grid_leak_rate": rate("final_grid_leak"),
            "fallback_rate": rate("fallback_used"),
            "target_exact_diagnostic_rate": rate("target_exact_diagnostic"),
            "avg_packet_tokens_est": round(sum(item["packet_tokens_est"] for item in items) / count, 2),
            "max_packet_tokens_est": max(item["packet_tokens_est"] for item in items),
            "avg_prompt_tokens": round(sum(item["prompt_tokens"] for item in items) / count, 2),
            "avg_model_calls": round(sum(item["model_calls"] for item in items) / count, 2),
            "failure_counts": dict(Counter(gate for item in items for gate in item.get("failed_gates", []))),
        }
    return {
        "generated_at": utc_now(),
        "episodes": len(rows),
        "variants": variants,
        "pressure_after": pressure,
    }


def render_report(summary: dict[str, Any], args: argparse.Namespace) -> str:
    lines = [
        "# Intellect-3 Campsite MCP Bench",
        "",
        f"- generated_at: `{summary['generated_at']}`",
        f"- split: `{args.split}`",
        f"- profiles: `{','.join(args.profiles)}`",
        f"- model: `{args.model}`",
        f"- episodes: `{summary['episodes']}`",
        "",
        "| variant | action accuracy | official pass | invalid commit | grid leak | max packet | avg calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["variants"].items():
        lines.append(
            f"| {name} | {row['action_accuracy']:.3f} | {row['official_semantic_pass_rate']:.3f} | "
            f"{row['invalid_commit_rate']:.3f} | {row['final_grid_leak_rate']:.3f} | "
            f"{row['max_packet_tokens_est']} | {row['avg_model_calls']:.2f} |"
        )
    lines.extend(
        [
            "",
            "Official pass uses the Prime-style perfect tent/tree matching verifier. Stored target equality is diagnostic only.",
        ]
    )
    return "\n".join(lines) + "\n"


async def run_bench(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = parse_normalized_campsite_rows(Path(args.source))
    if args.split == "holdout":
        selected = [(task, target) for task, target in source_rows if stable_holdout(task)]
    elif args.split == "train":
        selected = [(task, target) for task, target in source_rows if not stable_holdout(task)]
    elif args.split == "generated":
        selected = [
            (task, None)
            for task in generate_unseen_tasks(
                args.generated_count,
                seed=args.generated_seed,
                include_size_shift=args.generated_size_shift,
            )
        ]
    else:
        selected = source_rows
    selected = selected[args.offset : args.offset + args.limit] if args.limit > 0 else selected[args.offset :]

    run_id = args.run_id or f"int3-campsite-{args.split}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "created_at": utc_now(),
        "source": args.source,
        "split": args.split,
        "rows": len(selected),
        "variants": args.variants,
        "profiles": args.profiles,
        "endpoint": args.endpoint,
        "model": args.model,
        "context_ceiling": 12_000,
        "packet_hard_limit": args.packet_hard_limit,
        "temperature_gate_c": args.max_temp_c,
        "target_access": "scorer_only",
    }
    write_json(run_dir / "run_manifest.json", manifest)

    events_path = run_dir / "events.jsonl"
    rows_path = run_dir / "rows.jsonl"
    append_jsonl(events_path, {"ts": utc_now(), "event": "run_start", **manifest})
    records: list[dict[str, Any]] = []
    async with mcp_session() as session:
        catalog = await session.read_resource("int3://catalog/campsite/v1")
        append_jsonl(events_path, {"ts": utc_now(), "event": "mcp_catalog", "content": _resource_text(catalog)})
        for index, (task, target) in enumerate(selected):
            for profile in args.profiles:
                for variant in args.variants:
                    started = time.perf_counter()
                    record = await run_episode(
                        session,
                        task,
                        target,
                        variant=variant,
                        profile=profile,
                        order_seed=args.order_seed + index,
                        endpoint=args.endpoint,
                        model=args.model,
                        timeout=args.timeout,
                        max_tokens=args.max_tokens,
                        no_model=args.no_model,
                        production_fallback=args.production_fallback,
                    )
                    record["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                    record["packet_gate_pass"] = record["packet_tokens_est"] <= args.packet_hard_limit
                    records.append(record)
                    append_jsonl(rows_path, record)
                    append_jsonl(events_path, {"ts": utc_now(), "event": "episode", **record})

    pressure = gpu_pressure_snapshot(min_free_mb=args.min_free_mb, max_temp_c=args.max_temp_c)
    summary = summarize(records, pressure=pressure)
    summary["run_id"] = run_id
    hybrid = summary["variants"].get("hybrid", {})
    summary["promotion_ready"] = bool(
        records
        and hybrid.get("schema_valid_rate", 0.0) >= 0.99
        and hybrid.get("action_accuracy", 0.0) >= 0.95
        and hybrid.get("official_semantic_pass_rate", 0.0) >= 0.99
        and hybrid.get("invalid_commit_rate", 1.0) == 0.0
        and all(row["packet_gate_pass"] for row in records)
        and all(not row["final_grid_leak"] for row in records)
        and pressure.get("passed", False)
    )
    write_json(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(render_report(summary, args), encoding="utf-8")
    append_jsonl(events_path, {"ts": utc_now(), "event": "run_complete", "summary": summary})
    print(json.dumps({"run_dir": str(run_dir.resolve()), "summary": summary}, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real MCP Intellect-3 Campsite Bonsai bench.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--split", choices=("holdout", "train", "all", "generated"), default="holdout")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--generated-count", type=int, default=200)
    parser.add_argument("--generated-seed", type=int, default=73_000)
    parser.add_argument("--generated-size-shift", type=int, default=50)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["normal", "no_pass"])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--packet-hard-limit", type=int, default=2500)
    parser.add_argument("--min-free-mb", type=int, default=512)
    parser.add_argument("--max-temp-c", type=int, default=87)
    parser.add_argument("--order-seed", type=int, default=0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-model", action="store_true")
    parser.add_argument("--production-fallback", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    asyncio.run(run_bench(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
