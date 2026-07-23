"""Operational CLI for the Hermes ultra-lean runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.lean_contracts import phase_projection
from agent.lean_state import safe_id
from hermes_cli.config import load_config


def make_agent(runtime_mode: str = "lean"):
    import os

    os.environ["HERMES_RUNTIME_MODE"] = runtime_mode
    from hermes_agent import HermesLiteAgent

    config = load_config()
    model = config.get("model", {})
    return HermesLiteAgent(
        base_url=model.get("base_url", "http://127.0.0.1:8801/v1"),
        api_key="local",
        provider=model.get("provider", "local"),
        model=model.get("default", "local/bonsai-8b"),
        quiet_mode=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hermes-lite-lean")
    sub = parser.add_subparsers(dest="command", required=True)
    route = sub.add_parser("route")
    route.add_argument("query")
    preflight = sub.add_parser("preflight")
    preflight.add_argument("query")
    run = sub.add_parser("run")
    run.add_argument("query")
    run.add_argument("--task-id")
    resume = sub.add_parser("resume")
    resume.add_argument("task_id")
    resume.add_argument("--query", default="Continue from the durable lean snapshot.")
    status = sub.add_parser("status")
    status.add_argument("task_id")
    args = parser.parse_args(argv)

    if args.command == "status":
        config = load_config()
        root = Path(config.get("runtime", {}).get("lean", {}).get("state_dir", ".hermes/runtime")) / safe_id(args.task_id)
        snapshot = root / "snapshot.json"
        if not snapshot.exists():
            print(json.dumps({"error": "task not found", "task_id": args.task_id}))
            return 1
        print(snapshot.read_text(encoding="utf-8"))
        return 0

    agent = make_agent()
    runtime = agent._lean_runtime
    if runtime is None:
        raise SystemExit("Lean runtime failed to initialize.")
    if args.command == "route":
        contract, overlays, receipt = runtime.route(args.query)
        print(json.dumps({"contract_id": contract.get("name"), "overlays": [item.get("name") for item in overlays], "route": receipt}, indent=2))
        return 0
    if args.command == "preflight":
        contract, overlays, route_receipt = runtime.route(args.query)
        phase = phase_projection(contract, 0, overlays)
        _, budget = runtime.packet_builder.build(task_card={"instruction": args.query}, phase=phase, state={"phase_index": 0}, route=route_receipt)
        print(json.dumps({"contract_id": contract.get("name"), "phase": phase.get("phase", {}).get("id"), "budget": budget, "passed": budget["input_tokens"] <= runtime.max_input_tokens}, indent=2))
        return 0
    task_id = args.task_id if args.command == "run" else args.task_id
    query = args.query
    result = runtime.run(query, task_id=task_id)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
