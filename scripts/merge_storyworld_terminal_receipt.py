"""Merge terminal passthrough repairs into a storyworld phase receipt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-events", type=Path, required=True)
    parser.add_argument("--terminal-events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = rows(args.base_events)
    repairs = {row["encounter_id"]: row for row in rows(args.terminal_events)}
    if any(not row.get("terminal_passthrough") or not row.get("parse_ok") or row.get("fallback_used") for row in repairs.values()):
        raise SystemExit("terminal repair rows must be parsed passthroughs without fallback")

    merged = []
    applied = []
    for original in base:
        replacement = repairs.get(original["encounter_id"])
        if replacement is None:
            merged.append(original)
            continue
        if not original.get("fallback_used"):
            raise SystemExit(f"refusing to replace non-fallback row {original['encounter_id']}")
        row = dict(replacement)
        row["repair_provenance"] = {
            "base_parse_ok": original.get("parse_ok"),
            "base_fallback_used": original.get("fallback_used"),
            "repair": "terminal_zero_orx_passthrough",
        }
        merged.append(row)
        applied.append(row["encounter_id"])

    missing = sorted(set(repairs) - {row["encounter_id"] for row in base})
    if missing:
        raise SystemExit(f"terminal repairs absent from base receipt: {missing}")
    if len(applied) != len(repairs):
        raise SystemExit("not every terminal repair replaced a base row")

    generated = [row for row in merged if not row.get("terminal_passthrough")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_events.repaired.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in merged), encoding="utf-8"
    )
    summary = {
        "schema": "hermes.storyworld.phase_acceptance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_events": str(args.base_events.resolve()),
        "terminal_repair_events": str(args.terminal_events.resolve()),
        "rows": len(merged),
        "generated_rows": len(generated),
        "terminal_passthrough_rows": len(merged) - len(generated),
        "parse_passes": sum(row.get("parse_ok") is True for row in merged),
        "model_parse_passes": sum(row.get("model_parse_ok") is True for row in generated),
        "deterministic_syntax_repairs": sum(bool(row.get("repaired_used")) for row in generated),
        "fallback_rows": sum(bool(row.get("fallback_used")) for row in merged),
        "changed_generated_rows": sum(bool((row.get("block_quality") or {}).get("changed_from_target")) for row in generated),
        "max_prompt_estimated_tokens": max(int(row.get("prompt_estimated_tokens", 0)) for row in merged),
        "terminal_repairs_applied": applied,
    }
    summary["passed"] = (
        summary["rows"] == 20
        and summary["parse_passes"] == 20
        and summary["model_parse_passes"] == summary["generated_rows"]
        and summary["fallback_rows"] == 0
        and summary["max_prompt_estimated_tokens"] <= 8000
    )
    (args.output_dir / "summary.repaired.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
