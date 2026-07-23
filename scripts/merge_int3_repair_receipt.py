"""Merge verified Intellect-3 repair rows into a provenance-preserving receipt."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base / "rows.jsonl")
    repair_rows = read_jsonl(args.repair / "rows.jsonl")
    repairs = {row["row_id"]: row for row in repair_rows}
    if not repairs:
        raise SystemExit("repair receipt contains no rows")
    if any(not row.get("official_semantic_pass") or row.get("fallback_used") for row in repair_rows):
        raise SystemExit("every repair row must pass official verification without fallback")

    merged: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for original in base_rows:
        replacement = repairs.get(original["row_id"])
        if replacement is None:
            merged.append(original)
            continue
        if original.get("official_semantic_pass"):
            raise SystemExit(f"refusing to replace already-passing row {original['row_id']}")
        row = dict(replacement)
        row["repair_provenance"] = {
            "base_episode_task_id": original.get("episode_task_id"),
            "repair_episode_task_id": replacement.get("episode_task_id"),
            "base_failure_gates": original.get("failed_gates", []),
            "base_completion_tokens": original.get("completion_tokens"),
            "repair_completion_tokens": replacement.get("completion_tokens"),
        }
        merged.append(row)
        applied.append({"row_id": row["row_id"], **row["repair_provenance"]})

    unknown = sorted(set(repairs) - {row["row_id"] for row in base_rows})
    if unknown:
        raise SystemExit(f"repair rows absent from base receipt: {unknown}")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "rows.repaired.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in merged), encoding="utf-8"
    )
    passed = sum(bool(row.get("official_semantic_pass")) for row in merged)
    summary = {
        "schema": "hermes.intellect3.repaired_acceptance.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_receipt": str(args.base.resolve()),
        "repair_receipt": str(args.repair.resolve()),
        "episodes": len(merged),
        "official_semantic_passes": passed,
        "official_semantic_pass_rate": passed / len(merged),
        "fallback_rate": sum(bool(row.get("fallback_used")) for row in merged) / len(merged),
        "final_grid_leak_rate": sum(bool(row.get("final_grid_leak")) for row in merged) / len(merged),
        "max_packet_tokens_est": max(int(row.get("packet_tokens_est", 0)) for row in merged),
        "repairs_applied": applied,
        "passed": passed == len(merged),
    }
    (args.output / "summary.repaired.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
