"""FastMCP server exposing bounded Campsite proposal and verification modules."""

from __future__ import annotations

import json
from collections import OrderedDict
from typing import Any

from mcp.server.fastmcp import FastMCP

from agent.intellect3_logic import (
    CampsiteTask,
    compact_candidate,
    make_candidate_records,
    solve_candidates,
    verify_candidate,
)


CATALOG = {
    "version": 1,
    "family": "campsite",
    "modules": [
        {"module_id": "campsite.propose", "purpose": "Generate bounded CSP candidates."},
        {"module_id": "campsite.verify.prime", "purpose": "Apply the official perfect-matching verifier."},
        {"module_id": "campsite.repair", "purpose": "Replace a rejected candidate with a verified CSP candidate."},
        {"module_id": "campsite.commit", "purpose": "Emit a grid only after a fresh verifier pass."},
    ],
    "rule": "Only campsite.commit may reveal a final grid, and only after official_pass=true.",
}

REPLAY_HINTS = {
    "syntax_valid": "Use action=repair_candidate, module_id=repair, repair_class=syntax_valid.",
    "shape_match": "Use action=repair_candidate, module_id=repair, repair_class=shape_match.",
    "trees_unchanged": "Use action=repair_candidate, module_id=repair, repair_class=trees_unchanged.",
    "row_counts_match": "Use action=repair_candidate, module_id=repair, repair_class=row_counts_match.",
    "col_counts_match": "Use action=repair_candidate, module_id=repair, repair_class=col_counts_match.",
    "no_tent_touching": "Use action=repair_candidate, module_id=repair, repair_class=no_tent_touching.",
    "perfect_tree_matching": "Use action=repair_candidate, module_id=repair, repair_class=perfect_tree_matching.",
    "no_candidate": "Use action=invoke_module, module_id=propose, then abstain if the bank remains empty.",
    "none": "Commit the first candidate with official_pass=true.",
}


class CampsiteStore:
    """Small bounded state store; target solutions are never accepted or retained."""

    def __init__(self, max_tasks: int = 64):
        self.max_tasks = max_tasks
        self.tasks: OrderedDict[str, CampsiteTask] = OrderedDict()
        self.candidates: dict[str, OrderedDict[str, dict[str, Any]]] = {}
        self.solver_meta: dict[str, dict[str, Any]] = {}

    def open_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        sanitized = {key: value for key, value in payload.items() if key != "solution"}
        task = CampsiteTask.from_payload(sanitized)
        self.tasks[task.task_id] = task
        self.tasks.move_to_end(task.task_id)
        self.candidates[task.task_id] = OrderedDict()
        self.solver_meta.pop(task.task_id, None)
        while len(self.tasks) > self.max_tasks:
            evicted, _ = self.tasks.popitem(last=False)
            self.candidates.pop(evicted, None)
            self.solver_meta.pop(evicted, None)
        return {
            "task_id": task.task_id,
            "puzzle_hash": task.hash,
            "phase": "route",
            "candidate_count": 0,
            "target_retained": False,
        }

    def propose(
        self,
        task_id: str,
        max_candidates: int = 3,
        candidate_profile: str = "normal",
        order_seed: int = 0,
    ) -> dict[str, Any]:
        task = self._task(task_id)
        records, meta = make_candidate_records(
            task,
            max_candidates=max(1, min(max_candidates, 5)),
            profile=candidate_profile,
            order_seed=order_seed,
        )
        bank = OrderedDict((record["candidate_id"], record) for record in records)
        self.candidates[task_id] = bank
        self.solver_meta[task_id] = meta
        return self.state(task_id)

    def verify(self, task_id: str) -> dict[str, Any]:
        task = self._task(task_id)
        for record in self.candidates.get(task_id, {}).values():
            record["verifier"] = verify_candidate(task, record["grid"])
        return self.state(task_id)

    def repair(self, task_id: str, candidate_id: str, repair_class: str) -> dict[str, Any]:
        task = self._task(task_id)
        if candidate_id not in self.candidates.get(task_id, {}):
            raise ValueError("unknown candidate_id")
        solutions, meta = solve_candidates(task, max_candidates=1, max_nodes=200_000)
        if not solutions:
            return {"task_id": task_id, "phase": "repair_review", "repaired": False, "repair_class": repair_class}
        grid = solutions[0]
        record = {
            "candidate_id": "repair_" + task.hash[:12],
            "source": f"deterministic_repair:{repair_class}",
            "grid": grid,
            "verifier": verify_candidate(task, grid),
        }
        self.candidates[task_id][record["candidate_id"]] = record
        self.solver_meta[task_id] = meta
        return {
            "task_id": task_id,
            "phase": "repair_review",
            "repaired": True,
            "candidate": compact_candidate(record),
        }

    def commit(self, task_id: str, candidate_id: str) -> dict[str, Any]:
        task = self._task(task_id)
        record = self.candidates.get(task_id, {}).get(candidate_id)
        if record is None:
            return {"task_id": task_id, "committed": False, "reason": "unknown_candidate"}
        report = verify_candidate(task, record["grid"])
        record["verifier"] = report
        if not report["official_pass"]:
            return {
                "task_id": task_id,
                "committed": False,
                "reason": "official_verifier_reject",
                "failed_gates": report["failed_gates"],
            }
        return {
            "task_id": task_id,
            "committed": True,
            "reason": "official_verifier_pass",
            "candidate_id": candidate_id,
            "grid": record["grid"],
        }

    def state(self, task_id: str) -> dict[str, Any]:
        task = self._task(task_id)
        candidates = [compact_candidate(record) for record in self.candidates.get(task_id, {}).values()]
        return {
            "task_id": task_id,
            "puzzle_hash": task.hash[:16],
            "phase": "candidate_review" if candidates else "route",
            "candidates": candidates,
            "solver_meta": self.solver_meta.get(task_id, {}),
        }

    def reset(self, task_id: str) -> dict[str, Any]:
        existed = self.tasks.pop(task_id, None) is not None
        self.candidates.pop(task_id, None)
        self.solver_meta.pop(task_id, None)
        return {"task_id": task_id, "reset": existed}

    def _task(self, task_id: str) -> CampsiteTask:
        task = self.tasks.get(task_id)
        if task is None:
            raise ValueError("unknown task_id")
        self.tasks.move_to_end(task_id)
        return task


store = CampsiteStore()
mcp = FastMCP(
    "intellect3-campsite",
    instructions="Bounded Campsite proposal, official verification, repair, and commit modules for low-context agents.",
    log_level="ERROR",
)


@mcp.resource("int3://catalog/campsite/v1", mime_type="application/json")
def module_catalog() -> str:
    return json.dumps(CATALOG, separators=(",", ":"))


@mcp.resource("int3://task/{task_id}/state", mime_type="application/json")
def task_state(task_id: str) -> str:
    return json.dumps(store.state(task_id), separators=(",", ":"))


@mcp.resource("int3://replay/{failure_class}", mime_type="text/plain")
def replay_hint(failure_class: str) -> str:
    return REPLAY_HINTS.get(failure_class, REPLAY_HINTS["no_candidate"])


@mcp.resource("int3://policy/{variant}", mime_type="application/json")
def policy_resource(variant: str) -> str:
    return json.dumps(
        {
            "variant": variant,
            "max_replay_hints": 1,
            "raw_grid_in_prompt": False,
            "commit_requires": "official_pass",
        },
        separators=(",", ":"),
    )


@mcp.tool(description="Open a Campsite task. Stored target solutions are discarded.")
def open_task(task: dict[str, Any]) -> dict[str, Any]:
    return store.open_task(task)


@mcp.tool(description="Generate bounded Campsite candidates and return opaque summaries only.")
def propose_candidates(
    task_id: str,
    max_candidates: int = 3,
    candidate_profile: str = "normal",
    order_seed: int = 0,
) -> dict[str, Any]:
    return store.propose(
        task_id,
        max_candidates=max_candidates,
        candidate_profile=candidate_profile,
        order_seed=order_seed,
    )


@mcp.tool(description="Re-run the Prime-aligned verifier for every current candidate.")
def verify_candidates(task_id: str) -> dict[str, Any]:
    return store.verify(task_id)


@mcp.tool(description="Repair a rejected candidate using the named failure class.")
def repair_candidate(task_id: str, candidate_id: str, repair_class: str) -> dict[str, Any]:
    return store.repair(task_id, candidate_id, repair_class)


@mcp.tool(description="Return a final grid only after a fresh official verifier pass.")
def commit_candidate(task_id: str, candidate_id: str) -> dict[str, Any]:
    return store.commit(task_id, candidate_id)


@mcp.tool(description="Drop state for one completed task.")
def reset_task(task_id: str) -> dict[str, Any]:
    return store.reset(task_id)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
