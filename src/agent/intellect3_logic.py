"""Canonical Campsite core for the low-context Intellect-3 MCP harness.

The runtime never needs a stored target solution.  It proposes candidate grids,
checks them against the Prime Campsite semantics, and exposes only compact
candidate summaries to the coordinator model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PACKET_SCHEMA = "intellect3_campsite_mcp_packet_v2"
ACTION_SCHEMA = "intellect3_campsite_mcp_action_v2"
ACTION_KEYS = (
    "schema",
    "task_id",
    "phase",
    "action",
    "module_id",
    "candidate_id",
    "repair_class",
    "reason_code",
    "visible_output_allowed",
)
PHASES = {"route", "candidate_review", "repair_review"}
ACTIONS = {"invoke_module", "commit_candidate", "repair_candidate", "abstain"}
MODULES = {
    "propose",
    "verify",
    "repair",
    "commit",
}
REPAIR_CLASSES = {
    "none",
    "syntax_valid",
    "shape_match",
    "trees_unchanged",
    "row_counts_match",
    "col_counts_match",
    "no_tent_touching",
    "perfect_tree_matching",
    "no_candidate",
}


def _copy_grid(grid: Sequence[Sequence[str]]) -> list[list[str]]:
    return [list(row) for row in grid]


def _neighbors4(r: int, c: int, n: int, m: int) -> Iterable[tuple[int, int]]:
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < n and 0 <= cc < m:
            yield rr, cc


def _neighbors8(r: int, c: int, n: int, m: int) -> Iterable[tuple[int, int]]:
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < m:
                yield rr, cc


def puzzle_hash(grid: Sequence[Sequence[str]], rows: Sequence[int], cols: Sequence[int]) -> str:
    payload = json.dumps(
        {"grid": grid, "row_constraints": rows, "col_constraints": cols},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CampsiteTask:
    task_id: str
    grid: list[list[str]]
    row_constraints: list[int]
    col_constraints: list[int]

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CampsiteTask":
        task = cls(
            task_id=str(payload.get("task_id") or payload.get("row_id") or "task"),
            grid=_copy_grid(payload["grid"]),
            row_constraints=[int(v) for v in payload["row_constraints"]],
            col_constraints=[int(v) for v in payload["col_constraints"]],
        )
        task.validate()
        return task

    def validate(self) -> None:
        if not self.grid or not self.grid[0]:
            raise ValueError("grid must be non-empty")
        width = len(self.grid[0])
        if any(len(row) != width for row in self.grid):
            raise ValueError("grid must be rectangular")
        if any(cell not in {"T", "X"} for row in self.grid for cell in row):
            raise ValueError("input grid may contain only T and X")
        if len(self.row_constraints) != len(self.grid):
            raise ValueError("row constraint count does not match grid")
        if len(self.col_constraints) != width:
            raise ValueError("column constraint count does not match grid")
        if sum(self.row_constraints) != sum(self.col_constraints):
            raise ValueError("row and column tent totals disagree")
        if sum(self.row_constraints) != sum(cell == "T" for row in self.grid for cell in row):
            raise ValueError("tent total must equal tree total")

    @property
    def hash(self) -> str:
        return puzzle_hash(self.grid, self.row_constraints, self.col_constraints)

    def runtime_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "puzzle_hash": self.hash,
            "grid": self.grid,
            "row_constraints": self.row_constraints,
            "col_constraints": self.col_constraints,
        }


def _perfect_matching(grid: Sequence[Sequence[str]]) -> tuple[bool, dict[tuple[int, int], tuple[int, int]]]:
    n = len(grid)
    m = len(grid[0]) if n else 0
    tents = [(r, c) for r in range(n) for c in range(m) if grid[r][c] == "C"]
    trees = {(r, c) for r in range(n) for c in range(m) if grid[r][c] == "T"}
    if len(tents) != len(trees):
        return False, {}

    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for tent in tents:
        r, c = tent
        edges[tent] = [coord for coord in _neighbors4(r, c, n, m) if coord in trees]
        if not edges[tent]:
            return False, {}

    tree_to_tent: dict[tuple[int, int], tuple[int, int]] = {}

    def augment(tent: tuple[int, int], seen: set[tuple[int, int]]) -> bool:
        for tree in edges[tent]:
            if tree in seen:
                continue
            seen.add(tree)
            if tree not in tree_to_tent or augment(tree_to_tent[tree], seen):
                tree_to_tent[tree] = tent
                return True
        return False

    if not all(augment(tent, set()) for tent in tents):
        return False, {}
    return len(tree_to_tent) == len(trees), {tent: tree for tree, tent in tree_to_tent.items()}


def verify_candidate(task: CampsiteTask, candidate: Any) -> dict[str, Any]:
    gates = {
        "syntax_valid": False,
        "shape_match": False,
        "trees_unchanged": False,
        "row_counts_match": False,
        "col_counts_match": False,
        "no_tent_touching": False,
        "perfect_tree_matching": False,
    }
    details: dict[str, Any] = {}
    if not isinstance(candidate, list) or not candidate or any(not isinstance(row, list) for row in candidate):
        return _verifier_result(gates, details)
    gates["syntax_valid"] = all(cell in {"T", "X", "C"} for row in candidate for cell in row)
    n = len(task.grid)
    m = len(task.grid[0])
    gates["shape_match"] = len(candidate) == n and all(len(row) == m for row in candidate)
    if not gates["syntax_valid"] or not gates["shape_match"]:
        return _verifier_result(gates, details)

    gates["trees_unchanged"] = all(
        (task.grid[r][c] == "T") == (candidate[r][c] == "T") for r in range(n) for c in range(m)
    )
    row_counts = [sum(cell == "C" for cell in row) for row in candidate]
    col_counts = [sum(candidate[r][c] == "C" for r in range(n)) for c in range(m)]
    details["row_counts"] = row_counts
    details["col_counts"] = col_counts
    gates["row_counts_match"] = row_counts == task.row_constraints
    gates["col_counts_match"] = col_counts == task.col_constraints
    gates["no_tent_touching"] = all(
        not any(candidate[rr][cc] == "C" for rr, cc in _neighbors8(r, c, n, m))
        for r in range(n)
        for c in range(m)
        if candidate[r][c] == "C"
    )
    matching_ok, matching = _perfect_matching(candidate)
    gates["perfect_tree_matching"] = matching_ok
    if matching_ok:
        details["matching"] = [
            {"tent": list(tent), "tree": list(tree)} for tent, tree in sorted(matching.items())
        ]
    return _verifier_result(gates, details)


def _verifier_result(gates: dict[str, bool], details: dict[str, Any]) -> dict[str, Any]:
    failed = [name for name, passed in gates.items() if not passed]
    return {"official_pass": not failed, "failed_gates": failed, "gates": gates, "details": details}


def _eligible_cells(task: CampsiteTask) -> set[tuple[int, int]]:
    n, m = len(task.grid), len(task.grid[0])
    return {
        (r, c)
        for r in range(n)
        for c in range(m)
        if task.grid[r][c] == "X"
        and any(task.grid[rr][cc] == "T" for rr, cc in _neighbors4(r, c, n, m))
    }


def solve_candidates(
    task: CampsiteTask,
    *,
    max_candidates: int = 3,
    max_nodes: int = 100_000,
) -> tuple[list[list[list[str]]], dict[str, Any]]:
    """Enumerate verifier-valid candidates with row/column pruning."""
    n, m = len(task.grid), len(task.grid[0])
    eligible = _eligible_cells(task)
    row_options: list[list[tuple[int, ...]]] = []
    for r, needed in enumerate(task.row_constraints):
        cols = [c for c in range(m) if (r, c) in eligible]
        options = []
        for option in itertools.combinations(cols, needed):
            if any(b - a <= 1 for a, b in zip(option, option[1:])):
                continue
            options.append(option)
        row_options.append(options)

    solutions: list[list[list[str]]] = []
    chosen: list[tuple[int, ...]] = []
    col_counts = [0] * m
    nodes = 0

    def visit(r: int) -> None:
        nonlocal nodes
        if len(solutions) >= max_candidates or nodes >= max_nodes:
            return
        nodes += 1
        if r == n:
            if col_counts != task.col_constraints:
                return
            grid = _copy_grid(task.grid)
            for rr, cols in enumerate(chosen):
                for cc in cols:
                    grid[rr][cc] = "C"
            if verify_candidate(task, grid)["official_pass"]:
                solutions.append(grid)
            return

        previous = set(chosen[-1]) if chosen else set()
        for option in row_options[r]:
            if any(any(abs(c - p) <= 1 for p in previous) for c in option):
                continue
            if any(col_counts[c] + 1 > task.col_constraints[c] for c in option):
                continue
            chosen.append(option)
            for c in option:
                col_counts[c] += 1
            feasible = True
            remaining = n - r - 1
            for c, target in enumerate(task.col_constraints):
                if col_counts[c] > target or col_counts[c] + remaining < target:
                    feasible = False
                    break
            if feasible:
                visit(r + 1)
            for c in option:
                col_counts[c] -= 1
            chosen.pop()
            if len(solutions) >= max_candidates or nodes >= max_nodes:
                return

    visit(0)
    return solutions, {
        "nodes": nodes,
        "max_nodes": max_nodes,
        "truncated": nodes >= max_nodes and len(solutions) < max_candidates,
        "eligible_cells": len(eligible),
        "row_option_counts": [len(v) for v in row_options],
        "solution_count_returned": len(solutions),
    }


def candidate_id(grid: Sequence[Sequence[str]]) -> str:
    raw = json.dumps(grid, separators=(",", ":"))
    return "cand_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def _remove_first_tent(grid: Sequence[Sequence[str]]) -> list[list[str]]:
    out = _copy_grid(grid)
    for r, row in enumerate(out):
        for c, cell in enumerate(row):
            if cell == "C":
                out[r][c] = "X"
                return out
    return out


def _mutate_tree(grid: Sequence[Sequence[str]]) -> list[list[str]]:
    out = _copy_grid(grid)
    for r, row in enumerate(out):
        for c, cell in enumerate(row):
            if cell == "T":
                out[r][c] = "X"
                return out
    return out


def make_candidate_records(
    task: CampsiteTask,
    *,
    max_candidates: int = 3,
    profile: str = "normal",
    order_seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    solutions, meta = solve_candidates(task, max_candidates=max_candidates)
    records: list[dict[str, Any]] = []
    zero = _copy_grid(task.grid)
    records.append(
        {
            "candidate_id": candidate_id(zero),
            "source": "zero_tent_copy",
            "grid": zero,
            "verifier": verify_candidate(task, zero),
        }
    )
    for grid in solutions:
        records.append(
            {
                "candidate_id": candidate_id(grid),
                "source": "deterministic_csp",
                "grid": grid,
                "verifier": verify_candidate(task, grid),
            }
        )
    if solutions and profile in {"pressure", "no_pass"}:
        for source, grid in (
            ("row_col_near_miss", _remove_first_tent(solutions[0])),
            ("tree_mutation", _mutate_tree(solutions[0])),
        ):
            records.append(
                {
                    "candidate_id": candidate_id(grid),
                    "source": source,
                    "grid": grid,
                    "verifier": verify_candidate(task, grid),
                }
            )
    if profile == "no_pass":
        records = [record for record in records if not record["verifier"]["official_pass"]]
    elif profile == "pressure":
        random.Random(order_seed).shuffle(records)
    elif profile != "normal":
        raise ValueError(f"unknown candidate profile: {profile}")
    return records, meta


def compact_candidate(record: dict[str, Any]) -> dict[str, Any]:
    verifier = record["verifier"]
    return {
        "candidate_id": record["candidate_id"],
        "source": record["source"],
        "official_pass": bool(verifier["official_pass"]),
        "failed_gates": list(verifier["failed_gates"]),
    }


def expected_action(task_id: str, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    passing = next((candidate for candidate in candidates if candidate["official_pass"]), None)
    if passing:
        return {
            "schema": ACTION_SCHEMA,
            "task_id": task_id,
            "phase": "candidate_review",
            "action": "commit_candidate",
            "module_id": "commit",
            "candidate_id": passing["candidate_id"],
            "repair_class": "none",
            "reason_code": "official_verifier_pass",
            "visible_output_allowed": False,
        }
    return {
        "schema": ACTION_SCHEMA,
        "task_id": task_id,
        "phase": "candidate_review",
        "action": "repair_candidate",
        "module_id": "repair",
        "candidate_id": candidates[0]["candidate_id"] if candidates else None,
        "repair_class": failure_class(candidates),
        "reason_code": "no_verified_candidate",
        "visible_output_allowed": False,
    }


def failure_class(candidates: Sequence[dict[str, Any]]) -> str:
    failures = [gate for item in candidates for gate in item.get("failed_gates", [])]
    if not candidates:
        return "no_candidate"
    if "syntax_valid" in failures:
        return "syntax_valid"
    if "shape_match" in failures:
        return "shape_match"
    if "trees_unchanged" in failures:
        return "trees_unchanged"
    if "row_counts_match" in failures or "col_counts_match" in failures:
        return "row_counts_match" if "row_counts_match" in failures else "col_counts_match"
    if "no_tent_touching" in failures:
        return "no_tent_touching"
    if "perfect_tree_matching" in failures:
        return "perfect_tree_matching"
    return "no_candidate"


def build_coordinator_packet(
    task: CampsiteTask,
    candidates: Sequence[dict[str, Any]],
    *,
    variant: str,
    replay_hint: str = "",
    trm_proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "packet_version": 2,
        "task": {
            "task_id": task.task_id,
            "family": "campsite",
            "phase": "candidate_review",
            "puzzle_hash": task.hash[:16],
        },
        "state": {"candidates": list(candidates)},
        "variant": variant,
        "output_contract": {
            "schema": ACTION_SCHEMA,
            "keys": list(ACTION_KEYS),
            "actions": sorted(ACTIONS),
            "module_ids": sorted(MODULES),
            "repair_classes": sorted(REPAIR_CLASSES),
            "json_only": True,
            "visible_output_allowed": False,
        },
    }
    if variant in {"ldt", "hybrid"}:
        packet["ldt"] = {
            "allowed_actions": ["commit_candidate"] if any(c["official_pass"] for c in candidates) else ["repair_candidate", "abstain"],
            "rule": "Never commit a candidate unless official_pass is true.",
        }
    if variant in {"trm", "hybrid"} and trm_proposal:
        packet["trm_proposal"] = trm_proposal
    if replay_hint:
        packet["replay_hint"] = replay_hint
    return packet


def validate_action(action: Any, *, task_id: str, candidate_ids: set[str]) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(action, dict):
        return {"valid": False, "errors": ["not_object"]}
    if tuple(action.keys()) != ACTION_KEYS:
        errors.append("wrong_keys_or_order")
    if action.get("schema") != ACTION_SCHEMA:
        errors.append("bad_schema")
    if action.get("task_id") != task_id:
        errors.append("bad_task_id")
    if action.get("phase") not in PHASES:
        errors.append("bad_phase")
    if action.get("action") not in ACTIONS:
        errors.append("bad_action")
    if action.get("module_id") not in MODULES:
        errors.append("bad_module_id")
    if action.get("repair_class") not in REPAIR_CLASSES:
        errors.append("bad_repair_class")
    if action.get("visible_output_allowed") is not False:
        errors.append("visible_output_not_false")
    if action.get("action") in {"commit_candidate", "repair_candidate"} and action.get("candidate_id") not in candidate_ids:
        errors.append("unknown_candidate")
    return {"valid": not errors, "errors": errors}


def parse_normalized_campsite_rows(path: Path) -> list[tuple[CampsiteTask, list[list[str]]]]:
    rows: list[tuple[CampsiteTask, list[list[str]]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            action = json.loads(str(entry["action"]))
            if action.get("task") != "campsite":
                continue
            game = json.loads(str(action["game_data_str"]))
            meta = game.get("metadata") or {}
            task = CampsiteTask.from_payload(
                {
                    "task_id": entry.get("trajectory_id"),
                    "grid": meta["grid"],
                    "row_constraints": meta["row_constraints"],
                    "col_constraints": meta["col_constraints"],
                }
            )
            rows.append((task, _copy_grid(meta["solution"])))
    return rows


def stable_holdout(task: CampsiteTask, ratio: float = 0.25) -> bool:
    split_payload = json.dumps(
        {
            "row_id": task.task_id,
            "grid": task.grid,
            "rows": task.row_constraints,
            "cols": task.col_constraints,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(split_payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < ratio


def generate_unseen_tasks(count: int, *, seed: int = 73_000, include_size_shift: int = 0) -> list[CampsiteTask]:
    """Generate unique official-style puzzles without exposing their solutions."""
    rng = random.Random(seed)
    source_shapes = [(3, 5), (4, 4), (4, 5), (5, 3), (5, 4), (5, 5)]
    tasks: list[CampsiteTask] = []
    seen: set[str] = set()
    attempts = 0
    while len(tasks) < count and attempts < count * 500:
        attempts += 1
        if len(tasks) >= count - include_size_shift:
            n, m = 6, 6
        else:
            n, m = rng.choice(source_shapes)
        tree_count = max(1, int(n * m * 0.2))
        base = [["X" for _ in range(m)] for _ in range(n)]
        positions = [(r, c) for r in range(n) for c in range(m)]
        rng.shuffle(positions)
        for r, c in positions[:tree_count]:
            base[r][c] = "T"
        solved = _assign_one_tent_per_tree(base, rng)
        if solved is None:
            continue
        rows = [sum(cell == "C" for cell in row) for row in solved]
        cols = [sum(solved[r][c] == "C" for r in range(n)) for c in range(m)]
        try:
            task = CampsiteTask.from_payload(
                {"task_id": f"generated_{seed}_{len(tasks):04d}", "grid": base, "row_constraints": rows, "col_constraints": cols}
            )
        except ValueError:
            continue
        if task.hash in seen:
            continue
        candidates, _ = solve_candidates(task, max_candidates=1)
        if not candidates:
            continue
        seen.add(task.hash)
        tasks.append(task)
    if len(tasks) != count:
        raise RuntimeError(f"generated only {len(tasks)} of {count} requested puzzles")
    return tasks


def _assign_one_tent_per_tree(base: list[list[str]], rng: random.Random) -> list[list[str]] | None:
    n, m = len(base), len(base[0])
    trees = [(r, c) for r in range(n) for c in range(m) if base[r][c] == "T"]
    rng.shuffle(trees)
    chosen: set[tuple[int, int]] = set()

    def visit(index: int) -> bool:
        if index == len(trees):
            return True
        r, c = trees[index]
        options = [coord for coord in _neighbors4(r, c, n, m) if base[coord[0]][coord[1]] == "X"]
        rng.shuffle(options)
        for coord in options:
            if coord in chosen:
                continue
            rr, cc = coord
            if any(other in chosen for other in _neighbors8(rr, cc, n, m)):
                continue
            chosen.add(coord)
            if visit(index + 1):
                return True
            chosen.remove(coord)
        return False

    if not visit(0):
        return None
    grid = _copy_grid(base)
    for r, c in chosen:
        grid[r][c] = "C"
    return grid
