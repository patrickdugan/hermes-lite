import json
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from agent.intellect3_logic import (
    ACTION_KEYS,
    ACTION_SCHEMA,
    CampsiteTask,
    build_coordinator_packet,
    expected_action,
    generate_unseen_tasks,
    parse_normalized_campsite_rows,
    solve_candidates,
    stable_holdout,
    validate_action,
    verify_candidate,
)
from agent.intellect3_logic_mcp import CampsiteStore


SOURCE = Path(os.environ.get("HERMES_INTELLECT3_SOURCE", "data/intellect_3_logic.jsonl"))
PRIME_VERIFIER = Path(
    os.environ.get(
        "HERMES_INTELLECT3_VERIFIER",
        "vendor/prime-intellect/campsite_verifier.py",
    )
)


def simple_task() -> CampsiteTask:
    return CampsiteTask.from_payload(
        {
            "task_id": "simple",
            "grid": [["T", "X"], ["X", "X"]],
            "row_constraints": [1, 0],
            "col_constraints": [0, 1],
        }
    )


def test_solver_produces_official_valid_candidate():
    task = simple_task()
    candidates, meta = solve_candidates(task, max_candidates=2)

    assert len(candidates) == 1
    assert verify_candidate(task, candidates[0])["official_pass"] is True
    assert meta["nodes"] <= 4


def test_perfect_matching_rejects_hall_counterexample():
    original = [
        ["T", "X", "T", "X"],
        ["X", "X", "X", "X"],
        ["X", "T", "X", "X"],
        ["X", "X", "X", "X"],
    ]
    candidate = [
        ["T", "C", "T", "X"],
        ["X", "X", "X", "X"],
        ["C", "T", "C", "X"],
        ["X", "X", "X", "X"],
    ]
    task = CampsiteTask.from_payload(
        {
            "task_id": "hall",
            "grid": original,
            "row_constraints": [1, 0, 2, 0],
            "col_constraints": [1, 1, 1, 0],
        }
    )

    report = verify_candidate(task, candidate)

    assert report["gates"]["row_counts_match"] is True
    assert report["gates"]["col_counts_match"] is True
    assert report["gates"]["no_tent_touching"] is True
    assert report["gates"]["perfect_tree_matching"] is False
    assert report["official_pass"] is False


def test_compact_packet_contains_no_grid():
    task = simple_task()
    solutions, _ = solve_candidates(task, max_candidates=1)
    candidates = [
        {
            "candidate_id": "cand_valid",
            "source": "deterministic_csp",
            "official_pass": True,
            "failed_gates": [],
        }
    ]
    packet = build_coordinator_packet(
        task,
        candidates,
        variant="hybrid",
        replay_hint="Commit the verified candidate.",
        trm_proposal={"action": "commit_candidate", "candidate_id": "cand_valid"},
    )

    text = json.dumps(packet)
    assert solutions
    assert '"grid"' not in text
    assert len(text) // 4 < 500


def test_v2_action_schema_requires_exact_key_order():
    candidates = [{"candidate_id": "cand_valid", "official_pass": True, "failed_gates": []}]
    action = expected_action("row", candidates)

    assert tuple(action) == ACTION_KEYS
    assert action["schema"] == ACTION_SCHEMA
    assert validate_action(action, task_id="row", candidate_ids={"cand_valid"})["valid"] is True

    reordered = dict(reversed(list(action.items())))
    assert validate_action(reordered, task_id="row", candidate_ids={"cand_valid"})["valid"] is False


def test_mcp_store_discards_target_and_repairs_no_pass_bank():
    task = simple_task()
    store = CampsiteStore()
    payload = {**task.runtime_payload(), "solution": [["T", "C"], ["X", "X"]]}
    opened = store.open_task(payload)
    state = store.propose("simple", candidate_profile="no_pass")

    assert opened["target_retained"] is False
    assert state["candidates"]
    assert not any(item["official_pass"] for item in state["candidates"])

    candidate = state["candidates"][0]
    repaired = store.repair("simple", candidate["candidate_id"], "row_counts_match")
    committed = store.commit("simple", repaired["candidate"]["candidate_id"])
    assert committed["committed"] is True


def test_unseen_generator_is_deterministic_and_solvable():
    first = generate_unseen_tasks(4, seed=9001, include_size_shift=1)
    second = generate_unseen_tasks(4, seed=9001, include_size_shift=1)

    assert [task.hash for task in first] == [task.hash for task in second]
    assert len({task.hash for task in first}) == 4
    assert all(solve_candidates(task, max_candidates=1)[0] for task in first)


def test_tiny_recursive_policy_stays_below_parameter_cap():
    pytest.importorskip("torch")
    from agent.intellect3_policy_trm import TinyRecursivePolicy, build_frames

    model = TinyRecursivePolicy()
    frames = build_frames([simple_task()], seeds=1)

    assert model.manifest()["parameter_count"] < 50_000
    assert len(frames) == 2
    assert {frame["profile"] for frame in frames} == {"pressure", "no_pass"}


def test_frozen_source_split_is_80_29_when_source_is_available():
    if not SOURCE.exists():
        return
    rows = parse_normalized_campsite_rows(SOURCE)
    holdout = [task for task, _ in rows if stable_holdout(task)]

    assert len(rows) == 109
    assert len(holdout) == 29


def test_verifier_matches_prime_source_on_all_source_candidates():
    if not SOURCE.exists() or not PRIME_VERIFIER.exists():
        return

    verifier_module = types.ModuleType("logic_env.base.verifier")
    verifier_module.Verifier = object
    data_module = types.ModuleType("logic_env.base.data")
    data_module.Data = object
    saved = {name: sys.modules.get(name) for name in ("logic_env", "logic_env.base", "logic_env.base.verifier", "logic_env.base.data")}
    sys.modules["logic_env"] = types.ModuleType("logic_env")
    sys.modules["logic_env.base"] = types.ModuleType("logic_env.base")
    sys.modules["logic_env.base.verifier"] = verifier_module
    sys.modules["logic_env.base.data"] = data_module
    try:
        spec = importlib.util.spec_from_file_location("prime_campsite_verifier", PRIME_VERIFIER)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prime = module.CampsiteVerifier()

        for task, target in parse_normalized_campsite_rows(SOURCE):
            candidates, _ = solve_candidates(task, max_candidates=1)
            for candidate in [target, *candidates]:
                prime_pass = all(
                    (
                        prime._check_trees_unchanged(task.grid, candidate),
                        prime._check_row_constraints(candidate, task.row_constraints),
                        prime._check_col_constraints(candidate, task.col_constraints),
                        prime._check_tents_not_adjacent(candidate),
                        prime._check_tent_tree_matching(candidate),
                    )
                )
                assert verify_candidate(task, candidate)["official_pass"] is prime_pass
    finally:
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
