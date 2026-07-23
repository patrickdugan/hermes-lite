"""Tests for the MCP TRM/LDT experiment harness."""

import json
from unittest.mock import MagicMock, patch

from agent.mcp_trm_ldt_lab import (
    _hard_pressure_failed,
    build_variant_packet,
    create_run_manifest,
    gpu_preflight,
    gpu_pressure_snapshot,
    run_live_case_probe,
    run_live_smoke,
    run_marathon,
    run_packet_eval,
    train_retrieval_policy,
)


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_create_run_manifest_defaults():
    manifest = create_run_manifest(mcp="demo-mcp", task_id="tool_select")

    assert manifest["mcp"] == "demo-mcp"
    assert manifest["task_id"] == "tool_select"
    assert manifest["packet_budget_tokens"] == 1200
    assert manifest["caps"]["ram_mb"] == 2048
    assert "schema_gate" in manifest["promotion_gates"]
    assert manifest["artifacts"]["current_task"] == ".hermes/trm/current_task.json"


def test_run_packet_eval_scores_required_terms(tmp_path):
    cases = tmp_path / "cases.jsonl"
    _write_jsonl(cases, [
        {
            "task_id": "mcp_001",
            "instruction": "Choose the exact MCP resource reader.",
            "expected": {
                "required_terms": ["mcp_001", "resource_reader"],
                "forbidden_terms": ["broad_search"],
            },
            "replay_candidates": [
                {"task_id": "mcp_001", "score": 0, "passed": False, "failure": "wrong_tool", "output": "resource_reader"}
            ],
            "self_model": {"recent_pass_rate": 0.5},
        }
    ])

    summary = run_packet_eval(cases_path=cases, output_dir=tmp_path / "experiments", mcp="demo-mcp", run_id="run-a")

    assert summary["total_cases"] == 1
    assert summary["passed_cases"] == 1
    assert summary["pass_rate"] == 1.0
    assert summary["promotion_ready"] is True
    events = (tmp_path / "experiments" / "run-a" / "events.jsonl").read_text(encoding="utf-8")
    assert "case_eval" in events
    assert "mcp_001" in events


def test_run_packet_eval_detects_budget_failure(tmp_path):
    cases = tmp_path / "cases.jsonl"
    _write_jsonl(cases, [
        {
            "task_id": "mcp_002",
            "instruction": "x" * 5000,
            "expected": {"required_terms": ["mcp_002"]},
        }
    ])

    summary = run_packet_eval(
        cases_path=cases,
        output_dir=tmp_path / "experiments",
        mcp="demo-mcp",
        budget_tokens=200,
        run_id="run-budget",
    )

    assert summary["total_cases"] == 1
    assert summary["passed_cases"] == 0
    assert summary["failure_counts"]["packet_gate"] == 1


def test_gpu_preflight_parses_nvidia_smi():
    completed = MagicMock()
    completed.stdout = "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 2048, 70, 12, 555.99\n"

    with patch("agent.mcp_trm_ldt_lab.subprocess.run", return_value=completed):
        result = gpu_preflight(min_free_mb=1024, max_temp_c=86)

    assert result["available"] is True
    assert result["passed"] is True
    assert result["memory_total_mb"] == 4096
    assert result["memory_free_mb"] == 2048


def test_gpu_preflight_fails_on_low_free_memory():
    completed = MagicMock()
    completed.stdout = "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 512, 70, 12, 555.99\n"

    with patch("agent.mcp_trm_ldt_lab.subprocess.run", return_value=completed):
        result = gpu_preflight(min_free_mb=1024, max_temp_c=86)

    assert result["passed"] is False
    assert result["memory_free_mb"] == 512


def test_gpu_pressure_snapshot_parses_used_memory():
    completed = MagicMock()
    completed.stdout = "NVIDIA GeForce RTX 3050 Laptop GPU, 4096, 3040, 1056, 57, 0\n"

    with patch("agent.mcp_trm_ldt_lab.subprocess.run", return_value=completed):
        result = gpu_pressure_snapshot(min_free_mb=512, max_temp_c=86)

    assert result["passed"] is True
    assert result["memory_used_mb"] == 3040
    assert result["memory_free_mb"] == 1056
    assert result["memory_used_pct"] > 70


def test_run_packet_eval_pressure_gate_blocks_promotion(tmp_path):
    cases = tmp_path / "cases.jsonl"
    _write_jsonl(cases, [
        {
            "task_id": "mcp_pressure_001",
            "instruction": "Choose the exact MCP resource reader.",
            "expected": {"required_terms": ["mcp_pressure_001"]},
        }
    ])

    with patch(
        "agent.mcp_trm_ldt_lab.gpu_pressure_snapshot",
        return_value={"available": True, "passed": False, "memory_free_mb": 128, "temperature_c": 70},
    ):
        summary = run_packet_eval(
            cases_path=cases,
            output_dir=tmp_path / "experiments",
            mcp="demo-mcp",
            run_id="run-pressure",
            sample_pressure=True,
        )

    assert summary["passed_cases"] == 1
    assert summary["pressure_failures"] == 1
    assert summary["resource_ready"] is False
    assert summary["promotion_ready"] is False


def test_run_live_smoke_records_timings_and_pressure():
    response = {
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"total_tokens": 19},
        "timings": {"prompt_ms": 100.0},
    }
    pressure = {"available": True, "passed": True, "memory_free_mb": 900, "temperature_c": 60}

    with patch("agent.mcp_trm_ldt_lab.gpu_pressure_snapshot", return_value=pressure), \
            patch("agent.mcp_trm_ldt_lab._post_json", return_value=response):
        result = run_live_smoke(base_url="http://127.0.0.1:8801/v1", timeout_s=5)

    assert result["passed"] is True
    assert result["content"] == "ok"
    assert result["usage"]["total_tokens"] == 19
    assert result["pressure_after"]["passed"] is True


def test_run_live_smoke_can_delegate_pressure_to_external_wrapper():
    response = {"choices": [{"message": {"content": "ok"}}]}

    with patch(
        "agent.mcp_trm_ldt_lab.gpu_pressure_snapshot",
        side_effect=AssertionError("in-process pressure probe must not run"),
    ), patch("agent.mcp_trm_ldt_lab._post_json", return_value=response):
        result = run_live_smoke(
            base_url="http://127.0.0.1:8801/v1",
            timeout_s=5,
            sample_pressure=False,
        )

    assert result["passed"] is True
    assert result["pressure_before"]["source"] == "external_wrapper"
    assert result["pressure_after"]["source"] == "external_wrapper"


def test_run_live_case_probe_requires_task_id_line():
    response = {
        "choices": [{"message": {"content": "task_id=story_001 get_context_card page_0042 context_card"}}],
        "usage": {"total_tokens": 32},
        "timings": {"predicted_ms": 200.0},
    }
    pressure = {"available": True, "passed": True, "memory_free_mb": 900, "temperature_c": 60}
    captured = {}

    def fake_post_json(url, payload, *, timeout_s):
        captured["payload"] = payload
        captured["timeout_s"] = timeout_s
        return response

    case = {
        "task_id": "story_001",
        "instruction": "Retrieve page_0042 context.",
        "expected": {"required_terms": ["story_001", "get_context_card", "page_0042", "context_card"]},
    }

    with patch("agent.mcp_trm_ldt_lab.gpu_pressure_snapshot", return_value=pressure), \
            patch("agent.mcp_trm_ldt_lab._post_json", side_effect=fake_post_json):
        result = run_live_case_probe(case=case, packet="replay_hints: get_context_card page_0042 context_card")

    prompt = captured["payload"]["messages"][0]["content"]
    assert result["passed"] is True
    assert captured["payload"]["max_tokens"] == 40
    assert captured["payload"]["stop"] == ["\n", "\r\n"]
    assert captured["timeout_s"] == 90
    assert "task_id=story_001" in prompt
    assert "no JSON" in prompt


def test_build_variant_packet_controls_sections(tmp_path):
    case = {
        "task_id": "story_001",
        "instruction": "Choose context tool.",
        "bench_family": "storyworld_mcp_context",
        "self_model": {"recent_pass_rate": 0.5},
        "replay_candidates": [
            {"task_id": "story_001", "score": 0, "passed": False, "failure": "wrong_tool", "output": "get_context_card"}
        ],
    }
    root = tmp_path / "case"
    task = {"task_id": "story_001", "instruction": "Choose context tool.", "mcp": "demo"}
    (root / ".hermes" / "trm").mkdir(parents=True)
    (root / ".hermes" / "trm" / "current_task.json").write_text(json.dumps(task), encoding="utf-8")
    (root / ".hermes" / "trm" / "self_model.json").write_text(json.dumps(case["self_model"]), encoding="utf-8")
    (root / ".hermes" / "trm" / "replay_candidates.jsonl").write_text(
        json.dumps(case["replay_candidates"][0]) + "\n",
        encoding="utf-8",
    )

    baseline = build_variant_packet(case_root=root, case=case, task=task, variant="baseline", budget_tokens=1200, hard_context_tokens=12000)
    trm = build_variant_packet(case_root=root, case=case, task=task, variant="trm", budget_tokens=1200, hard_context_tokens=12000)
    ldt = build_variant_packet(case_root=root, case=case, task=task, variant="ldt", budget_tokens=1200, hard_context_tokens=12000)
    hybrid = build_variant_packet(case_root=root, case=case, task=task, variant="hybrid", budget_tokens=1200, hard_context_tokens=12000)

    assert "self_model" not in baseline
    assert "replay_hints" not in baseline
    assert "self_model" in trm
    assert "replay_hints" not in trm
    assert "replay_hints" in ldt
    assert "self_model" not in ldt
    assert "self_model" in hybrid
    assert "replay_hints" in hybrid


def test_train_retrieval_policy_writes_weights(tmp_path):
    events = tmp_path / "events.jsonl"
    _write_jsonl(events, [
        {"event": "case_eval", "failure": "packet_gate", "gates": {"packet_gate": {"passed": False}}},
        {"event": "live_case_probe", "passed": False},
    ])
    out = tmp_path / "policy.json"

    policy = train_retrieval_policy([events], out=out)

    assert out.exists()
    assert policy["event_count"] == 2
    assert policy["weights"]["packet_too_large"] > 2.0
    assert policy["weights"]["live_model_wrong_tool"] > 2.5


def test_hard_pressure_temperature_gate_is_inclusive():
    pressure = {"available": True, "memory_free_mb": 512, "temperature_c": 87}

    assert _hard_pressure_failed(pressure, hard_stop_free_mb=256, hard_stop_temp_c=87) == ""


def test_run_marathon_dry_run_writes_artifacts(tmp_path):
    cases = tmp_path / "cases.jsonl"
    _write_jsonl(cases, [
        {
            "task_id": "story_marathon_001",
            "instruction": "Choose the exact MCP context reader.",
            "bench_family": "storyworld_mcp_context",
            "expected": {"required_terms": ["story_marathon_001", "get_context_card"]},
            "replay_candidates": [
                {"task_id": "story_marathon_001", "score": 0, "passed": False, "failure": "wrong_tool", "output": "get_context_card"}
            ],
        }
    ])
    pressure = {"available": True, "passed": True, "memory_free_mb": 900, "temperature_c": 60}

    with patch("agent.mcp_trm_ldt_lab.gpu_pressure_snapshot", return_value=pressure):
        summary = run_marathon(
            cases_path=cases,
            output_dir=tmp_path / "marathons",
            run_id="dry-marathon",
            duration_minutes=10,
            dry_run=True,
        )

    run_dir = tmp_path / "marathons" / "dry-marathon"
    assert summary["status"] == "completed"
    assert summary["blocks_completed"] == 1
    assert (run_dir / "leaderboard.csv").exists()
    assert (run_dir / "retrieval_policy.json").exists()
    assert (run_dir / "report.md").exists()
