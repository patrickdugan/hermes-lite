import json
import os
import subprocess
import sys
from pathlib import Path

from agent.bitagent_role_adapters import ROLES, load_config


CONFIG_PATH = Path("configs/bitagent_bonsai_role_adapters_v1.json")


def _example(role, index, split):
    config = load_config(CONFIG_PATH)
    return {
        "schema": "hermes.bitagent_role_example.v1",
        "id": f"{role}-{index}",
        "role": role,
        "split": split,
        "source": {"path": "fixture", "recordId": str(index), "sha256": "0" * 64},
        "authority": {
            "proposeOnly": True,
            "allowedTools": config["roles"][role]["allowed_tools"],
            "forbiddenEffects": config["authority"]["forbidden_model_effects"],
        },
        "messages": [
            {"role": "system", "content": f"ROLE={role}; candidate only"},
            {"role": "user", "content": json.dumps({"task": f"fixture {index}"})},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "role": role,
                        "action": "collect_or_explain",
                        "execute": False,
                        "truthfulStateRequired": True,
                    }
                ),
            },
        ],
        "tags": ["fixture"],
    }


def test_validate_only_does_not_require_adapter_stack_or_load_weights(tmp_path):
    rows = []
    for role in sorted(ROLES):
        rows.extend(
            _example(role, index, "validation" if index == 0 else "test" if index == 1 else "train")
            for index in range(5)
        )
    corpus = tmp_path / "examples.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    output = tmp_path / "out"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path("src").resolve())
    command = [
        sys.executable,
        "-m",
        "agent.bitagent_qlora_train",
        "--config",
        str(CONFIG_PATH),
        "--examples",
        str(corpus),
        "--output-dir",
        str(output),
        "--training-task-id",
        "validate-only-fixture",
        "--role",
        "intent_planner",
        "--base-model",
        "prism-ml/Bonsai-8B-unpacked",
        "--ram-cap-mb",
        "2048",
        "--cpu-cap-pct",
        "50",
        "--io-cap-mb-s",
        "50",
        "--wall-seconds",
        "900",
        "--min-free-vram-mb",
        "512",
        "--checkpoint-steps",
        "50",
        "--chunk-strategy",
        "one-frozen-task-card-per-batch",
        "--validate-only",
    ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "validated"
    assert summary["training_started"] is False
    assert summary["weight_loading_started"] is False


def test_capped_wrapper_contains_nonnegotiable_cgroup_guards():
    wrapper = Path("scripts/run_capped_bitagent_qlora.sh").read_text()
    for required in (
        "MemoryMax=",
        "MemorySwapMax=0",
        "CPUQuota=",
        "IOReadBandwidthMax=",
        "IOWriteBandwidthMax=",
        "RuntimeMaxSec=",
        "BITAGENT_QLORA_CAP_WRAPPER_ACTIVE=1",
        "OOMPolicy=stop",
    ):
        assert required in wrapper
