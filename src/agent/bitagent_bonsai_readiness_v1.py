"""Seal the no-training readiness assessment for BitAgent Bonsai role adapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from agent.bitagent_role_adapters import ROLES, load_config, validate_corpus


SCHEMA = "hermes.bitagent_bonsai_role_adapter_readiness.v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot_operator_validation(
    role: str,
    operator_root: Path,
    receipt_root: Path,
    *,
    corpus_sha256: str,
    base_revision: str,
) -> dict[str, Any]:
    source_root = operator_root / f"validate-{role}"
    source_summary = source_root / "summary.json"
    source_manifest = source_root / "run_manifest.json"
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))

    if summary["status"] != "validated":
        raise ValueError(f"{role} did not validate: {summary['status']}")
    if summary["training_started"] or summary["weight_loading_started"]:
        raise ValueError(f"{role} crossed the no-training boundary")
    if manifest["weight_loading_enabled"]:
        raise ValueError(f"{role} enabled weight loading during validation")
    if manifest["corpus"]["examples_sha256"] != corpus_sha256:
        raise ValueError(f"{role} validated a different corpus")
    if manifest["base_revision"] != base_revision:
        raise ValueError(f"{role} validated a different base revision")

    snapshot_root = receipt_root / "operator_validations"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    summary_name = f"{role}.summary.json"
    manifest_name = f"{role}.manifest.json"
    shutil.copyfile(source_summary, snapshot_root / summary_name)
    shutil.copyfile(source_manifest, snapshot_root / manifest_name)

    return {
        "status": "validated",
        "training_started": False,
        "weight_loading_started": False,
        "weight_loading_enabled": False,
        "summary_path": f"operator_validations/{summary_name}",
        "summary_sha256": file_sha256(snapshot_root / summary_name),
        "manifest_path": f"operator_validations/{manifest_name}",
        "manifest_sha256": file_sha256(snapshot_root / manifest_name),
    }


def build_receipt(
    config_path: Path,
    corpus_path: Path,
    operator_root: Path,
    receipt_root: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    validation = validate_corpus(config, corpus_path, promotion=False)
    training = config["training"]
    counts = validation["counts_by_role"]
    split_counts = validation["counts_by_role_and_split"]
    total_floor = training["minimum_promotion_examples_per_role"]
    validation_floor = training["minimum_promotion_validation_examples_per_role"]
    test_floor = training["minimum_promotion_test_examples_per_role"]

    deficits: dict[str, dict[str, int]] = {}
    for role in sorted(ROLES):
        deficits[role] = {
            "total": max(0, total_floor - counts[role]),
            "validation": max(0, validation_floor - split_counts[role].get("validation", 0)),
            "test": max(0, test_floor - split_counts[role].get("test", 0)),
        }

    base_revision = config["models"]["training_base"]["revision"]
    operators = {
        role: _snapshot_operator_validation(
            role,
            operator_root,
            receipt_root,
            corpus_sha256=validation["examples_sha256"],
            base_revision=base_revision,
        )
        for role in sorted(ROLES)
    }

    return {
        "schema": SCHEMA,
        "status": "blocked_before_training",
        "study_id": config["study_id"],
        "claim_scope": (
            "Operator and corpus readiness only. No Bonsai weights were loaded, "
            "no adapter was trained, and no model-efficacy claim is made."
        ),
        "config": {
            "path": config_path.as_posix(),
            "sha256": file_sha256(config_path),
            "base_model": config["models"]["training_base"]["repo"],
            "base_revision": base_revision,
        },
        "corpus": {
            "source_path": str(corpus_path.resolve()),
            "sha256": validation["examples_sha256"],
            "examples": validation["examples"],
            "counts_by_role": counts,
            "counts_by_role_and_split": split_counts,
            "raw_transcripts_included": validation["raw_transcripts_included"],
            "secret_values_detected": validation["secret_values_detected"],
            "unauthorized_effects_detected": validation["unauthorized_effects_detected"],
        },
        "promotion_floors": {
            "total_per_role": total_floor,
            "validation_per_role": validation_floor,
            "test_per_role": test_floor,
        },
        "promotion_deficits": deficits,
        "total_additional_reviewed_examples_required": sum(
            role_deficit["total"] for role_deficit in deficits.values()
        ),
        "minimum_additional_held_examples_required": sum(
            role_deficit["validation"] + role_deficit["test"]
            for role_deficit in deficits.values()
        ),
        "operator_validations": operators,
        "gates": {
            "seed_corpus_contract_valid": True,
            "all_role_operators_validate_without_weights": all(
                item["status"] == "validated"
                and not item["training_started"]
                and not item["weight_loading_started"]
                and not item["weight_loading_enabled"]
                for item in operators.values()
            ),
            "promotion_corpus_ready": all(
                all(value == 0 for value in role_deficit.values())
                for role_deficit in deficits.values()
            ),
            "training_authorized": False,
        },
        "next_gate": (
            "Expand and independently review every role corpus to the registered total "
            "and held-split floors, then rerun this sealer before any capped QLoRA run."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--operator-root", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    args = parser.parse_args()

    args.receipt_root.mkdir(parents=True, exist_ok=True)
    receipt = build_receipt(
        args.config,
        args.corpus,
        args.operator_root,
        args.receipt_root,
    )
    receipt_path = args.receipt_root / "readiness.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
