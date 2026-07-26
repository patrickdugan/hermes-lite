"""Registration utilities for the BitAgent/Hermes cross-domain role mesh."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


REGISTRATION_SCHEMA = "hermes.bitagent_cross_domain_role_mesh_registration.v1"
MATRIX_SCHEMA = "hermes.bitagent_cross_domain_role_mesh_case.v1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _resolve(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _query_sha256(case: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(case["task"]))


def materialize_matrix(protocol: dict[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    source = protocol["source"]
    cases = read_jsonl(_resolve(source["cases_path"], repo_root))
    held_levels = set(map(int, source["held_levels"]))
    held_cases = [
        case for case in cases if int(case["complexity"]["level"]) in held_levels
    ]
    conditions = protocol["fault_conditions"]
    rows: list[dict[str, Any]] = []
    for case in held_cases:
        for condition in conditions:
            identity = {
                "task_id": case["task_id"],
                "condition": condition["condition"],
            }
            rows.append(
                {
                    "schema": MATRIX_SCHEMA,
                    "matrix_id": sha256_bytes(canonical_json_bytes(identity)),
                    "task_id": case["task_id"],
                    "query_sha256": _query_sha256(case),
                    "domain": case["domain"],
                    "complexity_level": case["complexity"]["level"],
                    "condition": condition["condition"],
                    "injection_stage": condition["injection_stage"],
                    "registered_expectation": condition["registered_expectation"],
                    "split": "held_perturbation",
                }
            )
    return rows


def _validate_protocol(protocol: dict[str, Any], repo_root: Path) -> None:
    if protocol.get("schema") != "hermes.bitagent_cross_domain_role_mesh_protocol.v1":
        raise ValueError("unexpected protocol schema")
    if len(protocol.get("arms", [])) != 4:
        raise ValueError("exactly four registered arms are required")
    if len(protocol.get("fault_conditions", [])) != 5:
        raise ValueError("exactly five registered fault conditions are required")

    source = protocol["source"]
    for path_key, hash_key in (
        ("cases_path", "cases_sha256"),
        ("contracts_path", "contracts_sha256"),
    ):
        path = _resolve(source[path_key], repo_root)
        if sha256_file(path) != source[hash_key]:
            raise ValueError(f"source hash mismatch: {path}")

    role_interface = protocol["role_interface"]
    role_config = _resolve(role_interface["config_path"], repo_root)
    if sha256_file(role_config) != role_interface["config_sha256"]:
        raise ValueError("role interface config hash mismatch")

    for controller in protocol["controllers"].values():
        if not isinstance(controller, dict) or "path" not in controller:
            continue
        path = _resolve(controller["path"], repo_root)
        if sha256_file(path) != controller["sha256"]:
            raise ValueError(f"controller hash mismatch: {path}")


def register_study(config_path: Path, output_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    protocol = read_json(config_path)
    _validate_protocol(protocol, repo_root)
    matrix = materialize_matrix(protocol, repo_root)
    if len(matrix) != 40:
        raise ValueError(f"expected 40 matrix rows, received {len(matrix)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    matrix_path = output_dir / "matrix.jsonl"
    write_json(protocol_path, protocol)
    write_jsonl(matrix_path, matrix)

    controller_hashes: dict[str, str] = {}
    controller_root = output_dir / "controller_artifacts"
    controller_root.mkdir(parents=True, exist_ok=True)
    for name, controller in protocol["controllers"].items():
        if not isinstance(controller, dict) or "path" not in controller:
            continue
        source_path = _resolve(controller["path"], repo_root)
        suffix = source_path.suffix
        destination = controller_root / f"{name}{suffix}"
        shutil.copyfile(source_path, destination)
        controller_hashes[f"{name}_sha256"] = sha256_file(destination)

    source = protocol["source"]
    role_interface = protocol["role_interface"]
    component_hashes = {
        "protocol_sha256": sha256_file(protocol_path),
        "matrix_sha256": sha256_file(matrix_path),
        "source_cases_sha256": sha256_file(_resolve(source["cases_path"], repo_root)),
        "contracts_sha256": sha256_file(_resolve(source["contracts_path"], repo_root)),
        "role_config_sha256": sha256_file(
            _resolve(role_interface["config_path"], repo_root)
        ),
        **controller_hashes,
    }
    registration_id = sha256_bytes(
        canonical_json_bytes(
            {
                "study_id": protocol["study_id"],
                "component_hashes": component_hashes,
            }
        )
    )
    receipt = {
        "schema": REGISTRATION_SCHEMA,
        "study_id": protocol["study_id"],
        "status": "registered_no_role_mesh_outcomes",
        "registration_id": registration_id,
        "source_task_status": protocol["source"]["source_task_status"],
        "held_source_tasks": len({row["task_id"] for row in matrix}),
        "fault_conditions": len({row["condition"] for row in matrix}),
        "matrix_rows": len(matrix),
        "arms": [arm["arm"] for arm in protocol["arms"]],
        "domains": sorted({row["domain"] for row in matrix}),
        "component_hashes": component_hashes,
        "claim_scope": protocol["claim_scope"],
    }
    write_json(output_dir / "registration_receipt.json", receipt)
    return receipt


def verify_registration(output_dir: Path) -> dict[str, Any]:
    receipt = read_json(output_dir / "registration_receipt.json")
    protocol = read_json(output_dir / "protocol.json")
    repo_root = Path(__file__).resolve().parents[2]
    expected = {
        "protocol_sha256": sha256_file(output_dir / "protocol.json"),
        "matrix_sha256": sha256_file(output_dir / "matrix.jsonl"),
        "source_cases_sha256": sha256_file(
            _resolve(protocol["source"]["cases_path"], repo_root)
        ),
        "contracts_sha256": sha256_file(
            _resolve(protocol["source"]["contracts_path"], repo_root)
        ),
        "role_config_sha256": sha256_file(
            _resolve(protocol["role_interface"]["config_path"], repo_root)
        ),
    }
    for name, controller in protocol["controllers"].items():
        if not isinstance(controller, dict) or "path" not in controller:
            continue
        expected[f"{name}_sha256"] = sha256_file(
            output_dir / "controller_artifacts" / f"{name}{Path(controller['path']).suffix}"
        )
    if expected != receipt["component_hashes"]:
        raise ValueError("registered component hash mismatch")
    registration_id = sha256_bytes(
        canonical_json_bytes(
            {
                "study_id": receipt["study_id"],
                "component_hashes": expected,
            }
        )
    )
    if registration_id != receipt["registration_id"]:
        raise ValueError("registration identity mismatch")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--config", type=Path, required=True)
    register.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--registration-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "register":
        result = register_study(args.config, args.output_dir)
    else:
        result = verify_registration(args.registration_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
