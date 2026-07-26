"""Seal the exact BitAgent/Hermes deterministic sidecar interface."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agent.bitagent_control_mesh_v0 import canonical_hash
from agent.bitagent_sidecar_v1 import OPERATIONS, load_sidecar_config


SCHEMA = "hermes.bitagent_sidecar_interface_receipt.v1"
INTERFACE_PATHS = (
    "configs/bitagent_hermes_sidecar_v1.json",
    "contracts/bitagent_hermes_sidecar_v1.schema.json",
    "docs/bitagent_hermes_sidecar_v1.md",
    "integrations/bitagent/README.md",
    "integrations/bitagent/hermes-role-mesh-sidecar.d.ts",
    "integrations/bitagent/hermes-role-mesh-sidecar.mjs",
    "integrations/bitagent/package.json",
    "src/agent/bitagent_sidecar_receipt_v1.py",
    "src/agent/bitagent_sidecar_v1.py",
    "tests/agent/test_bitagent_sidecar_v1.py",
    "tests/node/bitagent_hermes_stdio_sidecar.test.mjs",
)
REGISTRATION_PATH = (
    "evals/registered/bitagent_hermes_control_mesh_v0/"
    "registration_receipt.json"
)
CONFORMANCE_PATH = (
    "evals/registered/bitagent_hermes_control_mesh_v0/"
    "capability_fingerprint_conformance.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _canonical_lf_sha256(path: Path) -> tuple[str, int]:
    canonical = path.read_bytes()
    if canonical.startswith(b"\xef\xbb\xbf"):
        canonical = canonical[3:]
    canonical = canonical.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(canonical).hexdigest(), len(canonical)


def build_interface_receipt(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    config = load_sidecar_config(
        root / "configs/bitagent_hermes_sidecar_v1.json"
    )
    registration = _read_json(root / REGISTRATION_PATH)
    conformance = _read_json(root / CONFORMANCE_PATH)
    if registration.get("split_overlap_count") != 0:
        raise ValueError("control-mesh registration contains split overlap")
    if conformance.get("status") != "passed" or not conformance.get(
        "agreement"
    ):
        raise ValueError("capability fingerprint conformance is not passing")

    source_files = {}
    for relative in INTERFACE_PATHS:
        digest, canonical_bytes = _canonical_lf_sha256(root / relative)
        source_files[relative] = {
            "canonical_lf_sha256": digest,
            "canonical_bytes": canonical_bytes,
        }
    conformance_sha256, _ = _canonical_lf_sha256(root / CONFORMANCE_PATH)
    material = {
        "schema": SCHEMA,
        "study_id": config["study_id"],
        "status": "sealed_deterministic_interface",
        "transport": config["transport"]["kind"],
        "operations": sorted(OPERATIONS),
        "authority": {
            "candidate_only": True,
            "capability_material_is_authorization": False,
            "approval": False,
            "signing": False,
            "broadcast": False,
            "execution": False,
        },
        "control_mesh_registration_id": registration["registration_id"],
        "control_mesh_source_status": registration[
            "source_versioning_status"
        ],
        "capability_fingerprint_conformance_canonical_lf_sha256": (
            conformance_sha256
        ),
        "capability_fingerprint_result": conformance[
            "implementations"
        ]["hermes_python"]["result"],
        "capability_fingerprint_cross_language_agreement": True,
        "source_files": source_files,
        "verification_commands": [
            "python -m pytest tests/agent/test_bitagent_sidecar_v1.py -q",
            "python -m pytest tests/agent -q -k bitagent",
            "node --test tests/node/bitagent_hermes_stdio_sidecar.test.mjs",
        ],
        "claim_scope": config["claim_scope"],
    }
    return {
        **material,
        "receipt_sha256": canonical_hash(material),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = build_interface_receipt(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
