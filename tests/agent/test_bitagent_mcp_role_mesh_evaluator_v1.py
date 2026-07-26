import json
from pathlib import Path

from agent.bitagent_mcp_role_mesh_v1 import (
    DOMAIN_ORDER,
    _candidate_from_resources,
    inject_fault,
    validate_candidate,
    verify_evaluator,
    verify_role_receipts,
)


ROOT = Path("evals/registered/bitagent_hermes_cross_domain_role_mesh_v1")


def _resource(domain, skill, kind, action, ordinal, *, stale=False):
    version = "v0" if stale else "v1"
    provenance = "legacy_unattested" if stale else "attested"
    prefix = "Former action token" if stale else "Action token"
    return {
        "resource_id": f"{domain}.{skill}.{'legacy' if stale else 'current'}",
        "domain": domain,
        "skill_id": skill,
        "resource_type": kind,
        "conflict_group": f"{domain}.{skill}",
        "version": version,
        "provenance": provenance,
        "stale": stale,
        "catalog_ordinal": ordinal,
        "content": f"{prefix}: {action}.",
    }


def _case(domain, level=5):
    first = _resource(domain, "inspect", "schema", f"inspect_{domain}", 1)
    first_old = _resource(
        domain,
        "inspect",
        "schema",
        f"legacy_inspect_{domain}",
        0,
        stale=True,
    )
    second = _resource(domain, "verify", "validator", f"verify_{domain}", 3)
    second_old = _resource(
        domain,
        "verify",
        "validator",
        f"legacy_verify_{domain}",
        2,
        stale=True,
    )
    return {
        "task_id": f"{domain}.l{level:02d}",
        "domain": domain,
        "complexity": {"level": level},
        "task": {
            "task_id": f"{domain}.l{level:02d}",
            "domain": domain,
            "instruction": f"Inspect and verify {domain}.",
            "interface_contract": {
                "require_attested": True,
                "required_resource_types": ["schema", "validator"],
            },
        },
        "resources": [first_old, first, second_old, second],
        "expected": {
            "required_resource_ids": [first["resource_id"], second["resource_id"]],
            "action_sequence": [f"inspect_{domain}", f"verify_{domain}"],
        },
    }


def test_faults_are_distinct_and_typed_validation_rejects_each_pre_ldt_fault():
    cases = [_case(domain) for domain in DOMAIN_ORDER]
    case = cases[0]
    resources = [case["resources"][1], case["resources"][3]]
    clean = _candidate_from_resources(case, resources)
    assert validate_candidate(case, clean) == (True, [])
    for condition in (
        "stale_resource_swap",
        "wrong_domain_resource",
        "missing_required_step",
    ):
        candidate = inject_fault(case, clean, condition, cases)
        valid, reasons = validate_candidate(case, candidate)
        assert valid is False
        assert reasons


def test_post_ldt_condition_does_not_mutate_before_authorization():
    cases = [_case(domain) for domain in DOMAIN_ORDER]
    case = cases[0]
    clean = _candidate_from_resources(case, [case["resources"][1], case["resources"][3]])
    assert inject_fault(case, clean, "post_ldt_mutation", cases) == clean


def test_evaluator_addendum_and_receipt_chain_verify():
    addendum = verify_evaluator(ROOT)
    assert addendum["status"] == "frozen_no_role_mesh_outcomes"
    assert addendum["resource_execution"]["gpu"] == "disabled"
    assert verify_role_receipts([]) is True
