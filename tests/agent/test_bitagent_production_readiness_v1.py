import json
from pathlib import Path


RESULT = Path("reports/bitagent_hermes_production_readiness_v1.json")
REPORT = Path("reports/bitagent_hermes_production_readiness_v1.md")


def _result():
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_readiness_decision_keeps_live_authority_closed():
    result = _result()
    assert result["decision"]["offline_simulation_control_plane"] == "go"
    assert result["decision"]["live_learned_adapter_plane"] == "no_go"
    assert result["decision"]["production_financial_operation"] == "no_go"
    gates = {gate["id"]: gate["status"] for gate in result["gates"]}
    assert gates["node_python_sidecar"] == "ready"
    assert gates["bonsai_role_adapters"] == "blocked"
    assert gates["production_financial_authority"] == "blocked"


def test_readiness_numbers_preserve_registered_negative_controls():
    observations = _result()["observations"]
    assert observations["cross_domain_raw_invalid_acceptance"] == {
        "numerator": 35,
        "denominator": 35,
    }
    assert observations["cross_domain_ldt_invalid_acceptance"] == {
        "numerator": 0,
        "denominator": 35,
    }
    assert observations["cross_domain_identical_recovery"] == {
        "numerator": 0,
        "denominator": 35,
    }
    assert observations["cross_domain_distinct_recovery"] == {
        "numerator": 35,
        "denominator": 35,
    }


def test_next_workorder_cannot_skip_corpus_promotion_gate():
    result = _result()
    corpus = next(
        gate for gate in result["gates"] if gate["id"] == "bonsai_role_corpus"
    )
    assert corpus["status"] == "blocked"
    assert "717" in corpus["scope"]
    assert "152" in corpus["scope"]
    assert result["next_workorder"]["name"] == "bitagent_bonsai_role_corpus_v2"
    assert "production financial operation" in result[
        "claim_boundary"
    ].lower()


def test_human_report_points_to_machine_decision():
    report = REPORT.read_text(encoding="utf-8")
    assert RESULT.as_posix() in report
    assert "**GO**" in report
    assert report.count("**NO-GO**") == 2
