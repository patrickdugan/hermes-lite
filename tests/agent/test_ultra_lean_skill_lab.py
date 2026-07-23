from agent.ultra_lean_skill_lab import build_probe, parse_object, validate_response


CONTRACT = {
    "schema": "hermes.ultra_lean_skill.v1",
    "name": "test-trm",
    "conveyor": {"phases": ["ROUTE", "VERIFY", "COMMIT"]},
    "gates": ["Verify."],
}


def test_probe_stays_small_and_names_first_phase():
    expected, prompt = build_probe(CONTRACT)
    assert expected["phase"] == "ROUTE"
    assert expected["contract_id"] == "test-trm"
    assert len(prompt) < 2000


def test_parser_accepts_fenced_json_but_validator_enforces_contract():
    value = parse_object('```json\n{"phase":"ROUTE","operation":"select","gate":"pending"}\n```')
    expected, _ = build_probe(CONTRACT)
    assert validate_response(value, expected) == []


def test_validator_rejects_extra_keys_and_bad_action():
    expected, _ = build_probe(CONTRACT)
    value = {"phase": expected["phase"], "operation": "solve_everything", "gate": "pending", "extra": True}
    errors = validate_response(value, expected)
    assert "wrong_keys" in errors
    assert "invalid_operation" in errors
