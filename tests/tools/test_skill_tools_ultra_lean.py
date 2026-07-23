import json

from agent.prompt_builder import build_skills_system_prompt
from tools.skill_tools import skill_view_tool, skills_list_tool


def _make_skill(root, name="trm-test"):
    skill = root / name
    refs = skill / "references"
    refs.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: trm-test\ndescription: Compact TRM test skill\n---\n\nFULL SECRET BODY",
        encoding="utf-8",
    )
    (refs / "large.md").write_text("REFERENCE PAYLOAD " * 1000, encoding="utf-8")
    contract = {
        "schema": "hermes.ultra_lean_skill.v1",
        "name": name,
        "purpose": "Test lean loading",
        "context": {
            "hard_window_tokens": 12000,
            "active_working_set_tokens": 6000,
            "reserve_tokens": 6000,
        },
        "conveyor": {"phases": ["ROUTE", "VERIFY", "COMMIT"]},
        "gates": ["Commit only after verification."],
    }
    (skill / "ULTRA_LEAN.json").write_text(json.dumps(contract), encoding="utf-8")


def test_external_root_is_listed_and_indexed(monkeypatch, tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "catalog"
    _make_skill(root)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SKILLS_DIRS", str(root))

    listing = json.loads(skills_list_tool())
    assert listing["count"] == 1
    assert listing["skills"][0]["ultra_lean"] is True
    assert "trm-test" in build_skills_system_prompt()


def test_auto_mode_excludes_full_skill_and_references(monkeypatch, tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "catalog"
    _make_skill(root)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SKILLS_DIRS", str(root))

    result = skill_view_tool("trm-test")
    payload = json.loads(result)

    assert payload["mode"] == "ultra_lean"
    assert payload["activate"] is True
    assert payload["estimated_tokens"] < 700
    assert "FULL SECRET BODY" not in result
    assert "REFERENCE PAYLOAD" not in result


def test_full_mode_is_explicit_diagnostic_escape_hatch(monkeypatch, tmp_path):
    home = tmp_path / "home"
    root = tmp_path / "catalog"
    _make_skill(root)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SKILLS_DIRS", str(root))

    payload = json.loads(skill_view_tool("trm-test", mode="full"))

    assert payload["mode"] == "full"
    assert "FULL SECRET BODY" in payload["content"]
    assert "REFERENCE PAYLOAD" in next(iter(payload["references"].values()))
