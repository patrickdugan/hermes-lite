"""Skill discovery and ultra-lean contract loading.

The catalog supports the normal Hermes home plus read-only source roots.  It
does not import the tool registry, which keeps it safe for prompt assembly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator, Optional


ULTRA_LEAN_FILENAME = "ULTRA_LEAN.json"


def _configured_roots() -> list[Path]:
    roots: list[Path] = []
    raw_env = os.getenv("HERMES_SKILLS_DIRS", "")
    if raw_env:
        roots.extend(Path(value).expanduser() for value in raw_env.split(os.pathsep) if value.strip())

    try:
        from hermes_cli.config import load_config

        configured = load_config().get("skills", {}).get("roots", [])
        if isinstance(configured, list):
            roots.extend(Path(str(value)).expanduser() for value in configured if str(value).strip())
    except Exception:
        pass
    return roots


def skill_roots() -> list[Path]:
    """Return existing skill roots in deterministic precedence order."""
    home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes-lite"))
    candidates = [home / "skills", *_configured_roots()]
    roots: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key not in seen and resolved.is_dir():
            roots.append(resolved)
            seen.add(key)
    return roots


def iter_skill_files() -> Iterator[tuple[Path, Path]]:
    """Yield ``(root, SKILL.md)`` pairs, deduplicated by skill name."""
    seen_names: set[str] = set()
    for root in skill_roots():
        for skill_file in sorted(root.rglob("SKILL.md")):
            name = skill_file.parent.name
            if name in seen_names:
                continue
            seen_names.add(name)
            yield root, skill_file


def find_skill(name: str) -> Optional[tuple[Path, Path]]:
    wanted = name.strip()
    for root, skill_file in iter_skill_files():
        if skill_file.parent.name == wanted:
            return root, skill_file
    return None


def load_ultra_lean_contract(skill_file: Path) -> Optional[dict]:
    path = skill_file.parent / ULTRA_LEAN_FILENAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    if value.get("schema") not in {"hermes.ultra_lean_skill.v1", "hermes.ultra_lean_skill.v2"}:
        return None
    if value.get("name") != skill_file.parent.name:
        return None
    return value
