#!/usr/bin/env python3
"""
Skill Tools — Browse and load reusable skill definitions.

Skills are stored as SKILL.md files under ~/.hermes-lite/skills/{name}/SKILL.md.
The prompt builder includes a compact index in the system prompt; these tools
let the agent load full skill content on demand and list what's available.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from agent.skill_catalog import find_skill, iter_skill_files, load_ultra_lean_contract, skill_roots

logger = logging.getLogger(__name__)

DEFAULT_HERMES_HOME = os.path.expanduser("~/.hermes-lite")


def _skills_dir() -> Path:
    """Backward-compatible primary skill directory."""
    home = Path(os.getenv("HERMES_HOME", DEFAULT_HERMES_HOME))
    return home / "skills"


def _read_frontmatter(path: Path) -> dict:
    """Extract YAML frontmatter fields from a SKILL.md file."""
    try:
        content = path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)^---\s*\n", content, re.MULTILINE | re.DOTALL)
        if not match:
            return {}
        fm = {}
        for line in match.group(1).splitlines():
            if ":" in line:
                key, _, val = line.partition(":")
                fm[key.strip()] = val.strip()
        return fm
    except Exception:
        return {}


# =============================================================================
# skills_list
# =============================================================================

def skills_list_tool(category: Optional[str] = None) -> str:
    """List available skills with descriptions."""
    roots = skill_roots()
    if not roots:
        return json.dumps({"skills": [], "message": "No skills directory found."})

    skills = []
    for root, skill_file in iter_skill_files():
        name = skill_file.parent.name
        fm = _read_frontmatter(skill_file)
        desc = fm.get("description", "")
        # Truncate long descriptions for the listing
        if len(desc) > 120:
            desc = desc[:117] + "..."
        rel = skill_file.relative_to(root)
        cat = rel.parts[0] if len(rel.parts) > 1 else "general"
        if category and cat != category:
            continue
        skills.append({
            "name": name,
            "category": cat,
            "description": desc,
            "ultra_lean": load_ultra_lean_contract(skill_file) is not None,
        })

    return json.dumps({"skills": skills, "count": len(skills)}, ensure_ascii=False)


SKILLS_LIST_SCHEMA = {
    "name": "skills_list",
    "description": (
        "List available skills. Skills are reusable expertise modules "
        "(e.g. frontend-design, webapp-testing) that provide detailed instructions "
        "for specific task types. Use this to discover what skills are available."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": "Optional category filter.",
            },
        },
        "required": [],
    },
}


# =============================================================================
# skill_view
# =============================================================================

def skill_view_tool(name: str, mode: str = "auto") -> str:
    """Load a skill contract, preferring ultra-lean mode when available."""
    if not name or not name.strip():
        return json.dumps({"error": "Skill name is required."})

    name = name.strip()
    mode = (mode or "auto").strip().lower()
    if mode not in {"auto", "ultra_lean", "full"}:
        return json.dumps({"error": "mode must be auto, ultra_lean, or full"})

    found = find_skill(name)
    if not found:
        available = [p.parent.name for _, p in iter_skill_files()]
        return json.dumps({
            "error": f"Skill '{name}' not found.",
            "available": available,
        })
    _, skill_path = found

    try:
        contract = load_ultra_lean_contract(skill_path)
        if mode == "ultra_lean" and contract is None:
            return json.dumps({"error": f"Skill '{name}' has no valid ULTRA_LEAN.json contract."})
        if contract is not None and mode in {"auto", "ultra_lean"}:
            compact = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return json.dumps({
                "name": name,
                "mode": "ultra_lean",
                "activate": True,
                "estimated_tokens": (len(compact) + 3) // 4,
                "contract": contract,
            }, ensure_ascii=False, separators=(",", ":"))

        content = skill_path.read_text(encoding="utf-8")

        # Also load any reference files in the skill directory
        refs = {}
        for ref_file in skill_path.parent.rglob("*.md"):
            if ref_file.name != "SKILL.md":
                rel = str(ref_file.relative_to(skill_path.parent))
                ref_content = ref_file.read_text(encoding="utf-8")
                # Truncate very large reference files
                if len(ref_content) > 5000:
                    ref_content = ref_content[:5000] + "\n\n[Truncated]"
                refs[rel] = ref_content

        result = {"name": name, "mode": "full", "content": content}
        if refs:
            result["references"] = refs
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Failed to read skill: {e}"})


SKILL_VIEW_SCHEMA = {
    "name": "skill_view",
    "description": (
        "Load a skill's full instructions by name. Call this when you identify "
        "a skill that matches the current task (from the skills index in your "
        "system prompt). The skill content provides detailed instructions, "
        "patterns, and best practices to follow."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the skill to load (e.g. 'frontend-design').",
            },
            "mode": {
                "type": "string",
                "enum": ["auto", "ultra_lean", "full"],
                "description": "auto uses the lean contract when available; full is diagnostic only.",
            },
        },
        "required": ["name"],
    },
}


# =============================================================================
# Checks
# =============================================================================

def check_skills_requirements() -> bool:
    return bool(skill_roots())


# =============================================================================
# Registry
# =============================================================================

from tools.registry import registry

registry.register(
    name="skills_list",
    toolset="skills",
    schema=SKILLS_LIST_SCHEMA,
    handler=lambda args, **kw: skills_list_tool(category=args.get("category")),
    check_fn=check_skills_requirements,
)

registry.register(
    name="skill_view",
    toolset="skills",
    schema=SKILL_VIEW_SCHEMA,
    handler=lambda args, **kw: skill_view_tool(name=args.get("name", ""), mode=args.get("mode", "auto")),
    check_fn=check_skills_requirements,
)
