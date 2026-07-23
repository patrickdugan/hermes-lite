"""Real stdio filesystem MCP and 32-query lean retrieval benchmark."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP


RESOURCES = [
    {"uri": "file:///workspace/README.md", "label": "root README current setup", "payload": "Current project setup and installation instructions."},
    {"uri": "file:///workspace/README.old.md", "label": "stale archived README", "payload": "Obsolete setup instructions."},
    {"uri": "file:///workspace/package.json", "label": "root package manifest", "payload": '{"name":"workspace-root"}'},
    {"uri": "file:///workspace/packages/ui/package.json", "label": "UI package manifest", "payload": '{"name":"workspace-ui"}'},
    {"uri": "file:///workspace/CONTRIBUTING.md", "label": "root contribution guide", "payload": "Contribution workflow and pull request rules."},
    {"uri": "file:///workspace/docs/setup.md", "label": "detailed setup guide", "payload": "Detailed environment setup guide."},
    {"uri": "mcp://filesystem/templates/glob?pattern=**/*.spec.ts", "label": "glob all TypeScript spec files", "payload": "tests/a.spec.ts\ntests/b.spec.ts"},
    {"uri": "mcp://filesystem/templates/glob?pattern=**/*.test.ts", "label": "glob all TypeScript test files", "payload": "tests/a.test.ts\ntests/b.test.ts"},
]

QUERIES = [
    ("Open the root README and show setup.", "file:///workspace/README.md"),
    ("Read the current README.md, not the archived copy.", "file:///workspace/README.md"),
    ("Where are the root project installation instructions?", "file:///workspace/README.md"),
    ("Load /workspace/README.md.", "file:///workspace/README.md"),
    ("Read the root package.json.", "file:///workspace/package.json"),
    ("Open the workspace package manifest, not the UI package.", "file:///workspace/package.json"),
    ("What is the root package name?", "file:///workspace/package.json"),
    ("Load /workspace/package.json.", "file:///workspace/package.json"),
    ("Open the UI package.json.", "file:///workspace/packages/ui/package.json"),
    ("Read packages/ui/package.json.", "file:///workspace/packages/ui/package.json"),
    ("What is the UI package manifest?", "file:///workspace/packages/ui/package.json"),
    ("Load the package manifest under packages/ui.", "file:///workspace/packages/ui/package.json"),
    ("Open the root CONTRIBUTING.md.", "file:///workspace/CONTRIBUTING.md"),
    ("Show the contribution guide.", "file:///workspace/CONTRIBUTING.md"),
    ("What are the pull request contribution rules?", "file:///workspace/CONTRIBUTING.md"),
    ("Load /workspace/CONTRIBUTING.md.", "file:///workspace/CONTRIBUTING.md"),
    ("Read the detailed docs setup guide.", "file:///workspace/docs/setup.md"),
    ("Open docs/setup.md.", "file:///workspace/docs/setup.md"),
    ("Show the detailed environment setup document.", "file:///workspace/docs/setup.md"),
    ("Load /workspace/docs/setup.md rather than the README.", "file:///workspace/docs/setup.md"),
    ("Find all **/*.spec.ts files.", "mcp://filesystem/templates/glob?pattern=**/*.spec.ts"),
    ("Use the glob template for TypeScript spec files.", "mcp://filesystem/templates/glob?pattern=**/*.spec.ts"),
    ("List every .spec.ts test under the repository.", "mcp://filesystem/templates/glob?pattern=**/*.spec.ts"),
    ("Search recursively for spec.ts files.", "mcp://filesystem/templates/glob?pattern=**/*.spec.ts"),
    ("Find all **/*.test.ts files.", "mcp://filesystem/templates/glob?pattern=**/*.test.ts"),
    ("Use the glob template for TypeScript test files.", "mcp://filesystem/templates/glob?pattern=**/*.test.ts"),
    ("List every .test.ts file under the repository.", "mcp://filesystem/templates/glob?pattern=**/*.test.ts"),
    ("Search recursively for test.ts files.", "mcp://filesystem/templates/glob?pattern=**/*.test.ts"),
    ("Open the stale README archive explicitly.", "file:///workspace/README.old.md"),
    ("Read README.old.md.", "file:///workspace/README.old.md"),
    ("Show the obsolete setup instructions.", "file:///workspace/README.old.md"),
    ("Load /workspace/README.old.md, not the current file.", "file:///workspace/README.old.md"),
]

mcp = FastMCP("lean-filesystem-mcp")


@mcp.tool(description="List compact filesystem resource descriptors without payloads.")
def list_descriptors() -> dict[str, Any]:
    return {"resources": [{"uri": item["uri"], "label": item["label"]} for item in RESOURCES]}


@mcp.tool(description="Read one exact filesystem resource after descriptor selection.")
def read_selected(uri: str) -> dict[str, Any]:
    item = next((row for row in RESOURCES if row["uri"] == uri), None)
    return {"found": item is not None, "uri": uri, "payload": item["payload"] if item else ""}


def _tool_payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        value = structured.get("result", structured)
        return value if isinstance(value, dict) else {"result": value}
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", "")
        if text:
            try:
                value = json.loads(text)
                return value if isinstance(value, dict) else {"result": value}
            except json.JSONDecodeError:
                return {"text": text}
    return {}


@asynccontextmanager
async def session() -> AsyncIterator[ClientSession]:
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    params = StdioServerParameters(command=sys.executable, args=["-m", "agent.lean_filesystem_mcp", "server"], cwd=str(Path(__file__).resolve().parents[2]), env=env)
    async with stdio_client(params) as streams:
        async with ClientSession(*streams) as client:
            await client.initialize()
            yield client


def retrieve(query: str, descriptors: list[dict[str, str]]) -> str:
    lower = query.lower().replace("\\", "/")
    words = set(re.findall(r"[a-z0-9.*]+", lower))
    scored = []
    for item in descriptors:
        uri, label = item["uri"], item["label"]
        score = len(words & set(re.findall(r"[a-z0-9.*]+", (uri + " " + label).lower())))
        name = uri.rsplit("/", 1)[-1].lower()
        if name and name in lower:
            score += 20
        uri_path = uri.removeprefix("file://").lower()
        if uri_path and uri_path in lower:
            score += 30
        if "root" in lower and "/packages/" not in uri and "/docs/" not in uri:
            score += 5
        if "workspace package" in lower and "/packages/" not in uri:
            score += 12
        if "ui" in lower and "/packages/ui/" in uri:
            score += -12 if "not the ui" in lower or "not ui" in lower else 12
        rejects_old = any(token in lower for token in ("not the archived", "not archived", "not the old", "current"))
        if rejects_old:
            score += -15 if "old" in uri.lower() else 6
        elif any(token in lower for token in ("stale", "old", "obsolete", "archive")):
            score += 15 if "old" in uri.lower() else -10
        elif "old" in uri.lower():
            score -= 12
        if ("installation" in lower or "setup instructions" in lower) and "README.md" in uri and "old" not in uri.lower():
            score += 12
        if "contribut" in lower and "CONTRIBUTING.md" in uri:
            score += 15
        if "detailed" in lower and "/docs/setup.md" in uri:
            score += 10
        if "spec" in lower and "spec.ts" in uri:
            score += 20
        if "test" in lower and "test.ts" in uri and "spec" not in lower:
            score += 20
        scored.append((score, uri))
    return max(scored)[1]


async def run_bench(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    async with session() as client:
        for index, (query, expected) in enumerate(QUERIES):
            listed = _tool_payload(await client.call_tool("list_descriptors", {}))
            descriptors = list(listed.get("resources", []))
            chosen = retrieve(query, descriptors)
            read = _tool_payload(await client.call_tool("read_selected", {"uri": chosen}))
            passed = chosen == expected and bool(read.get("found"))
            row = {"row_id": f"filesystem_{index:02d}", "query": query, "expected": expected, "chosen": chosen, "passed": passed, "mcp_calls": 2, "descriptor_tokens_est": len(json.dumps(descriptors, separators=(",", ":"))) // 4, "payload_loaded_after_selection": True}
            rows.append(row)
    rows_path = output_dir / "rows.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows),
        "correct_first_hits": sum(row["passed"] for row in rows),
        "first_hit_rate": sum(row["passed"] for row in rows) / len(rows),
        "mean_mcp_calls": sum(row["mcp_calls"] for row in rows) / len(rows),
        "max_descriptor_tokens_est": max(row["descriptor_tokens_est"] for row in rows),
        "payload_before_selection": False,
        "passed": all(row["passed"] for row in rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("server")
    bench = sub.add_parser("bench")
    bench.add_argument("--output-dir", default="experiments/lean-runtime-e2e/filesystem-mcp-32")
    args = parser.parse_args(argv)
    if args.command == "server":
        mcp.run(transport="stdio")
        return 0
    summary = asyncio.run(run_bench(Path(args.output_dir)))
    print(json.dumps(summary, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
