# hermes-lite

A local-first coding agent for macOS with a native Rust TUI, multi-agent swarms, and Rust-accelerated internals. Built on [Hermes](https://github.com/NousResearch) by Nous Research.

https://github.com/user-attachments/assets/placeholder-demo-video

## What is this

hermes-lite takes the open-source Hermes Agent, strips it to a focused local coding tool, then extends it with:

- **Rust FSM** — PyO3 state machine replacing the Python conversation loop (12 states, 5 actions)
- **Rust SessionDB** — rusqlite + FTS5 + WAL replacing the Python SQLite layer
- **Native TUI** — ratatui terminal UI with multi-agent panes, @mentions, delegation, and inter-agent routing
- **Persistent memory** — global + project-level memories shared across all swarm agents via filesystem
- **Skills system** — reusable expertise modules agents load on demand for specialized tasks
- **Subprocess protocol** — JSON-over-pipes connecting TUI to Python agent processes
- **Integration test suite** — 26 live end-to-end tests driving the agent via subprocess protocol

## Quick start

```bash
# Python agent + CLI
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Optional TRM training and research harnesses
pip install -e ".[research]"

# Rust extensions (FSM + SessionDB)
pip install maturin
maturin develop --release -m hermes_rs/Cargo.toml

# Rust TUI
cargo build --release -p hermes_tui

# Configure
mkdir -p ~/.hermes-lite
echo "ANTHROPIC_API_KEY=sk-ant-..." > ~/.hermes-lite/.env
```

Requires Python 3.11+ and Rust 1.75+.

## Usage

```bash
hermes-lite                              # interactive REPL
hermes-lite chat -q "summarize this"     # single-shot
hermes-lite --continue                   # resume last session
./target/release/hermes-tui              # native TUI
```

### Bonsai 8B with 12k Context

Run a downloaded Bonsai GGUF through a local llama.cpp CUDA bundle on port
8801. Configure machine-local paths through environment variables or pass the
equivalent script parameters:

```powershell
$env:HERMES_LLAMA_CPP_DIR = "C:\path\to\llama.cpp"
$env:HERMES_BONSAI_MODEL = "C:\path\to\Bonsai-8B.gguf"
$env:HERMES_RUNTIME_DIR = "$HOME\.hermes-lite\runtime"
.\scripts\start_bonsai_llamacpp.ps1
hermes-lite config set model.provider local
hermes-lite config set model.default local/bonsai-8b
hermes-lite config set model.base_url http://127.0.0.1:8801/v1
hermes-lite config set OPENAI_BASE_URL http://127.0.0.1:8801/v1
```

The launcher also accepts explicit paths:

```powershell
.\scripts\start_bonsai_llamacpp.ps1 `
  -LlamaDir "C:\path\to\llama.cpp" `
  -ModelPath "C:\path\to\Bonsai-8B.gguf"
```

TRM/LDT skills use the 12k ultra-lean contract path documented in
[`docs/ultra_lean_trm_skills.md`](docs/ultra_lean_trm_skills.md). In `auto`
mode, requests route before prompt construction, project one phase from
`ULTRA_LEAN.json`, and keep state in artifact-backed snapshots instead of chat
history. Use `python -m agent.lean_cli` with `route`, `preflight`, `run`,
`status`, or `resume` to inspect and operate this path directly.

Use `.\scripts\start_bonsai_llamacpp.ps1 -Status` to check the recorded process
and `.\scripts\start_bonsai_llamacpp.ps1 -Stop` to stop only that recorded PID.

In the interactive CLI, `/provider bonsai` or `/model bonsai` selects this
local GGUF profile. Ollama remains available as an explicit fallback:

```bash
ollama pull digitsflow/bonsai-8b
ollama serve
hermes-lite config set model.provider local
hermes-lite config set model.default digitsflow/bonsai-8b
hermes-lite config set model.base_url http://127.0.0.1:11434/v1
hermes-lite config set OPENAI_BASE_URL http://127.0.0.1:11434/v1
```

For Ollama in the interactive CLI, use `/provider ollama-bonsai` or
`/model ollama-bonsai`.

For small local models, hermes-lite automatically injects a compact TRM/LDT
retrieval packet when artifacts exist under `.hermes/trm`, `.hermes/ldt`, or
`.hermes`. Preferred files are `current_task.json`, `self_model.json`,
`replay_candidates.jsonl`, `scores.jsonl`, and `decision_traces.jsonl`. The
default packet budget is 1200 tokens with one replay hint.

### MCP TRM/LDT Experiment Harness

Run eval-first MCP retrieval experiments before any local training:

```bash
hermes-lite-mcp-lab preflight --min-free-mb 1024 --max-temp-c 87
hermes-lite-mcp-lab pressure --min-free-mb 512 --max-temp-c 87
hermes-lite-mcp-lab live-smoke --base-url http://127.0.0.1:8801/v1 --model local/bonsai-8b --max-tokens 8
hermes-lite-mcp-lab manifest --mcp your-mcp --task-id mcp_tool_select_001 --out experiments/run_manifest.json
hermes-lite-mcp-lab eval --cases evals/mcp_cases.jsonl --mcp your-mcp --output-dir experiments
hermes-lite-mcp-lab eval --cases evals/mcp_cases.example.jsonl --mcp example-mcp --output-dir experiments
hermes-lite-mcp-lab eval --cases evals/storyworld_mcp_cases.example.jsonl --mcp storyworld-encounter --output-dir experiments --sample-pressure --min-free-mb 512
hermes-lite-mcp-lab marathon --suite storyworld --duration-minutes 240 --base-url http://127.0.0.1:8801/v1 --model local/bonsai-8b --warn-temp-c 87 --hard-stop-temp-c 87 --live-probe-max-tokens 40 --live-probe-timeout-s 90 --output-dir experiments/marathons
```

Each JSONL eval case can provide a task, expected packet terms, replay rows,
scores, and an optional actual MCP call:

```json
{"task_id":"mcp_tool_select_001","instruction":"Choose the exact MCP resource reader.","expected":{"required_terms":["resource_reader"],"forbidden_terms":["broad_search"]},"replay_candidates":[{"task_id":"mcp_tool_select_001","score":0,"passed":false,"failure":"wrong_tool","output":"resource_reader"}]}
```

The harness writes `events.jsonl`, `summary.json`, and per-case `.hermes/trm`
artifacts under the selected experiment run directory.

On the 4 GB RTX 3050 path, use `preflight` before starting Bonsai and
`pressure`/`live-smoke` after `scripts\start_bonsai_llamacpp.ps1` is running.
Packet promotion is blocked when packet size, pressure, or eval gates fail.
Use `marathon --dry-run` for a single non-live scheduling pass. The marathon
compares `baseline`, `trm`, `ldt`, `hybrid`, and `hybrid_reranked`, then writes
`run_manifest.json`, `events.jsonl`, `leaderboard.csv`, `retrieval_policy.json`,
`summary.json`, and `report.md` under the selected run directory.

For domain-general Bonsai 8B comparisons, `hermes-lite-mcp-mesh-gym` crosses
storyworld, logic, repository, and data/provenance skills over a six-level
complexity ladder. It seals construction before outcomes and compares full
context, static top-k, typed packets, TRM reranking, LDT verification, and
adaptive hybrid expansion. See
[`docs/bonsai_mcp_skill_mesh_gym_v0.md`](docs/bonsai_mcp_skill_mesh_gym_v0.md).
Use `scripts\run_capped_bonsai_mcp_mesh.ps1` for live Windows runs so the
owned server and evaluator share hard RAM and CPU caps and PID-specific
cleanup.

### BitAgent role adapters

The BitAgent path keeps a shared Bonsai 8B base and four separate,
candidate-only adapters for intent planning, UTXO/TradeLayer simulation,
approval-risk review, and interrupted-session recovery. Hermes Lite selects a
role deterministically, builds a compact retrieval packet, and activates only
that adapter per llama.cpp request. Wallet approval, signing, broadcast, and
verification remain outside the model.

The training harness refuses uncapped weight loading and the checked-in runtime
manifest remains disabled until adapters pass held-out and GGUF compatibility
gates. See
[`docs/bitagent_bonsai_role_adapters_v1.md`](docs/bitagent_bonsai_role_adapters_v1.md).

## Multi-agent mode

The Rust TUI supports multiple agent panes, each running an independent subprocess with its own session, model, and conversation.

```
/split              vertical split — spawn new agent
/hsplit             horizontal split
/tabs               switch to tab layout
/close              close focused pane
/name <n>           rename agent
/broadcast <msg>    send to all agents
/agents             list all agents
```

**@mentions:** `@frontend refactor this` routes to a named agent. `@frontend! do X` routes and pulls the response back. `@all run tests` broadcasts.

**Navigation:** `Ctrl+Left/Right` switches panes, `Alt+1-9` jumps by number.

**Delegation:** Agents can programmatically delegate tasks to other agents in the swarm via the `delegate_task` tool. Results are routed back automatically.

**Shared memory:** Project-level memories (`.hermes/MEMORY.md`) are shared across all agents in the swarm via the filesystem. One agent saves context, all agents can read it.

## Agent tools

| Tool | Description |
|------|-------------|
| `terminal` | Shell execution with dangerous-command approval (30 patterns) |
| `process` | Background process management (spawn, poll, kill, stdin) |
| `read_file` | Read files with line numbers, pagination, fuzzy filename suggestions |
| `write_file` | Create/overwrite with auto-mkdir and write-deny list |
| `patch` | Find-replace with 8 fuzzy matching strategies + unified diff |
| `search_files` | ripgrep-backed regex search and glob file find |
| `todo` | Task planning with status tracking (survives context compression) |
| `memory` | Persistent cross-session memory — global + project-level, shared across swarm agents |
| `skills_list` | Browse available skill modules (frontend-design, webapp-testing, etc.) |
| `skill_view` | Load a skill's full instructions for specialized tasks |
| `clarify` | Ask user questions mid-task (rendered as modal dialog in TUI) |
| `delegate_task` | Delegate work to another named agent in the swarm |

## Architecture

```
hermes_tui/          Rust TUI (ratatui) — multi-pane, subprocess management
hermes_rs/           Rust extensions (PyO3) — FSM + SessionDB
src/
  run_agent.py       Python agent loop — LLM calls, tool dispatch, streaming
  cli.py             Interactive REPL — 23 slash commands, session management
  agent/             Agent internals — prompt builder, compression, loop driver
  tools/             Tool implementations + 6 terminal backends
  hermes_cli/        CLI entry point, config, setup wizard
vendor/
  mini-swe-agent/    Terminal execution engine (vendored, MIT)
```

The Rust FSM drives the conversation loop. `agent/loop_driver.py` bridges Rust states to Python. The TUI spawns Python agents as subprocesses communicating via JSON protocol (11 event types each direction).

**Terminal backends:** local (default), Docker, SSH, Singularity, Modal, Daytona.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full breakdown.

## Subprocess protocol

JSON lines over stdin/stdout between Rust TUI and Python agent:

**TUI → Agent:** `UserInput`, `ClarifyResponse`, `DelegatedTask`, `CrossAgentContext`, `Interrupt`, `Shutdown`

**Agent → TUI:** `Ready`, `SessionInfo`, `Token`, `ToolCallStart`, `ToolCallResult`, `ResponseComplete`, `LoopStateChange`, `ClarifyRequest`, `DelegateTask`, `DelegationResult`, `ContextCompressed`, `Done`, `Error`

## Rust FSM

12 loop states: `Init` → `BuildPrompt` → `ApiCall` → `ParseResponse` → `CheckScratchpad` → `AdaptToolCalls` → `ExecuteTools` → `CheckInterrupt` → `CheckContext` → `HandleError` → `Summarize` → `Done`

5 actions: `Continue`, `Break`, `Retry`, `Nudge`, `Fail`

## Demo recording

The `demo/scripts/tui_demo_driver.py` automates full TUI demo recordings — 13 scripted scenes driven via tmux keystrokes, recorded with asciinema, and rendered to MP4:

```bash
python3 demo/scripts/tui_demo_driver.py --record      # full recording
python3 demo/scripts/tui_demo_driver.py --dry-run      # preview scenes
python3 demo/scripts/tui_demo_driver.py --scene 3      # start from scene 3
python3 demo/scripts/tui_demo_driver.py --fast          # 2x speed
```

Features:
- **Instant-paste input** — prompts appear immediately (no character-by-character typing)
- **Auto API key sourcing** — pulls from env, project `.env`, or `~/.hermes-lite/.env`
- **Pre-flight checks** — verifies agent subprocess connects before recording starts
- **Post-processing** — variable speed sections (3x/30x) with real-world elapsed timer overlay in the corner
- **Multiple output formats** — `.cast` (asciinema), `.gif` (via agg), `.mp4` (via ffmpeg)

The demo showcases: skill loading, single-agent app building, multi-agent swarm deployment (6 agents), inter-agent delegation, shared memory writes/reads, broadcast, and graceful shutdown.

## Testing

```bash
# Unit tests (1065 tests)
python3 -m pytest tests/ -q

# Live integration tests (26 tests, requires API key)
python3 -m pytest tests/prodpush/ -v -m prodpush --timeout=180
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| `tests/agent/` | 12 modules | FSM, compression, prompt caching, tool parsing |
| `tests/tools/` | 16 modules | All tools, memory, approval patterns, fuzzy matching |
| `tests/hermes_cli/` | 3 modules | Config, model choices, CLI behavior |
| `tests/prodpush/` | 26 tests | End-to-end via subprocess protocol |

## Safety

**Command approval** — 30 regex patterns trigger user confirmation: `rm -rf`, `chmod 777`, `dd`, `DROP TABLE`, fork bombs, pipe-to-shell, etc.

**Write protection** — blocks writes to `~/.ssh/`, `~/.aws/`, `/etc/sudoers`, shell rc files, credentials, and 19 other sensitive paths.

**API key redaction** — scrubs keys from logs (sk-*, ghp_*, xoxb-*, etc).

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Full architecture reference |
| [docs/AGENTS.md](docs/AGENTS.md) | Development guide |
| [docs/MULTI_AGENT_DESIGN.md](docs/MULTI_AGENT_DESIGN.md) | Multi-agent design decisions |
| [docs/COMPARISON.md](docs/COMPARISON.md) | Comparison with hermes-agent |
| [demo/README.md](demo/README.md) | Demo scenarios and scripts |
| [demo/QUICKSTART.md](demo/QUICKSTART.md) | Quick start guide |

## License

Built on **Hermes** by [Nous Research](https://nousresearch.com) and **mini-swe-agent** v2.2.6 by Kilian Lieret & Carlos Jimenez (MIT). The Rust extensions, TUI, subprocess protocol, delegation system, memory system, skills system, and test suite are original to hermes-lite.
