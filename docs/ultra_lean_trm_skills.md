# Ultra-Lean TRM Skills for Bonsai 8B

Hermes-lite has a separate stateless runtime for small local models. It does
not inject the standard Hermes system prompt, complete tool schemas, global
skill index, chat transcript, or full `SKILL.md` tree. `auto` mode selects this
path for configured small models at 12k context or less; `standard` keeps the
original agent unchanged.

## 12k allocation

```text
12,000-token harness envelope (12,288-token llama.cpp server)
+-- <=8,000 structured input
|   +-- fixed lean kernel (56 tokens)
|   +-- task card + one projected skill phase
|   +-- compact snapshot + selected evidence handles
|   `-- latest module/verifier result
+-- <=2,000 generated output
`-- >=2,000 safety margin for tokenizer and transport variance
```

The ordinary Bonsai system packet measured 5,297 tokens before user input.
The lean runtime starts with the 56-token kernel, routes from compact metadata,
and loads at most one phase projection. JSON sections are structurally compacted
and never truncated as raw strings. Input over 8,000 tokens fails closed.

## Execution architecture

```text
request
  -> auto/lean gate
  -> deterministic top-5 route
  -> tiny recursive reranker (optional checkpoint)
  -> Bonsai selects among top 3 only when confidence is low
  -> primary contract + at most 2 compatible overlays
  -> one phase packet
  -> deterministic/MCP/TRM module
  -> verifier gate
  -> commit artifact or one bounded repair
```

The harness owns phase identity, `task_id`, `contract_id`, artifact handles,
and snapshot updates. Bonsai emits only a compact phase action. Final output is
released only from a verifier-passed artifact. Durable state is stored under
`.hermes/runtime/<task_id>/` as `events.jsonl`, atomic `snapshot.json`, and
content artifacts; raw transcript replay is not used.

## Skill contract v2

Each `ULTRA_LEAN.json` contains route examples and hard negatives, overlay
compatibility, phase-local operations, retrieval slots, tool profile, output
cap, gates, output contract, and one repair policy. Full Markdown and references
remain authoring surfaces loaded only by named artifact handle.

Generate and validate all 25 TRM/LDT contracts:

```powershell
cd "<path-to-Hermes-Skills>"
python scripts\build_ultra_lean_contracts.py --write
python scripts\build_ultra_lean_contracts.py
```

## Operations

```powershell
python -m agent.lean_cli route "solve this campsite logic grid"
python -m agent.lean_cli preflight "retrieve the exact MCP resource"
python -m agent.lean_cli run "retrieve the exact MCP resource" --task-id mcp-001
python -m agent.lean_cli status mcp-001
python -m agent.lean_cli resume mcp-001
```

Train the 6,338-parameter recursive router under hard Windows resource caps:

```powershell
.\scripts\run_capped_lean_router.ps1 -MaxTempC 87
```

The wrapper limits memory to 2 GB, CPU to 50%, IO to 50 MB/s, checkpoints the
run, performs PID-scoped cleanup, and publishes only a successful checkpoint.
The reference run completed 100 steps in 8.17 seconds, used 640.3 MB peak RAM,
and scored 14/14 held route examples.

## Acceptance receipts

- Skill schema matrix: 25/25 first-pass, maximum prompt 1,127 estimated tokens,
  `experiments/ultra-lean-skills-v2/20260711-114737/`.
- Filesystem MCP: 32/32 descriptor-first retrieval, two MCP calls per query,
  maximum descriptor packet 166 estimated tokens,
  `experiments/lean-runtime-e2e/filesystem-mcp-32-v2/`.
- Intellect-3 Logic: 28/29 at an intentionally reduced 96-token coordinator
  cap; the sole truncated JSON row passed at the normal 180-token cap. The
  provenance-preserving merged receipt is 29/29 with no fallback or final-grid
  leakage under `experiments/lean-runtime-e2e/lean-runtime-int3-held29-repaired/`.
- Storyworld: the repaired receipt passes 20/20 at a 761-token maximum prompt:
  15 Bonsai-generated and changed blocks passed model parsing with deterministic
  stable-ID syntax repair, while five zero-choice terminal nodes used a
  zero-generation passthrough. No fallback remains. See
  `experiments/lean-runtime-e2e/storyworld/bonsai_lean_storyworld_20_repaired/`.

The 25-skill matrix measures routing and action-schema discipline, not domain
correctness. A skill is promotable only with its own held candidate generation,
deterministic verifier, resource receipt, and no-fallback evidence.
