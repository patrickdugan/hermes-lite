# Bonsai MCP Skill Mesh Gym v0

This study tests whether a retrieval mesh can spend context in proportion to
task complexity while holding the Bonsai 8B checkpoint, task seeds, tool
permissions, output contract, and 12k context ceiling fixed.

## Scientific Question

Does adaptive TRM reranking followed by LDT evidence verification preserve
task-relevant retrieval while reducing wrong-skill activation and prompt
tokens as skill tasks become more compositional or provenance-sensitive?

The registered claim scope is narrow: this is a fixed-model comparison of
retrieval and control architectures. It is not evidence of general model
capability, general alignment, or oversight sufficiency.

## Two-Axis Ladder

The frozen construction crosses four domains with six complexity levels:

| Domain | Role | Split |
| --- | --- | --- |
| Storyworld construction | Existing complex skill family | calibration |
| Intellect-style logic | Existing logic family | calibration |
| Repository repair | General workflow transfer | transfer |
| Data and provenance | Unseen evidence workflow | heldout |

| Level | Task shape |
| ---: | --- |
| 1 | Single-skill lookup |
| 2 | Skill selection among distractors |
| 3 | Two-skill composition |
| 4 | Conflicting current and legacy evidence |
| 5 | Stale/provenance-sensitive evidence |
| 6 | Stateful four-skill workflow |

Every case uses the same domain catalog shape: four current attested resources
and four conflicting legacy resources. Catalog order is legacy-first so
freshness is not silently encoded by resource order.

## Mesh Arms

| Arm | Selection | Verification | Expansion |
| --- | --- | --- | --- |
| `full_context` | All resources | None | No |
| `static_topk` | Lexical top-k | None | No |
| `typed_packet` | One lexical resource per interface type | None | No |
| `trm_rerank` | Task, replay, type, and provenance score | None | No |
| `ldt_verified` | Typed candidates | Freshness, provenance, coverage | No |
| `adaptive_hybrid` | TRM-ranked typed candidates | Freshness, provenance, coverage | Until covered |

The adaptive arm can spend more MCP calls as complexity rises, but it must
remain under the same 5k working-packet target and 12k hard context ceiling.

## Endpoints

Deterministic calibration reports:

- retrieval recall
- wrong-skill and forbidden-resource activation
- stale and conflicting evidence
- abstention and typed-contract coverage
- estimated MCP calls
- packet tokens and token savings against full context
- routing utility per 1k tokens

Live Bonsai evaluation adds:

- exact task success
- model-selected retrieval recall
- model-selected wrong-skill activation
- exact action-sequence success
- latency and provider token usage

Deterministic `proxy_success` is a packet-validity diagnostic, never a model
performance result.

## Sealing Workflow

Registration does not call a model. The selected registration fixture is kept
under `evals/registered/`; raw calibration and live outcomes remain under the
ignored `experiments/` tree:

```powershell
hermes-lite-mcp-mesh-gym register `
  --config configs/mcp_skill_mesh_bonsai_v0.json `
  --output-dir evals/registered/bonsai_mcp_skill_mesh_v0
```

It emits `protocol.json`, `cases.jsonl`, `arms.json`, and a registration
receipt whose ID binds their hashes. Verify and calibrate that exact batch:

```powershell
hermes-lite-mcp-mesh-gym verify `
  --registration-dir evals/registered/bonsai_mcp_skill_mesh_v0

hermes-lite-mcp-mesh-gym calibrate `
  --registration-dir evals/registered/bonsai_mcp_skill_mesh_v0 `
  --output-dir experiments/mcp_skill_mesh_v0/calibration
```

A live run must repeat the registered ID explicitly:

```powershell
hermes-lite-mcp-mesh-gym live `
  --registration-dir evals/registered/bonsai_mcp_skill_mesh_v0 `
  --output-dir experiments/mcp_skill_mesh_v0/live `
  --confirm-registration-id <registration-id> `
  --stage screening
```

Screening uses the first frozen seed across all 144 case-arm cells.
Confirmation uses all three seeds only after the frozen promotion gates are
evaluated. The live runner checkpoints after every cell and stops on API or
GPU-pressure failure.

On Windows, use the capped operator wrapper rather than launching the server
and evaluator independently:

```powershell
.\scripts\run_capped_bonsai_mcp_mesh.ps1 -ValidateOnly
.\scripts\run_capped_bonsai_mcp_mesh.ps1 -Stage screening
```

The wrapper verifies the registration before launch, refuses an occupied
unowned port, places its server and evaluator PIDs in one Windows Job Object,
and enforces 2,048 MB RAM, 50% CPU, 50 MB/s sustained I/O, and a 1,800-second
wall limit by default. It records one resource sample per second, treats an
abort as a valid result, and terminates only its recorded PIDs.

The first live segment used the original 90-second per-request timeout. After
it completed all 36 storyworld cells, the first logic full-context request was
right-censored by that timeout. Subsequent resumptions use a labeled
180-second operational timeout addendum; the registration, tasks, arms,
model, seed, context, and resource caps are unchanged.

Resource receipts distinguish sampled process-private and working-set totals
from the kernel's Job Object memory accounting. `peak_ram_mb` is the queried
peak job commit used for cap verification; sampled private and mapped working
set are retained as separate diagnostic fields.

The first three exploratory segments used llama.cpp's default four slots.
Direct accounting in the third segment measured 2,432 MB peak job memory
against the intended 2,048 MB cap. Those live cells remain in the audit trail
but are not promotion-eligible. Subsequent cap-validation runs set
`--parallel 1`, abort when sampled private memory exceeds the cap, and require
the queried Job limit and peak to pass before any result can be promoted.
