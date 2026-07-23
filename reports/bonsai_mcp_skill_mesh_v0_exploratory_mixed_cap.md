# Bonsai MCP Skill Mesh v0: Exploratory Mixed-Cap Screening

## Disposition

This seed-17 screening completed all 144 registered case-arm cells, but it is
not promotion-eligible. The first three segments used llama.cpp's default four
slots and a PowerShell nested-struct bug left the intended Windows Job memory
flags unset. Direct kernel accounting measured 2,432 MB peak job memory
against the intended 2,048 MB cap in the third segment.

The final seven cells used the corrected one-slot operator. Their terminal
receipt verified:

- Job limit flags: `768` (`PROCESS_MEMORY | JOB_MEMORY`)
- Configured process limit: 2,048 MB
- Configured job limit: 2,048 MB
- Peak job memory: 1,999.527 MB
- Sampled peak private memory: 1,961.070 MB
- Cleanup: passed, zero lingering PIDs and zero owned GPU memory

Because only seven cells have cap-valid receipts, the matrix is retained as an
exploratory architecture-neighborhood result. It cannot generate a
confirmation arm list.

## Aggregate Pattern

| Arm | Exact success | Retrieval recall | Wrong-skill rate | Tokens | Savings vs full |
| --- | ---: | ---: | ---: | ---: | ---: |
| `adaptive_hybrid` | 0.792 | 1.000 | 0.000 | 508.0 | 53.8% |
| `trm_rerank` | 0.542 | 0.958 | 0.104 | 619.0 | 43.8% |
| `typed_packet` | 0.417 | 0.844 | 0.156 | 427.5 | 61.2% |
| `ldt_verified` | 0.417 | 0.844 | 0.094 | 396.5 | 64.0% |
| `static_topk` | 0.208 | 0.823 | 0.229 | 427.8 | 61.1% |
| `full_context` | 0.083 | 0.979 | 0.292 | 1100.5 | 0.0% |

One 90-second timeout on `logic.l01/full_context` is retained as an error row.
The same cell completed under the labeled 180-second operational addendum and
is counted once in the aggregate.

## Domain Split

| Domain | Full | Static | Typed | TRM | LDT | Adaptive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Storyworld | 0.000 | 0.333 | 0.500 | 0.500 | 0.500 | 1.000 |
| Logic | 0.167 | 0.167 | 0.500 | 0.167 | 0.500 | 0.333 |
| Repository | 0.000 | 0.333 | 0.167 | 0.833 | 0.167 | 1.000 |
| Data/provenance | 0.167 | 0.000 | 0.500 | 0.667 | 0.500 | 0.833 |

The logic result is the important counterexample. Adaptive, typed, and
LDT-verified packets all have full retrieval recall and zero wrong-skill
activation in that domain, yet typed and LDT reach 0.500 exact success versus
adaptive at 0.333. The adaptive hybrid is therefore not uniformly dominant;
its expanded evidence can hurt action ordering even when retrieval metrics
are perfect.

Repository tasks produce the opposite ordering: adaptive reaches 1.000 and
TRM reranking 0.833, while typed and LDT each reach 0.167. Ranked redundancy
appears useful for repository action selection but harmful in logic. This is
the architecture-neighborhood split the domain-general construction was
designed to expose.

## Claim Boundary

The model, cases, arms, seed, context ceiling, and output contract were
registered before outcomes. However, the hard-cap failure prevents this
matrix from serving as the registered screening result. The result supports
only a hypothesis for the clean replay: retrieval architecture interacts with
domain-level action sequencing, and adaptive expansion has a measurable logic
counterexample.

The raw artifact is retained locally at
`experiments/mcp_skill_mesh_v0/exploratory_mixed_cap_20260723`.
