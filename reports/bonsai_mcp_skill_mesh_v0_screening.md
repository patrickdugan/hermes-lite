# Bonsai MCP Skill Mesh v0: Screening Result

## Scope

This fixed-model screening compares six retrieval and control architectures for
Bonsai 8B on 24 matched tasks spanning storyworld, logic, repository, and data
provenance workflows. It evaluates architecture selection under one registered
screening seed. It does not establish general model capability, general
alignment, or oversight sufficiency.

## Integrity

- Registration:
  `fdaa6dce3853b05e8dba52cf0b1c935b32123e25e746bdce317708616767679f`
- Completed cells: 144/144
- Matched blocks: 24/24
- API errors: 0
- Cell journal SHA-256:
  `a249bb165735cc6bfd2608f6da793c88bb94af6fb5170344df2a30124388b4e4`
- Cell-receipt manifest SHA-256:
  `431b1be17addd2281129ba0adba803c34e0007b1439c32ca56daa81ce2832256`
- Screening summary SHA-256:
  `8efd799fa12cba2791e959fc4de2e0aa2688e7a96a018bd48c7b77710b7d6674`
- Resource attestation SHA-256:
  `0f42f7f8450dc60e8cd883355220eba9bc61900b896f162579a07f9fd8b9b632`
- Resource attestation: passed for all 144 cells across three capped chunks.

## Results

| Arm | Success | Wrong-skill rate | Token savings | Mean latency |
|---|---:|---:|---:|---:|
| adaptive_hybrid | 0.7917 | 0.0000 | 0.5384 | 11.174 s |
| trm_rerank | 0.5417 | 0.1042 | 0.4376 | 13.061 s |
| typed_packet | 0.4167 | 0.1563 | 0.6116 | 10.221 s |
| ldt_verified | 0.4167 | 0.0938 | 0.6398 | 9.560 s |
| static_topk | 0.2083 | 0.2292 | 0.6113 | 9.697 s |
| full_context | 0.1250 | 0.2917 | 0.0000 | 18.284 s |

The frozen confirmation gates require success at least 0.60, wrong-skill
activation at most 0.10, and token savings versus full context at least 0.20.
Only `adaptive_hybrid` passes all three. `trm_rerank` misses both the success
gate and the wrong-skill ceiling; every other arm misses the success gate.

## Retained Weaknesses

The promoted arm is not uniformly strong:

| Domain | Success | Cells |
|---|---:|---:|
| storyworld | 1.0000 | 6 |
| repository | 1.0000 | 6 |
| data_provenance | 0.8333 | 6 |
| logic | 0.3333 | 6 |

Its success by complexity level is 1.00, 0.75, 0.75, 1.00, 0.75, and 0.50
for levels 1 through 6. Wrong-skill activation is zero in every adaptive cell,
so the remaining errors are execution or sequencing errors rather than
retrieval contamination under this scorer.

## Decision

Promote only `adaptive_hybrid`. Confirmation reruns that arm on all 24 tasks
and all three frozen seeds, for 72 expected cells. Screening is a selection
stage; final claims require the independently resource-attested confirmation
result and must retain per-domain and per-seed variation.
