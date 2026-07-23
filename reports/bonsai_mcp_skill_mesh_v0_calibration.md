# Bonsai MCP Skill Mesh v0: Deterministic Calibration

## Status

- Study: `bonsai_mcp_skill_mesh_complexity_v0`
- Registration: `fdaa6dce3853b05e8dba52cf0b1c935b32123e25e746bdce317708616767679f`
- Registered construction commit: `3783118`
- Cells: 24 cases x 6 mesh arms = 144
- Integrity failures: 0
- Deterministic cells SHA-256:
  `af39013684f6e7704560d62804574d45fd5a81fac2c0adb53ac457e607253969`
- Status: packet calibration only; no Bonsai task-performance outcomes

## Aggregate Results

| Arm | Retrieval recall | Wrong-skill activation | Abstention | Tokens | MCP calls | Token savings vs full | Proxy success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_context` | 1.000 | 0.750 | 0.000 | 1100.5 | 8.00 | 0.0% | 0.000 |
| `static_topk` | 0.823 | 0.271 | 0.000 | 427.8 | 2.00 | 61.1% | 0.333 |
| `typed_packet` | 0.844 | 0.156 | 0.000 | 427.5 | 2.00 | 61.2% | 0.750 |
| `trm_rerank` | 0.958 | 0.389 | 0.000 | 619.0 | 3.00 | 43.8% | 0.000 |
| `ldt_verified` | 0.844 | 0.000 | 0.250 | 396.5 | 2.00 | 64.0% | 0.750 |
| `adaptive_hybrid` | 1.000 | 0.000 | 0.000 | 508.0 | 2.04 | 53.8% | 1.000 |

The deterministic construction exhibits the intended separation:

- Full context guarantees exposure recall but includes every stale and
  irrelevant skill resource.
- Static retrieval is compact but loses coverage under composition.
- Typed retrieval improves selectivity, but without freshness verification it
  can still activate stale evidence.
- TRM reranking recovers most required evidence, but a fixed top-k continues
  to include non-required resources and under-covers four-skill workflows.
- LDT verification eliminates wrong-skill activation by rejecting invalid
  evidence, at the cost of abstention when it cannot expand.
- The adaptive hybrid preserves full coverage by expanding only until the
  typed contract is satisfied.

## Complexity Response

| Level | Adaptive recall | Adaptive tokens | Adaptive calls | Static recall | TRM recall | LDT recall | LDT abstention |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.000 | 389.8 | 1.00 | 1.000 | 1.000 | 0.750 | 0.250 |
| 2 | 1.000 | 390.0 | 1.00 | 1.000 | 1.000 | 0.750 | 0.250 |
| 3 | 1.000 | 502.3 | 2.00 | 1.000 | 1.000 | 1.000 | 0.000 |
| 4 | 1.000 | 511.8 | 2.00 | 0.750 | 1.000 | 0.875 | 0.250 |
| 5 | 1.000 | 508.3 | 2.00 | 0.750 | 1.000 | 0.875 | 0.250 |
| 6 | 1.000 | 745.8 | 4.25 | 0.438 | 0.750 | 0.813 | 0.500 |

The adaptive arm does not obtain its compactness from a fixed packet size.
Its estimated retrieval work rises from one call at levels 1-2 to 4.25 calls
at level 6, while its packet grows from about 390 to 746 tokens. This is the
registered behavior of interest: context and retrieval work increase with
contract complexity while remaining below the full-context packet.

## Claim Boundary

These results are deterministic properties of the registered retrieval
policies and synthetic skill catalogs. The adaptive arm's perfect calibration
is expected from its expansion rule; it is an implementation invariant, not a
learned result. `proxy_success` is not model task success, full-context
wrong-skill activation measures evidence exposure rather than model use, and
estimated MCP calls are not measured service latency.

The next evidentiary step is the frozen seed-17 Bonsai screening. It must test
whether the fixed model can use each packet to emit the exact required
resources and action sequence. Confirmation is unavailable until a screening
summary satisfies the registered task-success, wrong-skill, and token-savings
gates.
