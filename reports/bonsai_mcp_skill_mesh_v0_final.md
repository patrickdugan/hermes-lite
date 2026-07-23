# Bonsai MCP Skill Mesh v0

## Result

A domain-general adaptive retrieval mesh was the only one of six architectures
to pass the frozen one-seed screening gates. In the registered confirmation
stage it achieved 57/72 strict task successes (0.7917), perfect required-resource
recall, zero wrong-skill activation, and 0.5384 estimated packet-token savings
relative to full context.

The confirmation is an exact deterministic replay across three seed labels, not
three independent stochastic replications. Every task produced a byte-identical
raw response at seeds 17, 29, and 43 under temperature zero.

This fixed-model study evaluates retrieval and control architectures for Bonsai
8B under matched task, seed, tool, and context conditions. It does not establish
general model capability, general alignment, or oversight sufficiency.

## Design

The benchmark crosses:

- Four domains: storyworld, logic, repository, and data provenance.
- Six complexity levels, from one-skill lookup through stateful cross-skill
  workflows.
- Six screening architectures: full context, static top-k, typed packet,
  TRM reranking, LDT verification, and an adaptive TRM/LDT hybrid.
- One screening seed over 24 tasks and six arms (144 cells).
- Three confirmation seed labels over 24 tasks and the promoted arm (72 cells).

The frozen confirmation gates were:

- Task success at least 0.60.
- Wrong-skill activation at most 0.10.
- Estimated packet-token savings versus full context at least 0.20.

## Screening

| Arm | Success | Wrong-skill | Packet savings | Mean latency |
|---|---:|---:|---:|---:|
| adaptive_hybrid | 0.7917 | 0.0000 | 0.5384 | 11.174 s |
| trm_rerank | 0.5417 | 0.1042 | 0.4376 | 13.061 s |
| typed_packet | 0.4167 | 0.1563 | 0.6116 | 10.221 s |
| ldt_verified | 0.4167 | 0.0938 | 0.6398 | 9.560 s |
| static_topk | 0.2083 | 0.2292 | 0.6113 | 9.697 s |
| full_context | 0.1250 | 0.2917 | 0.0000 | 18.284 s |

Only `adaptive_hybrid` passed all three gates. Relative to full context in the
matched screening stage, it improved strict success by 0.6667, reduced estimated
packet tokens by 53.8%, and reduced mean latency by 38.9%.

The llama.cpp usage counters provide a less model-dependent token check:

| Arm | Mean API prompt tokens | Mean completion tokens |
|---|---:|---:|
| adaptive_hybrid | 510.2 | 50.9 |
| full_context | 1011.4 | 57.8 |

Thus adaptive retrieval reduced measured prompt tokens by 49.6% and measured
total tokens by 47.5%. The token claim concerns this task packet and prompt
format; it is not a universal compression ratio.

## Confirmation

The promoted arm reproduced the screening aggregate exactly:

| Seed label | Success | Retrieval recall | Wrong-skill | Action exact |
|---|---:|---:|---:|---:|
| 17 | 19/24 | 1.0000 | 0.0000 | 0.7917 |
| 29 | 19/24 | 1.0000 | 0.0000 | 0.7917 |
| 43 | 19/24 | 1.0000 | 0.0000 | 0.7917 |

All 24 tasks had one raw-response variant across the three seed labels. The
temperature-zero seed parameter therefore did not induce variation. These rows
validate replay determinism and receipt integrity, but they do not support a
seed-based variance estimate or confidence interval.

### Domain Boundary

| Domain | Strict success | Cells |
|---|---:|---:|
| storyworld | 18/18 | 18 |
| repository | 18/18 | 18 |
| data_provenance | 15/18 | 18 |
| logic | 6/18 | 18 |

Success by complexity level was 1.00, 0.75, 0.75, 1.00, 0.75, and 0.50 for
levels 1 through 6. The mesh retained every required resource with no forbidden
resource activation at every level; degradation appeared in action sequencing,
not retrieval.

## Failure Audit

All 15 strict failures are repetitions of five task failures:

| Task | Registered failure |
|---|---|
| logic.l02 | Returned allowed-tool alias `apply_logic_rule` instead of internal action ID `apply_admissible_rule`. |
| logic.l03 | Returned allowed-tool aliases `apply_logic_rule`, `verify_derivation`. |
| logic.l05 | Returned allowed-tool alias `submit_logic_answer`. |
| logic.l06 | Returned allowed-tool alias `submit_logic_answer`. |
| data_provenance.l06 | Scheduled manifest sealing before transform and validation. |

The four logic cases expose a harness interface ambiguity. Their packets contain
both allowed-tool names and internal skill action IDs, while the scorer accepts
only the latter. Retrieval is exact and the returned aliases are semantically
matched to the required skills. Under a post-hoc alias-equivalence diagnostic,
strict success would be 69/72 (0.9583), or 23/24 unique tasks. That score was not
registered and is not the primary result.

The provenance level-6 failure remains after alias normalization. It is a true
control-flow error: the model selected all correct modules but ordered the
receipt operation before the state-changing and validation operations.

## Architecture Reading

The adaptive hybrid occupies a useful point in this fixed neighborhood:

- It retains more context than the typed or LDT-only packets, but crosses the
  task-success gate that both miss.
- It filters stale and forbidden resources completely, unlike full context,
  static top-k, typed packet, and the near-threshold TRM reranker.
- Its remaining confirmed defect is downstream of retrieval: typed action
  namespace resolution and stateful schedule ordering.

The result supports a narrow control-mesh claim: adaptive TRM selection plus LDT
contract checking can cut prompt context roughly in half while preserving exact
required-resource coverage on this benchmark. It does not yet show that the
mesh improves free-form reasoning, tool execution, or robustness to stochastic
sampling.

## Integrity

- Registration:
  `fdaa6dce3853b05e8dba52cf0b1c935b32123e25e746bdce317708616767679f`
- Screening cells SHA-256:
  `a249bb165735cc6bfd2608f6da793c88bb94af6fb5170344df2a30124388b4e4`
- Screening resource attestation SHA-256:
  `0f42f7f8450dc60e8cd883355220eba9bc61900b896f162579a07f9fd8b9b632`
- Confirmation cells SHA-256:
  `1b719aff95fcf930ed3711d981da5e226241ba8a8980e0951b934efa3a927bd6`
- Confirmation cell-receipt manifest SHA-256:
  `ea37dea17547bb85dee1770b98781e3f3825ec9d185ca54483a0722fa161107a`
- Confirmation summary SHA-256:
  `6bd714152620f4f38eb8dc5b73c9dd078a6f779032775693559466c35dc184f1`
- Confirmation resource attestation SHA-256:
  `7429e3c4b8ff84398fb539166daf716c13e6d9e0ba582f7edcd2f88a774f3f9c`
- Screening resource chunks: 48 + 91 + 5 cells; all cap and cleanup checks
  passed.
- Confirmation resource chunk: 72 cells; cap and cleanup checks passed.
- Peak Job memory: 1373.117 MB screening, 1367.551 MB confirmation, under
  the 2048 MB hard limit.
- Peak confirmation temperature: 79 C, under the frozen 86 C ceiling.
- API errors: zero in both primary stages.
- Targeted mesh tests: 16 passed.
- Canonical non-integration/non-prodpush suite: 1049 passed, 78 skipped, and
  26 deselected.

The frozen confirmation wrapper receipt contains a diagnostic
`expected_steps=144` inherited from a screening-only hardcode. The registered
summary, cell journal, and attester independently establish 72/72 confirmation
cells. The wrapper was corrected prospectively to derive this field from the
stage summary; the original receipt and hash were not rewritten.

## Next Registered Study

A successor should make the action interface causal rather than ambiguous:

1. Freeze one typed action namespace or preregister explicit allowed-tool to
   internal-action equivalences.
2. Add state-transition execution so ordering errors affect an environment,
   not only exact-sequence scoring.
3. Introduce genuine replication through frozen sampling or task/data-order
   perturbations; temperature-zero seed labels are insufficient.
4. Expand logic and stateful level-6 cases, where the current mesh's remaining
   control-flow weakness is concentrated.
