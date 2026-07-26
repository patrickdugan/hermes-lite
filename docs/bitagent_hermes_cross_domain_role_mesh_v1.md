# BitAgent/Hermes Cross-Domain Role Mesh v1

## Question

Can the candidate-only role topology developed for BitAgent preserve complex
Hermes MCP workflows under stale evidence, wrong-domain retrieval, incomplete
plans, and post-validation mutation while keeping each role packet bounded?

This is a registered perturbation bridge. The eight source tasks were held and
solved in `hermes_lite_executable_bridge_v1`; their answers are no longer
blind. The five fault conditions and all role-mesh outcomes are new and frozen
before evaluation.

## Interface Transfer

The study uses four abstract interface positions:

| Interface | BitAgent binding | Cross-domain responsibility |
| --- | --- | --- |
| Planner | `intent_planner` | Propose a candidate action sequence |
| Specialist | `utxo_tradelayer_specialist` | Check domain-action correspondence |
| Guard | `risk_approval_guard` | Apply typed LDT and attested-registry checks |
| Recovery | `recovery_operator` | Reconstruct a rejected plan without execution |

Only the interface shape transfers. No financial adapter is treated as a
storyworld, logic, repository, or data-provenance expert.

All outputs remain candidate-only. The deterministic host owns
materialization, and it rechecks the candidate hash after LDT validation.

## Frozen Crossing

Eight level-5/6 workflows span storyworld, logic, repository, and
data-provenance tasks. Each is crossed with:

1. clean candidate;
2. stale-resource swap before LDT;
3. wrong-domain resource before LDT;
4. missing required step before LDT;
5. post-LDT mutation before materialization.

Four arms separate the control effects:

| Arm | LDT | Fallback | Retrieval |
| --- | --- | --- | --- |
| `raw_direct` | No | None | Typed lexical |
| `ldt_identical_fallback` | Yes | Candidate unchanged | Typed lexical |
| `ldt_distinct_recovery` | Yes | Attested reconstruction | Typed lexical |
| `adaptive_role_mesh` | Yes | Attested reconstruction | Frozen RAM/TRM |

The identical-fallback arm distinguishes containment from useful recovery.
The raw arm measures how many planted faults reach materialization without the
control boundary.

## Endpoints

Co-primary endpoints are unsafe candidate acceptance, final strict success,
post-LDT mutation blocking, and distinct-recovery success. Secondary endpoints
measure raw proposals, wrong-skill activation, role routing, role-local token
cost, fallback, latency, and receipt integrity.

Token savings compare role-local packets with replaying the broad task packet
to all four roles. This is prompt accounting, not measured model-server KV
cache or billing.

## Claim Boundary

The study can support a claim about typed interface transfer and fault
containment on forty registered perturbation cases. It cannot establish
trained BitAgent-adapter transfer, blind task generalization, real tool
execution, financial behavior, Bonsai capability, general safety, or
oversight sufficiency.

## Commands

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m agent.bitagent_mcp_role_mesh_v1 register `
  --config configs/bitagent_hermes_cross_domain_role_mesh_v1.json `
  --output-dir evals/registered/bitagent_hermes_cross_domain_role_mesh_v1

python -m agent.bitagent_mcp_role_mesh_v1 verify `
  --registration-dir evals/registered/bitagent_hermes_cross_domain_role_mesh_v1
```
