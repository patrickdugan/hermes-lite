# BitAgent + Hermes Lite Production Readiness v1

## Decision

The paired system has a production-oriented **deterministic control plane**,
but it is not a production agent.

| Deployment target | Decision |
| --- | --- |
| Offline and simulation integration | **GO** |
| Live Bonsai role-adapter operation | **NO-GO** |
| Wallet or financial execution | **NO-GO** |

The ready component is an out-of-process, candidate-only boundary. The blocked
component is everything that would turn model output into live behavior:
trained adapters, live MCP execution, native BitAgent deployment, and
financial authority.

## Implemented Flow

```text
BitAgent host state
        |
        v
owned Node -> Python stdio sidecar
        |
        +--> deterministic role route
        +--> bounded MCP/context packet
        |
        +--> [Bonsai role adapter: DISABLED]
        |
candidate -> typed LDT
        |          |
        | valid    | rejected
        v          v
fingerprinted   distinct deterministic
request data    recovery/fallback
        \          /
         v        v
BitAgent native authority membrane
        |
        +--> approval/signing/broadcast/execution: OUT OF SCOPE
```

The sidecar can packetize, validate an externally supplied candidate, and
materialize fingerprinted capability-request data. It cannot approve, sign,
broadcast, execute, or load model weights.

## Evidence

| Evidence layer | Result | Interpretation |
| --- | --- | --- |
| Launch mesh | 120/120 final success; 0 receipt failures | Typed fallback preserves a saturated host baseline |
| Adaptive RAM/TRM | 23/30 raw success; fallback 7/30 | 58.8% less fallback than RAM or TRM alone |
| Narrow lexical control | 30/30 raw success | Keep lexical routing as launch-neighborhood default |
| Attack matrix | 14/14 attacks rejected; 4/4 controls accepted | Registered deterministic containment passed |
| Cross-domain mesh | 160 cells; 0 receipt failures | Harness and receipt path are reproducible |
| Raw cross-domain arm | 35/35 invalid candidates accepted | Without LDT, the fault lane is unsafe by construction |
| LDT + identical fallback | 0/35 invalid accepted; 0/35 recovered | Containment alone does not provide utility |
| LDT + distinct recovery | 0/35 invalid accepted; 35/35 recovered | Counterfactual fallback supplies the measured utility |
| Adaptive cross-domain arm | 7/8 route accuracy; 8/8 mutations blocked | Better routing, with one retained route miss |
| Context | 584-token maximum role packet; 67.8% estimated savings | Strong prompt-budget headroom, not measured KV or billing savings |
| Stdio integration | 80 focused tests passed | Cross-language transport and integrity controls are executable |

The cross-domain study failed its two global lexical gates because three clean
lexical cells were rejected before the mutation manipulation could be
reached. Those negative results remain part of the evidence. The adaptive arm
reached and blocked all eight post-LDT mutations.

## Readiness Matrix

| Layer | Status | Boundary |
| --- | --- | --- |
| External source pinning | Ready | BitAgent commit and six critical files are hash-pinned; its dirty tree is read-only |
| Typed roles and packets | Ready | Four roles, deterministic routing, bounded context, candidate-only output |
| Node/Python transport | Ready | Owned stdio, 1 MiB bound, no shell, collision rejection, response hashes |
| LDT containment | Experimental | Passed planted attacks; no general-safety or adaptive-gaming claim |
| Distinct recovery | Experimental | Exact in frozen tasks; no live learned recovery result |
| RAM/TRM arbitration | Experimental | Useful on complex routing, unnecessary on saturated launch cases |
| Bonsai corpus | Blocked | 83 examples; 717-example promotion deficit |
| Bonsai adapters | Blocked | No training, no promotion, no weight loading |
| Live MCP providers | Blocked | No provider outage, authentication, or live tool-execution study |
| Clean BitAgent integration | Blocked | Package not yet installed or tested on a clean BitAgent branch |
| Financial authority | Blocked | Wallet, approval, signing, broadcast, and execution intentionally absent |

## Adapter Gate

The current role corpus is insufficient:

| Role | Current | Required | Deficit |
| --- | ---: | ---: | ---: |
| Intent planner | 50 | 200 | 150 |
| UTXO/TradeLayer specialist | 10 | 200 | 190 |
| Risk/approval guard | 11 | 200 | 189 |
| Recovery operator | 12 | 200 | 188 |

At least 152 of the 717 additions must fill validation and test splits to
reach 20 examples per held split per role. Benchmark-generated failures are
discovery material, not automatic promotion data; additions require
independent review and split decontamination.

## Next Workorder

1. Build `bitagent_bonsai_role_corpus_v2` to 200 reviewed examples per role,
   including normal, stale, interrupted, rejected, routing, recovery, and
   post-validation mutation cases.
2. Reseal provenance and verify no train/validation/test overlap.
3. Run one role per resource-capped QLoRA job only after the corpus gate
   passes.
4. Require held role accuracy, tool-argument validity, recovery, zero
   unauthorized effects, zero secret requests, and GGUF compatibility before
   promotion.
5. Rerun the frozen launch, attack, and cross-domain suites using actual
   Bonsai candidates.
6. Integrate the ESM sidecar on a clean BitAgent branch and test simulation-only
   native fingerprint verification, timeout, restart, load, and MCP-provider
   outage behavior.

No wallet-connected test belongs in this workorder.

## Provenance

- Control mesh: `694b8a07fbaa7b31ff21d1436a61a2be5a1b1e5e`
- Containment: `96e5c50e515b2204d04e36e50700adba07d921a3`
- Adapter readiness: `b40b12ed7fb29f3ab459e4a5f646be7b4dd1b664`
- Cross-domain result: `aa4bb2e20cc96a2f5aff3f4d89412842a62ec44d`
- Stdio sidecar: `67e63f096311cd891ab5beb40224180646f48f04`
- Sidecar receipt: `efa73738c784330e679ec26fe93abf8aacb6d584`
- Capability fingerprint:
  `6b1fd73a1156011947e6fd743279a3e85e03393bb220d1dc6c402cf63d879ad6`
- Canonical cross-domain records:
  `911630bb35e7b50c7581130e41be3323f948d1bb8cbe71ce54aa77a6866cd728`

The machine-readable decision is
`reports/bitagent_hermes_production_readiness_v1.json`.

## Claim Boundary

This synthesis supports a production-oriented deterministic integration
substrate for offline and simulation evaluation. It does not establish
trained-adapter efficacy, blind generalization, general safety, live MCP
reliability, wallet readiness, profitable strategy, or production financial
operation. Token savings are deterministic packet estimates; latency excludes
model servers, networks, wallets, and chains.
