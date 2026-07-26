# BitAgent/Hermes Lite Paired Integration v1

## Result

The deterministic BitAgent/Hermes integration substrate is implemented and
tested without modifying the dirty local BitAgent checkout. A standalone
Node/TypeScript client owns a Python Hermes child process and exposes three
typed operations: bounded packetization, LDT candidate validation, and
fingerprinted capability-request materialization.

The deterministic lane is ready. The live Bonsai 8B adapter lane remains
blocked because its adapter artifacts have not been trained or promoted.

## Transport Evidence

The cross-language test invokes the actual ESM client and Python child. It:

1. routes a simulation task to `utxo_tradelayer_specialist`;
2. constructs a packet below the 6,000-token working cap;
3. validates an evidence-bound candidate through the LDT;
4. materializes a capability fingerprint without authorization or execution;
5. rejects the same candidate after its state-evidence hash is mutated.

The client starts Python with `shell: false`, correlates request ID and
operation, validates authority per operation, and recomputes the canonical
response SHA-256. The Python boundary caps each line at 1 MiB, rejects unknown
or secret-bearing input without echo, and keeps a bounded idempotency cache
that rejects request-ID reuse with changed bytes.

## Prior Benchmarks

This transport sits on the previously sealed control-mesh evidence:

- 160 cross-domain cells completed under frozen resource caps with zero
  receipt failures.
- Raw proposals accepted all 35 invalid fault injections; every LDT arm
  accepted zero.
- Identical fallback recovered 0/35 invalid cases.
- Distinct deterministic fallback recovered 35/35.
- Adaptive routing recovered 32/32 eligible cases, selected the intended role
  on 7/8 source workflows, and estimated 67.8% replay-token savings.
- The three lexical-selection misses and their resulting global-gate failures
  remain retained negative results; the adaptive arm passed the corresponding
  8/8 post-LDT mutation-block check.

These results measure a frozen deterministic harness. They are not evidence
for trained Bonsai adapters or live financial execution.

## Verification

Observed on the sealed interface:

```text
python -m pytest tests/agent/test_bitagent_sidecar_v1.py -q
13 passed

python -m pytest tests/agent -q -k bitagent
80 passed, 450 deselected

node --test tests/node/bitagent_hermes_stdio_sidecar.test.mjs
1 passed
```

The active Python environment emits one unrelated warning because
`pytest-asyncio` is absent while `pyproject.toml` contains `asyncio_mode`.

The machine-readable, canonical-LF interface receipt is
`evals/registered/bitagent_hermes_paired_sidecar_v1/interface_receipt.json`.
It binds the source bytes to control-mesh registration
`af4d4d23d888736a3200a73b87017cb2f0d6ddd867236473b659f3fdc8071042`
and to the cross-language capability fingerprint
`6b1fd73a1156011947e6fd743279a3e85e03393bb220d1dc6c402cf63d879ad6`.

## Readiness Boundary

Ready:

- typed role contracts and deterministic host routing;
- bounded MCP/context packets;
- LDT containment and fallback semantics;
- hash-linked receipts and cross-language capability fingerprints;
- standalone BitAgent host transport with no upstream-core modification.

Blocked:

- role-adapter training and promotion: 83 examples exist, with 717 registered
  promotion deficits and at least 152 additional held examples required;
- live Bonsai inference: runtime status is
  `adapter_artifacts_not_trained`;
- wallet, approval, signing, broadcast, and execution integration.

## Claim Boundary

This release establishes a production-oriented, candidate-only integration
interface and reproducible deterministic benchmark evidence. It does not
establish adapter efficacy, general safety, blind task generalization,
profitable trading, live wallet readiness, or production financial
operation. The local BitAgent source remains byte-pinned at commit
`ad578df3fa1f1e337b631f898613987594151439` with uncommitted upstream-local
files; those files were read but not modified.
