# BitAgent/Hermes Control Mesh v0: Registration Checkpoint

## Result

The paired integration is registered before new mesh outcomes. It now has:

- four frozen proposal arms (`lexical_ldt`, `ram_ldt`, `trm_ldt`,
  `adaptive_mesh_ldt`);
- a deterministic host-owned role boundary;
- a 6k working / 12k hard MCP packet contract;
- an LDT that checks role, tools, evidence binding, capability effects,
  secrets, and expiry;
- deterministic fallback on every rejection;
- exact BitAgent capability-request materialization after validation only;
- a hash chain over envelope, proposal, route, LDT decision, and fallback;
- a frozen 40-train / 10-held split with zero overlap;
- separately guarded Bonsai role-adapter training and runtime seams.

No RAM, TRM, adaptive-mesh, Bonsai, or QLoRA outcome is reported at this
checkpoint.

## Host Baseline

The current local BitAgent checkout passed its existing deterministic host
checks before registration:

| Command | Result |
| --- | ---: |
| `npm run test:launch` | 27/27 tests |
| `npm run test:sovereign` | 5/5 tests |
| `npm run test:economic` | 4/4 tests |
| `npm run eval:launch` | 50/50 cases; all eight reported metrics = 1.0 |

These establish the current host baseline, not a model or mesh improvement.
The Hermes path must preserve it. The new Hermes-focused suite passes 33/33
tests without loading model weights.

## Interface Inventory

BitAgent is a TypeScript sovereign economic agent, not a generic agent
framework. Its useful integration surface is the authority membrane:

- typed capability requests and effects;
- exact invocation fingerprints;
- external policy evidence for capital reservation;
- one-shot capability leases;
- explicit refusal to delegate broadcast.

Hermes Lite contributes:

- compact MCP packet construction;
- deterministic role-scoped adapter activation;
- sparse RAM retrieval;
- tiny recursive routing recommendations;
- typed LDT validation and fallback;
- resource-capped training entry points.

The resulting seam does not modify BitAgent core files. Hermes consumes their
byte-pinned interfaces and emits candidate material for BitAgent to authorize.

## Conformance

A cross-language golden vector was executed through the current BitAgent
TypeScript `capabilityRequestFingerprint` and Hermes Python
`capability_request_fingerprint` implementations. Both returned:

```text
6b1fd73a1156011947e6fd743279a3e85e03393bb220d1dc6c402cf63d879ad6
```

This proves equality for the registered capability-request value shape. It is
not a general JSON-canonicalization proof.

## Provenance

The relevant local BitAgent files are not represented by its current Git head:
the launch cases, harness, sovereign types, capabilities, and survival policy
are untracked, while `package.json` is modified. The source manifest therefore
records `local_uncommitted_source_byte_pinned`, exact SHA-256 values, file
sizes, and per-file Git status.

The local Git head is
`ad578df3fa1f1e337b631f898613987594151439`, but that commit alone cannot
reproduce the registered interfaces.

## Registered Limitations

- The 50-case corpus has no recovery-operator cases.
- Ten deterministic held cases do not establish broad financial-agent
  performance.
- The current host baseline is saturated, so the first mesh study primarily
  tests preservation, routing efficiency, rejection, and context cost.
- The QLoRA seed lane is below its 200-reviewed-examples-per-role promotion
  floor.
- No 8B weights were loaded and no live-funds path was exercised.

The full non-integration repository suite was also attempted. The isolated
first-failure run reached 492 passes and 48 skips before failing because
`prompt_toolkit`, a declared base dependency, is absent from the active Python
environment. An unrestricted attempt later hit an existing Windows access
violation in `hermes_state` initialization. Neither failure enters the focused
33-test result, and the broad suite is not claimed green.

## Next Boundary

The next run should train only the small RAM and TRM routing components under
the registered 2 GB / 50% CPU / 50 MB/s I/O / 900 s caps. It should calibrate
on the 40 training cases, evaluate the 10 held cases once, emit per-case
receipts for all four arms, and stop without a four-role claim because recovery
is absent.
