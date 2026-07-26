# BitAgent/Hermes Stdio Sidecar v1

## Boundary

The sidecar gives an unmodified BitAgent Node host one process-owned NDJSON
lane into Hermes Lite. It exposes three deterministic operations:

- `packetize`: route one role and build a bounded MCP/context packet;
- `validate`: run the typed LDT and seal a hash-linked mesh receipt;
- `materialize_capability_request`: emit fingerprinted request data after a
  fresh LDT decision.

The child never approves, signs, broadcasts, executes, calls a wallet, or
loads model weights. Capability material remains untrusted input to
BitAgent's native authority membrane.

Live Bonsai proposal generation is not part of this sidecar. Its independently
gated runtime remains disabled until the role corpora and adapter promotion
criteria pass.

## Transport Controls

- The Node host owns one child process and uses `shell: false`.
- Requests and responses use one JSON object per line.
- Input is read with a 1 MiB hard line bound rather than buffered without
  limit.
- Unknown fields and recursively detected secret-bearing inputs are rejected
  without echo.
- A bounded 256-entry cache replays byte-equivalent request IDs and rejects
  request-ID collisions.
- Responses bind request ID, operation, and result with canonical SHA-256;
  the Node host recomputes that hash before accepting a result.
- The result receipt hashes canonical-LF artifact bytes so checkout line-ending
  translation does not invalidate provenance.
- The child returns only `no_effect` or
  `candidate_only_or_capability_material_no_execution` authority labels.

## Preflight

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m agent.bitagent_sidecar_v1 `
  --config configs/bitagent_hermes_sidecar_v1.json `
  preflight
```

The expected deterministic status is `ready_deterministic_sidecar`.
`live_adapter_runtime_ready` remains false. Preflight imports neither Torch
nor model weights.

Start the local child directly for protocol inspection:

```powershell
python -m agent.bitagent_sidecar_v1 `
  --config configs/bitagent_hermes_sidecar_v1.json `
  serve
```

Normal BitAgent use should instantiate
`integrations/bitagent/hermes-role-mesh-sidecar.mjs`, which starts and
terminates only its owned child.

## Upstream Isolation

No file is written to the local BitAgent repository. The ESM client is
packaged under `integrations/bitagent` in Hermes Lite so it can later be
published or vendored from a clean BitAgent integration branch.

## Claim Boundary

The tests establish candidate-only cross-language transport, hard input
bounding, secret and schema rejection, deterministic evidence binding, LDT
decisions, response integrity, idempotency, and fingerprint materialization.
They do not establish adapter efficacy, live model readiness, wallet
integration, profitable strategy, or production financial operation.
