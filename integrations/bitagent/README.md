# BitAgent Host Integration

`hermes-role-mesh-sidecar.mjs` gives a Node/TypeScript BitAgent host a stable
out-of-process boundary without modifying the BitAgent core or importing
Python.

The sidecar supports only:

- `packetize`: deterministic role routing plus a bounded Hermes packet;
- `validate`: typed LDT decision and hash-linked mesh receipt;
- `materialize_capability_request`: fingerprinted request material after a
  fresh LDT validation.

It never approves, signs, broadcasts, executes, or calls wallet code.
Materialization returns data for BitAgent's existing authority membrane; it is
not authorization.

```javascript
import { HermesRoleMeshSidecar } from "@moralitylab/bitagent-hermes-role-mesh";

const sidecar = new HermesRoleMeshSidecar({
  cwd: "C:/projects/hermes-lite",
});

const packet = await sidecar.request({
  operation: "packetize",
  task_card: { phase: "simulation", message: "Simulate 10000 sats." },
  workflow_state: { workflowId: "w1", status: "draft" },
  tool_contracts: [
    { name: "bitagent.strategy.simulate", input: { amountSats: "string" } },
  ],
});

await sidecar.close();
```

The host keeps one sidecar child, uses unique request IDs, and treats every
response as untrusted until its native BitAgent policy layer verifies the
returned fingerprint. The client owns and terminates only the process it
starts.

The live Bonsai adapter client remains a separate Python component and refuses
discovery while `configs/bitagent_bonsai_runtime_v1.json` is not `ready`.
