import assert from "node:assert/strict";
import { resolve } from "node:path";
import test from "node:test";

import { HermesRoleMeshSidecar } from "../../integrations/bitagent/hermes-role-mesh-sidecar.mjs";

test("Node host packetizes, validates, and materializes through stdio", async () => {
  const sidecar = new HermesRoleMeshSidecar({
    cwd: resolve("."),
    python: process.env.BITAGENT_TEST_PYTHON || "python",
    timeoutMs: 10_000,
  });
  const taskCard = {
    phase: "simulation",
    operation: "simulate",
    message: "Simulate a bounded TradeLayer intake.",
  };
  const workflowState = {
    workflowId: "sidecar-workflow-1",
    status: "draft",
  };
  const toolContracts = [
    {
      name: "bitagent.strategy.simulate",
      input: { amountSats: "string" },
    },
  ];
  try {
    const packet = await sidecar.request({
      request_id: "sidecar-packet-1",
      operation: "packetize",
      task_card: taskCard,
      workflow_state: workflowState,
      tool_contracts: toolContracts,
    });
    assert.equal(packet.authority, "no_effect");
    assert.equal(packet.result.role, "utxo_tradelayer_specialist");
    assert.ok(packet.result.packet_receipt.estimated_tokens <= 6000);

    const expiresAt = new Date(Date.now() + 120_000).toISOString();
    const candidate = {
      schema: "hermes.bitagent_candidate.v0",
      role: packet.result.role,
      evidence: packet.result.evidence_binding,
      capability_request: {
        requestId: "capability-sidecar-1",
        agentId: "bitagent-sidecar-test",
        capability: "propose_tradelayer_intake",
        effects: ["read_state", "reserve_capital"],
        scope: {
          workflowId: "sidecar-workflow-1",
          rail: "tradelayer",
        },
        expiresAt,
        intent: {
          id: "intent-sidecar-1",
          rail: "tradelayer",
        },
      },
    };
    const validation = await sidecar.request({
      request_id: "sidecar-validate-1",
      operation: "validate",
      task_card: taskCard,
      workflow_state: workflowState,
      tool_contracts: toolContracts,
      candidate,
    });
    assert.equal(validation.authority, "no_effect");
    assert.equal(validation.result.ldt_decision.decision, "candidate_valid");

    const materialized = await sidecar.request({
      request_id: "sidecar-materialize-1",
      operation: "materialize_capability_request",
      task_card: taskCard,
      workflow_state: workflowState,
      tool_contracts: toolContracts,
      candidate,
    });
    assert.equal(
      materialized.authority,
      "candidate_only_or_capability_material_no_execution",
    );
    assert.equal(materialized.result.authorization, false);
    assert.equal(materialized.result.execution, false);
    assert.match(
      materialized.result.capability_request.invocationFingerprint,
      /^[a-f0-9]{64}$/,
    );

    const mutated = structuredClone(candidate);
    mutated.evidence.state_sha256 = "0".repeat(64);
    await assert.rejects(
      () =>
        sidecar.request({
          request_id: "sidecar-materialize-mutated",
          operation: "materialize_capability_request",
          task_card: taskCard,
          workflow_state: workflowState,
          tool_contracts: toolContracts,
          candidate: mutated,
        }),
      (error) => error.code === "contract_rejected",
    );
  } finally {
    await sidecar.close();
  }
});
