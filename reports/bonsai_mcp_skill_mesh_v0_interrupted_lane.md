# Bonsai MCP Skill Mesh v0: Interrupted Primary Lane

## Status

The first clean-cap replay attempt, `bonsai-mcp-mesh-screening-20260723-133544`,
is a construction failure. It produced three readable cell rows, then its
detached supervisor and both owned children terminated without sealing a
terminal resource receipt. The cell and resource journals contain zero-filled
tails, so none of the rows are admissible for screening, promotion, or effect
estimation.

The raw lane is preserved at
`experiments/mcp_skill_mesh_v0/interrupted_supervision_20260723-133544`.

## Evidence

- Registration ID:
  `fdaa6dce3853b05e8dba52cf0b1c935b32123e25e746bdce317708616767679f`
- Cell journal SHA-256:
  `55ab85b30706fe5aa0a897c412984373741b4d208c8509aaacf1ad40f7198147`
- Cell journal zero bytes: `2077`
- Resource-event journal SHA-256:
  `49f6af31d0ba09caa8a65cf0635766562335384468ba847359d01ab56752c9d6`
- Resource-event journal zero bytes: `257`
- Readable cell rows: `3`
- Admissible cell rows: `0`
- Terminal resource receipt: absent
- Lingering owned PIDs: none

The final resource sample before termination reported `1875.625 MB` sampled
private memory, below the `2048 MB` cap. This does not prove the run was
cap-valid because terminal kernel Job accounting is absent. It does show that
the available evidence does not support labeling the stop as a sampled RAM-cap
breach.

## Corrective Boundary

Commit `dfb0e3c` adds atomic per-cell sidecar receipts, fsynced journal appends,
per-sample kernel Job accounting, and terminal checkpoint-integrity handling.
Subsequent benchmark chunks must run under foreground supervision rather than a
detached launcher.

The fresh lane uses the official matching-version llama.cpp `b10064` CUDA 12.4
runtime under a labeled post-registration addendum. The model digest, cases,
arms, prompts, seeds, context, inference temperature, and resource caps are
unchanged. No interrupted CPU row may be pooled into the fresh CUDA lane.

## Claim Boundary

This artifact documents an execution failure and its exclusion. It carries no
model-performance result.
