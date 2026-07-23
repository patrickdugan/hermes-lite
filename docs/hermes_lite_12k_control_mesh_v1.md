# Hermes Lite 12k Control Mesh v1

This registered gym adapts the ultra-lean Hermes skill corpus into a compact
agentic control plane. It tests whether learned retrieval and action modules can
recover a useful fraction of full Hermes behavior while operating inside a
12,000-token envelope.

## Components

- **TRM router:** the existing 6,338-parameter recursive candidate reranker,
  retrained on a frozen development split.
- **RAM policy:** a dependency-light sparse Retrieval and Action Memory trained
  in system RAM from route examples, hard negatives, and typed phase records.
- **Typed LDT:** a deterministic validator that owns phase identity, operation
  vocabulary, ordering, and bounded repair.

The registered control-flow arms are:

1. `lexical_typed`
2. `trm_typed`
3. `ram_typed`
4. `trm_then_ram_typed`
5. `ram_then_trm_typed`
6. `adaptive_mesh`

The two orderings are intentionally separate. TRM-first restricts RAM to a
lexical/TRM neighborhood; RAM-first tests whether associative memory can surface
a better shortlist before recursive scoring. The adaptive mesh combines both
with the lexical control and records TRM/RAM agreement.

## Split

Registration snapshots every valid `ULTRA_LEAN.json` contract from the
read-only source corpus and records both raw and normalized hashes. Source Git
state is provenance only; no source file is modified.

- Non-transfer contracts contribute two development examples and one held
  query.
- Six named contracts contribute no positive training examples and form the
  contract-transfer lane.
- Held positive queries are crossed with plain, checkpoint-state, and stale-hint
  contexts.
- The first hard negative for every routable contract is a held forbidden-skill
  control; the second is development-only.

The query split is hash-checked before registration. Temperature-zero model
seeds are not treated as replication; the registered context perturbations
change the actual input.

## Baseline

The final efficacy fraction is:

```text
Hermes Lite strict executable success / full Hermes strict executable success
```

The full Hermes lane is explicit and must produce its own receipt on the same
held cases with a 160k context ceiling. No final fraction is emitted from skill
metadata, deterministic calibration, or a missing baseline. The provisional
promotion target is 0.65 of full-Hermes strict success while respecting the 12k
packet cap.

## Commands

Register before any training or held evaluation:

```powershell
python -m agent.lean_control_mesh_v1 register `
  --config configs/hermes_lite_12k_control_mesh_v1.json `
  --output-dir evals/registered/hermes_lite_12k_control_mesh_v1
```

Train only through the Windows Job wrapper:

```powershell
.\scripts\run_capped_lean_control_mesh_v1.ps1
```

Run deterministic held calibration after the training receipt is sealed:

```powershell
python -m agent.lean_control_mesh_v1 calibrate `
  --registration-dir evals/registered/hermes_lite_12k_control_mesh_v1 `
  --model-dir $HOME/.hermes-lite/models/control-mesh-v1 `
  --output-dir experiments/lean-control-mesh-v1/calibration
```

Calibration measures routing, typed schedule construction, LDT repair demand,
negative-control rejection, transfer, and packet size. It is not the executable
Bonsai or full-Hermes result.
