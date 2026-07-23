# Hermes Lite Executable Bridge v1

## Question

Can a 12k Bonsai agent recover a useful fraction of a full Hermes agent's
strict task success when a trained sparse RAM routes compact domain skills and
an inner MCP mesh retrieves only current task resources?

The frozen comparison uses eight level-5/6 tasks from the sealed Bonsai MCP mesh
case corpus. The source cases and contracts are referenced by hash rather than
copied or regenerated.

## Split

- Levels 1-4: 16 domain-RAM training tasks.
- Levels 5-6: 8 held executable tasks.
- Domains: storyworld, logic, repository, and data provenance.
- Exact task and query overlap: zero.

The domain RAM is a four-label sparse Bernoulli log-odds memory. Its labels are
existing Hermes skill contracts:

| Domain | Specialist contract |
|---|---|
| storyworld | `comprehensive-storyworld-building` |
| logic | `intellect3-logic-hermes` |
| repository | `mcp-trm-retrieval-optimization` |
| data provenance | `pure-trm-trainer` |

For label \(j\) and sparse feature \(f\), the fitted memory weight is

\[
w_{jf}
=
\log\frac{n_{jf}+1}{N_j+2}
-
\log\frac{n_{\neg j,f}+1}{N_{\neg j}+2}.
\]

The adaptive arm first runs the v1.1 TRM/RAM router, then permits the
support-backed domain RAM to override its selected specialist. The lexical arm
uses the v1.1 lexical route directly. Both arms use the same inner
`adaptive_hybrid` resource selector, typed LDT, task cases, model, seed,
temperature, and output scorer.

## Action Interface

The prior Bonsai mesh exposed both allowed-tool aliases and internal action
tokens. This successor freezes one output namespace: only canonical
`Action token` values extracted from current selected resources are legal.
Legacy tokens and allowed-tool aliases are omitted. The packet never contains
the expected answer or action sequence.

## Baseline

The full-Hermes lane runs the actual `C:\projects\hermes-agent` `AIAgent` with:

- direct OpenAI `gpt-4.1`;
- `context_length_override=160000`;
- no tool surface;
- no project context or memory;
- the complete skill catalog and complete task resource catalog;
- the same canonical output namespace and strict scorer.

The primary efficacy fraction is

\[
\rho
=
\frac{\text{adaptive Bonsai strict task success}}
{\text{full Hermes strict task success}}.
\]

The registered target is \(\rho \ge 0.65\), with zero API errors. A zero
full-Hermes denominator does not license an efficacy ratio.

## Commands

Register before training or inference:

```powershell
python -m agent.executable_control_mesh_v1 register `
  --config configs/hermes_lite_executable_bridge_v1.json `
  --output-dir evals/registered/hermes_lite_executable_bridge_v1
```

Train the domain RAM only through the hard-cap wrapper:

```powershell
.\scripts\run_capped_lean_control_mesh_v1.ps1 `
  -TrainingKind executable_domain_ram `
  -TrainingTaskId hermes-lite-executable-domain-ram-v1 `
  -PublishedModelName executable-domain-ram-v1 `
  -RegistrationDir evals\registered\hermes_lite_executable_bridge_v1 `
  -RamEpochs 40 `
  -Seed 8819 `
  -WallSeconds 300
```

Run the Bonsai lane through the existing local-model cap wrapper:

```powershell
.\scripts\run_capped_bonsai_mcp_mesh.ps1 `
  -StudyMode executable_bridge_v1 `
  -RegistrationDir evals\registered\hermes_lite_executable_bridge_v1 `
  -RegistrationId 9255e3b4e1395ab18d73f6bc33bdae05a93c165ade71ce06944f0fddfe043d00 `
  -Stage screening
```

The full-Hermes prompts are materialized and hashed before the remote run. The
credential path is passed to the runner, but the key is never written to a
receipt or log.

## Claim Boundary

This is a post-outcome bridge study over synthetic MCP action-sequencing tasks.
It can measure strict response efficacy, domain routing, retrieval, and context
cost in this task family. It cannot establish general reasoning, real tool
execution, stochastic robustness, or a universal RAM advantage.
