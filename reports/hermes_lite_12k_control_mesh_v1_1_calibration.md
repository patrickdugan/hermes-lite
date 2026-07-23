# Hermes Lite 12k Control Mesh v1.1: Adaptive Calibration

## Status

The post-outcome v1.1 successor completed the unchanged 768-cell calibration:
128 held cases crossed with the six registered control-flow arms. Five arms
passed every frozen calibration gate. `ram_then_trm_typed` failed only the
negative-control gate, at 0.8696 against the registered 0.90 minimum.

This is deterministic routing and typed-plan calibration. It is not executable
task success and does not license a fraction-of-full-Hermes efficacy claim.

## Provenance

- Parent registration:
  `20affa942b608ac960a17a98d56ef6cccaa1f59c4a77346be0d69b93e798988e`
- Post-outcome control addendum:
  `a47183dcf42977ad0487ee35e58de53a23920bc975de7f76c37e8dbc06146662`
- Held cases: 128, unchanged from v1
- Calibration cells: 768
- Cells SHA-256:
  `a0542f84d8266dbe372036d165ee68800744d16ff0bf9c5906739bc77cb3139f`
- Summary SHA-256:
  `5c4c541aa7512a5e1fb85350184881131c6d9f57ab8ad299f4d9a2f072573f57`
- RAM policy SHA-256:
  `d7dfbac0ddffe65645f749622fc7ccaa3562e95db6068ffc330dfdc063c2466f`
- TRM router SHA-256:
  `ce8fc00f23830b9650ab06044667f65af751ff467ffabfb6434ab5f5f20b6543`

## Results

| Arm | Strict | Transfer | Negative | Pre-LDT plan | Repairs | Max packet | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| lexical_typed | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 735 | pass |
| trm_typed | 0.9922 | 1.0000 | 0.9565 | 1.0000 | 0 | 739 | pass |
| ram_typed | 1.0000 | 1.0000 | 1.0000 | 0.9143 | 9 | 740 | pass |
| trm_then_ram_typed | 0.9766 | 1.0000 | 1.0000 | 0.8857 | 9 | 741 | pass |
| ram_then_trm_typed | 0.9766 | 1.0000 | 0.8696 | 0.9143 | 9 | 856 | **fail** |
| adaptive_mesh | 1.0000 | 1.0000 | 1.0000 | 0.9143 | 9 | 862 | pass |

All arms remain well below both the 6,000-token promotion ceiling and the
12,000-token hard window.

## v1 Comparison

| Arm | v1 strict | v1.1 strict | Delta | Recovered cells |
|---|---:|---:|---:|---:|
| lexical_typed | 0.7266 | 1.0000 | +0.2734 | 35 |
| trm_typed | 0.7188 | 0.9922 | +0.2734 | 35 |
| ram_typed | 0.1719 | 1.0000 | +0.8281 | 106 |
| trm_then_ram_typed | 0.4922 | 0.9766 | +0.4844 | 62 |
| ram_then_trm_typed | 0.2500 | 0.9766 | +0.7266 | 93 |
| adaptive_mesh | 0.4922 | 1.0000 | +0.5078 | 65 |

Every arm now makes one invariant selection across the plain,
checkpoint-state, and stale-hint forms of each positive task. For the lexical,
TRM, and adaptive arms, each perturbation lane is 35/35 strict.

The lexical and TRM controls isolate the largest causal correction: parsing
`CURRENT_TASK` as the authoritative route text recovers 35 positive cells in
each arm without changing either model. The larger RAM-arm deltas combine that
typed boundary with the registered support gate and cannot be attributed to RAM
learning alone.

The LDT correction is observable rather than decorative. Each RAM-plan arm
produces nine allowed-but-noncanonical operations; the typed LDT repairs all
nine to the registered plan. Post-LDT exactness therefore reaches 1.0 whenever
the route itself is correct.

## Residual Failures

- `trm_then_ram_typed` chooses the wrong contract for all three perturbations of
  `held.metta-composition-hermes.2`.
- `ram_then_trm_typed` accepts three forbidden-skill controls.
- `trm_typed` accepts the `pure-trm-trainer` forbidden-skill control.

These failures are invariant across rerendered context where applicable. They
are controller decisions, not prompt-window overflows.

## Promotion Decision

Five arms satisfy the frozen numerical gates. The lexical control nevertheless
Pareto-dominates the perfect learned arms on this calibration: it has the same
strict, transfer, and negative-control rates as `ram_typed` and
`adaptive_mesh`, while using fewer tokens and requiring no LDT repairs.

The executable stage will therefore carry:

1. `lexical_typed` as the minimal typed-boundary control.
2. `adaptive_mesh` as the learned RAM/TRM challenger required to test whether
   learned retrieval adds value on actual task execution.

This two-arm selection is made after v1.1 calibration and must be frozen in a
separate executable-stage protocol before model outcomes are observed. A full
Hermes 160k lane must run the same task set before any relative-efficacy result
is reported.

## Claim Boundary

The supported result is that typed task compartmentalization, RAM support
gating, and canonical LDT repair eliminate the registered routing failures on
this deterministic held set. The data do not show learned routing outperforming
the lexical control, executable Bonsai competence, or parity with 160k Hermes.
