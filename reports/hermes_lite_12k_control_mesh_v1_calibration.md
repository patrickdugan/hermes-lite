# Hermes Lite 12k Control Mesh v1: Calibration

## Status

The registered held calibration completed 768 cells: 128 held cases crossed
with six control-flow arms. No arm passed the frozen strict-success gate of
0.75. The best arm was the dependency-free lexical typed control at 0.7266.

This calibration measures routing and schedule construction only. The required
full-Hermes 160k baseline and executable Bonsai stage are not present, so no
relative-efficacy claim is licensed.

## Provenance

- Registration:
  `20affa942b608ac960a17a98d56ef6cccaa1f59c4a77346be0d69b93e798988e`
- Held cases: 128
- Calibration cells: 768
- Cells SHA-256:
  `f41332c2bf8a136ce33805d79c4a4d5ec2783bdc85f06495846e868838c84437`
- Summary SHA-256:
  `643b18eb162533c39a0b9275aff0e95b6e8a97045f897679b877fde0544e5295`
- RAM policy SHA-256:
  `9b3268db07da4196619327850df70fdad502d4de792e16ff068ef0d7a378d160`
- TRM router SHA-256:
  `ce8fc00f23830b9650ab06044667f65af751ff467ffabfb6434ab5f5f20b6543`

## Results

| Arm | Strict success | Positive route | Transfer route | Negative pass | Max packet |
|---|---:|---:|---:|---:|---:|
| lexical_typed | 0.7266 | 0.6667 | 0.7778 | 1.0000 | 738 |
| trm_typed | 0.7188 | 0.6667 | 0.7222 | 0.9565 | 714 |
| trm_then_ram_typed | 0.4922 | 0.4190 | 0.3704 | 1.0000 | 719 |
| adaptive_mesh | 0.4922 | 0.4286 | 0.5000 | 0.9565 | 712 |
| ram_then_trm_typed | 0.2500 | 0.1048 | 0.0185 | 0.9130 | 723 |
| ram_typed | 0.1719 | 0.0000 | 0.0000 | 0.9565 | 692 |

Every packet is below 738 estimated tokens and therefore far below the 12k
hard ceiling. Context size is not the failure mode in this stage.

## Perturbation Boundary

| Arm | Plain | Checkpoint state | Stale hint |
|---|---:|---:|---:|
| lexical_typed | 1.0000 | 0.7429 | 0.2571 |
| trm_typed | 1.0000 | 0.8857 | 0.1143 |
| trm_then_ram_typed | 0.7429 | 0.3714 | 0.0286 |
| adaptive_mesh | 0.8857 | 0.2857 | 0.0000 |
| ram_then_trm_typed | 0.2000 | 0.0857 | 0.0286 |
| ram_typed | 0.0000 | 0.0000 | 0.0000 |

TRM adds useful invariance to the checkpoint-state wrapper but is more
susceptible than lexical routing to a competing stale-hint segment. The
perturbation therefore succeeds at creating real variation that temperature-zero
seed labels did not create in v0.

The RAM-only arm fails for a principled reason: six transfer contracts have no
positive RAM support by construction, and the current direct classifier gives
supported labels an unconstrained advantage over unsupported labels. RAM-first
then removes the correct unseen contract from the five-candidate shortlist, so
the downstream TRM cannot recover it. TRM-first retains more of the lexical
neighborhood and is correspondingly less destructive.

## Construction Defect

The v1 typed validator checks whether an operation is in a phase's allowed set,
but it does not enforce the one canonical operation registered for that phase.
RAM can therefore choose a legal but noncanonical operation without triggering
repair. The zero-repair result is not evidence that RAM action plans are exact.

Routing outcomes remain valid because route selection is upstream of the
validator. Lexical and TRM plans are generated canonically, so their strict
scores are not changed by this defect. RAM-arm plan and repair metrics are
construction-invalid and must not be used as evidence for typed action quality.

## Successor Decision

No arm is promoted from v1. A labeled v1.1 control addendum will be frozen
before rerun with three changes:

1. Enforce the registered canonical operation in the typed LDT and record every
   repair.
2. Parse `CURRENT_TASK`, `STATE`, and `STALE_HINT` as typed compartments; only
   the current task may drive the primary route.
3. Treat RAM as a support-bounded residual. A label with zero positive training
   support cannot be suppressed or promoted by RAM, and RAM-first may not evict
   the lexical leader from its shortlist.

The held cases and their hashes remain unchanged. Because these design choices
were made after observing v1 outcomes, v1.1 will be reported as an adaptive
successor, never as the original preregistered run.
