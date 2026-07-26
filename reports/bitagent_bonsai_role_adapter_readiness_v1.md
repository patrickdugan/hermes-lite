# BitAgent Bonsai Role-Adapter Readiness v1

## Decision

Bonsai 8B adapter training is blocked at the corpus-promotion gate. The current
83-example corpus passes the seed contract, and all four QLoRA operators pass
validation-only mode without importing or loading model weights, but none of
the role corpora meets the registered 200-example promotion floor.

No Bonsai weights were loaded and no adapter was trained in this assessment.

## Corpus Readiness

| Role | Total | Train | Validation | Test | Total deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| `intent_planner` | 50 | 48 | 1 | 1 | 150 |
| `utxo_tradelayer_specialist` | 10 | 8 | 1 | 1 | 190 |
| `risk_approval_guard` | 11 | 9 | 1 | 1 | 189 |
| `recovery_operator` | 12 | 10 | 1 | 1 | 188 |

The minimum expansion is 717 independently reviewed examples across the four
roles. Each role also requires at least 20 validation and 20 test examples, so
at least 152 of the added examples must fill held splits if the total floor is
met exactly.

Seed validation found no raw transcripts, secret values, or unauthorized
effects. That establishes schema and hygiene readiness only; it is not
evidence that the data volume, coverage, or label quality is sufficient for
training.

## Operator Readiness

The four role-specific invocations used the pinned
`prism-ml/Bonsai-8B-unpacked` revision
`376f381570d6115bc03f82adcfa4af0c7672ae54`, explicit 2,048 MB RAM, 50% CPU,
50 MB/s I/O, 900-second wall, and 512 MB free-VRAM requirements.

All four returned:

- `status=validated`;
- `training_started=false`;
- `weight_loading_started=false`;
- `weight_loading_enabled=false`.

The exact run manifests and summaries are committed beside the machine
receipt. They validate the operator contract, not local feasibility for an 8B
QLoRA run.

## Remaining Gates

Before any adapter training:

1. Expand every role to at least 200 independently reviewed examples.
2. Allocate at least 20 validation and 20 test examples per role with no
   cross-split duplicates.
3. Cover normal, stale, interrupted, rejected, and adversarial candidate
   behavior; planted containment attacks are tests, not SFT promotion data.
4. Rerun seed and promotion validation and reseal this readiness receipt.
5. Use one role per run under the WSL cgroup wrapper on a training-capable GPU
   environment. The existing 4 GB RTX 3050 profile is an inference/smoke
   target, not established 8B QLoRA capacity.
6. Keep adapters candidate-only and separately gated by the LDT and
   deterministic host through held evaluation and GGUF compatibility tests.

The documented local WSL Python 3.10 and inactive user systemd manager also
remain operator prerequisites; Hermes Lite requires Python 3.11 or newer and
the hard-cap wrapper requires a working user systemd bus.

## Claim Boundary

This result establishes deterministic corpus and no-weight operator preflight
behavior. It makes no claim about Bonsai 8B role accuracy, adapter quality,
general task capability, financial utility, or production readiness.

Source corpus SHA-256:
`db59fa18169feca866e56f334db142fdc0986c2e201d95a6ea839484dfe611e5`.
The source remains local uncommitted BitAgent material and is identified by
hash rather than represented as reproducible from the BitAgent Git HEAD.
