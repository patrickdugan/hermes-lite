# BitAgent/Hermes Controller Addendum v0.1

## Status

The small-controller implementation is frozen before RAM/TRM training or held
model evaluation. The addendum SHA-256 is:

```text
97559567ff765b46d1d6c253ae007c2c0fa430f1dab29d7942fffecbc31699d8
```

It binds the benchmark module, Windows Job Object cap wrapper, original
protocol, and frozen 40/10 split.

## Frozen Controllers

- RAM: sparse multiclass perceptron over word, adjacent-bigram, wallet-phase,
  and route-phase features; 40 epochs.
- TRM: 96 input features, 24 hidden units, four recursive updates, fewer than
  10,000 parameters; 400 AdamW updates.
- Adaptive: learned output is used only when RAM and TRM agree and the minimum
  RAM confidence, RAM coverage, and TRM confidence is at least 0.6.
- LDT: one shared typed gate validates host role, scoped tool, authority,
  evidence hashes, secrets, and observable wallet/input preconditions.
- Fallback: deterministic lexical host candidate on LDT rejection.

The three registered seeds are `26072026`, `26072027`, and `26072028`.
Training is CPU-only under 2,048 MB RAM, 50% CPU, 50 MB/s I/O, and 900 seconds,
with 50-step checkpoints and PID-owned cleanup.

## Verification

- BitAgent-focused contract tests: 40/40 passed.
- New plus legacy shared-wrapper tests: 16/16 passed.
- No model training, held model prediction, 8B weight loading, or live-funds
  action occurred before this checkpoint.

## Claim Boundary

The forthcoming run can compare proposal accuracy, final accuracy after typed
fallback, unsafe acceptance, routing recommendations, fallback rate, packet
tokens, latency, and receipt integrity on ten deterministic launch cases. It
cannot support a general four-role claim because recovery is absent.
