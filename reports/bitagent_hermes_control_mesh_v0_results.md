# BitAgent/Hermes Control Mesh v0 Results

## Outcome

The shared typed control boundary preserved the saturated BitAgent launch
baseline across all 120 arm-case receipts:

- final task success: 120/120;
- unsafe candidate acceptance: 0/120;
- receipt failures: 0/120;
- host baseline replay: 36/36 critical tests and 50/50 launch cases.

That is not a learned-controller capability result. The LDT and deterministic
fallback are responsible for correcting learned proposal errors.

## Arm Comparison

Totals across three preregistered seeds and ten held cases per seed:

| Arm | Raw proposals | Final success | Role accuracy | Fallback | Registered gates |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lexical + LDT | 30/30 | 30/30 | 30/30 | 0/30 | 3/3 seeds |
| RAM + LDT | 13/30 | 30/30 | 28/30 | 17/30 | 3/3 seeds |
| TRM + LDT | 13/30 | 30/30 | 28/30 | 17/30 | 2/3 seeds |
| Adaptive + LDT | 23/30 | 30/30 | 30/30 | 7/30 | 3/3 seeds |

Adaptive arbitration improved proposal success from 0.433 to 0.767 relative
to either learned controller and reduced fallback demand by 58.8% (17 to 7).
Every one of the seven retained errors was caught before finalization.

Lexical policy remains Pareto-best in this narrow neighborhood: it produced
perfect proposals with no fallback and no higher context cost. The result
therefore supports adaptive containment, not replacing the lexical host on
these launch cases.

## Held Transfer

The train split contains no labels for:

- `confirmed_deposit`;
- `validDestinationAddress`.

These are exactly the held preconditions that drive most learned-controller
errors. The adaptive arm still selected an incorrect learned proposal seven
times, concentrated in `strategy-09`, `withdraw-10`, and three single-seed
cases. The LDT compared each proposal against observable wallet and input
preconditions, rejected all seven, and invoked the deterministic host.

Rejection counts across seeds:

| Arm | Host-precondition mismatches | Host-role mismatches |
| --- | ---: | ---: |
| Adaptive | 7 | 0 |
| RAM | 15 | 2 |
| TRM | 15 | 2 |

This is the useful architecture result: learned components can be allowed to
operate selectively without making their confidence authoritative.

## Seed Variance

TRM raw proposal success was 0.6, 0.4, and 0.3 by seed. In seed `26072028`,
role recommendation accuracy fell to 0.8, below the registered 0.9 gate.
Because the addendum forbids pooling seeds, TRM does not pass uniformly.

Adaptive raw proposal success was 0.8, 0.7, and 0.8, with fallback rates 0.2,
0.3, and 0.2. It preserved role accuracy at 1.0 in every seed.

## Context And Latency

The bounded packet averaged 535.6 estimated tokens for every arm:

- 91.1% headroom under the 6,000-token working limit;
- 95.5% headroom under the 12,000-token hard limit.

This headroom comes from typed MCP packet construction, not from adaptive
arbitration. Controller-plus-LDT latency was below 0.33 ms on average, excluding
model-server, network, wallet, and chain execution.

## Training And Cleanup

Each seed used 40 registered training cards, 40 RAM epochs, and 400 CPU TRM
updates. The TRM has 3,754 parameters. All runs:

- read zero held cases during training;
- completed eight 50-step checkpoints;
- stayed below 429.6 MB peak process RAM;
- stayed below 18.6 MB/s peak measured process I/O;
- completed PID-owned cleanup with no lingering owned process or GPU compute
  application.

The initial host-baseline replay attempt failed because the orchestration
passed `"run test:launch"` as one npm argument. That failure is retained. The
corrected replay used separate arguments and passed all registered checks.

## Safety Boundary

The held set contains secret-handling cases, and all arms proposed safe
responses for them. Consequently, zero unsafe acceptance is not an adversarial
containment estimate: no learned arm emitted an unsafe proposal in these 30
secret-case opportunities. Authority-bypass unit tests separately verify
rejection of broadcast, post-LDT mutation, secret-request actions, fabricated
approval states, and non-role tools.

## Decision

1. Keep lexical policy as the launch-neighborhood default.
2. Keep adaptive RAM/TRM arbitration as an experimental lane behind the shared
   LDT; do not promote TRM alone.
3. Add a frozen recovery-role family and adversarial candidate-injection lane
   before making a four-role or containment-rate claim.
4. Move learned-controller evaluation to more complex MCP skill tasks, where
   lexical policy is not already saturated and context selection can create
   measurable utility.

## Claim Boundary

This result covers ten deterministic launch cases across three seeds. It has no
recovery-role tasks, no 8B adapter inference, no live funds, no profitability
measurement, and no general alignment claim. Final 1.0 success is a
typed-fallback preservation result, not evidence that RAM or TRM learned the
held task.
