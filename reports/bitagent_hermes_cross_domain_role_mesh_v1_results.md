# BitAgent/Hermes Cross-Domain Role Mesh v1 Results

## Outcome

The registered study completed all 160 cells with zero receipt failures, but
failed two global gates:

- exact clean success failed because lexical resource selection produced an
  invalid plan on 3/8 source tasks;
- exact post-LDT mutation blocking failed because those same three lexical
  cells were rejected before the post-LDT manipulation could occur.

The adaptive arm passed both manipulation checks: 8/8 clean plans reached the
LDT, and all 8/8 post-LDT mutations were blocked at materialization.

## Arm Comparison

Each arm crossed eight level-5/6 workflows with five fault conditions.

| Arm | Raw valid | Final strict | Invalid accepted | Recovery | Route | Token savings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `raw_direct` | 10/40 | 5/40 | 35/35 | n/a | 2/8 | 90.1%* |
| `ldt_identical_fallback` | 10/40 | 5/40 | 0/35 | 0/35 | 2/8 | 67.0% |
| `ldt_distinct_recovery` | 10/40 | 40/40 | 0/35 | 35/35 | 2/8 | 67.0% |
| `adaptive_role_mesh` | 16/40 | 40/40 | 0/32 | 32/32 | 7/8 | 67.8% |

`raw_direct` uses only one planner packet, so its token number is not a
four-role operational comparison.

The identical-fallback control isolates containment from usefulness. LDT
changed invalid-candidate acceptance from 1.00 to 0.00, but identical fallback
did not improve strict success. Replacing it with attested-registry recovery
added 35 successful cells, an absolute strict-success gain of 0.875.

Adaptive RAM/TRM reduced recovery demand from 35 to 32 cells and raised outer
specialist routing from 2/8 to 7/8. Its single route miss was
`data_provenance.l05`, selected as `comprehensive-storyworld-building` instead
of `pure-trm-trainer`. The inner adaptive resource mesh nevertheless produced
the exact task plan, so the route miss did not change final success.

## Lexical Failure

The frozen lexical `typed_packet` selector failed before planted faults on:

- `storyworld.l06`;
- `repository.l06`;
- `data_provenance.l05`.

It selected stale or unattested resources and achieved contract coverage of
0.75, 0.50, and 0.50 respectively. The LDT rejected all three. Distinct
recovery then reconstructed the exact plan from current required registry
resources.

This makes the global post-LDT gate failure a manipulation-check failure, not
a demonstrated TOCTOU blind spot. The mutation was actually injected after
authorization in only 5/8 lexical cells, and all five were blocked. The
adaptive arm exercised and blocked all eight.

## Context And Resources

Role-local packets stayed small:

- maximum single-role packet: 584 estimated tokens;
- mean gross role traffic: 1,340.7 tokens for lexical role meshes and 1,307.2
  for adaptive;
- mean savings versus replaying the broad task packet to four roles: 67.0% and
  67.8%.

These are deterministic prompt estimates, not server KV-cache, latency, or
billing measurements.

The CPU run used the frozen 2,048 MB RAM, 50% CPU, 50 MB/s I/O telemetry, and
300-second wall contract:

- wrapper elapsed: 4.037 seconds;
- wrapper peak working set: 369.309 MB;
- evaluator peak process working set: 367.496 MB;
- evaluator peak average I/O: 7.008 MB/s;
- CUDA disabled;
- cleanup passed with no lingering owned PID.

RAM and CPU were hard-capped by a Windows Job Object, wall time by the wrapper,
and I/O by evaluator telemetry abort rather than an operating-system hard
throttle.

## Provenance

- Registration:
  `bf3973f14198efb0778821b38e1ba94414a03ac2d2e3f77dd3eece7f702b7692`
- Evaluator addendum:
  `4c285b50d925bbd86cea82537927b798a96720d83ed80fb33a83b72ce8894267`
- Raw Windows records:
  `a8caa107eb08380a8be048c3cef0038feea9d533f610519ce043fe4cd135e155`
- Canonical-LF packaged records:
  `911630bb35e7b50c7581130e41be3323f948d1bb8cbe71ce54aa77a6866cd728`

Packaging canonicalized CRLF and optional UTF-8 BOM bytes to the repository's
required UTF-8 LF convention. It did not parse, reorder, or change record
content. The packaged summary and result receipt were rebound to the
canonical-LF hashes, and the manifest preserves both raw-run digests.

## Adapter Implication

This run does not authorize Bonsai adapter training, but it identifies the
highest-value corpus additions:

1. four-step current-versus-stale resource selection, especially storyworld
   and repository workflows;
2. data-provenance specialist routing around `l05`-style validation/manifest
   requests;
3. recovery examples that reconstruct from an attested registry rather than
   repeating a rejected proposal;
4. post-validation mutation examples whose expected behavior is materializer
   rejection followed by a fresh recovery receipt.

These should be independently reviewed additions to the relevant role corpora,
not generated directly into promotion splits from this benchmark.

## Claim Boundary

The result supports typed interface transfer on forty registered
perturbations of eight previously solved synthetic workflows. It shows that
LDT supplies containment, distinct recovery supplies utility, and adaptive
RAM/TRM reduces avoidable recovery while preserving bounded packets.

It does not evaluate trained BitAgent adapters, blind task generalization,
Bonsai 8B generation, real MCP execution, financial behavior, general safety,
or oversight sufficiency.
