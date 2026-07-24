# Hermes Lite Executable Bridge v1

## Result

On eight frozen level-5/6 synthetic MCP action-sequencing tasks spanning
storyworld, logic, repository, and data-provenance work, both 12k Bonsai arms
matched the post-registration full-Hermes substitute:

| System | Context budget | Mean packet | Strict success | Resource recall | Domain specialist |
|---|---:|---:|---:|---:|---:|
| Bonsai `lexical_typed` | 12,000 | 1,030.0 | 8/8 | 1.000 | 2/8 |
| Bonsai `adaptive_mesh` | 12,000 | 1,076.9 | 8/8 | 1.000 | 7/8 |
| Full Hermes `gpt-5.6-sol` | 160,000 configured | 7,001.3 | 8/8 | 1.000 | n/a |

The registered point-estimate efficacy gate is therefore met:

```text
Hermes Lite efficacy fraction
  = strict_success(Hermes Lite) / strict_success(full Hermes)
  = 1.00 / 1.00
  = 1.00 >= 0.65
```

This is parity on eight deterministic synthetic tasks, not a general capability
or stochastic-robustness result. One task changes the rate by 0.125.

## Context Efficiency

The adaptive 12k packet used 84.6% fewer estimated prompt tokens than the full
Hermes packet, a 6.50x reduction. The lexical packet used 85.3% fewer, a 6.80x
reduction. The packets differ intentionally: full Hermes receives the broad
resource set, while Hermes Lite must retrieve and compose a bounded subset.

This establishes the useful control-mesh result in this study: typed retrieval
can preserve strict action-sequence performance while discarding most of the
candidate context.

## RAM Contribution

The trained sparse domain RAM raised registered specialist accuracy from 0.25
to 0.875, a gain of 5/8 tasks. It did not improve strict task success or
resource recall because the typed inner resource router already solved all
eight held tasks.

The adaptive arm paid for that routing improvement:

- +46.875 estimated prompt tokens on average versus `lexical_typed`.
- +7,386 ms mean local latency, about 7.0%, in these sequential Bonsai calls.
- No strict-success gain on the frozen set.

Accordingly, `lexical_typed` Pareto-dominates `adaptive_mesh` for this task set
when the objective is only strict success, prompt size, and latency. RAM remains
behaviorally real rather than decorative, but its selected specialist was not
causally necessary for these outcomes. A harder successor must include tasks
whose executable plan depends on domain-specific instructions that the shared
inner router cannot recover.

## Full-Hermes Substitution

The preregistered direct OpenAI `gpt-4.1` denominator could not run because the
available credential was invalid. The registered OpenRouter failover also
failed authentication. Both lanes produced zero valid model responses and are
sealed as construction failures.

Before any benchmark response was obtained, runtime addendum
`b1c62a073a36cac7c98e78152f0bf94f3db6804444815b425cc6eb793c0d5526`
registered the available full-Hermes route:

- Provider: `openai-codex`
- Model: `gpt-5.6-sol`
- Context budget: 160,000 tokens
- Tool surface: none
- Frozen prompts and scorer: unchanged

The ChatGPT Codex route required a transport adapter: account binding,
omission of null `tools` and unsupported `max_output_tokens` fields, and
verbatim reconstruction from streamed text deltas. An exact-response provider
smoke passed before benchmark outcomes. The concise output contract was
prompt-enforced, with a 4,000-character receipt cap.

The substitute is a stricter, newer-model comparison. It is not the
preregistered GPT-4.1 result and must not be cited as one. The largest prompt
was 7,048 estimated tokens, so the experiment configures a 160k budget but does
not test 160k-length retrieval.

## Provenance

- Registration:
  `9255e3b4e1395ab18d73f6bc33bdae05a93c165ade71ce06944f0fddfe043d00`
- Codex runtime addendum:
  `b1c62a073a36cac7c98e78152f0bf94f3db6804444815b425cc6eb793c0d5526`
- Frozen prompt SHA-256:
  `93017a1cd71885dd654d323b51d8833832f30b604f8d3904a69bc83572b3ad5e`
- Bonsai cells SHA-256:
  `4f706f8b0a5ec2df129f4f059380eea5344a89308faa51399dccaaa0281b3ed9`
- Bonsai summary SHA-256:
  `32c25b5e939a03c7476450eff20e08fa64d6797db948a11849afc5f91fc425f4`
- Full-Hermes raw SHA-256:
  `e1c03a430654501824026d8a3da9b21d110e19c9cc9a0d955dc680a3ba37b8e0`
- Full-Hermes scored cells SHA-256:
  `79336f048addeab63eebef24c6ca2ac081beb1e71b69819f51e4f985d8a818b7`
- Full-Hermes summary SHA-256:
  `27566faea782f80183c83c49bf9a097cc7766a15b062779228d91e899465f8e9`

The first ten Bonsai cells have sampled resource telemetry but no final wrapper
receipt because their monitoring process was externally interrupted. The
six-cell resume completed under the 2,048 MB cap with cleanup passing. No
unowned process was terminated.

Targeted executable-bridge and control-mesh tests pass 22/22. The full
non-`prodpush` suite could not complete on this Windows host: its first isolated
failure was the pre-existing concurrent atomic-write test receiving
`PermissionError(13)` in two of ten writer threads, and pytest later crashed
inside its own numbered-temp symlink creation. No changed-module test failed.

## Claim Boundary

The supported claim is narrow: on eight held synthetic MCP tasks, typed
retrieval packets of about 1.0k tokens let Bonsai 8B match a no-tool
`gpt-5.6-sol` full-Hermes substitute receiving about 7.0k tokens, while a
trained RAM improved specialist routing without improving already-saturated
task success. This does not establish GPT-4.1 parity, 160k-length capability,
general MCP competence, tool execution, safety, or broad agent efficacy.
