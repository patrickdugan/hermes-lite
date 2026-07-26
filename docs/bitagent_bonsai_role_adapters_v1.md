# BitAgent Bonsai 8B Role Adapters v1

## Current status

The role contracts, deterministic corpus exporter, Hermes packet path, capped
QLoRA entry point, GGUF packaging seam, and llama.cpp per-request adapter
client are implemented. No 8B weights have been loaded and no adapter has been
trained yet.

The current BitAgent seed corpus contains 83 sanitized examples:

| Role | Seed examples | Financial authority |
| --- | ---: | --- |
| `intent_planner` | 50 | Candidate intent, missing fields, and typed next step |
| `utxo_tradelayer_specialist` | 10 | Candidate simulation and supplied-state interpretation |
| `risk_approval_guard` | 11 | Read-only allow/deny evidence review; cannot approve |
| `recovery_operator` | 12 | Read/verify recovery; cannot replace or duplicate execution |

The promotion floor is 200 independently reviewed examples per role. The seed
corpus is suitable for contract and trainer smoke tests, not production
promotion.

## Authority boundary

The deterministic host chooses exactly one role from persisted workflow state.
The model cannot choose its role. Every adapter output is an untrusted
candidate JSON object.

BitAgent still owns:

- wallet balances, UTXOs, confirmations, quotes, and transaction state;
- exact simulation and fee calculation;
- user approval;
- input selection, signing, and broadcast;
- order or position verification.

No role is allowed to call approval or execution tools. Candidate validation
rejects secret-bearing fields, unauthorized tools, and any assertion that the
model approved, signed, broadcast, or executed an action.

## Model and runtime

Training is pinned to `prism-ml/Bonsai-8B-unpacked` revision
`376f381570d6115bc03f82adcfa4af0c7672ae54`. Low-end inference is pinned to
the Q1 GGUF revision `48516770dd04643643e9f9019a2a349cf26c5dbd`.

Role adapters remain separate. They are not merged into the Q1 base. After
PEFT-to-GGUF conversion and compatibility tests, llama.cpp loads all promoted
adapters at scale zero. The BitAgent host sends one per-request `lora` entry
for the deterministically selected role.

The local Q1 file is about 1.16 GB, but that is weight storage only. Context
KV cache, runtime buffers, and adapters add memory. Device profiles therefore
reduce context on 2 GB, 4 GB, and 8 GB CPU targets and allow only one active
adapter.

## Hermes Lite context contract

The working packet target is 5k tokens, with a hard 6k working limit and a
separate 6k summary lane under the 12k Bonsai ceiling. Packet order is:

1. task card;
2. selected role contract;
3. compact persisted workflow state;
4. only the role's typed tool contracts;
5. current evidence;
6. at most one matching failure replay.

Raw transcripts are excluded. If the packet is too large, replay and excess
evidence are removed before required state. A packet that still exceeds the
limit is rejected.

## Prepare and validate the seed corpus

From the BitAgent starter:

```powershell
npm run export:bonsai-data
```

From Hermes Lite:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m agent.bitagent_role_adapters `
  --config configs/bitagent_bonsai_role_adapters_v1.json `
  validate-corpus `
  --examples C:\projects\BitAgent\BitAgent\tradelayer-thorchain-starter\tradelayer-thorchain-starter\training\artifacts\bonsai-role-corpus-v1\examples.jsonl
```

Use `--promotion` only for the reviewed expanded corpus. It intentionally fails
for the current seed set.

## Capped training

Actual weight loading is refused unless
`scripts/run_capped_bitagent_qlora.sh` starts the process under a WSL systemd
unit. The unit applies:

- `MemoryHigh` and `MemoryMax`;
- `MemorySwapMax=0`;
- machine-relative CPU quota;
- block-device read and write bandwidth caps;
- runtime and task-count limits.

The trainer also samples process RAM, CPU, I/O, swap growth, and CUDA memory,
checkpoints by steps or elapsed seconds, and writes `events.jsonl`,
`summary.json`, and `cleanup_summary.json`. Cap aborts are valid outcomes.

The training output, Hugging Face cache, and any local base model must reside
on the Linux block filesystem named by `--io-device`; a `/mnt/c` or `/mnt/d`
training directory is rejected because the cgroup cannot reliably enforce the
declared block-device I/O cap there.

The wrapper requires explicit values for:

- training task ID;
- RAM, CPU, I/O, wall-clock, and minimum free VRAM caps;
- checkpoint interval;
- chunk strategy;
- base path, or explicit model-download authorization.

Run `--validate-only` first. It never imports torch or loads weights.

The current WSL installation has Python 3.10, while Hermes Lite requires Python
3.11 or newer. Create a dedicated Python 3.11+ WSL environment and install
`pip install -e '.[adapters]'` before an actual run.

The current WSL system manager is enabled, but its user manager is not active
(`systemd-run --user` cannot connect to the bus). The wrapper now probes that
bus and refuses training when it is unavailable. Repairing or enabling the
user session is an operator-owned machine configuration step; the training
harness does not modify WSL configuration automatically.

## Packaging and serving

Convert one completed PEFT adapter:

```powershell
.\scripts\package_bitagent_lora.ps1 `
  -Role intent_planner `
  -AdapterDir C:\path\to\adapter `
  -OutputPath C:\path\to\intent_planner.gguf `
  -DryRun
```

After all held-out and compatibility gates pass, start the server with the
four exact GGUF paths and initialize all at scale zero:

```powershell
.\scripts\start_bonsai_llamacpp.ps1 `
  -LoraPaths @(
    "C:\path\intent_planner.gguf",
    "C:\path\utxo_tradelayer_specialist.gguf",
    "C:\path\risk_approval_guard.gguf",
    "C:\path\recovery_operator.gguf"
  ) `
  -LoraInitWithoutApply
```

The runtime manifest remains
`"status": "adapter_artifacts_not_trained"` until an operator has trained,
converted, evaluated, and explicitly promoted all four artifacts. The client
refuses live discovery while the manifest is not `ready`.

## Promotion gates

An adapter set is not deployable until:

- the expanded corpus meets per-role floors with contamination-safe heldouts;
- BitAgent launch and committed-signal critical tests pass;
- intent, tool-argument, recovery, and truthfulness thresholds pass;
- secret requests, fabricated state, and unauthorized effects are all zero;
- each GGUF adapter is compatible with the pinned Q1 base;
- low-end device profiles pass memory and latency measurements;
- the operator changes the runtime status to `ready`.

## Known limitations

- The seed corpus is small and weighted toward intent planning.
- No funded execution example belongs in training; financial effects remain
  host-tested.
- PEFT adapters trained against the unpacked checkpoint still require an
  empirical compatibility test against the Q1 GGUF runtime.
- The RTX 3050 Laptop GPU has 4 GB VRAM and is an inference/smoke-test target,
  not the registered full 8B QLoRA training target.
