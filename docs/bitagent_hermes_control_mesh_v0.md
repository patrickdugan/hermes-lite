# BitAgent/Hermes Control Mesh v0

## Purpose

This integration pairs Hermes Lite's bounded retrieval and small-controller
surfaces with BitAgent's typed economic authority membrane. It is deliberately
asymmetric: Hermes proposes; BitAgent decides and executes.

The registered study compares four candidate-generation arms:

| Arm | Candidate-generation policy | Authority |
| --- | --- | --- |
| `lexical_ldt` | Deterministic lexical route and typed template | Candidate only |
| `ram_ldt` | Sparse RAM retrieval recommendation | Candidate only |
| `trm_ldt` | Tiny recursive routing recommendation | Candidate only |
| `adaptive_mesh_ldt` | Host-selected lexical/RAM/TRM proposal lane | Candidate only |

RAM and TRM may recommend a role or proposal lane. They cannot select the
authoritative financial role. The deterministic host computes the role from
persisted workflow state, and the LDT rejects any mismatch.

The exact small-controller implementation and adaptive threshold are frozen in
`evals/registered/bitagent_hermes_control_mesh_v0/controller_addendum_v0_1.json`.
It was authored before any RAM/TRM training or held model outcome.

## Control Flow

```text
task card + compact state + typed tools + evidence + one replay
                              |
                              v
                 deterministic host role
                              |
             +----------------+----------------+
             |                |                |
          lexical            RAM              TRM
             |                |                |
             +---------- candidate JSON -------+
                              |
                              v
             LDT: role, tool, evidence, expiry,
             effects, secrets, authority, hashes
                    |                   |
                 reject               valid
                    |                   |
          deterministic fallback       v
                                fingerprint request
                                         |
                                         v
                             BitAgent authorization
                                         |
                       deny / manual / one-shot lease
                                         |
                                         v
                     external approval, signer, broadcast
```

The LDT can materialize a BitAgent `CapabilityRequest` only after the candidate
passes. It computes `invocationFingerprint` from the same canonical material
as BitAgent. A cross-language golden vector currently produces
`6b1fd73a...d879ad6` in both TypeScript and Python.

## Context Contract

The MCP-assisted packet is ordered as:

1. current task card;
2. deterministic role contract;
3. compact persisted workflow state;
4. only role-scoped typed tool contracts;
5. current evidence;
6. at most one matching failure replay.

The working target is 5k tokens, the working hard limit is 6k, and a separate
6k durable-summary lane keeps the complete envelope at or below 12k. Raw
transcripts are excluded. Required state and authority contracts outrank replay
examples when trimming.

## Authority Contract

Model output is always an untrusted candidate. The model cannot:

- approve an action;
- sign or broadcast;
- request or retain secret material;
- invent wallet, UTXO, quote, or chain state;
- choose its own financial role;
- issue or consume a capability lease.

BitAgent remains responsible for state truth, policy evaluation, exact
capability authorization, approval, signing, broadcast, and verification.
`broadcast` is nondelegable and is rejected before request materialization.

Every mesh decision binds hashes of the envelope, proposal, route, LDT
decision, and fallback. Any mutation invalidates the decision receipt.

## Registered Data

The local BitAgent launch corpus contains 50 deterministic cases in five
families: deposit, strategy, withdrawal, unsupported, and secret handling.
Within every family, suffixes `09` and `10` are held, producing 40 train and
10 held cases with zero exact ID overlap.

The current case set covers:

- 10 intent-planner cases;
- 20 UTXO/TradeLayer-specialist cases;
- 20 risk-guard cases;
- zero recovery-operator cases.

The recovery gap is registered. No general four-role result is admissible
until a separately frozen recovery family is added.

## Training Safety

Future RAM/TRM training uses hard caps of 2,048 MB RAM, 50% CPU, 50 MB/s I/O,
and 900 seconds, with checkpoints every 50 steps or 60 seconds. Training must
run under an OS cap wrapper and use PID-owned cleanup.

The optional Bonsai 8B QLoRA lane is separately configured. It refuses weight
loading unless launched through its WSL systemd cgroup wrapper. No 8B adapter
has been trained or promoted by this registration.

## Reproduction

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m agent.bitagent_control_mesh_v0 register `
  --protocol configs/bitagent_hermes_control_mesh_v0.json `
  --output-dir evals/registered/bitagent_hermes_control_mesh_v0

python -m pytest `
  tests/agent/test_bitagent_role_adapters.py `
  tests/agent/test_bitagent_role_client.py `
  tests/agent/test_bitagent_control_mesh_v0.py `
  tests/agent/test_bitagent_qlora_guardrails.py -q
```

Registration is intentionally strict about the current local BitAgent bytes.
Those files are uncommitted in the BitAgent checkout, so external reproduction
requires the byte-pinned source corpus or a later BitAgent commit containing
the same hashes. Commit `ad578df` alone is not sufficient.

## Claim Boundary

This registered local integration study measures bounded-context role routing,
typed candidate validity, LDT rejection, fallback behavior, and compatibility
with byte-pinned local BitAgent interfaces. It does not grant financial
authority, establish profitable behavior, certify the uncommitted BitAgent
source as reproducible from git, or authorize approval, signing, broadcast, or
live funds.
