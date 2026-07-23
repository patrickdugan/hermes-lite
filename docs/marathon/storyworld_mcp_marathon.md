# Storyworld MCP Marathon Runbook

This runbook is for Bonsai 8B on the 4 GB RTX 3050 path. It compares compact
retrieval packet variants for storyworld MCP tasks before any model-weight
training is attempted.

## Preconditions

- Bonsai is served by llama.cpp at `http://127.0.0.1:8801/v1`.
- `scripts\start_bonsai_llamacpp.ps1 -Status` reports a recorded process and an open TCP port.
- `hermes-lite-mcp-lab pressure --min-free-mb 512 --max-temp-c 87` passes.
- `hermes-lite-mcp-lab live-smoke --base-url http://127.0.0.1:8801/v1 --model local/bonsai-8b` passes.

## Pilot Command

```powershell
hermes-lite-mcp-lab marathon `
  --suite storyworld `
  --duration-minutes 240 `
  --block-minutes 20 `
  --cooldown-minutes 5 `
  --base-url http://127.0.0.1:8801/v1 `
  --model local/bonsai-8b `
  --warn-temp-c 87 `
  --hard-stop-temp-c 87 `
  --live-probe-max-tokens 40 `
  --live-probe-timeout-s 90 `
  --output-dir experiments/marathons
```

Use this first for a non-live scheduling check:

```powershell
hermes-lite-mcp-lab marathon --suite storyworld --duration-minutes 10 --dry-run
```

## Variants

- `baseline`: task card only.
- `trm`: task card, self-model, and failure note.
- `ldt`: task card and one replay/decision hint.
- `hybrid`: task card, self-model, failure note, and one replay hint.
- `hybrid_reranked`: hybrid packet with replay selected by `retrieval_policy.json`.

## Artifacts

Each run writes:

- `run_manifest.json`
- `events.jsonl`
- `leaderboard.csv`
- `retrieval_policy.json`
- `summary.json`
- `report.md`

Treat aborts as valid outcomes. Inspect `abort_reason`, pressure snapshots, and
live probe events before rerunning.

## Promotion Rule

A variant is ready for the next run only if:

- deterministic pass rate is at least 80%;
- max packet size stays within the configured token budget;
- max packet context ratio stays below 20% of 12k context;
- pressure gates pass;
- live probes do not exceed the configured failure cap.
