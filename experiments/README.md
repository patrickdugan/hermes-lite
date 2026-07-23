# Experiment Outputs

This directory is the local output root for Hermes Lite research harnesses. Run-level JSONL, checkpoints, logs,
and generated storyworld artifacts are intentionally ignored by Git.

Retain publishable evidence by distilling a run into:

- a report under `docs/` or a dedicated tracked report directory;
- a compact machine-readable summary with hashes of the source run;
- the frozen config or eval fixture needed to reproduce it; and
- an explicit note when raw local artifacts are required for independent replay.

Do not force-add credentials, provider caches, model weights, or local session databases.
