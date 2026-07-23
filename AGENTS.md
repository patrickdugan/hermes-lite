# Hermes Lite Development

The full development guide is in `docs/AGENTS.md`.

## Repository Rules

- Keep runtime credentials in `~/.hermes-lite/.env`, never in this repository.
- Treat `.hermes/`, `.airis-pilot/`, and `codex-chat-sessions/` as local state.
- Keep raw experiment runs under `experiments/`; commit only distilled reports or a deliberately selected fixture.
- Preserve the upstream-compatible core and isolate TRM/LDT additions behind explicit runtime or CLI entry points.
- Run targeted tests for changed modules, followed by the full non-integration suite before release.
- Do not terminate GPU or model-server processes that were not started by the current command.
