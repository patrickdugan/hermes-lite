"""Capped tiny recursive reranker for lean skill routing."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agent.lean_contracts import deterministic_route_score, load_contracts, route_feature_vector


FEATURE_DIM = 32
MAX_CANDIDATES = 5


class TinyRecursiveSkillRouter(nn.Module):
    def __init__(self, hidden_size: int = 48, recursive_steps: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.recursive_steps = recursive_steps
        self.input_projection = nn.Linear(FEATURE_DIM, hidden_size)
        self.recurrent = nn.Linear(hidden_size * 2, hidden_size)
        self.score_head = nn.Linear(hidden_size, 1)
        self.abstain_head = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = torch.tanh(self.input_projection(features))
        hidden = torch.zeros_like(encoded)
        for _ in range(self.recursive_steps):
            hidden = torch.tanh(self.recurrent(torch.cat([encoded, hidden], dim=-1)))
        scores = self.score_head(hidden).squeeze(-1)
        abstain = self.abstain_head(hidden.mean(dim=1)).squeeze(-1)
        return scores, abstain

    def manifest(self) -> dict[str, Any]:
        return {
            "model_type": "tiny_recursive_skill_router",
            "feature_dim": FEATURE_DIM,
            "max_candidates": MAX_CANDIDATES,
            "hidden_size": self.hidden_size,
            "recursive_steps": self.recursive_steps,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }


def build_rows(contracts: list[dict[str, Any]]) -> list[tuple[list[list[float]], int]]:
    primaries = [item for item in contracts if item.get("route", {}).get("kind") != "overlay"]
    rows = []
    for target in primaries:
        for query in target.get("route", {}).get("positive_examples", []):
            ranked = sorted(primaries, key=lambda item: deterministic_route_score(query, item), reverse=True)[:MAX_CANDIDATES]
            if target not in ranked:
                ranked[-1] = target
            random.shuffle(ranked)
            features = [route_feature_vector(query, item, deterministic_route_score(query, item)) for item in ranked]
            rows.append((features, ranked.index(target)))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--checkpoint-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=73011)
    parser.add_argument("--ram-cap-mb", type=int, default=2048)
    parser.add_argument("--io-cap-mb-s", type=float, default=50.0)
    args = parser.parse_args(argv)
    if os.getenv("LEAN_ROUTER_CAP_WRAPPER_ACTIVE") != "1":
        raise SystemExit("Refusing uncapped training; use scripts/run_capped_lean_router.ps1")

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    events = out / "events.jsonl"
    process = psutil.Process()
    started = time.monotonic()
    peak_ram = peak_io = 0.0
    last_io = process.io_counters().read_bytes + process.io_counters().write_bytes
    last_time = time.monotonic()
    io_excess_streak = 0

    contracts = load_contracts()
    rows = build_rows(contracts)
    random.shuffle(rows)
    split = max(1, int(len(rows) * 0.8))
    train_rows, held_rows = rows[:split], rows[split:]
    train = TensorDataset(torch.tensor([row[0] for row in train_rows]), torch.tensor([row[1] for row in train_rows], dtype=torch.long))
    held = TensorDataset(torch.tensor([row[0] for row in held_rows]), torch.tensor([row[1] for row in held_rows], dtype=torch.long))
    model = TinyRecursiveSkillRouter()
    if model.manifest()["parameter_count"] >= 10_000:
        raise RuntimeError("router parameter cap exceeded")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loader = DataLoader(train, batch_size=min(16, len(train)), shuffle=True)
    loss_fn = nn.CrossEntropyLoss()
    step = 0
    checkpoints = []
    status = "failed"
    abort_reason = ""
    try:
        while step < args.steps:
            for features, labels in loader:
                step += 1
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(features.float())
                loss = loss_fn(logits, labels)
                loss.backward()
                optimizer.step()
                now = time.monotonic()
                rss = process.memory_info().rss / 1024 / 1024
                io = process.io_counters().read_bytes + process.io_counters().write_bytes
                io_rate = max(0, io - last_io) / 1024 / 1024 / max(0.001, now - last_time)
                peak_ram, peak_io = max(peak_ram, rss), max(peak_io, io_rate)
                last_io, last_time = io, now
                io_excess_streak = io_excess_streak + 1 if io_rate > args.io_cap_mb_s else 0
                if rss > args.ram_cap_mb:
                    raise RuntimeError("ram_cap_exceeded")
                if io_excess_streak >= 3:
                    raise RuntimeError("sustained_io_cap_exceeded")
                if step % args.checkpoint_steps == 0 or step >= args.steps:
                    checkpoint = out / f"step-{step:06d}.pt"
                    torch.save({"model": model.state_dict(), "manifest": model.manifest(), "step": step}, checkpoint)
                    checkpoints.append(str(checkpoint))
                    with events.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "event": "checkpoint", "step": step, "loss": float(loss.detach()), "rss_mb": rss, "io_mb_s": io_rate, "path": str(checkpoint)}) + "\n")
                if step >= args.steps:
                    break
        status = "completed"
    except Exception as exc:
        status, abort_reason = "aborted", str(exc)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for features, labels in DataLoader(held, batch_size=32):
            logits, _ = model(features.float())
            correct += int((logits.argmax(-1) == labels).sum())
            total += len(labels)
    latest = out / "router.pt"
    if checkpoints:
        shutil.copy2(checkpoints[-1], latest)
    else:
        torch.save({"model": model.state_dict(), "manifest": model.manifest(), "step": step}, latest)
    summary = {
        "status": status,
        "abort_reason": abort_reason,
        "steps_completed": step,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "train_rows": len(train_rows),
        "held_rows": len(held_rows),
        "held_top1_accuracy": correct / max(1, total),
        "peak_ram_mb": round(peak_ram, 3),
        "peak_io_mb_s": round(peak_io, 3),
        "cpu_pct": 50,
        "checkpoints": checkpoints,
        "latest": str(latest),
        "manifest": model.manifest(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    del model, optimizer, loader, train, held
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    exit_code = main()
    # Windows PyTorch can fault while unloading native DLLs inside a Job
    # Object. All owned objects and artifacts are already finalized above.
    os._exit(exit_code)
