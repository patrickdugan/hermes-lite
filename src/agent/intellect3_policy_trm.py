"""CPU-only tiny recurrent policy for Campsite MCP action and candidate ranking."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from agent.intellect3_logic import (
    CampsiteTask,
    compact_candidate,
    expected_action,
    make_candidate_records,
    parse_normalized_campsite_rows,
    stable_holdout,
)


TRAINING_TASK_ID = "int3-campsite-policy-trm-v1"
ACTION_LABELS = ("invoke_module", "commit_candidate", "repair_candidate", "abstain")
REPAIR_LABELS = (
    "none",
    "syntax_valid",
    "shape_match",
    "trees_unchanged",
    "row_counts_match",
    "col_counts_match",
    "no_tent_touching",
    "perfect_tree_matching",
    "no_candidate",
)
GATES = (
    "syntax_valid",
    "shape_match",
    "trees_unchanged",
    "row_counts_match",
    "col_counts_match",
    "no_tent_touching",
    "perfect_tree_matching",
)
MAX_CANDIDATES = 5
FEATURE_DIM = 1 + MAX_CANDIDATES * (2 + len(GATES))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def encode_candidates(candidates: list[dict[str, Any]]) -> list[float]:
    features = [len(candidates) / MAX_CANDIDATES]
    for index in range(MAX_CANDIDATES):
        if index >= len(candidates):
            features.extend([0.0] * (2 + len(GATES)))
            continue
        candidate = candidates[index]
        failed = set(candidate.get("failed_gates") or [])
        features.extend(
            [
                1.0,
                float(bool(candidate.get("official_pass"))),
                *[float(gate not in failed) for gate in GATES],
            ]
        )
    return features


def build_frames(tasks: list[CampsiteTask], *, seeds: int = 4) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    for task in tasks:
        for order_seed in range(seeds):
            for profile in ("pressure", "no_pass"):
                records, _ = make_candidate_records(
                    task,
                    max_candidates=3,
                    profile=profile,
                    order_seed=order_seed,
                )
                candidates = [compact_candidate(record) for record in records]
                target = expected_action(task.task_id, candidates)
                candidate_index = next(
                    (index for index, item in enumerate(candidates) if item["candidate_id"] == target["candidate_id"]),
                    -100,
                )
                frames.append(
                    {
                        "task_id": task.task_id,
                        "profile": profile,
                        "order_seed": order_seed,
                        "features": encode_candidates(candidates),
                        "action": ACTION_LABELS.index(target["action"]),
                        "candidate": candidate_index if target["action"] == "commit_candidate" else -100,
                        "repair": REPAIR_LABELS.index(target["repair_class"])
                        if target["action"] == "repair_candidate"
                        else -100,
                    }
                )
    return frames


class TinyRecursivePolicy(nn.Module):
    def __init__(self, hidden_size: int = 64, recursive_steps: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.recursive_steps = recursive_steps
        self.input_projection = nn.Linear(FEATURE_DIM, hidden_size)
        self.recurrent = nn.Linear(hidden_size * 2, hidden_size)
        self.action_head = nn.Linear(hidden_size, len(ACTION_LABELS))
        self.candidate_head = nn.Linear(hidden_size, MAX_CANDIDATES)
        self.repair_head = nn.Linear(hidden_size, len(REPAIR_LABELS))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = torch.tanh(self.input_projection(features))
        hidden = torch.zeros_like(encoded)
        for _ in range(self.recursive_steps):
            hidden = torch.tanh(self.recurrent(torch.cat([encoded, hidden], dim=-1)))
        return self.action_head(hidden), self.candidate_head(hidden), self.repair_head(hidden)

    def manifest(self) -> dict[str, Any]:
        return {
            "model_type": "tiny_recursive_action_ranker",
            "feature_dim": FEATURE_DIM,
            "hidden_size": self.hidden_size,
            "recursive_steps": self.recursive_steps,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "action_labels": list(ACTION_LABELS),
            "repair_labels": list(REPAIR_LABELS),
            "max_candidates": MAX_CANDIDATES,
        }


@dataclass
class ResourceMonitor:
    process: psutil.Process
    started: float
    io_cap_mb_s: float
    peak_ram_mb: float = 0.0
    ram_total_mb: float = 0.0
    ram_samples: int = 0
    peak_io_mb_s: float = 0.0
    io_excess_streak: int = 0
    last_io_bytes: int = 0
    last_io_time: float = 0.0
    swap_in_start: int = 0

    @classmethod
    def create(cls, io_cap_mb_s: float) -> "ResourceMonitor":
        process = psutil.Process()
        io = process.io_counters()
        swap = psutil.swap_memory()
        return cls(
            process=process,
            started=time.monotonic(),
            io_cap_mb_s=io_cap_mb_s,
            last_io_bytes=int(io.read_bytes + io.write_bytes),
            last_io_time=time.monotonic(),
            swap_in_start=int(getattr(swap, "sin", 0)),
        )

    def sample(self) -> dict[str, float]:
        rss_mb = self.process.memory_info().rss / (1024 * 1024)
        self.peak_ram_mb = max(self.peak_ram_mb, rss_mb)
        self.ram_total_mb += rss_mb
        self.ram_samples += 1
        now = time.monotonic()
        io = self.process.io_counters()
        io_bytes = int(io.read_bytes + io.write_bytes)
        elapsed = max(0.001, now - self.last_io_time)
        io_mb_s = max(0.0, io_bytes - self.last_io_bytes) / (1024 * 1024) / elapsed
        self.peak_io_mb_s = max(self.peak_io_mb_s, io_mb_s)
        self.io_excess_streak = self.io_excess_streak + 1 if io_mb_s > self.io_cap_mb_s else 0
        self.last_io_bytes = io_bytes
        self.last_io_time = now
        swap_in = int(getattr(psutil.swap_memory(), "sin", 0))
        return {"rss_mb": round(rss_mb, 3), "io_mb_s": round(io_mb_s, 3), "swap_in_delta": swap_in - self.swap_in_start}

    @property
    def avg_ram_mb(self) -> float:
        return self.ram_total_mb / max(1, self.ram_samples)


class ResourceAbort(RuntimeError):
    pass


def _dataset(frames: list[dict[str, Any]]) -> TensorDataset:
    return TensorDataset(
        torch.tensor([frame["features"] for frame in frames], dtype=torch.float32),
        torch.tensor([frame["action"] for frame in frames], dtype=torch.long),
        torch.tensor([frame["candidate"] for frame in frames], dtype=torch.long),
        torch.tensor([frame["repair"] for frame in frames], dtype=torch.long),
    )


def evaluate(model: TinyRecursivePolicy, frames: list[dict[str, Any]]) -> dict[str, float]:
    model.eval()
    dataset = _dataset(frames)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    action_ok = candidate_ok = repair_ok = count = candidate_count = repair_count = 0
    with torch.no_grad():
        for features, actions, candidates, repairs in loader:
            action_logits, candidate_logits, repair_logits = model(features)
            action_ok += int((action_logits.argmax(-1) == actions).sum())
            count += len(features)
            candidate_mask = candidates != -100
            repair_mask = repairs != -100
            if candidate_mask.any():
                candidate_ok += int((candidate_logits.argmax(-1)[candidate_mask] == candidates[candidate_mask]).sum())
                candidate_count += int(candidate_mask.sum())
            if repair_mask.any():
                repair_ok += int((repair_logits.argmax(-1)[repair_mask] == repairs[repair_mask]).sum())
                repair_count += int(repair_mask.sum())
    model.train()
    return {
        "action_accuracy": action_ok / max(1, count),
        "candidate_accuracy": candidate_ok / max(1, candidate_count),
        "repair_accuracy": repair_ok / max(1, repair_count),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the capped CPU Campsite action/ranker TRM.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-steps", type=int, default=250)
    parser.add_argument("--checkpoint-seconds", type=int, default=60)
    parser.add_argument("--io-cap-mb-s", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=73_000)
    args = parser.parse_args(argv)

    if os.getenv("INT3_CAP_WRAPPER_ACTIVE") != "1":
        raise SystemExit("Refusing uncapped training: run scripts/run_capped_int3_trm.ps1")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    events = run_dir / "events.jsonl"
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    monitor = ResourceMonitor.create(args.io_cap_mb_s)
    started = time.monotonic()
    status = "failed"
    abort_reason = ""
    step = 0
    checkpoint_paths: list[str] = []
    model: TinyRecursivePolicy | None = None
    optimizer: torch.optim.Optimizer | None = None

    append_event(events, {"ts": utc_now(), "event": "start", "training_task_id": TRAINING_TASK_ID})
    try:
        rows = parse_normalized_campsite_rows(Path(args.source))
        development = sorted((task for task, _ in rows if not stable_holdout(task)), key=lambda task: task.hash)
        train_tasks, val_tasks = development[:64], development[64:]
        train_frames = build_frames(train_tasks)
        val_frames = build_frames(val_tasks)
        write_payload = {
            "training_task_id": TRAINING_TASK_ID,
            "train_task_count": len(train_tasks),
            "validation_task_count": len(val_tasks),
            "train_frame_count": len(train_frames),
            "validation_frame_count": len(val_frames),
            "chunk_strategy": "256-frame shards; minibatch 32",
            "target_access": "verifier-derived actions only",
        }
        (run_dir / "dataset_summary.json").write_text(json.dumps(write_payload, indent=2), encoding="utf-8")

        model = TinyRecursivePolicy()
        if model.manifest()["parameter_count"] >= 50_000:
            raise RuntimeError("TRM exceeds 50k parameter limit")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loader = DataLoader(_dataset(train_frames), batch_size=args.batch_size, shuffle=True)
        action_loss = nn.CrossEntropyLoss()
        candidate_loss = nn.CrossEntropyLoss(ignore_index=-100)
        repair_loss = nn.CrossEntropyLoss(ignore_index=-100)
        last_checkpoint = time.monotonic()

        while step < args.max_steps:
            for features, actions, candidates, repairs in loader:
                step += 1
                optimizer.zero_grad(set_to_none=True)
                action_logits, candidate_logits, repair_logits = model(features)
                loss = action_loss(action_logits, actions)
                if bool((candidates != -100).any()):
                    loss = loss + candidate_loss(candidate_logits, candidates)
                if bool((repairs != -100).any()):
                    loss = loss + repair_loss(repair_logits, repairs)
                loss.backward()
                optimizer.step()

                sample = monitor.sample()
                if monitor.io_excess_streak >= 3:
                    raise ResourceAbort("sustained_io_cap_exceeded")
                if sample["swap_in_delta"] > 64 * 1024 * 1024:
                    raise ResourceAbort("swap_activity_detected")

                checkpoint_due = step % args.checkpoint_steps == 0 or time.monotonic() - last_checkpoint >= args.checkpoint_seconds
                if checkpoint_due or step >= args.max_steps:
                    validation = evaluate(model, val_frames)
                    checkpoint = checkpoints / f"step-{step:06d}.pt"
                    torch.save(
                        {"step": step, "model": model.state_dict(), "manifest": model.manifest(), "validation": validation},
                        checkpoint,
                    )
                    checkpoint_paths.append(str(checkpoint))
                    append_event(
                        events,
                        {
                            "ts": utc_now(),
                            "event": "checkpoint",
                            "step": step,
                            "loss": round(float(loss.detach()), 6),
                            "validation": validation,
                            "resources": sample,
                            "path": str(checkpoint),
                        },
                    )
                    last_checkpoint = time.monotonic()
                if step >= args.max_steps:
                    break
        status = "completed"
    except ResourceAbort as exc:
        status = "aborted"
        abort_reason = str(exc)
        append_event(events, {"ts": utc_now(), "event": "abort", "step": step, "reason": abort_reason})
    finally:
        final_validation = evaluate(model, val_frames) if model is not None and "val_frames" in locals() else {}
        summary = {
            "training_task_id": TRAINING_TASK_ID,
            "status": status,
            "abort_reason": abort_reason,
            "steps_completed": step,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_ram_mb": round(monitor.peak_ram_mb, 3),
            "avg_ram_mb": round(monitor.avg_ram_mb, 3),
            "peak_io_mb_s": round(monitor.peak_io_mb_s, 3),
            "cpu_pct": 50,
            "checkpoints": checkpoint_paths,
            "validation": final_validation,
            "model": model.manifest() if model is not None else {},
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        append_event(events, {"ts": utc_now(), "event": "complete", "summary": summary})
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    return 0 if status in {"completed", "aborted"} else 1


if __name__ == "__main__":
    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # This process is always wrapper-owned. Avoid a Windows PyTorch DLL
    # teardown fault after the trainer has already released objects and logged
    # its final summary; the Job Object remains the authoritative cleanup edge.
    os._exit(exit_code)
