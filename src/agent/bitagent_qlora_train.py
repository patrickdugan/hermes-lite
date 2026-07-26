"""Hard-cap-aware QLoRA SFT entry point for one BitAgent Bonsai role.

Use the capped wrapper. Validation mode does not import torch or load weights.
Actual training is one role adapter at a time so runtime authority stays
separable and llama.cpp can activate only the host-selected adapter.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from agent.bitagent_role_adapters import ROLES, load_config, validate_corpus


WRAPPER_FLAG = "BITAGENT_QLORA_CAP_WRAPPER_ACTIVE"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_event(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": utc_now(), **value}, ensure_ascii=False) + "\n")


def load_rows(path: Path, *, role: str, split: str) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("role") == role and row.get("split") == split:
            rows.append(row)
    return rows


def directory_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def validate_operator_inputs(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    errors = []
    if not args.training_task_id.strip():
        errors.append("training_task_id is required")
    if args.ram_cap_mb <= 0:
        errors.append("ram_cap_mb must be positive")
    if not 1 <= args.cpu_cap_pct <= 100:
        errors.append("cpu_cap_pct must be in [1, 100]")
    if args.io_cap_mb_s <= 0:
        errors.append("io_cap_mb_s must be positive")
    if args.wall_seconds <= 0:
        errors.append("wall_seconds must be positive")
    if args.checkpoint_steps <= 0 and args.checkpoint_seconds <= 0:
        errors.append("checkpoint_steps or checkpoint_seconds must be positive")
    if not args.chunk_strategy.strip():
        errors.append("chunk_strategy is required")
    if not args.base_model.strip():
        errors.append("base_model path or repo is required")
    base_path = Path(args.base_model)
    pinned_repo = config["models"]["training_base"]["repo"]
    if not base_path.exists() and args.base_model != pinned_repo:
        errors.append(f"remote base must be the pinned repo {pinned_repo}")
    if not base_path.exists() and not args.allow_download and not args.validate_only:
        errors.append("remote weight loading requires explicit --allow-download")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "training_task_id": args.training_task_id,
        "role": args.role,
        "caps": {
            "ram_mb": args.ram_cap_mb,
            "cpu_pct": args.cpu_cap_pct,
            "io_mb_s": args.io_cap_mb_s,
            "wall_seconds": args.wall_seconds,
            "minimum_free_vram_mb": args.min_free_vram_mb,
        },
        "checkpoint": {
            "steps": args.checkpoint_steps,
            "seconds": args.checkpoint_seconds,
        },
        "chunk_strategy": args.chunk_strategy,
        "base_model": args.base_model,
        "base_revision": config["models"]["training_base"]["revision"],
        "allow_download": args.allow_download,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one capped BitAgent Bonsai role adapter")
    parser.add_argument("--config", default="configs/bitagent_bonsai_role_adapters_v1.json")
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-task-id", required=True)
    parser.add_argument("--role", required=True, choices=sorted(ROLES))
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--ram-cap-mb", required=True, type=int)
    parser.add_argument("--cpu-cap-pct", required=True, type=int)
    parser.add_argument("--io-cap-mb-s", required=True, type=float)
    parser.add_argument("--wall-seconds", required=True, type=int)
    parser.add_argument("--min-free-vram-mb", required=True, type=int)
    parser.add_argument("--checkpoint-steps", type=int, default=0)
    parser.add_argument("--checkpoint-seconds", type=int, default=0)
    parser.add_argument("--chunk-strategy", required=True)
    parser.add_argument("--resume-from-checkpoint", default="")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def run_training(
    args: argparse.Namespace,
    config: dict[str, Any],
    validation: dict[str, Any],
    manifest: dict[str, Any],
    output: Path,
) -> tuple[str, str, dict[str, Any]]:
    if os.getenv(WRAPPER_FLAG) != "1":
        raise RuntimeError("Refusing uncapped training; use scripts/run_capped_bitagent_qlora.sh")

    # Heavy imports happen only after all cap, corpus, and provenance checks.
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the registered 8B QLoRA path")
    free_vram, total_vram = torch.cuda.mem_get_info()
    if free_vram / 1024 / 1024 < args.min_free_vram_mb:
        raise RuntimeError(
            f"preflight_vram_below_minimum: free={free_vram / 1024 / 1024:.0f} MB "
            f"required={args.min_free_vram_mb} MB"
        )

    events_path = output / "events.jsonl"
    training = config["training"]
    process = psutil.Process()
    started = time.monotonic()
    initial_swap = psutil.swap_memory().used
    samples: list[dict[str, float]] = []
    initial_io = process.io_counters()
    state = {
        "last_io_bytes": initial_io.read_bytes + initial_io.write_bytes,
        "last_sample": time.monotonic(),
        "last_checkpoint": time.monotonic(),
        "io_excess_streak": 0,
        "cpu_excess_streak": 0,
        "abort_reason": "",
    }

    class RoleDataset(Dataset):
        def __init__(self, items: list[dict[str, Any]]):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index: int):
            return self.items[index]

    class ResourceCallback(TrainerCallback):
        def on_step_end(self, callback_args, callback_state, control, **kwargs):
            now = time.monotonic()
            memory = process.memory_info()
            io = process.io_counters()
            io_bytes = io.read_bytes + io.write_bytes
            elapsed = max(0.001, now - state["last_sample"])
            io_rate = max(0, io_bytes - state["last_io_bytes"]) / 1024 / 1024 / elapsed
            cpu_pct = process.cpu_percent(interval=None) / max(1, psutil.cpu_count(logical=True))
            rss_mb = memory.rss / 1024 / 1024
            swap_delta_mb = max(0, psutil.swap_memory().used - initial_swap) / 1024 / 1024
            allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
            sample = {
                "step": float(callback_state.global_step),
                "rss_mb": rss_mb,
                "cpu_pct": cpu_pct,
                "io_mb_s": io_rate,
                "swap_delta_mb": swap_delta_mb,
                "cuda_allocated_mb": allocated_mb,
                "cuda_reserved_mb": reserved_mb,
            }
            samples.append(sample)
            append_event(events_path, {"event": "resource_sample", **sample})
            state["last_io_bytes"] = io_bytes
            state["last_sample"] = now
            state["io_excess_streak"] = state["io_excess_streak"] + 1 if io_rate > args.io_cap_mb_s else 0
            state["cpu_excess_streak"] = state["cpu_excess_streak"] + 1 if cpu_pct > args.cpu_cap_pct + 5 else 0
            reason = ""
            if rss_mb > args.ram_cap_mb:
                reason = "ram_cap_exceeded"
            elif state["io_excess_streak"] >= 3:
                reason = "sustained_io_cap_exceeded"
            elif state["cpu_excess_streak"] >= 5:
                reason = "sustained_cpu_cap_exceeded"
            elif swap_delta_mb > 512:
                reason = "swap_growth_exceeded"
            elif now - started > args.wall_seconds:
                reason = "wall_clock_cap"
            if reason:
                state["abort_reason"] = reason
                append_event(events_path, {"event": "abort_requested", "reason": reason, "step": callback_state.global_step})
                control.should_training_stop = True
            if args.checkpoint_seconds > 0 and now - state["last_checkpoint"] >= args.checkpoint_seconds:
                control.should_save = True
                state["last_checkpoint"] = now
            return control

    class EventCallback(TrainerCallback):
        def on_save(self, callback_args, callback_state, control, **kwargs):
            append_event(events_path, {"event": "checkpoint", "step": callback_state.global_step})

        def on_log(self, callback_args, callback_state, control, logs=None, **kwargs):
            append_event(events_path, {"event": "trainer_log", "step": callback_state.global_step, "metrics": logs or {}})

    model = tokenizer = trainer = train_dataset = eval_dataset = None
    status, abort_reason = "failed", ""
    metrics: dict[str, Any] = {}
    cleanup: dict[str, Any] = {}
    try:
        revision = config["models"]["training_base"]["revision"] if not Path(args.base_model).exists() else None
        local_only = not args.allow_download
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            revision=revision,
            local_files_only=local_only,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        compute_dtype = torch.bfloat16 if training["compute_dtype"] == "bfloat16" else torch.float16
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=training["base_quantization"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            revision=revision,
            local_files_only=local_only,
            trust_remote_code=False,
            quantization_config=quantization,
            device_map={"": 0},
            torch_dtype=compute_dtype,
        )
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(training["gradient_checkpointing"]),
        )
        lora = LoraConfig(
            r=int(training["lora_rank"]),
            lora_alpha=int(training["lora_alpha"]),
            lora_dropout=float(training["lora_dropout"]),
            target_modules=list(training["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)

        def tokenize(row: dict[str, Any]) -> dict[str, Any]:
            messages = row["messages"]
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            max_length = int(training["max_sequence_tokens"])
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            encoded = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)
            labels = list(encoded["input_ids"])
            labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
            encoded["labels"] = labels
            return encoded

        train_rows = load_rows(Path(args.examples), role=args.role, split="train")
        validation_rows = load_rows(Path(args.examples), role=args.role, split="validation")
        if not train_rows or not validation_rows:
            raise RuntimeError(f"role {args.role} needs non-empty train and validation splits")
        train_dataset = RoleDataset([tokenize(row) for row in train_rows])
        eval_dataset = RoleDataset([tokenize(row) for row in validation_rows])
        training_args_kwargs = {
            "output_dir": str(output / "checkpoints"),
            "per_device_train_batch_size": int(training["per_device_train_batch_size"]),
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "num_train_epochs": float(training["epochs"]),
            "learning_rate": float(training["learning_rate"]),
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "logging_steps": 1,
            "save_strategy": "steps",
            "save_steps": max(1, args.checkpoint_steps or sys.maxsize),
            "save_total_limit": 3,
            "eval_strategy": "epoch",
            "bf16": compute_dtype == torch.bfloat16,
            "fp16": compute_dtype == torch.float16,
            "gradient_checkpointing": bool(training["gradient_checkpointing"]),
            "report_to": [],
            "seed": int(training["seed"]),
            "data_seed": int(training["seed"]),
            "remove_unused_columns": False,
        }
        try:
            trainer_args = TrainingArguments(**training_args_kwargs)
        except TypeError:
            training_args_kwargs["evaluation_strategy"] = training_args_kwargs.pop("eval_strategy")
            trainer_args = TrainingArguments(**training_args_kwargs)
        trainer = Trainer(
            model=model,
            args=trainer_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
            callbacks=[ResourceCallback(), EventCallback()],
        )
        append_event(events_path, {
            "event": "training_started",
            "role": args.role,
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
        })
        result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
        metrics = dict(result.metrics)
        if state["abort_reason"]:
            status, abort_reason = "aborted", state["abort_reason"]
        else:
            status = "completed"
            metrics.update(trainer.evaluate())
            adapter_dir = output / "adapter"
            trainer.save_model(str(adapter_dir))
            tokenizer.save_pretrained(str(adapter_dir))
            write_json(adapter_dir / "bitagent_adapter_manifest.json", {
                "schema": "hermes.bitagent_role_adapter_manifest.v1",
                "training_task_id": args.training_task_id,
                "role": args.role,
                "base_model": config["models"]["training_base"],
                "corpus_sha256": validation["examples_sha256"],
                "authority": config["authority"],
                "allowed_tools": config["roles"][args.role]["allowed_tools"],
                "adapter_tree_sha256": directory_hash(adapter_dir),
                "runtime_status": "candidate_pending_gguf_conversion_and_heldout_promotion",
            })
    except Exception as exc:
        if not abort_reason:
            abort_reason = str(exc)
        if status != "aborted":
            status = "failed"
        append_event(events_path, {
            "event": "training_exception",
            "status": status,
            "reason": abort_reason,
            "traceback": traceback.format_exc(limit=12),
        })
    finally:
        before = {
            "rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
        }
        try:
            del trainer, model, tokenizer, train_dataset, eval_dataset
        except Exception:
            pass
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.reset_peak_memory_stats()
        gc.collect()
        after = {
            "rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
        }
        cleanup = {
            "before": before,
            "after": after,
            "cuda_cache_cleared": True,
            "owned_pid": process.pid,
            "broad_process_termination": False,
        }
        write_json(output / "cleanup_summary.json", cleanup)

    resource_summary = {
        "sample_count": len(samples),
        "peak_ram_mb": round(max((item["rss_mb"] for item in samples), default=0), 2),
        "average_ram_mb": round(sum(item["rss_mb"] for item in samples) / max(1, len(samples)), 2),
        "peak_cpu_pct": round(max((item["cpu_pct"] for item in samples), default=0), 2),
        "average_cpu_pct": round(sum(item["cpu_pct"] for item in samples) / max(1, len(samples)), 2),
        "peak_io_mb_s": round(max((item["io_mb_s"] for item in samples), default=0), 2),
        "average_io_mb_s": round(sum(item["io_mb_s"] for item in samples) / max(1, len(samples)), 2),
        "peak_cuda_reserved_mb": round(max((item["cuda_reserved_mb"] for item in samples), default=0), 2),
    }
    return status, abort_reason, {
        "metrics": metrics,
        "resources": resource_summary,
        "cleanup": cleanup,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    validation = validate_corpus(config, args.examples, promotion=False)
    operator = validate_operator_inputs(args, config)
    manifest = {
        "schema": "hermes.bitagent_qlora_run_manifest.v1",
        "created_at": utc_now(),
        **operator,
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "corpus": validation,
        "training": config["training"],
        "weight_loading_enabled": not args.validate_only,
    }
    write_json(output / "run_manifest.json", manifest)
    if args.validate_only:
        summary = {
            "schema": "hermes.bitagent_qlora_run_summary.v1",
            "status": "validated",
            "training_started": False,
            "weight_loading_started": False,
            "manifest": manifest,
        }
        write_json(output / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return 0
    status, abort_reason, details = run_training(args, config, validation, manifest, output)
    summary = {
        "schema": "hermes.bitagent_qlora_run_summary.v1",
        "status": status,
        "abort_reason": abort_reason,
        "training_started": True,
        "weight_loading_started": True,
        "manifest": manifest,
        **details,
        "completed_at": utc_now(),
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0 if status == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
