"""Reproducible MLX LoRA/QLoRA training launcher.

mlx-lm exposes several important LoRA settings only through its YAML config.
JSON is a valid YAML subset, so this launcher writes a versionable JSON config
and invokes the supported unified CLI.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch reproducible MLX LoRA training.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model-path", default="training/models/phi-3-mini-4k-instruct")
    parser.add_argument("--data-path", default="training/data/mlx_data")
    parser.add_argument("--adapter-path", default="training/models/checkpoints_phi3_mlx")
    parser.add_argument("--config-path", default=None, help="Defaults to <adapter-path>/training_config.json")

    parser.add_argument("--iters", type=int, default=1_250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="Peak AdamW learning rate; conservative MLX-LM default for LoRA fine-tuning",
    )
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--max-seq-length", type=int, default=2_048)
    parser.add_argument("--num-layers", type=int, default=16, help="Last transformer layers to adapt; -1 means all")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=20.0,
        help="Direct MLX LoRA delta multiplier (not PEFT alpha/r)",
    )
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-keys",
        default="",
        help="Optional comma-separated module paths. Empty uses mlx-lm model defaults.",
    )
    parser.add_argument("--resume-adapter-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    model = Path(args.model_path)
    data = Path(args.data_path)
    if not model.exists():
        raise FileNotFoundError(f"Model path not found: {model}")
    for name in ("train.jsonl", "valid.jsonl"):
        file = data / name
        if not file.exists():
            raise FileNotFoundError(f"Dataset file not found: {file}")
        with file.open(encoding="utf-8") as handle:
            first_line = next((line for line in handle if line.strip()), None)
        if not first_line:
            raise ValueError(f"Dataset file is empty: {file}")
        sample = json.loads(first_line)
        messages = sample.get("messages")
        if not messages or messages[-1].get("role") != "assistant":
            raise ValueError(
                f"{file} must use mlx-lm chat format and end each sample with an assistant message"
            )
    if args.iters <= 0 or args.batch_size <= 0 or args.grad_accumulation_steps <= 0:
        raise ValueError("iters, batch size, and gradient accumulation must be positive")
    if args.lora_rank <= 0 or args.lora_scale <= 0:
        raise ValueError("LoRA rank and scale must be positive")
    if args.warmup_steps < 0 or args.warmup_steps >= args.iters:
        raise ValueError("warmup-steps must be >= 0 and smaller than iters")


def build_config(args: argparse.Namespace) -> dict:
    lora_parameters: dict[str, object] = {
        "rank": args.lora_rank,
        "scale": args.lora_scale,
        "dropout": args.lora_dropout,
    }
    keys = [key.strip() for key in args.lora_keys.split(",") if key.strip()]
    if keys:
        lora_parameters["keys"] = keys

    decay_steps = max(1, args.iters - args.warmup_steps)
    return {
        "model": args.model_path,
        "train": True,
        "fine_tune_type": "lora",
        "data": args.data_path,
        "seed": args.seed,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "grad_accumulation_steps": args.grad_accumulation_steps,
        "iters": args.iters,
        "val_batches": args.val_batches,
        "learning_rate": args.learning_rate,
        "lr_schedule": {
            "name": "cosine_decay",
            "arguments": [args.learning_rate, decay_steps],
            "warmup": args.warmup_steps,
            "warmup_init": args.learning_rate * 0.1,
        },
        "optimizer": "adamw",
        "optimizer_config": {
            "adam": {},
            "adamw": {"weight_decay": args.weight_decay},
            "muon": {},
            "sgd": {},
            "adafactor": {},
        },
        "steps_per_report": args.steps_per_report,
        "steps_per_eval": args.steps_per_eval,
        "adapter_path": args.adapter_path,
        "save_every": args.save_every,
        "max_seq_length": args.max_seq_length,
        "grad_checkpoint": True,
        "mask_prompt": True,
        "resume_adapter_file": args.resume_adapter_file,
        "lora_parameters": lora_parameters,
    }


def command_for(args: argparse.Namespace, config_path: Path) -> list[str]:
    return [args.python, "-m", "mlx_lm", "lora", "--config", str(config_path)]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        installed_version = version("mlx-lm")
    except (ValueError, FileNotFoundError, PackageNotFoundError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    config = build_config(args)
    config_path = Path(args.config_path or Path(args.adapter_path) / "training_config.json")
    command = command_for(args, config_path)
    print(f"mlx-lm {installed_version}")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print("Command:", " ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return 0

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
