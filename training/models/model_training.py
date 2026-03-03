"""
MLX LoRA training launcher for this repository.

This script wraps `python -m mlx_lm.lora` and only forwards options
that are supported by the installed mlx_lm version.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch MLX LoRA training.")
    parser.add_argument("--python", default=sys.executable, help="Python executable to run mlx_lm.lora")
    parser.add_argument("--model-path", default="training/models/phi-3-mini-4k-instruct")
    parser.add_argument("--data-path", default="training/data/mlx_data")
    parser.add_argument("--adapter-path", default="training/models/checkpoints_phi3_mlx")

    parser.add_argument("--iters", type=int, default=1250)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=200)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-scale", type=float, default=20.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--resume-adapter-file", default=None)
    parser.add_argument("--extra-args", default="", help="Extra raw args forwarded to mlx_lm.lora")
    parser.add_argument("--dry-run", action="store_true")

    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument("--mask-prompt", dest="mask_prompt", action="store_true")
    mask_group.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false")
    parser.set_defaults(mask_prompt=True)

    gc_group = parser.add_mutually_exclusive_group()
    gc_group.add_argument("--grad-checkpoint", dest="grad_checkpoint", action="store_true")
    gc_group.add_argument("--no-grad-checkpoint", dest="grad_checkpoint", action="store_false")
    parser.set_defaults(grad_checkpoint=True)

    return parser.parse_args()


def discover_supported_flags(python_exec: str) -> set[str]:
    probe = subprocess.run(
        [python_exec, "-m", "mlx_lm.lora", "--help"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        details = (probe.stdout or "") + "\n" + (probe.stderr or "")
        raise RuntimeError(
            "Unable to query mlx_lm.lora help. Install MLX dependencies first.\n"
            "Try: pip install -r requirements-macos.txt\n\n"
            f"Details:\n{details.strip()}"
        )

    text = (probe.stdout or "") + "\n" + (probe.stderr or "")
    return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", text))


def _append_option(
    cmd: list[str],
    supported: set[str],
    aliases: Sequence[str],
    value: Optional[object] = None,
    bool_flag: bool = False,
) -> Optional[str]:
    for flag in aliases:
        if flag not in supported:
            continue
        if bool_flag:
            if bool(value):
                cmd.append(flag)
            return flag

        if value is not None:
            cmd.extend([flag, str(value)])
        return flag
    return None


def _validate_data_path(data_path: Path) -> None:
    if data_path.is_dir():
        train = data_path / "train.jsonl"
        valid = data_path / "valid.jsonl"
        if not train.exists() or not valid.exists():
            raise FileNotFoundError(
                f"Missing train/valid JSONL files in {data_path}. Expected train.jsonl and valid.jsonl"
            )
    elif not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")


def build_command(args: argparse.Namespace, supported: set[str]) -> list[str]:
    data_path = Path(args.data_path)
    _validate_data_path(data_path)

    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [args.python, "-m", "mlx_lm.lora"]

    _append_option(cmd, supported, ["--model"], args.model_path)
    _append_option(cmd, supported, ["--data"], args.data_path)
    _append_option(cmd, supported, ["--adapter-path"], args.adapter_path)
    _append_option(cmd, supported, ["--resume-adapter-file", "--resume-adapter"], args.resume_adapter_file)

    # Training mode
    _append_option(cmd, supported, ["--train"], True, bool_flag=True)

    # Main hyperparameters
    _append_option(cmd, supported, ["--iters", "--num-iters"], args.iters)
    _append_option(cmd, supported, ["--batch-size"], args.batch_size)
    _append_option(cmd, supported, ["--learning-rate", "--lr"], args.learning_rate)
    _append_option(cmd, supported, ["--save-every"], args.save_every)
    _append_option(cmd, supported, ["--steps-per-report"], args.steps_per_report)
    _append_option(cmd, supported, ["--steps-per-eval"], args.steps_per_eval)
    _append_option(cmd, supported, ["--val-batches"], args.val_batches)
    _append_option(cmd, supported, ["--max-seq-length", "--max-length"], args.max_seq_length)
    _append_option(cmd, supported, ["--num-layers", "--lora-layers"], args.num_layers)
    _append_option(cmd, supported, ["--seed"], args.seed)

    # LoRA options (version-dependent names)
    _append_option(cmd, supported, ["--lora-rank", "--rank"], args.lora_rank)
    _append_option(cmd, supported, ["--lora-scale", "--lora-alpha", "--alpha"], args.lora_scale)
    _append_option(cmd, supported, ["--lora-dropout", "--dropout"], args.lora_dropout)

    # Bool options
    _append_option(cmd, supported, ["--mask-prompt"], args.mask_prompt, bool_flag=True)
    _append_option(cmd, supported, ["--grad-checkpoint"], args.grad_checkpoint, bool_flag=True)

    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args.strip()))

    return cmd


def main() -> int:
    args = parse_args()
    try:
        supported = discover_supported_flags(args.python)
        command = build_command(args, supported)
    except Exception as exc:
        print(f"❌ {exc}")
        return 1

    print("🚀 Launching MLX LoRA training:")
    print(" ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        print("ℹ️ Dry run only.")
        return 0

    process = subprocess.run(command)
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
