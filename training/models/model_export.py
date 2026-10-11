"""Fuse a LoRA adapter and optionally export a 4-bit MLX deployment model."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse LoRA adapters for deployment.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--backend", choices=["auto", "mlx", "peft"], default="auto")
    parser.add_argument("--base-model", default="training/models/phi-3-mini-4k-instruct")
    parser.add_argument("--adapter", default="training/models/checkpoints_phi3_mlx")
    parser.add_argument("--fused-output", default="training/models/phi3-medprof-fused-mlx")
    parser.add_argument("--quantized-output", default="training/models/Phi3_Medical_Latest")
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, choices=[2, 3, 4, 6, 8], default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument("--q-mode", choices=["affine", "mxfp4"], default="affine")
    parser.add_argument("--max-shard-size", default="4GB", help="PEFT/HF merge only")
    parser.add_argument("--trust-remote-code", action="store_true", help="PEFT/HF merge only")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def detect_backend(adapter: Path) -> str:
    if (adapter / "adapters.safetensors").exists():
        return "mlx"
    if (adapter / "adapter_model.safetensors").exists() or (adapter / "adapter_model.bin").exists():
        return "peft"
    raise FileNotFoundError(f"No MLX or PEFT adapter weights found in {adapter}")


def validate(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, str]:
    base = Path(args.base_model)
    adapter = Path(args.adapter)
    fused = Path(args.fused_output)
    quantized = Path(args.quantized_output)
    if not base.exists():
        raise FileNotFoundError(f"Base model not found: {base}")
    if not adapter.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter}")
    backend = detect_backend(adapter) if args.backend == "auto" else args.backend
    expected = "adapters.safetensors" if backend == "mlx" else "adapter_config.json"
    if not (adapter / expected).exists():
        raise FileNotFoundError(f"Expected {expected} for {backend} adapter in {adapter}")
    if fused.exists():
        raise FileExistsError(f"Refusing to overwrite fused output: {fused}")
    if not args.no_quantize and quantized.exists():
        raise FileExistsError(f"Refusing to overwrite quantized output: {quantized}")
    if args.q_group_size <= 0:
        raise ValueError("q-group-size must be positive")
    return base, adapter, fused, quantized, backend


def mlx_commands(args: argparse.Namespace, base: Path, adapter: Path, fused: Path, quantized: Path) -> list[list[str]]:
    commands = [
        [
            args.python,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            str(base),
            "--adapter-path",
            str(adapter),
            "--save-path",
            str(fused),
        ]
    ]
    if not args.no_quantize:
        commands.append(
            [
                args.python,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                str(fused),
                "--mlx-path",
                str(quantized),
                "--quantize",
                "--q-bits",
                str(args.q_bits),
                "--q-group-size",
                str(args.q_group_size),
                "--q-mode",
                args.q_mode,
            ]
        )
    return commands


def merge_peft(args: argparse.Namespace, base: Path, adapter: Path, fused: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
        device_map={"": "cpu"},
        attn_implementation="eager",
    )
    peft_model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    merged = peft_model.merge_and_unload(safe_merge=True)
    fused.mkdir(parents=True, exist_ok=False)
    merged.save_pretrained(
        fused,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer_source = adapter if (adapter / "tokenizer_config.json").exists() else base
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
    tokenizer.save_pretrained(fused)


def deployment_files_exist(path: Path) -> bool:
    return (path / "config.json").exists() and (path / "tokenizer.json").exists() and any(
        path.glob("*.safetensors")
    )


def write_manifest(path: Path, args: argparse.Namespace, backend: str) -> None:
    packages = {}
    for package in ("mlx", "mlx-lm", "torch", "transformers", "peft"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "quantization": None
        if args.no_quantize
        else {"bits": args.q_bits, "group_size": args.q_group_size, "mode": args.q_mode},
        "packages": packages,
    }
    (path / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        base, adapter, fused, quantized, backend = validate(args)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    commands: list[list[str]] = []
    if backend == "mlx":
        commands = mlx_commands(args, base, adapter, fused, quantized)
    elif not args.no_quantize:
        commands = [
            [
                args.python,
                "-m",
                "mlx_lm",
                "convert",
                "--hf-path",
                str(fused),
                "--mlx-path",
                str(quantized),
                "--quantize",
                "--q-bits",
                str(args.q_bits),
                "--q-group-size",
                str(args.q_group_size),
                "--q-mode",
                args.q_mode,
            ]
        ]

    print(f"Backend: {backend}")
    if backend == "peft":
        print(f"PEFT merge: {base} + {adapter} -> {fused}")
    for command in commands:
        print("Command:", " ".join(shlex.quote(part) for part in command))
    if args.dry_run:
        return 0

    try:
        if backend == "peft":
            merge_peft(args, base, adapter, fused)
        for command in commands:
            subprocess.run(command, check=True)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Export failed: {error}", file=sys.stderr)
        return 1

    deployment = fused if args.no_quantize else quantized
    if not deployment_files_exist(deployment):
        print(f"Export incomplete: required model/tokenizer files are missing in {deployment}", file=sys.stderr)
        return 1
    write_manifest(deployment, args, backend)
    print(f"Deployment model ready: {deployment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
