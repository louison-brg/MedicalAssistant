"""Generate a smoke-test response from a merged model or PEFT checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a model or LoRA checkpoint.")
    parser.add_argument("--checkpoint", default="training/models/checkpoints_phi3_lora")
    parser.add_argument("--base-model", default=None, help="Override PEFT adapter base model")
    parser.add_argument("--prompt", default="I have felt tired and dizzy for three days. What should I do?")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def is_mlx_model(checkpoint: Path) -> bool:
    config_path = checkpoint / "config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return bool(config.get("quantization"))


def smoke_test_mlx(checkpoint: Path, prompt_text: str, max_new_tokens: int) -> int:
    from mlx_lm import generate, load

    model, tokenizer = load(str(checkpoint))
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
    )
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        verbose=True,
    )
    if not response.strip():
        raise RuntimeError("The MLX model generated an empty response")
    return 0


def main() -> int:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    if is_mlx_model(checkpoint):
        return smoke_test_mlx(checkpoint, args.prompt, args.max_new_tokens)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    adapter_config_path = checkpoint / "adapter_config.json"

    if adapter_config_path.exists() and (
        (checkpoint / "adapter_model.safetensors").exists() or (checkpoint / "adapter_model.bin").exists()
    ):
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_path = args.base_model or adapter_config.get("base_model_name_or_path")
        if not base_path:
            raise ValueError("Unable to determine the adapter base model; pass --base-model")
        from peft import PeftModel

        tokenizer_source = checkpoint if (checkpoint / "tokenizer_config.json").exists() else base_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=args.trust_remote_code)
        base = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base, checkpoint)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=args.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            trust_remote_code=args.trust_remote_code,
            attn_implementation="eager",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device).eval()
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generated_tokens = output[0, inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
