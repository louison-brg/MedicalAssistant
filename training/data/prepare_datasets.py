"""Create a Hugging Face dataset with assistant-only causal-LM labels."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
DATASET_CACHE = BASE_DIR / ".dataset_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize chat JSONL for the legacy HF trainer.")
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "mlx_data")
    parser.add_argument("--output", type=Path, default=BASE_DIR / "processed_professor_phi3_hf")
    parser.add_argument("--tokenizer", default="training/models/phi-3-mini-4k-instruct")
    parser.add_argument("--max-length", type=int, default=2_048)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def legacy_text_to_messages(text: str) -> Optional[list[dict[str, str]]]:
    text = (text or "").strip()
    if not text:
        return None

    phi3 = re.search(
        r"<\|user\|>\s*(.*?)\s*<\|end\|>\s*<\|assistant\|>\s*(.*?)\s*<\|end\|>",
        text,
        re.DOTALL,
    )
    if phi3:
        return [
            {"role": "user", "content": phi3.group(1).strip()},
            {"role": "assistant", "content": phi3.group(2).strip()},
        ]

    for pattern in (
        r"Student:\s*(.*?)\s*Professor:\s*(.*)",
        r"Patient:\s*(.*?)\s*Doctor:\s*(.*)",
    ):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return [
                {"role": "user", "content": match.group(1).strip()},
                {"role": "assistant", "content": match.group(2).strip()},
            ]
    return None


def normalize_messages(example: dict) -> dict:
    messages = example.get("messages")
    if not messages:
        messages = legacy_text_to_messages(example.get("text", ""))
    if not messages or len(messages) < 2:
        return {"messages": []}

    normalized = []
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in {"system", "user", "assistant"} and content:
            normalized.append({"role": role, "content": content})
    if not normalized or normalized[-1]["role"] != "assistant":
        return {"messages": []}
    return {"messages": normalized}


def tokenize_chat(example: dict, tokenizer, max_length: int) -> dict:
    messages = example["messages"]
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
    )

    full_ids = full_ids[:max_length]
    prompt_length = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_length + list(full_ids[prompt_length:])
    return {
        "input_ids": list(full_ids),
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def main() -> int:
    args = parse_args()
    train_path = args.data_dir / "train.jsonl"
    valid_path = args.data_dir / "valid.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(f"Expected train.jsonl and valid.jsonl in {args.data_dir}")
    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output}. Pass --overwrite to replace it.")
        shutil.rmtree(args.output)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(valid_path)},
        cache_dir=str(DATASET_CACHE),
    )
    dataset = dataset.map(normalize_messages)
    dataset = dataset.filter(lambda example: bool(example["messages"]))
    original_columns = dataset["train"].column_names
    tokenized = dataset.map(
        lambda example: tokenize_chat(example, tokenizer, args.max_length),
        remove_columns=original_columns,
    )
    tokenized = tokenized.filter(lambda example: any(label != -100 for label in example["labels"]))
    tokenized.save_to_disk(args.output)
    print(
        f"Saved {len(tokenized['train']):,} train and "
        f"{len(tokenized['validation']):,} validation samples to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
