"""Fail-fast quality checks for raw chat JSONL or tokenized HF datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit training dataset structure and leakage.")
    parser.add_argument("--data-path", type=Path, default=Path("training/data/mlx_data"))
    parser.add_argument("--max-duplicate-ratio", type=float, default=0.01)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {error}") from error
    return rows


def fingerprint(row: dict) -> str:
    if row.get("messages"):
        value = "\n".join(str(item.get("content", "")).strip().casefold() for item in row["messages"])
    elif row.get("input_ids"):
        value = ",".join(map(str, row["input_ids"]))
    else:
        value = str(row.get("text", "")).strip().casefold()
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def audit_chat_rows(rows: list[dict], split: str) -> tuple[list[str], set[str]]:
    errors: list[str] = []
    hashes = set()
    duplicate_count = 0
    assistant_lengths = []
    source_counts = Counter()

    for index, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            errors.append(f"{split}[{index}] has no valid messages list")
            continue
        roles = [message.get("role") for message in messages]
        if roles[-1] != "assistant" or "user" not in roles:
            errors.append(f"{split}[{index}] must contain a user turn and end with assistant")
        if any(not str(message.get("content", "")).strip() for message in messages):
            errors.append(f"{split}[{index}] contains empty content")
        if "<|assistant|>" in json.dumps(messages) or "<|user|>" in json.dumps(messages):
            errors.append(f"{split}[{index}] contains preformatted control tokens")

        assistant_lengths.append(len(str(messages[-1].get("content", ""))))
        source_counts[str(row.get("source", "unknown"))] += 1
        digest = fingerprint(row)
        if digest in hashes:
            duplicate_count += 1
        hashes.add(digest)

    ratio = duplicate_count / max(1, len(rows))
    print(f"{split}: {len(rows):,} rows, {duplicate_count:,} duplicates ({ratio:.2%})")
    if assistant_lengths:
        sorted_lengths = sorted(assistant_lengths)
        p95 = sorted_lengths[min(len(sorted_lengths) - 1, int(len(sorted_lengths) * 0.95))]
        print(
            f"  assistant chars: median={statistics.median(assistant_lengths):.0f}, "
            f"p95={p95:,}, max={max(assistant_lengths):,}"
        )
    print("  sources:", ", ".join(f"{name}={count:,}" for name, count in sorted(source_counts.items())))
    return errors, hashes


def audit_tokenized(path: Path) -> int:
    from datasets import DatasetDict, load_from_disk

    dataset = load_from_disk(path)
    splits = dataset if isinstance(dataset, DatasetDict) else {"data": dataset}
    failures = []
    for name, split in splits.items():
        lengths = [len(row["input_ids"]) for row in split]
        supervised = [sum(label != -100 for label in row.get("labels", [])) for row in split]
        if not lengths or any(length == 0 for length in lengths):
            failures.append(f"{name} contains empty token sequences")
        if any(count == 0 for count in supervised):
            failures.append(f"{name} contains examples without assistant labels")
        print(
            f"{name}: {len(split):,} rows, median={statistics.median(lengths):.0f} tokens, "
            f"max={max(lengths):,}, median supervised={statistics.median(supervised):.0f}"
        )
    if failures:
        print("FAILED")
        for failure in failures:
            print(" -", failure)
        return 1
    print("Dataset checks passed")
    return 0


def main() -> int:
    args = parse_args()
    if (args.data_path / "train.jsonl").exists():
        train = read_jsonl(args.data_path / "train.jsonl")
        valid = read_jsonl(args.data_path / "valid.jsonl")
        train_errors, train_hashes = audit_chat_rows(train, "train")
        valid_errors, valid_hashes = audit_chat_rows(valid, "valid")
        overlap = train_hashes & valid_hashes
        duplicate_ratio = (len(train) - len(train_hashes) + len(valid) - len(valid_hashes)) / max(
            1, len(train) + len(valid)
        )
        failures = train_errors + valid_errors
        if overlap:
            failures.append(f"train/validation leakage: {len(overlap)} identical examples")
        if duplicate_ratio > args.max_duplicate_ratio:
            failures.append(
                f"duplicate ratio {duplicate_ratio:.2%} exceeds {args.max_duplicate_ratio:.2%}"
            )
        if failures:
            print(f"FAILED with {len(failures)} issue(s)")
            for failure in failures[:25]:
                print(" -", failure)
            return 1
        print("Dataset checks passed")
        return 0

    if args.data_path.exists():
        return audit_tokenized(args.data_path)
    raise FileNotFoundError(f"Dataset not found: {args.data_path}")


if __name__ == "__main__":
    raise SystemExit(main())
