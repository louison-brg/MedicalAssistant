"""Deterministic factual benchmark for local MLX medical checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark an MLX model on held-out MedQA rows.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True, help="Validation JSONL in messages format")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return " ".join(re.findall(r"[a-z0-9]+", decomposed))


def reference_answer(target: str) -> str:
    patterns = (
        r"^The most appropriate choice is (.+)\.$",
        r"^In this clinical scenario, (.+) is the best answer\.$",
        r"^Based on the findings described, choose (.+)\.$",
    )
    for pattern in patterns:
        match = re.match(pattern, target.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return target.strip().rstrip(".")


def answer_option(prompt: str, reference: str) -> str | None:
    normalized_reference = normalize(reference)
    for key, option in re.findall(r"^([A-Z0-9]+)\.\s+(.+)$", prompt, flags=re.MULTILINE):
        if normalize(option) == normalized_reference:
            return key
    return None


def is_correct(response: str, reference: str, option: str | None) -> bool:
    normalized_response = normalize(response)
    normalized_reference = normalize(reference)
    if normalized_reference and normalized_reference in normalized_response:
        return True
    if option:
        stripped = response.strip().upper().strip(".():[] ")
        if stripped == option:
            return True
        return bool(
            re.search(
                rf"\b(?:answer|choice|option)\s*(?:is|:)?\s*{re.escape(option)}\b",
                response,
                flags=re.IGNORECASE,
            )
        )
    return False


def load_rows(path: Path, max_samples: int, seed: int) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row.get("source") == "medqa"]
    if not rows:
        raise ValueError(f"No MedQA rows found in {path}")
    random.Random(seed).shuffle(rows)
    return rows[: min(max_samples, len(rows))]


def main() -> int:
    args = parse_args()
    if args.max_samples <= 0 or args.max_tokens <= 0:
        raise ValueError("max-samples and max-tokens must be positive")

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    rows = load_rows(args.data, args.max_samples, args.seed)
    model, tokenizer = load(args.model)
    sampler = make_sampler(temp=0.0)
    results = []

    for index, row in enumerate(rows, start=1):
        user_message = row["messages"][0]
        target = row["messages"][-1]["content"]
        reference = reference_answer(target)
        prompt = tokenizer.apply_chat_template(
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            sampler=sampler,
            verbose=False,
        ).strip()
        option = answer_option(user_message["content"], reference)
        correct = is_correct(response, reference, option)
        results.append(
            {
                "correct": correct,
                "reference": reference,
                "reference_option": option,
                "response": response,
                "question": user_message["content"],
            }
        )
        if index % 10 == 0 or index == len(rows):
            hits = sum(item["correct"] for item in results)
            print(f"{index}/{len(rows)} | exact-answer accuracy: {hits / index:.1%}")

    correct = sum(item["correct"] for item in results)
    summary = {
        "model": args.model,
        "data": str(args.data),
        "seed": args.seed,
        "samples": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "empty_responses": sum(not item["response"] for item in results),
        "failures": [item for item in results if not item["correct"]][:20],
    }
    print(json.dumps({key: value for key, value in summary.items() if key != "failures"}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
