"""Build leakage-resistant chat JSONL files for MLX LoRA training.

The output uses mlx-lm's native ``messages`` format. This is important: prompt
masking is not supported for preformatted ``text`` examples, so the older
pipeline inadvertently trained on both user and assistant tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent
DATASET_CACHE = BASE_DIR / ".dataset_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare medical chat data for mlx-lm.")
    parser.add_argument("--raw-dir", type=Path, default=BASE_DIR / "raw")
    parser.add_argument("--meddialog-path", type=Path, default=BASE_DIR / "processed/english-train.json")
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "mlx_data")
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-medqa-examples", type=int, default=2_500)
    parser.add_argument("--max-textbook-examples", type=int, default=15_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in normalized.split("\n"):
        compact = re.sub(r"[ \t]+", " ", line).strip()
        if compact:
            lines.append(compact)
    return "\n".join(lines).strip()


def looks_noisy(text: str) -> bool:
    lowered = text.lower().strip()
    if not lowered:
        return True
    banned_fragments = (
        "the correct answer is",
        "explanation: step",
        "this is a riddle",
    )
    if any(fragment in lowered for fragment in banned_fragments):
        return True
    words = re.findall(r"[a-zA-ZÀ-ÿ']+", lowered)
    return len(words) >= 12 and len(set(words)) / len(words) < 0.34


def chat_example(user: str, assistant: str, *, source: str, group_id: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": clean_text(user)},
            {"role": "assistant", "content": clean_text(assistant)},
        ],
        "source": source,
        "group_id": group_id,
    }


def format_medqa(example: dict, index: int) -> Optional[dict]:
    question = clean_text(example.get("question"))
    answer = clean_text(example.get("answer"))
    options = example.get("options")
    if not question or not answer or looks_noisy(answer):
        return None

    option_lines = []
    if isinstance(options, dict):
        values = ((str(key), options[key]) for key in sorted(options))
    elif isinstance(options, list):
        values = ((str(i), value) for i, value in enumerate(options, start=1))
    else:
        values = ()
    for key, value in values:
        cleaned = clean_text(value)
        if cleaned:
            option_lines.append(f"{key}. {cleaned}")

    user_parts = ["Choose the single best answer for this medical exam question.", question]
    if option_lines:
        user_parts.append("Options:\n" + "\n".join(option_lines))
    answer = (
        f"The most appropriate choice is {answer}."
        if index % 3 == 0
        else f"In this clinical scenario, {answer} is the best answer."
        if index % 3 == 1
        else f"Based on the findings described, choose {answer}."
    )
    return chat_example(
        "\n\n".join(user_parts),
        answer,
        source="medqa",
        group_id=f"medqa:{index}",
    )


def sentence_chunks(text: str, max_chars: int = 1_800, overlap_chars: int = 180) -> Iterable[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text).replace("\n", " "))
    buffer: list[str] = []
    current_length = 0

    for sentence in (item.strip() for item in sentences):
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if buffer:
                yield " ".join(buffer)
                buffer = []
                current_length = 0
            for start in range(0, len(sentence), max_chars):
                chunk = sentence[start : start + max_chars].strip()
                if chunk:
                    yield chunk
            continue

        projected = current_length + len(sentence) + (1 if buffer else 0)
        if projected <= max_chars:
            buffer.append(sentence)
            current_length = projected
            continue

        yield " ".join(buffer)
        overlap: list[str] = []
        overlap_length = 0
        for previous in reversed(buffer):
            if overlap_length + len(previous) > overlap_chars:
                break
            overlap.insert(0, previous)
            overlap_length += len(previous) + 1
        buffer = overlap + [sentence]
        current_length = sum(map(len, buffer)) + max(0, len(buffer) - 1)

    if buffer:
        yield " ".join(buffer)


def parse_turn(raw_turn: str) -> Optional[Tuple[str, str]]:
    if ":" not in raw_turn:
        return None
    speaker, content = raw_turn.split(":", 1)
    role = speaker.strip().lower()
    text = clean_text(content)
    if role not in {"patient", "doctor"} or not text:
        return None
    return role, text


def meddialog_rows(example: dict, index: int) -> list[dict]:
    description = clean_text(example.get("description"))
    utterances = example.get("utterances") or []
    parsed = [turn for turn in (parse_turn(item) for item in utterances) if turn]
    rows = []
    history: list[Tuple[str, str]] = []

    for position, (role, text) in enumerate(parsed):
        if role != "patient":
            history.append((role, text))
            continue

        doctor_reply = next(
            (reply for next_role, reply in parsed[position + 1 :] if next_role == "doctor"),
            "",
        )
        if doctor_reply and not looks_noisy(doctor_reply):
            dialogue = [
                f"{'Patient' if turn_role == 'patient' else 'Doctor'}: {turn_text}"
                for turn_role, turn_text in history[-4:] + [("patient", text)]
            ]
            user_parts = []
            if description:
                user_parts.append(f"Case summary: {description}")
            user_parts.extend(
                [
                    "Conversation:\n" + "\n".join(dialogue),
                    "Reply to the latest patient message. Be clear about uncertainty and urgent warning signs.",
                ]
            )
            rows.append(
                chat_example(
                    "\n\n".join(user_parts),
                    doctor_reply,
                    source="meddialog",
                    group_id=f"meddialog:{index}",
                )
            )
        history.append((role, text))
    return rows


def fingerprint(row: dict) -> str:
    canonical = "\n".join(message["content"].casefold() for message in row["messages"])
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def deduplicate(rows: Sequence[dict]) -> list[dict]:
    seen: set[str] = set()
    unique = []
    for row in rows:
        key = fingerprint(row)
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def grouped_split(rows: Sequence[dict], validation_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_ratio < 0.5:
        raise ValueError("validation_ratio must be between 0 and 0.5")

    by_source: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_source[row["source"]][row["group_id"]].append(row)

    train: list[dict] = []
    valid: list[dict] = []
    for source, groups_by_id in sorted(by_source.items()):
        groups = list(groups_by_id.values())
        random.Random(f"{seed}:{source}").shuffle(groups)
        target = max(1, round(sum(map(len, groups)) * validation_ratio)) if len(groups) > 1 else 0
        validation_count = 0
        for group in groups:
            if validation_count < target and len(groups) > 1:
                valid.extend(group)
                validation_count += len(group)
            else:
                train.extend(group)

    random.Random(seed).shuffle(train)
    random.Random(seed + 1).shuffle(valid)
    return train, valid


def save_jsonl(rows: Sequence[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def build_rows(
    raw_dir: Path,
    meddialog_path: Path,
    max_medqa_examples: int,
    max_textbook_examples: int,
    seed: int,
) -> list[dict]:
    medqa_path = raw_dir / "med_qa/data_clean/data_clean/questions/US/train.jsonl"
    textbook_dir = raw_dir / "med_qa/data_clean/data_clean/textbooks/en"
    for required in (medqa_path, textbook_dir):
        if not required.exists():
            raise FileNotFoundError(f"Required data source not found: {required}")

    rows: list[dict] = []
    medqa = load_dataset(
        "json",
        data_files=str(medqa_path),
        split="train",
        cache_dir=str(DATASET_CACHE),
    )
    medqa_rows = list(filter(None, (format_medqa(example, i) for i, example in enumerate(medqa))))
    if max_medqa_examples > 0 and len(medqa_rows) > max_medqa_examples:
        random.Random(seed).shuffle(medqa_rows)
        medqa_rows = medqa_rows[:max_medqa_examples]
    rows.extend(medqa_rows)

    textbook_rows = []
    for file in sorted(textbook_dir.glob("*.txt")):
        content = file.read_text(encoding="utf-8", errors="ignore")
        if len(content) < 300:
            continue
        title = file.stem.replace("_", " ").strip()
        prompt = f"Explain this medical topic for a clinical assistant: {title}"
        for chunk in sentence_chunks(content, max_chars=900, overlap_chars=0):
            if len(chunk) >= 120 and not looks_noisy(chunk):
                textbook_rows.append(
                    chat_example(
                        prompt,
                        chunk,
                        source="textbook",
                        group_id=f"textbook:{file.stem}",
                    )
                )

    if max_textbook_examples > 0 and len(textbook_rows) > max_textbook_examples:
        random.Random(seed).shuffle(textbook_rows)
        textbook_rows = textbook_rows[:max_textbook_examples]
    rows.extend(textbook_rows)

    if meddialog_path.exists():
        meddialog = load_dataset(
            "json",
            data_files=str(meddialog_path),
            split="train",
            cache_dir=str(DATASET_CACHE),
        )
        for index, sample in enumerate(meddialog):
            rows.extend(meddialog_rows(sample, index))
    else:
        print(f"Optional MedDialog source not found; skipping: {meddialog_path}")
    return rows


def main() -> int:
    args = parse_args()
    print("Preparing medical chat dataset for MLX…")
    rows = build_rows(
        args.raw_dir,
        args.meddialog_path,
        args.max_medqa_examples,
        args.max_textbook_examples,
        args.seed,
    )
    unique_rows = deduplicate(rows)
    train, valid = grouped_split(unique_rows, args.validation_ratio, args.seed)
    save_jsonl(train, args.out_dir / "train.jsonl")
    save_jsonl(valid, args.out_dir / "valid.jsonl")

    sources = Counter(row["source"] for row in unique_rows)
    print(f"Saved {len(train):,} train / {len(valid):,} validation examples")
    print(f"Removed {len(rows) - len(unique_rows):,} exact duplicates")
    print("Sources:", ", ".join(f"{name}={count:,}" for name, count in sorted(sources.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
