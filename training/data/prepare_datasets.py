import os
import re
from typing import List

from datasets import load_dataset
from transformers import AutoTokenizer

# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "mlx_data")
SAVE_DIR = os.path.join(BASE_DIR, "processed_professor_phi3_hf")

TOKENIZER_MODEL = "microsoft/phi-3-mini-4k-instruct"
MAX_LENGTH = 2048

# =========================
# Setup
# =========================
print(f"🚀 Processing MLX JSONL from {DATA_DIR}...")
os.makedirs(SAVE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("ℹ️ Added pad_token = eos_token")

# =========================
# 1. Load Data
# =========================
train_path = os.path.join(DATA_DIR, "train.jsonl")
valid_path = os.path.join(DATA_DIR, "valid.jsonl")

print(f"📂 Loading {train_path} and {valid_path}...")
if not os.path.exists(train_path) or not os.path.exists(valid_path):
    raise FileNotFoundError(f"Missing one of the data files in {DATA_DIR}")

dataset = load_dataset("json", data_files={"train": train_path, "validation": valid_path})


# =========================
# 2. Normalize input format
# =========================
def to_phi3_chat(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    if "<|user|>" in text and "<|assistant|>" in text:
        return text

    # Backward compatibility for old samples.
    student_prof = re.search(r"Student:\s*(.*?)\s*Professor:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if student_prof:
        user_content = student_prof.group(1).strip()
        assistant_content = student_prof.group(2).strip()
        if user_content and assistant_content:
            return f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n{assistant_content}<|end|>"

    patient_doctor = re.search(r"Patient:\s*(.*?)\s*Doctor:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if patient_doctor:
        user_content = patient_doctor.group(1).strip()
        assistant_content = patient_doctor.group(2).strip()
        if user_content and assistant_content:
            return f"<|user|>\n{user_content}<|end|>\n<|assistant|>\n{assistant_content}<|end|>"

    return ""


def format_chat(example):
    return {"text": to_phi3_chat(example["text"])}


print("🧩 Normalizing to Phi-3 chat format...")
dataset = dataset.map(format_chat)
dataset = dataset.filter(lambda ex: bool(ex["text"]))

# =========================
# 3. Tokenization & assistant-only masking
# =========================
assistant_markers: List[List[int]] = []
for marker in ["<|assistant|>\n", "<|assistant|>"]:
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    if marker_ids and marker_ids not in assistant_markers:
        assistant_markers.append(marker_ids)

if not assistant_markers:
    raise RuntimeError("Could not encode assistant markers with tokenizer.")


def find_subsequence(sequence: List[int], pattern: List[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    for idx in range(len(sequence) - len(pattern), -1, -1):
        if sequence[idx:idx + len(pattern)] == pattern:
            return idx
    return -1


def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=False,
    )
    input_ids = tokens["input_ids"]
    labels = list(input_ids)

    assistant_start = -1
    marker_length = 0
    for marker in assistant_markers:
        idx = find_subsequence(input_ids, marker)
        if idx >= 0 and idx >= assistant_start:
            assistant_start = idx
            marker_length = len(marker)

    if assistant_start < 0:
        labels = [-100] * len(labels)
    else:
        content_start = assistant_start + marker_length
        for i in range(content_start):
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": tokens["attention_mask"],
        "labels": labels,
    }


print("🔠 Tokenizing and masking...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=False,
    remove_columns=dataset["train"].column_names,
)
tokenized_dataset = tokenized_dataset.filter(lambda ex: any(label != -100 for label in ex["labels"]))

# =========================
# 4. Save
# =========================
print(f"💾 Saving to {SAVE_DIR}...")
tokenized_dataset.save_to_disk(SAVE_DIR)
print(
    f"✅ Created {len(tokenized_dataset['train'])} train and "
    f"{len(tokenized_dataset['validation'])} validation samples."
)
print(f"🎉 Dataset saved to {SAVE_DIR}")
