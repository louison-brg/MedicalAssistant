import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset

# ==========================================================
# ⚙️ Configuration
# ==========================================================
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
OUT_DIR = BASE_DIR / "mlx_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("🚀 Préparation du dataset médical pour MLX (format Phi-3 chat)...\n")


def format_phi3(user_text: str, assistant_text: str) -> str:
    """Encapsule le dialogue dans le format attendu par Phi-3 Instruct."""
    return f"<|user|>\n{user_text}<|end|>\n<|assistant|>\n{assistant_text}<|end|>"


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
    lowered = text.lower()
    if not lowered:
        return True
    if "the correct answer is" in lowered or "explanation: step" in lowered:
        return True
    if "this is a riddle" in lowered:
        return True
    words = re.findall(r"[a-zA-ZÀ-ÿ']+", lowered)
    if len(words) >= 12:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.34:
            return True
    return False


def format_medqa(example: dict) -> dict:
    question = clean_text(example.get("question"))
    answer = clean_text(example.get("answer"))
    options = example.get("options")

    if not question or not answer:
        return {"text": ""}

    option_lines: List[str] = []
    if isinstance(options, dict):
        for key in sorted(options):
            value = clean_text(options[key])
            if value:
                option_lines.append(f"{key}. {value}")
    elif isinstance(options, list):
        for i, value in enumerate(options, start=1):
            text = clean_text(value)
            if text:
                option_lines.append(f"{i}. {text}")

    user_parts = [
        "A medical exam question was asked. Provide the best clinical answer.",
        question,
    ]
    if option_lines:
        user_parts.append("Options:\n" + "\n".join(option_lines))

    assistant = answer
    if assistant and assistant[-1] not in ".!?":
        assistant += "."

    if looks_noisy(assistant):
        return {"text": ""}
    return {"text": format_phi3("\n\n".join(user_parts), assistant)}


def sentence_chunks(text: str, max_chars: int = 1800, overlap_chars: int = 250) -> Iterable[str]:
    # Préférer découper sur les paragraphes d'abord pour garder la cohérence sémantique
    paragraphs = re.split(r"\n{2,}", text)
    sentences = []
    for p in paragraphs:
        if p.strip():
            sentences.extend(re.split(r"(?<=[.!?])\s+", p.strip()))

    buffer: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(sentence) > max_chars:
            if buffer:
                yield " ".join(buffer).strip()
                buffer = []
                current_len = 0
            for i in range(0, len(sentence), max_chars):
                chunk = sentence[i:i + max_chars].strip()
                if chunk:
                    yield chunk
            continue

        projected = current_len + len(sentence) + (1 if buffer else 0)
        if projected <= max_chars:
            buffer.append(sentence)
            current_len = projected
        else:
            yield " ".join(buffer).strip()
            
            # Ajout d'overlap sémantique pour ne pas perdre le contexte
            overlap = []
            overlap_len = 0
            for s in reversed(buffer):
                if overlap_len + len(s) <= overlap_chars:
                    overlap.insert(0, s)
                    overlap_len += len(s) + 1
                else:
                    break
            
            buffer = overlap + [sentence]
            current_len = overlap_len + len(sentence)

    if buffer:
        yield " ".join(buffer).strip()


def parse_turn(raw_turn: str) -> Optional[Tuple[str, str]]:
    if ":" not in raw_turn:
        return None
    speaker, content = raw_turn.split(":", 1)
    role = speaker.strip().lower()
    text = clean_text(content)
    if role not in {"patient", "doctor"} or not text:
        return None
    return role, text


def meddialog_rows(example: dict) -> List[dict]:
    description = clean_text(example.get("description"))
    utterances = example.get("utterances") or []
    parsed = [turn for turn in (parse_turn(u) for u in utterances) if turn is not None]
    rows: List[dict] = []

    if len(parsed) < 2:
        return rows

    history: List[Tuple[str, str]] = []
    for i, (role, text) in enumerate(parsed):
        if role == "patient":
            doctor_reply = ""
            for j in range(i + 1, len(parsed)):
                if parsed[j][0] == "doctor":
                    doctor_reply = parsed[j][1]
                    break

            if doctor_reply and not looks_noisy(doctor_reply):
                context_window = history[-4:] + [("patient", text)]
                dialogue = []
                for turn_role, turn_text in context_window:
                    prefix = "Patient" if turn_role == "patient" else "Doctor"
                    dialogue.append(f"{prefix}: {turn_text}")

                user_parts = []
                if description:
                    user_parts.append(f"Case summary: {description}")
                user_parts.append("Conversation:\n" + "\n".join(dialogue))
                user_parts.append("Reply as the medical assistant to the latest patient message.")
                user_message = "\n\n".join(user_parts)
                rows.append({"text": format_phi3(user_message, doctor_reply)})

        history.append((role, text))

    return rows


def save_mlx_format(dataset: Dataset, filename: str) -> None:
    path = OUT_DIR / filename
    with open(path, "w", encoding="utf-8") as handle:
        for entry in dataset:
            text = clean_text(entry["text"])
            if text:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"✅ Sauvegardé : {path} ({len(dataset)} exemples)")


# ==========================================================
# 1️⃣ Chargement des sources
# ==========================================================
print("📘 Chargement de MedQA...")
MEDQA_PATH = RAW_DIR / "med_qa/data_clean/data_clean/questions/US/train.jsonl"
medqa_raw = load_dataset("json", data_files=str(MEDQA_PATH))["train"]
medqa = medqa_raw.map(format_medqa, remove_columns=medqa_raw.column_names)
medqa = medqa.filter(lambda ex: bool(ex["text"]))
print(f"   ↳ {len(medqa)} exemples retenus")

print("📚 Chargement des Textbooks...")
TEXTBOOK_DIR = RAW_DIR / "med_qa/data_clean/data_clean/textbooks/en"
textbook_rows: List[dict] = []
for file in sorted(TEXTBOOK_DIR.glob("*.txt")):
    content = clean_text(file.read_text(encoding="utf-8", errors="ignore"))
    if len(content) < 300:
        continue
    title = file.stem.replace("_", " ").strip()
    user_prompt = f"Explain this medical topic for a clinical assistant: {title}"
    for chunk in sentence_chunks(content, max_chars=1800):
        if len(chunk) >= 120 and not looks_noisy(chunk):
            textbook_rows.append({"text": format_phi3(user_prompt, chunk)})

textbooks = Dataset.from_list(textbook_rows) if textbook_rows else Dataset.from_list([{"text": ""}]).filter(lambda _: False)
print(f"   ↳ {len(textbooks)} exemples retenus")

print("💬 Chargement de MedDialog...")
MEDDIALOG_PATH = BASE_DIR / "processed/english-train.json"
meddialog_raw = load_dataset("json", data_files=str(MEDDIALOG_PATH))["train"]
dialog_rows: List[dict] = []
for sample in meddialog_raw:
    dialog_rows.extend(meddialog_rows(sample))
meddialog = Dataset.from_list(dialog_rows) if dialog_rows else Dataset.from_list([{"text": ""}]).filter(lambda _: False)
print(f"   ↳ {len(meddialog)} exemples retenus")

# ==========================================================
# 2️⃣ Fusion + split
# ==========================================================
print("🧩 Fusion des sources...")
parts = [ds for ds in [medqa, textbooks, meddialog] if len(ds) > 0]
if not parts:
    raise RuntimeError("Aucun exemple valide généré. Vérifiez les données sources.")

combined = concatenate_datasets(parts).shuffle(seed=42)
print(f"✅ Total final: {len(combined)} exemples")

split_idx = max(1, int(len(combined) * 0.9))
if split_idx >= len(combined):
    split_idx = len(combined) - 1
train_data = combined.select(range(split_idx))
valid_data = combined.select(range(split_idx, len(combined)))

# ==========================================================
# 3️⃣ Export JSONL pour MLX
# ==========================================================
save_mlx_format(train_data, "train.jsonl")
save_mlx_format(valid_data, "valid.jsonl")

print("\n🎉 Dataset prêt pour MLX.")