"""
prepare_professor_dataset_mlx_final.py — version 1536-safe
----------------------------------------------------------
Crée un dataset “Student–Professor” compatible MLX :
- Fusionne MedQA, Textbooks et MedDialog
- Découpe proprement les textes à <= 1536 tokens (Phi-3 tokenizer)
- Nettoyage + split 90/10
- Sauvegarde en JSONL (MLX-ready)
"""

import os
import json
import random
import re
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer

# =========================
# Config
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "mlx_data")
RAW_DIR  = os.path.join(BASE_DIR, "raw")

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_RATIO   = 0.9
MIN_TEXT_LEN  = 50
MAX_TOKENS    = 1536  # cohérent avec l'entraînement MLX

print("🚀 Création du dataset Prof–Étudiant (format MLX)…\n")

# =========================
# Tokenizer Phi-3
# =========================
TOKENIZER_MODEL = "microsoft/phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

def ntoks(txt: str) -> int:
    return len(tokenizer.encode(txt, add_special_tokens=False))

# =========================
# Découpage intelligent ≤ MAX_TOKENS
# =========================
def smart_split(text: str, max_tokens: int = MAX_TOKENS):
    """
    - coupe aux frontières de phrases si possible,
    - si une phrase dépasse à elle seule max_tokens, on "hard-slice" au token près.
    Garantit que chaque chunk <= max_tokens.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks, cur_tokens = [], []

    def flush_current():
        if cur_tokens:
            chunks.append(tokenizer.decode([t for tt in cur_tokens for t in tt]))
            cur_tokens.clear()

    for sent in sentences:
        if not sent:
            continue
        toks = tokenizer.encode(sent, add_special_tokens=False)

        # Cas 1 : la phrase seule dépasse max_tokens -> hard-slice par paquets
        if len(toks) > max_tokens:
            # Vider le buffer courant
            flush_current()
            for i in range(0, len(toks), max_tokens):
                piece = toks[i:i+max_tokens]
                chunks.append(tokenizer.decode(piece))
            continue

        # Cas 2 : peut-on ajouter cette phrase au chunk courant ?
        cur_len = sum(len(x) for x in cur_tokens)
        if cur_len + len(toks) <= max_tokens:
            cur_tokens.append(toks)
        else:
            # On flush et on recommence un chunk
            flush_current()
            cur_tokens.append(toks)

    # Dernier flush
    flush_current()
    return [c for c in chunks if len(c.strip()) > MIN_TEXT_LEN]

# =========================
# 1) MedQA (USMLE)
# =========================
def format_medqa(example):
    q   = (example.get("question") or "").strip()
    opts = example.get("options") or {}
    ans  = example.get("answer") or ""
    meta = example.get("meta_info") or ""

    if isinstance(opts, dict):
        options_text = "\n".join([f"{k}. {v}" for k, v in opts.items()])
    else:
        options_text = str(opts)

    text = (
        f"Student: {q}\n"
        f"Options:\n{options_text}\n"
        f"Professor: The correct answer is {ans}. "
        f"Explanation: {meta if meta else 'This involves physiological and pharmacological reasoning.'}"
    )
    return {"text_chunks": smart_split(text)}

print("📘 Chargement de MedQA (USMLE)…")
MEDQA_PATH = os.path.join(RAW_DIR, "med_qa/data_clean/data_clean/questions/US/train.jsonl")
medqa_raw = load_dataset("json", data_files=MEDQA_PATH)["train"]
medqa     = medqa_raw.map(format_medqa, remove_columns=medqa_raw.column_names)
print(f"✅ MedQA formaté : {len(medqa)} exemples\n")

# =========================
# 2) Textbooks anglais
# =========================
print("📚 Chargement des textbooks anglais…")
TEXTBOOK_DIR = os.path.join(RAW_DIR, "med_qa/data_clean/data_clean/textbooks/en")
text_rows = []
for filename in os.listdir(TEXTBOOK_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(TEXTBOOK_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = (f.read() or "").strip()
        if len(content) > MIN_TEXT_LEN:
            title = os.path.splitext(filename)[0].replace("_", " ").title()
            base  = f"Student: Can you explain the topic of {title}?\nProfessor: {content}"
            chunks = smart_split(base)
            for c in chunks:
                text_rows.append({"text_chunks": [c]})
textbooks = Dataset.from_list(text_rows)
print(f"✅ Textbooks formatés : {len(textbooks)} exemples\n")

# =========================
# 3) MedDialog (anglais)
# =========================
print("💬 Chargement de MedDialog…")
def format_meddialog(example):
    desc = example.get("description", "") or ""
    utts = example.get("utterances", []) or []
    dialogue = " ".join(utts).replace("patient:", "student:").replace("doctor:", "professor:")
    chunks = smart_split(f"Case: {desc}\n{dialogue}")
    return {"text_chunks": chunks}

MEDDIALOG_PATH = os.path.join(BASE_DIR, "processed/english-train.json")
meddialog_raw  = load_dataset("json", data_files=MEDDIALOG_PATH)["train"]
meddialog      = meddialog_raw.map(format_meddialog, remove_columns=meddialog_raw.column_names)
print(f"✅ MedDialog formaté : {len(meddialog)} exemples\n")

# =========================
# 4) Fusion + flatten (garanti <= MAX_TOKENS)
# =========================
print("🧩 Fusion de tous les datasets…")
combined = concatenate_datasets([medqa, textbooks, meddialog])

print("🔧 Flatten des sous-listes…")
flat = []
for ex in combined:
    for c in ex["text_chunks"]:
        if c and len(c.strip()) > MIN_TEXT_LEN:
            # Assurance ultime : clip si jamais (par précaution)
            toks = tokenizer.encode(c, add_special_tokens=False)
            if len(toks) > MAX_TOKENS:
                toks = toks[:MAX_TOKENS]
                c = tokenizer.decode(toks)
            flat.append({"text": c})

final_dataset = Dataset.from_list(flat)
print(f"✅ Total final : {len(final_dataset)} exemples (tous ≤ {MAX_TOKENS} tokens)\n")

# Sanity check
lens = [ntoks(r["text"]) for r in final_dataset]
print(f"🔎 Vérif max tokens: max={max(lens)}  mean={sum(lens)//len(lens)}  n>{MAX_TOKENS}={sum(1 for x in lens if x>MAX_TOKENS)}")

# =========================
# 5) Split train / eval
# =========================
print("✂️ Split 90/10…")
final_dataset = final_dataset.shuffle(seed=42)
cut = int(len(final_dataset) * TRAIN_RATIO)
train_ds = final_dataset.select(range(cut))
eval_ds  = final_dataset.select(range(cut, len(final_dataset)))
print(f"✅ Train : {len(train_ds)}  |  Eval : {len(eval_ds)}\n")

# =========================
# 6) Sauvegarde JSONL (MLX)
# =========================
def save_jsonl(ds, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in ds:
            f.write(json.dumps({"text": ex["text"].strip().replace("\n\n","\n")}, ensure_ascii=False) + "\n")

TRAIN_PATH = os.path.join(SAVE_DIR, "train.jsonl")
VALID_PATH = os.path.join(SAVE_DIR, "valid.jsonl")

print("💾 Sauvegarde des fichiers…")
save_jsonl(train_ds, TRAIN_PATH)
save_jsonl(eval_ds,  VALID_PATH)
print(f"✅ Train : {TRAIN_PATH}")
print(f"✅ Valid : {VALID_PATH}\n")

# =========================
# 7) Exemples aléatoires
# =========================
print("🔍 Exemples aléatoires :\n")
for i in random.sample(range(min(3, len(train_ds))), k=min(3, len(train_ds))):
    print("---")
    print(train_ds[i]["text"][:500], "…\n")

print("🎉 Dataset MLX prêt, sans troncature runtime !")
