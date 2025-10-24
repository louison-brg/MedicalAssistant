"""
prepare_professor_dataset_phi3.py
---------------------------------
Crée un dataset “Student–Professor” pour le fine-tuning du modèle Phi-3-mini.
Sources :
 - MedQA (USMLE)
 - Textbooks médicaux anglais (.txt)
 - MedDialog (anglais)
"""

import os
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# ==========================================================
# ⚙️ Configuration
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "processed_professor_phi3")
RAW_DIR = os.path.join(BASE_DIR, "raw")

TOKENIZER_MODEL = "microsoft/phi-3-mini-4k-instruct"
MAX_LENGTH = 512

os.makedirs(SAVE_DIR, exist_ok=True)

print("🚀 Création du dataset Prof–Étudiant pour Phi-3...\n")

# Initialisation du tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

print(f"✅ Tokenizer chargé : {TOKENIZER_MODEL}\n")

# ==========================================================
# 1️⃣ MedQA (USMLE)
# ==========================================================
def format_medqa(example):
    q = example.get("question", "")
    opts = example.get("options", {})
    ans = example.get("answer", "")
    meta = example.get("meta_info", "")

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
    return {"text": text}

print("📘 Chargement de MedQA (USMLE)...")
MEDQA_PATH = os.path.join(RAW_DIR, "med_qa/data_clean/data_clean/questions/US/train.jsonl")
medqa = load_dataset("json", data_files=MEDQA_PATH)["train"].map(format_medqa)
print(f"✅ MedQA formaté : {len(medqa)} exemples\n")

# ==========================================================
# 2️⃣ Textbooks anglais (.txt)
# ==========================================================
print("📚 Chargement des textbooks anglais...")
TEXTBOOK_DIR = os.path.join(RAW_DIR, "med_qa/data_clean/data_clean/textbooks/en")
text_data = []

for filename in os.listdir(TEXTBOOK_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(TEXTBOOK_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if len(content) > 200:
                title = os.path.splitext(filename)[0].replace("_", " ").title()
                text_data.append({
                    "text": f"Student: Can you explain the topic of {title}?\nProfessor: {content}"
                })

textbooks = Dataset.from_list(text_data)
print(f"✅ Textbooks formatés : {len(textbooks)} exemples\n")

# ==========================================================
# 3️⃣ MedDialog (anglais)
# ==========================================================
print("💬 Chargement de MedDialog...")

def format_meddialog(example):
    desc = example.get("description", "")
    utts = example.get("utterances", [])
    dialogue = " ".join(utts).replace("patient:", "student:").replace("doctor:", "professor:")
    text = f"Case: {desc}\n{dialogue}"
    return {"text": text}

MEDDIALOG_PATH = os.path.join(BASE_DIR, "processed/english-train.json")
meddialog = load_dataset("json", data_files=MEDDIALOG_PATH)["train"].map(format_meddialog)
print(f"✅ MedDialog formaté : {len(meddialog)} exemples\n")

# ==========================================================
# 4️⃣ Fusion
# ==========================================================
print("🧩 Fusion de tous les datasets...")
combined = concatenate_datasets([medqa, textbooks, meddialog])
print(f"✅ Total : {len(combined)} exemples combinés\n")

# ==========================================================
# 5️⃣ Sauvegarde non-tokenisée
# ==========================================================
RAW_SAVE_PATH = os.path.join(SAVE_DIR, "raw_text_dataset")
combined.save_to_disk(RAW_SAVE_PATH)
print(f"💾 Dataset texte sauvegardé : {RAW_SAVE_PATH}\n")

# ==========================================================
# 6️⃣ Tokenisation
# ==========================================================
def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=False)
    input_ids = tokens["input_ids"]
    result_input_ids, result_attention_masks = [], []

    for i in range(0, len(input_ids), MAX_LENGTH):
        chunk = input_ids[i:i + MAX_LENGTH]
        attention_mask = [1] * len(chunk)

        if len(chunk) < MAX_LENGTH:
            pad_len = MAX_LENGTH - len(chunk)
            chunk += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        result_input_ids.append(chunk)
        result_attention_masks.append(attention_mask)

    return {"input_ids": result_input_ids, "attention_mask": result_attention_masks}

print("🔠 Tokenisation avec Phi-3 tokenizer...")
temp_dataset = combined.map(tokenize_function, batched=False, remove_columns=combined.column_names)

flat_input_ids, flat_attention_masks = [], []
for ex in temp_dataset:
    for i in range(len(ex["input_ids"])):
        flat_input_ids.append(ex["input_ids"][i])
        flat_attention_masks.append(ex["attention_mask"][i])

tokenized_dataset = Dataset.from_dict({
    "input_ids": flat_input_ids,
    "attention_mask": flat_attention_masks
})

print(f"✅ {len(tokenized_dataset):,} séquences prêtes pour l’entraînement")

# ==========================================================
# 7️⃣ Sauvegarde finale
# ==========================================================
TOKENIZED_PATH = os.path.join(SAVE_DIR, "tokenized")
tokenized_dataset.save_to_disk(TOKENIZED_PATH)
print(f"🎉 Dataset tokenisé sauvegardé dans : {TOKENIZED_PATH}\n")
