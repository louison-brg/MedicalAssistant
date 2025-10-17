"""
prepare_datasets.py
-------------------
Fusionne et tokenise les datasets :
- MedQA (USMLE)
- Textbooks médicaux anglais (.txt)
- MedDialog (english-train.json)

Sortie :
Dataset fusionné, nettoyé et tokenisé sauvegardé dans training/data/processed/
"""

import os
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# =======================================================================
# 0️⃣ Configuration de base
# =======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "processed")
TOKENIZER_MODEL = "gpt2"
MAX_LENGTH = 512

os.makedirs(SAVE_DIR, exist_ok=True)

# Initialisation du tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("🚀 Préparation des datasets médicaux...\n")


# =======================================================================
# 1️⃣ MedQA (USMLE)
# =======================================================================
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
        f"Question: {q}\n"
        f"Options:\n{options_text}\n"
        f"Answer: {ans}\n"
        f"Metadata: {meta}"
    )
    return {"text": text}


print("📘 Chargement de MedQA (USMLE)...")

MEDQA_PATH = os.path.join(BASE_DIR, "raw/med_qa/data_clean/data_clean/questions/US/train.jsonl")

medqa = load_dataset("json", data_files=MEDQA_PATH)["train"].map(format_medqa)

print(f"✅ MedQA: {len(medqa)} exemples chargés et formatés\n")


# =======================================================================
# 2️⃣ Textbooks anglais (.txt)
# =======================================================================
print("📚 Chargement des textbooks anglais...")

TEXTBOOK_DIR = os.path.join(BASE_DIR, "raw/med_qa/data_clean/data_clean/textbooks/en")
text_data = []

for filename in os.listdir(TEXTBOOK_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(TEXTBOOK_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if len(content) > 200:
                text_data.append({"text": content})

textbooks = Dataset.from_list(text_data)

print(f"✅ Textbooks: {len(textbooks)} documents chargés\n")


# =======================================================================
# 3️⃣ MedDialog (anglais)
# =======================================================================
print("💬 Chargement de MedDialog (anglais)...")

def format_meddialog(example):
    desc = example.get("description", "")
    utts = example.get("utterances", [])
    # Les utterances sont déjà sous forme de texte ("patient: ..." / "doctor: ...")
    dialogue = " ".join(utts)
    text = f"Case: {desc}\nDialogue: {dialogue}"
    return {"text": text}

MEDDIALOG_PATH = os.path.join(BASE_DIR, "processed/english-train.json")

meddialog = load_dataset("json", data_files=MEDDIALOG_PATH)["train"].map(format_meddialog)

print(f"✅ MedDialog: {len(meddialog)} dialogues chargés\n")


# =======================================================================
# 4️⃣ Fusion des datasets
# =======================================================================
print("🧩 Fusion des datasets...")
combined = concatenate_datasets([medqa, textbooks, meddialog])
print(f"✅ Total: {len(combined)} exemples combinés\n")


# =======================================================================
# 5️⃣ Tokenisation
# =======================================================================
def tokenize_function(example):
    # Tokenisation sans troncature globale
    tokens = tokenizer(example["text"], truncation=False)
    input_ids = tokens["input_ids"]

    result_input_ids = []
    result_attention_masks = []

    # Découpage en morceaux de 512 tokens
    for i in range(0, len(input_ids), MAX_LENGTH):
        chunk = input_ids[i:i + MAX_LENGTH]
        attention_mask = [1] * len(chunk)

        # Padding si besoin
        if len(chunk) < MAX_LENGTH:
            pad_len = MAX_LENGTH - len(chunk)
            chunk += [tokenizer.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        result_input_ids.append(chunk)
        result_attention_masks.append(attention_mask)

    return {
        "input_ids": result_input_ids,
        "attention_mask": result_attention_masks
    }



print("🔠 Tokenisation en cours...")
temp_dataset = combined.map(tokenize_function, batched=False, remove_columns=combined.column_names)

# =====================================================
# 🧩 Flatten manuel des sous-listes (chunks)
# =====================================================
from datasets import Dataset

print("🔧 Flatten des séquences multiples par document...")
flat_input_ids = []
flat_attention_masks = []

for ex in temp_dataset:

    if isinstance(ex["input_ids"][0], list):
        for i in range(len(ex["input_ids"])):
            flat_input_ids.append(ex["input_ids"][i])
            flat_attention_masks.append(ex["attention_mask"][i])
    else:
        flat_input_ids.append(ex["input_ids"])
        flat_attention_masks.append(ex["attention_mask"])

tokenized_dataset = Dataset.from_dict({
    "input_ids": flat_input_ids,
    "attention_mask": flat_attention_masks
})

print(f"✅ Dataset aplati : {len(tokenized_dataset):,} séquences prêtes à l’entraînement")


# =======================================================================
# 6️⃣ Sauvegarde finale
# =======================================================================
print("💾 Sauvegarde du dataset tokenizé...")
tokenized_dataset.save_to_disk(SAVE_DIR)

print(f"🎉 Dataset final prêt pour l'entraînement !\n📂 Emplacement : {SAVE_DIR}")
