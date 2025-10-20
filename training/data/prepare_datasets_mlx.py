"""
prepare_professor_dataset_mlx_final.py
--------------------------------------
Crée un dataset “Student–Professor” propre et compatible MLX :
- Fusionne MedQA, Textbooks et MedDialog
- Nettoie et vérifie les exemples
- Split en train / eval
- Sauvegarde en JSONL (MLX-ready)
"""

import os
import json
import random
from datasets import load_dataset, concatenate_datasets, Dataset

# ==========================================================
# ⚙️ Configuration
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "mlx_data")
RAW_DIR = os.path.join(BASE_DIR, "raw")

os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_RATIO = 0.9  # 90% train, 10% eval
MIN_TEXT_LEN = 50  # éviter les dialogues trop courts

print("🚀 Création du dataset Prof–Étudiant (format MLX)...\n")

# ==========================================================
# 1️⃣ MedQA (USMLE)
# ==========================================================
def format_medqa(example):
    q = example.get("question", "").strip()
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
# 2️⃣ Textbooks anglais
# ==========================================================
print("📚 Chargement des textbooks anglais...")
TEXTBOOK_DIR = os.path.join(RAW_DIR, "med_qa/data_clean/data_clean/textbooks/en")
text_data = []

for filename in os.listdir(TEXTBOOK_DIR):
    if filename.endswith(".txt"):
        path = os.path.join(TEXTBOOK_DIR, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if len(content) > MIN_TEXT_LEN:
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
    return {"text": f"Case: {desc}\n{dialogue}"}

MEDDIALOG_PATH = os.path.join(BASE_DIR, "processed/english-train.json")
meddialog = load_dataset("json", data_files=MEDDIALOG_PATH)["train"].map(format_meddialog)
print(f"✅ MedDialog formaté : {len(meddialog)} exemples\n")


# ==========================================================
# 4️⃣ Fusion + Nettoyage
# ==========================================================
print("🧩 Fusion de tous les datasets...")
combined = concatenate_datasets([medqa, textbooks, meddialog])
print(f"✅ Total initial : {len(combined)} exemples combinés")

print("🧹 Nettoyage des exemples trop courts...")
filtered = combined.filter(lambda x: len(x["text"]) > MIN_TEXT_LEN)
print(f"✅ {len(filtered)} exemples conservés après nettoyage\n")


# ==========================================================
# 5️⃣ Split en train / eval
# ==========================================================
print("✂️ Split en train / validation (90/10)...")
filtered = filtered.shuffle(seed=42)
split_idx = int(len(filtered) * TRAIN_RATIO)
train_dataset = filtered.select(range(split_idx))
eval_dataset = filtered.select(range(split_idx, len(filtered)))

print(f"✅ Train : {len(train_dataset)} exemples")
print(f"✅ Eval  : {len(eval_dataset)} exemples\n")


# ==========================================================
# 6️⃣ Sauvegarde JSONL (format MLX)
# ==========================================================
def save_jsonl(dataset, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in dataset:
            text = ex["text"].strip().replace("\n\n", "\n")
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

TRAIN_PATH = os.path.join(SAVE_DIR, "train.jsonl")
VALID_PATH = os.path.join(SAVE_DIR, "valid.jsonl")

print("💾 Sauvegarde des fichiers...")
save_jsonl(train_dataset, TRAIN_PATH)
save_jsonl(eval_dataset, VALID_PATH)

print(f"✅ Train : {TRAIN_PATH}")
print(f"✅ valid  : {VALID_PATH}\n")

# ==========================================================
# 7️⃣ Vérification rapide du contenu
# ==========================================================
print("🔍 Exemples aléatoires :\n")
for i in random.sample(range(len(train_dataset)), min(3, len(train_dataset))):
    print(f"--- Exemple {i} ---")
    print(train_dataset[i]["text"][:500], "...\n")

print("🎉 Dataset MLX complet et prêt à l'emploi !")
