"""
model_training.py — version finale avec affichage bf16 garanti
"""

import os
import torch
import mlflow
from datasets import load_from_disk
from transformers import (
    OPTForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)

# ============================================================
# 0️⃣ Configuration générale
# ============================================================

MODEL_NAME = "facebook/opt-125m"
DATA_PATH = "training/data/processed"
SAVE_DIR = "training/models/checkpoints"
MLFLOW_URI = "file:./mlruns"
MAX_LENGTH = 512

set_seed(42)

# ============================================================
# 1️⃣ Chargement du dataset
# ============================================================

print("🔄 Chargement du dataset...")
dataset = load_from_disk(DATA_PATH)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"✅ Dataset chargé et divisé :")
print(f"   - Train : {len(train_dataset)} exemples")
print(f"   - Validation : {len(eval_dataset)} exemples")

# ============================================================
# 2️⃣ Tokenizer + Modèle
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = OPTForCausalLM.from_pretrained(MODEL_NAME)
model.resize_token_embeddings(len(tokenizer))

# Force init MPS avant détection bf16
if torch.backends.mps.is_available():
    torch.zeros(1, device="mps")

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✅ Utilisation du device : {device}")

# ============================================================
# 3️⃣ Vérification bf16
# ============================================================

use_bf16 = False
bf16_supported = False

if device == "mps":
    try:
        x = torch.randn(2, 2, device="mps", dtype=torch.bfloat16)
        _ = x @ x
        bf16_supported = True
    except Exception as e:
        print(f"⚠️ bf16 non supporté sur MPS : {e}")

elif device == "cuda":
    bf16_supported = torch.cuda.is_bf16_supported()

use_bf16 = bf16_supported
if bf16_supported:
    print("✅ bf16 activé et pris en charge sur ce device.")
else:
    print("⚠️ bf16 non pris en charge, fallback en float32.")

print(f"📦 Type de poids initiaux : {next(model.parameters()).dtype}")
print(f"🎯 Mode d’entraînement choisi : {'bfloat16' if use_bf16 else 'float32'}")

# ============================================================
# 4️⃣ Configuration de l'entraînement
# ============================================================

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to=["mlflow"],
    bf16=use_bf16,
    fp16=False,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("medical-llm-training")

# ============================================================
# 5️⃣ Entraînement
# ============================================================

# Vérifie le dtype effectif du modèle
print(f"🧠 Dtype effectif du modèle après move : {next(model.parameters()).dtype}")
print(f"🧮 bf16 flag dans TrainingArguments : {training_args.bf16}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

print("🚀 Début de l'entraînement...")
trainer.train()


# ============================================================
# 6️⃣ Sauvegarde
# ============================================================

print("💾 Sauvegarde du modèle final...")
trainer.save_model()
tokenizer.save_pretrained(SAVE_DIR)
print("✨ Entraînement terminé avec succès !")
