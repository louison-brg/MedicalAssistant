"""
evaluate_model.py
-----------------
Évalue un modèle GPT2-like fine-tuné :
- Calcule la perplexité
- Génère quelques exemples médicaux
- Sauvegarde un rapport d’évaluation (.txt)
- Logge les résultats dans MLflow
"""

import os
import math
import torch
from transformers import (
    GPT2LMHeadModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline
)
from datasets import load_from_disk
import mlflow

# ======================
# ⚙️ 1. Configuration
# ======================
MODEL_PATH = "training/models/checkpoints/checkpoint-9633"
DATA_PATH = "training/data/processed"
REPORTS_DIR = "training/reports"

DEVICE = "mps" if torch.backends.mps.is_available() else \
         "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(REPORTS_DIR, exist_ok=True)
print(f"✅ Utilisation du device : {DEVICE}")

# ======================
# 📦 2. Chargement du modèle et tokenizer
# ======================
print("📦 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH).to(DEVICE)

# ======================
# 📚 3. Chargement du dataset tokenisé
# ======================
print("📥 Chargement du dataset pour évaluation...")
dataset = load_from_disk(DATA_PATH)
subset = dataset.select(range(2000)) if len(dataset) > 2000 else dataset

# 🧠 Ajout des labels nécessaires pour la perplexité
if "labels" not in subset.column_names:
    subset = subset.map(lambda ex: {"labels": ex["input_ids"]})

# ======================
# ⚙️ 4. Configuration du Trainer pour évaluation
# ======================
args = TrainingArguments(
    output_dir="training/models/eval_output",
    per_device_eval_batch_size=1,
    dataloader_drop_last=True,
    report_to=[],
)

trainer = Trainer(model=model, args=args, eval_dataset=subset)

# ======================
# 📊 5. Calcul de la perplexité
# ======================
print("📊 Calcul de la perplexité...")
eval_results = trainer.evaluate()
print("📊 Résultats bruts :", eval_results)

loss_key = "eval_loss" if "eval_loss" in eval_results else "loss" if "loss" in eval_results else None

if loss_key:
    perplexity = math.exp(eval_results[loss_key])
    print(f"✅ Perplexité : {perplexity:.2f}")
else:
    print("⚠️ Impossible de calculer la perplexité : aucune perte trouvée.")
    perplexity = None

# ======================
# 💬 6. Génération de texte médical
# ======================
print("\n🩺 Exemples de génération :")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE != "cpu" else -1
)

prompts = [
    "Patient: I have a persistent cough and chest pain. What could this be?\nDoctor:",
    "Question: What is the treatment for type 2 diabetes in obese patients?\nAnswer:",
    "Patient: My throat hurts and I feel feverish for 3 days. What should I do?\nDoctor:",
]

generated_texts = []
for prompt in prompts:
    print(f"\n🧩 Prompt: {prompt}\n---")
    result = generator(prompt, max_new_tokens=120, temperature=0.7, top_p=0.95, do_sample=True)
    output = result[0]["generated_text"]
    print(output)
    generated_texts.append((prompt, output))

# ======================
# 🧾 7. Création d’un rapport d’évaluation
# ======================
report_path = os.path.join(REPORTS_DIR, "evaluation_report.txt")
with open(report_path, "w") as f:
    f.write("=== ÉVALUATION DU MODÈLE MÉDICAL ===\n\n")
    if perplexity:
        f.write(f"Perplexité : {perplexity:.2f}\n\n")
    else:
        f.write("Perplexité : non calculée\n\n")

    f.write("=== Exemples de génération ===\n")
    for i, (prompt, gen) in enumerate(generated_texts, 1):
        f.write(f"\n--- Exemple {i} ---\n")
        f.write(f"Prompt:\n{prompt}\n\nRéponse générée:\n{gen}\n")

print(f"\n📄 Rapport d’évaluation sauvegardé : {report_path}")

# ======================
# 📈 8. Log des résultats dans MLflow
# ======================
print("\n📈 Log des résultats dans MLflow...")
mlflow.set_experiment("Medical-LLM")
with mlflow.start_run(run_name="Evaluation_Run") as run:
    if perplexity:
        mlflow.log_metric("perplexity", perplexity)
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(f"{MODEL_PATH}/config.json")
    mlflow.log_artifact(f"{MODEL_PATH}/pytorch_model.bin" if os.path.exists(f"{MODEL_PATH}/pytorch_model.bin") else f"{MODEL_PATH}/model.safetensors")

print("✅ Évaluation terminée et loggée dans MLflow !")
