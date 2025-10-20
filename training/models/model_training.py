"""
Fine-tuning Phi-3-mini local sur Mac (MPS) avec LoRA + MLflow — version finale stable
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"     # bascule auto CPU si op MPS manque
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # éviter OOM silencieux MPS

import torch
import mlflow
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from peft import LoraConfig, get_peft_model

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "training/models/phi-3-mini-4k-instruct"
DATA_PATH  = "training/data/processed_professor_phi3/tokenized"
SAVE_DIR   = "training/models/checkpoints_phi3_lora"
MLFLOW_URI = "file:./mlruns"

EPOCHS, BATCH_SIZE, GRAD_ACCUM_STEPS = 1, 1, 8
LR, WARMUP_STEPS, WEIGHT_DECAY = 2e-4, 200, 0.01

set_seed(42)
os.makedirs(SAVE_DIR, exist_ok=True)
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ============================================================
# FIX MLflow for MacOS compatibility
# ============================================================
import mlflow.utils.autologging_utils
mlflow.utils.autologging_utils._is_testing = False
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

# ============================================================
# DEVICE
# ============================================================
if torch.backends.mps.is_available():
    device = "mps"
    dtype = torch.float32
    IS_MAC = True
elif torch.cuda.is_available():
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    IS_MAC = False
else:
    device = "cpu"
    dtype = torch.float32
    IS_MAC = False

print(f"🧠 Device: {device} | dtype: {dtype}")

# ============================================================
# DATA
# ============================================================
print("🔄 Chargement du dataset...")
dataset = load_from_disk(DATA_PATH)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset, eval_dataset = split["train"], split["test"]

if IS_MAC:
    # mode test allégé
    train_dataset = train_dataset.select(range(500))
    eval_dataset  = eval_dataset.select(range(100))
    print("⚠️ Mode test activé : 500 exemples train / 100 val.")

print(f"✅ Dataset : {len(train_dataset)} train / {len(eval_dataset)} val")

# ============================================================
# TOKENIZER + MODEL
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("📦 Chargement du modèle Phi-3 local…")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=dtype)
model.to(device)
print("✅ Modèle chargé.")

# ============================================================
# LORA CONFIG + INJECTION
# ============================================================
targets = []
for name, _ in model.named_modules():
    if any(x in name for x in ["q_proj", "v_proj", "o_proj", "dense", "Wqkv"]):
        targets.append(name.split('.')[-1])
targets = list(set(targets))
print(f"🎯 Modules LoRA détectés: {targets}")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=targets,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.to(device)
model.train()

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"✅ {len(trainable)} paramètres entraînables détectés.")
if len(trainable) == 0:
    raise RuntimeError("❌ Aucun paramètre entraînable ! Vérifie le device ou les modules LoRA.")

# ============================================================
# MLFLOW INIT
# ============================================================
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("phi3-medical-professor")

# ============================================================
# TRAINING SETUP
# ============================================================
training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    weight_decay=WEIGHT_DECAY,
    warmup_steps=WARMUP_STEPS,
    logging_steps=50,
    save_steps=500,
    save_strategy="steps",
    dataloader_pin_memory=not IS_MAC,
    report_to=["mlflow"],
    fp16=False,
    bf16=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

# ============================================================
# TRAIN
# ============================================================
print("🚀 Début du fine-tuning LoRA (MPS + MLflow)…")
torch.set_grad_enabled(True)
model.train()
trainer.train()

# ============================================================
# EVAL
# ============================================================
print("📊 Évaluation post-entraînement…")
results = trainer.evaluate()
print("✅ Résultats :", results)

# ============================================================
# SAVE
# ============================================================
print("💾 Sauvegarde du modèle LoRA…")
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print("✨ Terminé avec succès !")
