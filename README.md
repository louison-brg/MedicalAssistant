# 🧠 MedLLM iOS

### Local Medical Language Model for iPhone & iPad

---

## 📋 Overview

**MedLLM iOS** is an experimental project that aims to build a **medical language model (LLM)** capable of running **entirely offline** on iPhone and iPad devices.

The goal is to combine:
- 🤖 **Deep learning & NLP** (PyTorch / Hugging Face)
- 📱 **On-device inference** (CoreML + SwiftUI)
- ⚙️ **MLOps tools** (MLflow, export tooling)

---

## 🧩 Project Goals

- Build and fine-tune a **custom medical LLM** (MedQA, MedDialog, textbooks)
- Optimize the model for **Apple Silicon** and on-device runtime
- Deploy locally via **CoreML** in a native **SwiftUI app**
- Track experiments and keep export steps reproducible

---

## 📁 Repository Layout

- `training/` — data preparation, training, evaluation
- `model_export/` — CoreML conversion + validation tools
- `ios_app/` — SwiftUI app + on-device inference code
- `models/`, `adapters/`, `exports/`, `mlruns/` — local artifacts (not meant for Git)

## 🧹 Artifact Management

Large binaries should live outside Git (Git LFS, DVC, or an artifact store):
- model weights (`*.safetensors`, `*.bin`, `*.npz`)
- CoreML packages (`*.mlpackage`)
- experiment logs (`mlruns/`)

---

## ✅ Quickstart (Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

macOS export + MLX extras:

```bash
pip install -r requirements-macos.txt
```

Dev tools:

```bash
pip install -r requirements-dev.txt
```

---

## 📚 Data Preparation

> ⚠️ Ensure you have the rights to use any datasets or textbooks.

Prepare the HF tokenized dataset (Phi‑3):

```bash
python training/data/prepare_datasets.py
```

Prepare MLX JSONL (optional):

```bash
python training/data/prepare_datasets_mlx.py
```

---

## 🏋️ Training (LoRA)

```bash
python training/models/model_training.py \
  --model-path training/models/phi-3-mini-4k-instruct \
  --data-path training/data/processed_professor_phi3/tokenized \
  --save-dir training/models/checkpoints_phi3_lora
```

Merge LoRA into a standalone checkpoint (for export):

```bash
python training/models/merge_lora.py \
  --base-model training/models/phi-3-mini-4k-instruct \
  --adapter training/models/checkpoints_phi3_lora \
  --output training/models/phi3-medprof-merged
```

---

## 📊 Evaluation

```bash
python training/models/model_evaluate.py \
  --model-path training/models/phi3-medprof-merged \
  --data-path training/data/processed_professor_phi3/tokenized
```

---

## 📦 CoreML Export

```bash
python model_export/convert_phi3_to_coreml.py \
  --model-path training/models/phi3-medprof-merged \
  --output ios_app/MedicalAssistant/MedicalAssistant/MedicalLLM.mlpackage
```

Optional: generate a text response with CoreML locally:

```bash
python model_export/generate_coreml.py --model exports/MedicalLLM_fp16.mlpackage --prompt "Patient: I have chest pain. Doctor:"
```

---

## 📱 iOS App

1) Open `ios_app/MedicalAssistant/MedicalAssistant.xcodeproj` in Xcode
2) Ensure `MedicalLLM.mlpackage` is included in **Copy Bundle Resources**
3) Keep `tokenizer.json`, `tokenizer_config.json`, and `generation_config.json` in sync with the merged model
4) Run on device

The app supports:
- CPU/GPU toggle
- Partial streaming output
- Cancel generation
- Local chat persistence

---

## 🧪 Tests / Sanity Checks

```bash
python tests/test_pytorch.py --model-path training/models/phi3-medprof-merged
python tests/test_model_mlflow.py --model-path training/models/phi3-medprof-merged
```

---

## 🔐 Safety & Compliance

- This app is **not medical advice** and should not be used for diagnosis.
- Always include clear disclaimers in UI and documentation.
- Verify dataset licenses before distribution.

---

## 👤 Author

**Louison Beranger**  
AI Engineer & Developer
