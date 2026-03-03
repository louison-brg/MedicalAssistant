# MedLLM iOS (MLX)

Medical assistant LLM that runs locally on iPhone via a native Swift + MLX runtime.

## Repository layout
- `training/` data prep, tokenization, fine-tuning, eval
- `model_export/` MLX debugging utilities
- `ios_app/` iOS SwiftUI app with MLX inference
- `training/models/` local model checkpoints and merged/quantized weights

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-macos.txt
```

## Data pipeline (MLX)
1. Build conversation-style JSONL:
```bash
python3 training/data/make_mlx_jsonl.py
```

2. (Optional) Build Hugging Face tokenized dataset (for HF diagnostics only):
```bash
python3 training/data/prepare_datasets.py
```

Output path:
- `training/data/processed_professor_phi3_hf`

## Fine-tuning (LoRA)
```bash
python3 training/models/model_training.py \
  --model-path training/models/phi-3-mini-4k-instruct \
  --data-path training/data/mlx_data \
  --adapter-path training/models/checkpoints_phi3_mlx
```

Legacy PyTorch trainer is still available at:
- `training/models/model_training_hf.py`

## Evaluation
```bash
python3 training/models/model_evaluate.py \
  --model-path training/models/phi3-medprof-merged \
  --data-path training/data/processed_professor_phi3_hf
```

## iOS app (MLX inference)
1. Open `/Users/louison/Projets/MedicalAssistant/ios_app/MedicalAssistant/MedicalAssistant.xcodeproj`
2. Ensure your merged + quantized MLX model folder is present in the app bundle:
   - `/Users/louison/Projets/MedicalAssistant/ios_app/MedicalAssistant/MedicalAssistant/Phi3_Medical_4bit/`
   - Required files: `model.safetensors`, `config.json`, `generation_config.json`, tokenizer files
3. Build and run on device

Notes:
- History reset button is the top-right `arrow.counterclockwise.circle` icon.
- The app stores chat history locally (`messages.json`) and clears it from this button.

## Sanity checks
```bash
python3 tests/test_pytorch.py --model-path training/models/phi3-medprof-merged
python3 tests/test_model_mlflow.py --model-path training/models/phi3-medprof-merged
python3 tests/test_tokenizer_assets.py
```

## Safety
- Not medical advice.
- Keep explicit emergency disclaimer in UI.
- Verify dataset licenses before distribution.
