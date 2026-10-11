# MedLLM iOS

Private medical-assistant prototype running a quantized Phi-3 model entirely on iPhone with SwiftUI and MLX.

> This is an educational prototype, not a medical device or a diagnosis tool. Clinical safety, dataset licensing, bias evaluation, and regulatory review are required before real-world use.

## What is in the repository

- `ios_app/MedicalAssistant/MedicalAssistant/`: SwiftUI chat, local history, tokenizer, and 4-bit MLX inference.
- `training/data/`: dataset construction, leakage/quality checks, and optional Hugging Face tokenization.
- `training/models/`: MLX LoRA training, PEFT fallback, evaluation, checkpoint smoke tests, and model export.
- `training/reports/`: evaluation outputs and benchmark workbook.
- `tests/`: fast pipeline and deployment-asset contract tests.

The primary Apple Silicon path is MLX. The Hugging Face/PyTorch scripts are retained as diagnostics and as a PEFT fallback, not as the recommended Mac training route.

## App performance design

The UI is structured to keep typing independent from inference and chat rendering:

- The composer owns its draft as local SwiftUI state, so a keystroke does not invalidate the message list, Markdown, or mesh background.
- Stream updates are coalesced to roughly 20 UI refreshes per second.
- Partial responses use plain `Text`; Markdown is parsed only once the response is complete.
- Session JSON encoding and protected-file writes run in an actor outside the main actor.
- Token decoding is incremental rather than repeatedly decoding the full response.
- Repetition penalty remains on MLX instead of copying the full vocabulary logits to the CPU for every token.
- The transformer graph is evaluated once per generation step instead of synchronizing Metal after every layer.

Measure on a physical iPhone with Instruments (SwiftUI, Time Profiler, Metal System Trace, and Allocations). Simulator timings are not representative of MLX inference.

## Environment

Python 3.11 or 3.12 and Apple Silicon are recommended for MLX training. Keep
the MLX and PEFT fallbacks in separate environments: current MLX-LM uses the
Transformers 5 family, while the retained PEFT fallback is pinned to the
Transformers 4 family.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-macos.txt -r requirements-dev.txt
```

Only when the legacy PyTorch/PEFT route is needed:

```bash
python3 -m venv .venv-hf
source .venv-hf/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
```

`requirements-macos.txt` tracks the validated MLX/MLX-LM release family. Do not mix adapters and base models produced by incompatible conversion versions without re-running the asset tests.

## 1. Build the training dataset

Expected local sources:

- `training/data/raw/med_qa/data_clean/data_clean/questions/US/train.jsonl`
- `training/data/raw/med_qa/data_clean/data_clean/textbooks/en/*.txt`
- `training/data/processed/english-train.json` (optional MedDialog source)

Build native chat-format JSONL and audit it:

```bash
python training/data/make_mlx_jsonl.py
python training/data/check_dataset_quality.py --data-path training/data/mlx_data
```

The output rows use `messages`, not preformatted `text`. This lets MLX-LM apply the tokenizer chat template and mask the prompt from the loss. The split is group-aware: chunks from one textbook or turns from one source dialogue cannot appear in both train and validation.

Textbook responses are capped near the app's 256-token generation budget and
the source is limited to 15,000 examples by default. MedQA is capped at 2,500
examples and converted to complete-sentence answers. The source contains only
answer keys, not rationales, so the prompt never asks the model to imitate a
justification that is absent from the target data.

To prepare the legacy Hugging Face dataset:

```bash
python training/data/prepare_datasets.py \
  --tokenizer training/models/phi-3-mini-4k-instruct \
  --overwrite
python training/data/check_dataset_quality.py \
  --data-path training/data/processed_professor_phi3_hf
```

## 2. Train a LoRA adapter with MLX

First inspect the full, effective configuration without starting training:

```bash
python training/models/model_training.py --dry-run
```

Then train:

```bash
python training/models/model_training.py \
  --model-path training/models/phi-3-mini-4k-instruct \
  --data-path training/data/mlx_data \
  --adapter-path training/models/checkpoints_phi3_mlx
```

Defaults use assistant-only loss, AdamW, cosine decay with warm-up, gradient checkpointing, gradient accumulation, validation, and the conservative MLX-LM LoRA values (learning rate `1e-5`, rank `8`, direct scale `20`, dropout `0`). MLX's `scale` directly multiplies the LoRA delta; it is not PEFT's `alpha / rank`. Tune the number of adapted layers and iteration count from frozen task benchmarks rather than assuming that a lower training loss means a better medical model.

Resume with:

```bash
python training/models/model_training.py \
  --resume-adapter-file training/models/checkpoints_phi3_mlx/adapters.safetensors
```

## 3. Fuse and quantize for iPhone

MLX-trained adapters must be fused with MLX-LM, not PEFT. The exporter auto-detects MLX versus PEFT adapters and refuses to overwrite an existing model directory.

```bash
python training/models/model_export.py \
  --base-model training/models/phi-3-mini-4k-instruct \
  --adapter training/models/checkpoints_phi3_mlx \
  --fused-output training/models/phi3-medprof-fused-v2 \
  --quantized-output training/models/Phi3_Medical_Latest_v2 \
  --q-bits 4 \
  --q-group-size 64
```

The exporter performs the operations in the safe order:

1. Fuse the floating-point LoRA delta into the base model.
2. Quantize the fused standalone model.
3. Verify model, tokenizer, and config assets.
4. Write `export_manifest.json` with source paths, quantization, and tool versions.

For the app, place the verified deployment folder at:

```text
ios_app/MedicalAssistant/MedicalAssistant/Phi3_Medical_Latest/
```

Required assets include `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json`, and `tokenizer_config.json`.

## 4. Evaluate

Evaluate assistant-only perplexity and deterministic smoke prompts:

```bash
python training/models/model_evaluate.py \
  --model-path training/models/phi3-medprof-merged \
  --data-path training/data/processed_professor_phi3_hf \
  --disable-mlflow
```

Perplexity alone is insufficient for a medical assistant. Run the same deterministic held-out MedQA sample against the deployed model and every candidate:

```bash
python training/models/model_benchmark.py \
  --model training/models/Phi3_Medical_Latest_v2 \
  --data training/data/mlx_data/valid.jsonl \
  --max-samples 100 \
  --seed 20260805
```

This benchmark checks exact answer/option agreement; it is useful for detecting regressions but is not a clinical validation. Add clinician-reviewed rubrics for factuality, uncertainty, contraindications, demographic bias, hallucination rate, refusal behavior, and escalation only when the symptoms actually warrant it.

Do not promote a checkpoint solely because its validation loss improved. A
candidate must answer diagnosis and treatment questions correctly, avoid
invented contraindicated advice, express appropriate uncertainty, and escalate
only when clinically indicated. Keep the deployed model unchanged whenever a
candidate regresses against the factual baseline.

## 5. Build the iOS app

Open:

```text
ios_app/MedicalAssistant/MedicalAssistant.xcodeproj
```

The current project targets iOS 26 and resolves MLX Swift plus MarkdownUI through Swift Package Manager. Xcode may require its optional Metal toolchain:

```bash
xcodebuild -downloadComponent MetalToolchain
```

Build from the command line:

```bash
xcodebuild build \
  -project ios_app/MedicalAssistant/MedicalAssistant.xcodeproj \
  -scheme MedicalAssistant \
  -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO
```

Run on a recent physical iPhone with enough memory for the approximately 2 GB bundled model. The app clears MLX weights when backgrounded and reloads them lazily for the next request.

## Verification

```bash
python -m compileall -q training tests
ruff check training tests
pytest -q
```

The deployment-asset tests are skipped when the local 4-bit model is absent. When present, they verify quantization metadata, tokenizer stop tokens, and the exact tensor naming contract required by the Swift runner.

## Security and privacy notes

- Conversations are stored in the app container using complete file protection and unlocked with device-owner authentication.
- Inference is local; the app contains no network client.
- Deleting the app removes its local conversation container, but device backups and crash diagnostics require a separate deployment policy.
- Never distribute raw patient data or datasets without an explicit lawful basis, de-identification review, license review, and retention policy.
