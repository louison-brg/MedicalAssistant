# MedLLM iOS — Private On-Device Medical Assistant

[![Platform](https://img.shields.io/badge/Platform-iOS%2018%2B%20%7C%20iOS%2026-blue?logo=apple)](https://developer.apple.com/ios/)
[![Swift](https://img.shields.io/badge/Swift-5.0%20%7C%20SwiftUI-orange?logo=swift)](https://developer.apple.com/swift/)
[![MLX](https://img.shields.io/badge/Inference-Apple%20MLX%204--bit-purple)](https://github.com/ml-explore/mlx-swift)
[![Model](https://img.shields.io/badge/Model-Phi--3%20Mini%20(Fine--Tuned)-emerald)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Swift%20Testing%20%26%20Pytest-brightgreen)](tests/)
[![Privacy](https://img.shields.io/badge/Privacy-100%25%20On--Device%20%7C%20Offline-success)](#security--privacy)

An end-to-end, privacy-first medical assistant prototype running a specialized **4-bit quantized Phi-3 model** entirely on-device with **SwiftUI** and **Apple MLX**.

From custom LoRA fine-tuning and factual regression benchmarking to zero-network iOS inference, MedLLM is engineered to deliver fast, secure, and accessible medical guidance while strictly adhering to on-device memory limits.

> **Disclaimer**: This is an educational research prototype, not a certified medical device or diagnostic system. It does not replace professional medical judgment. Before clinical use, regulatory certification (MDR/FDA), dataset licensing compliance, and clinician evaluations are required.

---

## Architecture Overview

```
                                      ┌─────────────────────────────────────────────────────────┐
                                      │              Python / MLX Training Pipeline              │
                                      │                                                         │
[MedQA & Clinical Textbooks] ────────►│  Group-Aware Split  ──►  LoRA Fine-Tuning (MLX-LM)      │
                                      │                                  │                      │
                                      │  Asset Contract Tests  ◄──  4-bit Quantization & Export  │
                                      └──────────────────────────────────┼──────────────────────┘
                                                                         │ (safetensors + vocab)
                                                                         ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 iOS / SwiftUI Application Engine                               │
│                                                                                               │
│  ┌───────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐  │
│  │     Swift UI / UX     │◄────►│   Core App & Storage    │◄────►│   MLX Inference Engine  │  │
│  │                       │      │                         │      │                         │  │
│  │ • Biometric Lock (LA) │      │ • Actor MessageStore    │      │ • Custom BPE Tokenizer  │  │
│  │ • Smart Auto-Scroll   │      │ • Complete Protection   │      │ • 4-bit Quantized RoPE  │  │
│  │ • Offline Voice (TTS) │      │ • Backup Exclusion      │      │ • KV-Cache Optimization │  │
│  │ • Consultation Export │      │ • Memory Watcher        │      │ • Post-Inference Purge  │  │
│  │ • Multi-Session Search│      │   (Jetsam Resilience)   │      │   (Metal GPU Cache)     │  │
│  └───────────────────────┘      └─────────────────────────┘      └─────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Engineering Highlights

### 1. On-Device LLM Inference & Memory Optimization
- **Quantized 4-Bit Execution**: Evaluates weights on the Apple Neural Engine and GPU using Apple's MLX Swift library.
- **Metal Cache Eviction**: Purges temporary activation graphs and intermediate Metal buffers immediately upon generation completion (`GPU.clearCache()`).
- **Jetsam Resilience**: Handles `UIApplication.didReceiveMemoryWarningNotification` by evicting inference caches or unmapping idle model weights, preventing iOS memory kills.
- **Single Graph Evaluation**: Batches tensor operations into a single compute graph evaluation per token rather than synchronizing Metal after each transformer layer.
- **Custom BPE Tokenizer**: Fully native Swift implementation with incremental streaming decoding ($O(n)$ total response decoding instead of quadratic $O(n^2)$ re-parsing) and bounded cache eviction.

### 2. High-Performance UI / UX Architecture
- **Decoupled Keystroke Rendering**: Input composer maintains its text state locally, ensuring individual keystrokes never invalidate the Markdown tree, message list, or ambient mesh background.
- **Coalesced Token Streaming**: Limits UI redraws to ~20 FPS while synchronizing lightweight tactile haptic feedback (`UIImpactFeedbackGenerator`).
- **Smart Auto-Scroll**: Tracks user scroll position. If the user scrolls up to review prior answers, incoming tokens will not hijack the viewport; a floating **« Dernier message ↓ »** button lets the user smoothly snap back to the stream.
- **Interactive Emergency Actions**: Directly connects patients to emergency services (**SAMU 15** and **Europe 112**) via one-tap native URL schemes within the medical disclaimer screen.

### 3. Patient Accessibility & Clinical Workflow
- **Offline Text-to-Speech (TTS)**: Built-in speech synthesis via `AVSpeechSynthesizer` with Markdown syntax stripping, enabling hands-free access for patients with visual impairment or motor fatigue.
- **Consultation Export**: Formats complete discussions into clean Markdown / clipboard-ready summaries for physicians and patient records.
- **Session Search & Management**: Fast in-memory full-text search across conversations, with inline renaming and confirmation dialogs.

---

## Repository Structure

```text
├── ios_app/
│   └── MedicalAssistant/
│       ├── MedicalAssistant/
│       │   ├── MLXRunner.swift               # 4-bit Phi-3 transformer inference & Metal memory management
│       │   ├── Tokenizer.swift               # Fast Swift BPE tokenizer with incremental decoding
│       │   ├── ChatViewModel.swift           # Session coordinator, streaming task, memory warning observer
│       │   ├── ContentView.swift             # Main SwiftUI interface, biometric gate, floating scroll pill
│       │   ├── ChatComposer.swift            # Decoupled input bar with character count hints
│       │   ├── MessageBubble.swift           # Markdown bubbles, TTS voice player, ShareLink, copy actions
│       │   ├── ChatHistoryView.swift         # Conversation drawer with live search, renaming, swipe actions
│       │   ├── SpeechSynthesizerManager.swift# On-device speech synthesis with markdown cleanup
│       │   ├── MedicalDisclaimerView.swift   # Emergency numbers (15 / 112) and legal safety notices
│       │   ├── MessageStore.swift            # Actor-isolated encrypted JSON persistence
│       │   ├── Message.swift                 # Data models and structured consultation Markdown export
│       │   └── Phi3_Medical_Latest/          # Bundled 4-bit quantized safetensors model and vocabulary
│       └── MedicalAssistantTests/
│           └── MedicalAssistantTests.swift   # Swift unit tests (storage, tokenizer, speech, sessions)
├── training/
│   ├── data/
│   │   ├── make_mlx_jsonl.py                 # Multi-turn dataset compilation with group-aware splitting
│   │   ├── check_dataset_quality.py          # Data leakage, repetition, and quality auditing
│   │   └── prepare_datasets.py               # Hugging Face fallback tokenization
│   ├── models/
│   │   ├── model_training.py                 # MLX LoRA training with assistant-only loss masking
│   │   ├── model_benchmark.py                # Factual evaluation on held-out MedQA benchmarks
│   │   ├── model_export.py                   # Weight fusion & 4-bit quantization pipeline
│   │   └── model_checkpoint.py               # Checkpoint validation and integrity smoke tests
│   └── reports/                              # Retraining and benchmark validation logs
├── tests/
│   ├── test_data_pipeline.py                 # Leakage prevention and dataset integrity tests
│   ├── test_model_assets.py                  # Quantization contract and safetensors tensor tests
│   └── test_training_pipeline.py             # Config parser and dry-run training pipeline tests
├── pyproject.toml                            # Pytest and Ruff configuration
├── requirements-macos.txt                    # Validated Apple Silicon MLX dependencies
└── requirements.txt                          # Fallback PyTorch / Transformers dependencies
```

---

## Getting Started

### Prerequisites

- **macOS Sonoma** or newer with Apple Silicon (M1/M2/M3/M4)
- **Xcode 16+** (supports iOS 18+ and iOS 26 deployment targets)
- **Python 3.11** or **3.12**

### 1. iOS App Setup & Build

1. Open the project in Xcode:
   ```bash
   open ios_app/MedicalAssistant/MedicalAssistant.xcodeproj
   ```
2. Swift Package Manager will resolve dependencies automatically (`mlx-swift`, `swift-markdown-ui`).
3. Build and test from the terminal:
   ```bash
   # Run Swift unit tests on iPhone Simulator
   xcodebuild test \
     -project ios_app/MedicalAssistant/MedicalAssistant.xcodeproj \
     -scheme MedicalAssistant \
     -destination "platform=iOS Simulator,name=iPhone 17" \
     -only-testing:MedicalAssistantTests
   ```
4. Deploy to a physical device with adequate memory for the ~2 GB quantized model.

### 2. Python Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install validated MLX training dependencies
pip install --upgrade pip
pip install -r requirements-macos.txt -r requirements-dev.txt
```

---

## Machine Learning Pipeline

### 1. Dataset Preparation & Leakage Audit
Compiles clinical question-answering pairs and textbook explanations into native chat-format JSONL with **group-aware cross-validation** (preventing passages from the same textbook from appearing in both train and validation sets):

```bash
python training/data/make_mlx_jsonl.py
python training/data/check_dataset_quality.py --data-path training/data/mlx_data
```

### 2. MLX LoRA Fine-Tuning
Fine-tunes adapter layers on Apple Silicon using assistant-only loss masking:

```bash
# Dry run to inspect configuration
python training/models/model_training.py --dry-run

# Run LoRA training
python training/models/model_training.py \
  --model-path training/models/phi-3-mini-4k-instruct \
  --data-path training/data/mlx_data \
  --adapter-path training/models/checkpoints_phi3_mlx
```

### 3. Factual Regression Benchmarking
Prioritizes factual clinical accuracy over artificially rewarded generic emergency referrals on a held-out MedQA dataset:

```bash
python training/models/model_benchmark.py \
  --model training/models/Phi3_Medical_Latest \
  --data training/data/mlx_data/valid.jsonl \
  --max-samples 100 \
  --seed 20260805
```

### 4. 4-Bit Model Fusion & Quantization
Safely fuses the LoRA deltas into the base model and quantizes weights for low-memory iOS deployment:

```bash
python training/models/model_export.py \
  --base-model training/models/phi-3-mini-4k-instruct \
  --adapter training/models/checkpoints_phi3_mlx \
  --fused-output training/models/phi3-medprof-fused \
  --quantized-output ios_app/MedicalAssistant/MedicalAssistant/Phi3_Medical_Latest \
  --q-bits 4 \
  --q-group-size 64
```

---

## Quality Assurance & Verification

The repository enforces automated contract testing across both Swift and Python:

```bash
# 1. Run Python test suite (13 tests)
pytest -q

# 2. Run Ruff linter and static compilation check
ruff check training tests
python -m compileall -q training tests

# 3. Run Swift unit test suite (8 tests)
xcodebuild test \
  -project ios_app/MedicalAssistant/MedicalAssistant.xcodeproj \
  -scheme MedicalAssistant \
  -destination "platform=iOS Simulator,name=iPhone 17" \
  -only-testing:MedicalAssistantTests
```

---

## Security & Privacy

- **100% Offline**: Zero remote telemetry, network sockets, or analytics trackers.
- **Biometric Security**: Conversations are gated behind LocalAuthentication (Face ID / Touch ID / Device Passcode).
- **At-Rest Encryption**: Files are written with `NSDataWritingOptions.completeFileProtection`.
- **Backup Exclusion**: Conversation storage files explicitly set `isExcludedFromBackup = true` to prevent unencrypted cloud syncing.

---

## Author & Contact

**Louison Béranger** — [GitHub](https://github.com/louison-brg)  
*Machine Learning & iOS Engineer*
