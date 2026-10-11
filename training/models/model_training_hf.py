"""
Fine-tuning Phi-3-mini local (MPS/CUDA/CPU) with LoRA + optional MLflow.
Provides a CLI to keep paths and hyperparams consistent.
"""

import argparse
import inspect
import os
from typing import List

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import mlflow
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Phi-3-mini.")
    parser.add_argument("--model-path", default="training/models/phi-3-mini-4k-instruct", help="HF model path")
    parser.add_argument("--data-path", default="training/data/processed_professor_phi3_hf", help="Tokenized dataset")
    parser.add_argument("--save-dir", default="training/models/checkpoints_phi3_lora", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=float, default=0.1, help="If dataset is not a dict")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="phi3-medical-professor")
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-targets", default="", help="Comma-separated target modules (optional)")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable dynamic model code loading from local/Hub files when required.",
    )
    parser.add_argument("--quick", action="store_true", help="Limit data for a fast smoke run")
    return parser.parse_args()


def resolve_device(requested: str) -> tuple[str, torch.dtype, bool]:
    if requested != "auto":
        device = requested
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if device == "mps":
        dtype = torch.float16
    elif device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype, device == "mps"


def detect_lora_targets(model) -> List[str]:
    targets: set[str] = set()
    target_modules = {
        "q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj",
        "gate_up_proj", "down_proj", "up_proj", "gate_proj",
    }
    for name, _ in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in target_modules:
            targets.add(leaf)
    return sorted(targets)


class AssistantOnlyCollator:
    """Dynamically pads inputs while preserving precomputed -100 label masks."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        model_inputs = [
            {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
            for feature in features
        ]
        batch = self.tokenizer.pad(model_inputs, padding=True, return_tensors="pt")
        padded_length = batch["input_ids"].shape[1]
        padded_labels = []
        for feature in features:
            labels = list(feature["labels"])
            padding = [-100] * (padded_length - len(labels))
            labels = padding + labels if self.tokenizer.padding_side == "left" else labels + padding
            padded_labels.append(labels)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"

    device, dtype, is_mac = resolve_device(args.device)
    print(f"🧠 Device: {device} | dtype: {dtype}")

    # Data loading
    print("🔄 Loading dataset...")
    dataset = load_from_disk(args.data_path)
    if isinstance(dataset, DatasetDict):
        train_dataset = dataset.get("train") or dataset.get("training") or next(iter(dataset.values()))
        eval_dataset = dataset.get("validation") or dataset.get("eval") or dataset.get("test")
        if eval_dataset is None:
            split = train_dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
            train_dataset, eval_dataset = split["train"], split["test"]
    else:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]

    if args.quick:
        max_train = args.max_train_samples or 500
        max_eval = args.max_eval_samples or 100
        train_dataset = train_dataset.select(range(min(max_train, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(max_eval, len(eval_dataset))))
        print(f"⚠️ Quick mode: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        if args.max_train_samples:
            train_dataset = train_dataset.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_dataset))))
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    print(f"✅ Dataset: {len(train_dataset)} train / {len(eval_dataset)} eval")

    def truncate(example):
        return {
            key: example[key][: args.max_length]
            for key in ("input_ids", "attention_mask", "labels")
            if key in example
        }

    train_dataset = train_dataset.map(truncate)
    eval_dataset = eval_dataset.map(truncate)
    train_dataset = train_dataset.filter(lambda example: any(label != -100 for label in example["labels"]))
    eval_dataset = eval_dataset.filter(lambda example: any(label != -100 for label in example["labels"]))

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📦 Loading Phi-3 model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    model.to(device)
    print("✅ Model loaded.")

    # LoRA config
    if args.lora_targets:
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    else:
        targets = detect_lora_targets(model)
    print(f"🎯 LoRA target modules: {targets}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=targets,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.to(device)
    model.enable_input_require_grads()
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}% of {all_params:,})")
    if trainable_params == 0:
        raise RuntimeError("No trainable parameters. Check device or LoRA targets.")

    # MLflow
    report_to = []
    if not args.disable_mlflow:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        report_to = ["mlflow"]

    # Training args
    training_kwargs = dict(
        output_dir=args.save_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_steps=args.eval_steps,
        dataloader_pin_memory=not is_mac,
        report_to=report_to,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        label_names=["labels"],
    )
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    training_kwargs[strategy_key] = "steps"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=AssistantOnlyCollator(tokenizer),
    )

    print("🚀 Starting LoRA fine-tuning...")
    torch.set_grad_enabled(True)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    print("📊 Post-training evaluation...")
    results = trainer.evaluate()
    print("✅ Eval results:", results)

    print("💾 Saving LoRA adapters...")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    print("✨ Done.")


if __name__ == "__main__":
    main()
