"""
Evaluate a Phi-3 (or compatible) model:
- Perplexity on a tokenized or text dataset
- Sample generations
- Optional MLflow logging
"""

import argparse
import math
import os
from typing import List, Optional

import torch
import mlflow
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a causal LM checkpoint.")
    parser.add_argument("--model-path", default="training/models/phi3-medprof_final_full")
    parser.add_argument("--data-path", default="training/data/processed_professor_phi3/tokenized")
    parser.add_argument("--reports-dir", default="training/reports")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--mlflow-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="model-evaluation")
    parser.add_argument("--prompt-file", default=None, help="Optional file with one prompt per line")
    return parser.parse_args()


def resolve_device(requested: str) -> tuple[str, torch.dtype]:
    if requested != "auto":
        device = requested
    elif torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if device == "mps":
        dtype = torch.float32
    elif device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def load_prompts(path: Optional[str]) -> List[str]:
    if not path:
        return [
            "Patient: I have a persistent cough and chest pain. What could this be?\nDoctor:",
            "Question: What is the treatment for type 2 diabetes in obese patients?\nAnswer:",
            "Patient: My throat hurts and I feel feverish for 3 days. What should I do?\nDoctor:",
        ]
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def main() -> None:
    args = parse_args()
    os.makedirs(args.reports_dir, exist_ok=True)

    device, dtype = resolve_device(args.device)
    print(f"✅ Using device: {device} | dtype: {dtype}")

    print("📦 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)

    # Dataset
    dataset = None
    if args.data_path and os.path.exists(args.data_path):
        dataset = load_from_disk(args.data_path)
    else:
        print("⚠️ No dataset found. Skipping perplexity.")

    eval_dataset = None
    if dataset is not None:
        if isinstance(dataset, DatasetDict):
            eval_dataset = dataset.get("validation") or dataset.get("eval") or dataset.get("test")
            if eval_dataset is None:
                eval_dataset = dataset.get("train") or next(iter(dataset.values()))
        else:
            eval_dataset = dataset

        # Prepare data
        if "text" in eval_dataset.column_names:
            def tokenize(batch):
                return tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=args.max_length,
                )
            eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=eval_dataset.column_names)

        if "labels" not in eval_dataset.column_names and "input_ids" in eval_dataset.column_names:
            eval_dataset = eval_dataset.map(lambda ex: {"labels": ex["input_ids"]})

        if args.max_samples and len(eval_dataset) > args.max_samples:
            eval_dataset = eval_dataset.select(range(args.max_samples))

    # Perplexity
    perplexity = None
    if eval_dataset is not None:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        eval_args = TrainingArguments(
            output_dir="training/models/eval_output",
            per_device_eval_batch_size=1,
            dataloader_drop_last=True,
            report_to=[],
        )
        trainer = Trainer(model=model, args=eval_args, eval_dataset=eval_dataset, data_collator=data_collator)
        print("📊 Computing perplexity...")
        eval_results = trainer.evaluate()
        loss_key = "eval_loss" if "eval_loss" in eval_results else "loss" if "loss" in eval_results else None
        if loss_key:
            perplexity = math.exp(eval_results[loss_key])
            print(f"✅ Perplexity: {perplexity:.2f}")
        else:
            print("⚠️ No loss found; perplexity not computed.")

    # Generation
    prompts = load_prompts(args.prompt_file)
    generated = []
    print("\n🩺 Sample generations:")
    for prompt in prompts:
        print(f"\nPrompt: {prompt}\n---")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=120,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(text)
        generated.append((prompt, text))

    # Report
    report_path = os.path.join(args.reports_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== MODEL EVALUATION ===\n\n")
        f.write(f"Model: {args.model_path}\n")
        if perplexity is not None:
            f.write(f"Perplexity: {perplexity:.2f}\n\n")
        else:
            f.write("Perplexity: not computed\n\n")
        f.write("=== Generations ===\n")
        for i, (prompt, gen) in enumerate(generated, 1):
            f.write(f"\n--- Example {i} ---\nPrompt:\n{prompt}\n\nGenerated:\n{gen}\n")

    print(f"\n📄 Report saved to: {report_path}")

    if not args.disable_mlflow:
        print("\n📈 Logging to MLflow...")
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name="model_evaluation"):
            if perplexity is not None:
                mlflow.log_metric("perplexity", perplexity)
            mlflow.log_artifact(report_path)

            # Save a few generations as artifacts
            for i, (_, gen) in enumerate(generated, 1):
                tmp_path = os.path.join(args.reports_dir, f"gen_{i}.txt")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(gen)
                mlflow.log_artifact(tmp_path)

    print("✅ Evaluation complete.")


if __name__ == "__main__":
    main()
