import argparse
import os
import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

TEST_PROMPTS = [
    "Explain the typical symptoms and causes of acute otitis media.",
    "What are the main risk factors for type 2 diabetes?",
    "Describe the typical presentation of bacterial pneumonia.",
    "What are the common side effects of ACE inhibitors?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLflow eval + prompt checks")
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "training/models/phi3-medprof-merged"),
    )
    parser.add_argument("--eval-data", default="training/data/processed_professor_phi3/tokenized")
    parser.add_argument("--mlflow-uri", default="file:./mlruns")
    parser.add_argument("--mlflow-experiment", default="model-evaluation")
    parser.add_argument("--max-samples", type=int, default=100)
    return parser.parse_args()


def evaluate_perplexity(model, tokenizer, eval_dataset, device, max_samples=100):
    model.eval()
    total_loss = 0.0
    total_length = 0

    if len(eval_dataset) > max_samples:
        eval_indices = np.random.choice(len(eval_dataset), max_samples, replace=False)
        eval_dataset = eval_dataset.select(eval_indices)

    for item in tqdm(eval_dataset, desc="Calculating perplexity"):
        with torch.no_grad():
            if "text" in item:
                inputs = tokenizer(item["text"], return_tensors="pt").to(device)
            else:
                inputs = {"input_ids": torch.tensor([item["input_ids"]], device=device)}
                if "attention_mask" in item:
                    inputs["attention_mask"] = torch.tensor([item["attention_mask"]], device=device)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_length += inputs["input_ids"].size(1)

    return torch.exp(torch.tensor(total_loss / total_length))


def generate_response(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    print(f"Using device: {device} | dtype: {dtype}")

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    eval_dataset = None
    if os.path.exists(args.eval_data):
        try:
            eval_dataset = load_from_disk(args.eval_data)
        except Exception:
            eval_dataset = None

    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_params({
            "model_path": args.model_path,
            "device": device,
            "dtype": str(model.dtype),
        })

        if eval_dataset is not None:
            print("Calculating perplexity...")
            perplexity = evaluate_perplexity(model, tokenizer, eval_dataset, device, max_samples=args.max_samples)
            mlflow.log_metric("perplexity", perplexity.item())
            print(f"Perplexity: {perplexity:.2f}")
        else:
            print("No evaluation dataset found, skipping perplexity")

        print("\nTesting medical prompts...")
        for i, prompt in enumerate(TEST_PROMPTS, 1):
            print(f"\nPrompt {i}: {prompt}")
            response = generate_response(model, tokenizer, prompt, device)
            mlflow.log_text(response, f"response_{i}.txt")
            mlflow.log_metric(f"response_{i}_length", len(response.split()))
            print(f"Response: {response}\n")


if __name__ == "__main__":
    main()
