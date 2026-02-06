import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load a checkpoint and generate a quick response.")
    parser.add_argument("--checkpoint", default="training/models/checkpoints/checkpoint-3700")
    parser.add_argument(
        "--prompt",
        default="Patient: I have been feeling tired and dizzy for 3 days. What should I do?\nDoctor:",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(device)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("\n🩺 Réponse générée :\n")
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
