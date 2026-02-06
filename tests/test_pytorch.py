import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick generation sanity check.")
    parser.add_argument(
        "--model-path",
        default=os.getenv("MODEL_PATH", "training/models/phi3-medprof-merged"),
    )
    parser.add_argument(
        "--prompt",
        default="Explain the typical symptoms and causes of seborrheic dermatitis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if device == "mps" else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    print(f"🔧 Using device: {device} | dtype: {dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation="eager",
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
            no_repeat_ngram_size=3,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📝 Model response:\n{response}")


if __name__ == "__main__":
    main()
