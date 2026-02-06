import json
from pathlib import Path
from transformers import AutoTokenizer

IOS_TOKENIZER = Path("ios_app/MedicalAssistant/MedicalAssistant/tokenizer.json")
MODEL_PATH = Path("training/models/phi3-medprof-merged")

SPECIAL_TOKENS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<unk>"]


def main() -> None:
    if not IOS_TOKENIZER.exists():
        raise FileNotFoundError(f"Missing iOS tokenizer: {IOS_TOKENIZER}")

    data = json.loads(IOS_TOKENIZER.read_text(encoding="utf-8"))
    vocab = data.get("model", {}).get("vocab", {})

    missing = [t for t in SPECIAL_TOKENS if t not in vocab]
    if missing:
        raise ValueError(f"Missing special tokens in iOS tokenizer: {missing}")

    if MODEL_PATH.exists():
        hf = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
        for token in ["<|end|>", "<unk>"]:
            if token in vocab and hf.convert_tokens_to_ids(token) != vocab[token]:
                raise ValueError(f"Token id mismatch for {token}: HF={hf.convert_tokens_to_ids(token)} iOS={vocab[token]}")

    print("✅ Tokenizer assets look consistent.")


if __name__ == "__main__":
    main()
