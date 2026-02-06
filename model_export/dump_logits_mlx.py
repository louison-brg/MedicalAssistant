import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump du dernier vecteur de logits via MLX.")
    parser.add_argument(
        "--model",
        default="models/phi3-int4",
        help="Chemin vers le modèle MLX (fusionné ou quantifié).",
    )
    parser.add_argument(
        "--tokenizer",
        default="training/models/phi3-medprof-merged",
        help="Chemin vers le tokenizer Hugging Face.",
    )
    parser.add_argument(
        "--prompt",
        default="<|user|>What are the warning symptoms of lung cancer?<|end|>\n<|assistant|>",
        help="Prompt utilisé pour générer les logits.",
    )
    parser.add_argument(
        "--output",
        default="logs/logits_mlx.json",
        help="Fichier de sortie JSON.",
    )
    parser.add_argument(
        "--dtype",
        choices=["uint16", "uint32"],
        default="uint16",
        help="Type utilisé pour encoder les tokens (uint16 pour vocabs < 65536).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    print(f"🧠 Chargement du modèle MLX : {args.model}")
    model, tokenizer = mlx_lm.load(args.model, tokenizer_path=args.tokenizer)

    tokens = tokenizer.encode(args.prompt)
    dtype = getattr(mx, args.dtype)
    inputs = mx.array([tokens], dtype=dtype)
    logits = model(inputs)[0, -1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "tokenizer": args.tokenizer,
        "prompt": args.prompt,
        "values": logits.astype(mx.float32).tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"✅ Logits MLX sauvegardés dans {output_path}")


if __name__ == "__main__":
    main()
