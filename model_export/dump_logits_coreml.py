import argparse
import json
from pathlib import Path

import coremltools as ct
import numpy as np
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exécute un .mlpackage CoreML localement et sauvegarde le dernier vecteur de logits."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Chemin vers le .mlpackage exporté (ex: exports/MedicalLLM_fp16_q4.mlpackage).",
    )
    parser.add_argument(
        "--tokenizer",
        default="training/models/phi3-medprof-merged",
        help="Chemin du tokenizer Hugging Face utilisé lors du fine-tune.",
    )
    parser.add_argument(
        "--prompt",
        default="<|user|>What are the warning symptoms of lung cancer?<|end|>\n<|assistant|>",
        help="Prompt à injecter pour générer les logits.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Longueur maximale acceptée par le modèle CoreML (doit correspondre à la conversion).",
    )
    parser.add_argument(
        "--output",
        default="logs/logits_coreml.json",
        help="Fichier cible pour enregistrer les logits.",
    )
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu_and_ne", "cpu_only"],
        default="cpu_only",
        help="Unité de calcul utilisée pour l'exécution CoreML locale.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not model_path.exists():
        raise FileNotFoundError(f"Impossible de trouver le modèle CoreML : {model_path}")

    compute_units_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    compute_units = compute_units_map[args.compute_units]

    print(f"🧠 Chargement du modèle CoreML : {model_path}")
    mlmodel = ct.models.MLModel(str(model_path), compute_units=compute_units)

    print(f"🔤 Chargement du tokenizer : {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    encoded = tokenizer(
        args.prompt,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=args.seq_len,
    )
    input_ids = encoded["input_ids"].astype(np.int32)
    if input_ids.shape != (1, args.seq_len):
        raise ValueError(
            f"Tensor input_ids doit être (1, {args.seq_len}), obtenu {input_ids.shape}. "
            "Vérifiez --seq-len et le prompt."
        )

    print("⏱️ Exécution CoreML…")
    prediction = mlmodel.predict({"input_ids": input_ids})
    logits = prediction["logits"]
    if logits.ndim == 3:
        # Batch, sequence, vocab
        if logits.shape[0] != 1 or logits.shape[1] != args.seq_len:
            raise ValueError(f"Shape inattendue pour logits: {logits.shape}")
        last_logits = logits[0, -1]
    elif logits.ndim == 2:
        # Sequence, vocab
        if logits.shape[0] != args.seq_len:
            raise ValueError(f"Shape inattendue pour logits: {logits.shape}")
        last_logits = logits[-1]
    else:
        raise ValueError(f"Rank inattendu pour logits: {logits.shape}")
    last_logits = last_logits.astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": args.prompt,
        "values": last_logits.tolist(),
        "model": str(model_path),
        "tokenizer": args.tokenizer,
        "seq_len": args.seq_len,
    }
    output_path.write_text(json.dumps(payload, indent=2))
    print(f"✅ Logits CoreML sauvegardés dans {output_path}")


if __name__ == "__main__":
    main()
