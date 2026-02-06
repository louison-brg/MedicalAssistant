import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Génération simple via un modèle CoreML causal.")
    parser.add_argument(
        "--model",
        required=True,
        help="Chemin vers la .mlpackage (ex: exports/MedicalLLM_fp16_pal6.mlpackage).",
    )
    parser.add_argument(
        "--tokenizer",
        default="training/models/phi3-medprof-merged",
        help="Tokenizer Hugging Face correspondant.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt utilisateur (plain text).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Longueur max compatible avec la conversion.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Nombre maximum de tokens générés.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Paramètre de sampling (0 = greedy).")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p (nucleus) sampling.")
    parser.add_argument("--logits-output", default=None, help="Optionnel: fichier JSON pour sauvegarder la dernière distribution.")
    parser.add_argument(
        "--compute-units",
        choices=["all", "cpu_and_ne", "cpu_only"],
        default="cpu_only",
        help="Unité de calcul utilisée localement.",
    )
    parser.add_argument(
        "--tmpdir",
        default=".coreml_tmp",
        help="Répertoire temporaire dédié pour CoreML (évite les permissions /tmp).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Graine aléatoire pour reproduire le sampling (None = aléatoire).",
    )
    return parser.parse_args()


def top_k_top_p_filtering(scores: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]

    if top_k > 0:
        sorted_scores = sorted_scores[:top_k]
        sorted_indices = sorted_indices[:top_k]

    if 0 < top_p < 1.0:
        probs = np.exp(sorted_scores - sorted_scores.max())
        probs /= probs.sum()
        cumulative = np.cumsum(probs)
        cutoff = cumulative > top_p
        if cutoff.any():
            keep = np.argmax(cutoff)
            sorted_scores = sorted_scores[: keep + 1]
            sorted_indices = sorted_indices[: keep + 1]

    filtered = np.full_like(scores, -np.inf)
    filtered[sorted_indices] = scores[sorted_indices]
    return filtered


def sample_token(scores: np.ndarray, temperature: float, top_k: int, top_p: float, rng: np.random.Generator) -> int:
    if temperature <= 0:
        return int(np.argmax(scores))

    filtered = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
    filtered = filtered / max(temperature, 1e-4)
    probs = np.exp(filtered - np.max(filtered))
    mask = np.isfinite(filtered)
    probs = probs * mask
    probs_sum = probs.sum()
    if probs_sum <= 0:
        return int(np.argmax(scores))
    probs /= probs_sum
    return int(rng.choice(len(probs), p=probs))


def main() -> None:
    args = parse_args()

    compute_units_map = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    compute_units = compute_units_map[args.compute_units]

    tmp_path = Path(args.tmpdir).resolve()
    tmp_path.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_path)

    print(f"🧠 Chargement du modèle : {args.model}")
    mlmodel = ct.models.MLModel(args.model, compute_units=compute_units)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    bos = tokenizer.bos_token_id or tokenizer.eos_token_id
    eos = tokenizer.eos_token_id

    rng = np.random.default_rng(args.seed)

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    input_ids: List[int] = ([bos] if bos is not None else []) + prompt_ids

    if len(input_ids) >= args.seq_len:
        raise ValueError(f"Prompt trop long ({len(input_ids)}) pour seq_len={args.seq_len}")

    generated: List[int] = []
    current_ids = input_ids.copy()

    for _ in range(args.max_new_tokens):
        if len(current_ids) >= args.seq_len:
            print("⚠️ Séquence pleine, arrêt de la génération (ajustez --seq-len ou --max-new-tokens).")
            break
        padded = current_ids + [tokenizer.pad_token_id or 0] * (args.seq_len - len(current_ids))
        np_input = np.array([padded], dtype=np.int32)
        prediction = mlmodel.predict({"input_ids": np_input})
        logits = prediction["logits"]
        next_scores = logits[0, len(current_ids) - 1]
        token_id = sample_token(next_scores, args.temperature, args.top_k, args.top_p, rng)
        generated.append(token_id)
        current_ids.append(token_id)
        if eos is not None and token_id == eos:
            break

    full_sequence = input_ids + generated
    text = tokenizer.decode(full_sequence, skip_special_tokens=True)
    print("📝 Génération :")
    print(text)

    if args.logits_output:
        Path(args.logits_output).write_text(json.dumps({"values": next_scores.tolist()}))
        print(f"📤 Derniers logits enregistrés dans {args.logits_output}")


if __name__ == "__main__":
    main()
