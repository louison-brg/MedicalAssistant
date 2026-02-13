import argparse
import warnings
from pathlib import Path

import numpy as np
import coremltools as ct
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from coremltools.optimize.coreml import (  # type: ignore
        linear_quantize_weights,
        palettize_weights,
    )
except Exception:  # pragma: no cover
    linear_quantize_weights = None
    palettize_weights = None


class Phi3Wrapper(torch.nn.Module):
    """Petit wrapper pour stabiliser le tracé Torch -> CoreML."""

    def __init__(self, model: AutoModelForCausalLM, output_dtype: torch.dtype = torch.float32):
        super().__init__()
        self.model = model
        self.config = model.config
        self.output_dtype = output_dtype

        # Recalibre les embeddings rotatoires (sinon erreurs lors du traçage)
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, "rotary_emb"):
                rotary_emb = layer.self_attn.rotary_emb
                if rotary_emb.inv_freq is None:
                    dim = rotary_emb.dim
                    base = getattr(rotary_emb, "base", 10000.0)
                    device = next(self.model.parameters()).device
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
                    rotary_emb.inv_freq = inv_freq.to(dtype=self.model.dtype)
                else:
                    rotary_emb.inv_freq = rotary_emb.inv_freq.to(dtype=self.model.dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(dtype=torch.int32)
        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = outputs.logits.to(dtype=self.output_dtype)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convertir Phi-3 fine-tuné en CoreML optimisé.")
    parser.add_argument(
        "--model-path",
        default="training/models/phi3-medprof-merged",
        help="Checkpoint fusionné (LoRA + base).",
    )
    parser.add_argument(
        "--output",
        default="ios_app/MedicalAssistant/MedicalAssistant/MedicalLLM.mlpackage",
        help="Chemin de sortie du modèle CoreML (nom aligné avec l'app iOS).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
        help="Longueur de séquence maximale (doit correspondre à CoreMLRunner).",
    )
    parser.add_argument(
        "--precision",
        choices=["float32", "float16"],
        default="float16",
        help="Précision principale lors de la conversion.",
    )
    parser.add_argument(
        "--deployment-target",
        default="iOS15",
        help="Cible minimale (ct.target.*). Ajusté automatiquement pour la quantization 4-bit.",
    )
    parser.add_argument(
        "--quantize-bits",
        type=int,
        choices=[4, 8],
        default=None,
        help="Quantification post-traitement (4 ou 8 bits). Requiert coremltools>=7.0.",
    )
    parser.add_argument(
        "--quant-mode",
        choices=["linear", "linear_symmetric"],
        default="linear",
        help="Mode de quantification (impact sur le centrage des poids).",
    )
    parser.add_argument(
        "--quant-granularity",
        choices=["per_tensor", "per_channel", "per_block"],
        default="per_block",
        help="Granularité de la quantification.",
    )
    parser.add_argument(
        "--quant-block-size",
        type=int,
        default=64,
        help="Taille de bloc pour la granularité per_block (multiple de 16 recommandé).",
    )
    parser.add_argument(
        "--quant-weight-threshold",
        type=int,
        default=4096,
        help="Nombre minimal d'éléments pour quantifier un tenseur (évite les petites matrices sensibles).",
    )
    parser.add_argument(
        "--quant-dtype",
        choices=["int4", "uint4", "int8", "uint8"],
        default=None,
        help="Type exact à utiliser (sinon déterminé via quantize-bits).",
    )
    parser.add_argument(
        "--palettize-bits",
        type=int,
        choices=[2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Palettization (K-Means) des poids avec n bits (ex: 6 pour int6).",
    )
    parser.add_argument(
        "--palettize-mode",
        choices=["kmeans", "scalar"],
        default="kmeans",
        help="Mode de palettization (kmeans recommandé).",
    )
    parser.add_argument(
        "--palettize-weight-threshold",
        type=int,
        default=4096,
        help="Nombre minimal d'éléments pour palettiser un tenseur.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"📦 Chargement du checkpoint : {model_path}")

    torch.set_grad_enabled(False)
    torch._C._jit_set_profiling_executor(False)

    # Charge en float32 puis cast si besoin (plus sûr côté CPU).
    # Manual loading to bypass transformers 4.43 sharding bug
    print("⏳ Manually loading model weights (bypassing from_pretrained bug)...")
    from transformers import AutoConfig, Phi3ForCausalLM
    from safetensors.torch import load_file
    import glob

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    # Ensure no auto_map if present (we removed it, but good to be safe)
    if hasattr(config, "auto_map"):
        del config.auto_map

    base_model = Phi3ForCausalLM(config)
    
    # Load all safetensors shards
    state_dict = {}
    safetensor_files = sorted(glob.glob(str(Path(model_path) / "*.safetensors")))
    print(f"📂 Found {len(safetensor_files)} safetensor shards: {[f.split('/')[-1] for f in safetensor_files]}")
    
    for sf in safetensor_files:
        print(f"   -> Loading {sf}...")
        shard = load_file(sf)
        state_dict.update(shard)

    print("🔧 Applying state_dict...")
    keys = base_model.load_state_dict(state_dict, strict=False)
    print(f"✅ Weights loaded. Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")
    if len(keys.missing_keys) > 0:
        print(f"⚠️ Missing keys details: {keys.missing_keys[:5]}...")
    
    base_model.eval()

    target_dtype = torch.float16 if args.precision == "float16" else torch.float32
    if target_dtype != base_model.dtype:
        print(f"🔁 Conversion du modèle en {target_dtype}")
        base_model.to(dtype=target_dtype)

    print("⏳ Loading tokenizer from base model (to avoid JSON version mismatch)...")
    tokenizer = AutoTokenizer.from_pretrained("training/models/phi-3-mini-4k-instruct")

    print("🔍 Préparation d'un exemple pour le traçage…")
    example = tokenizer(
        "Patient: I have a headache. Doctor:",
        return_tensors="pt",
        padding="max_length",
        max_length=args.seq_len,
        truncation=True,
    )
    input_ids = example["input_ids"].to(dtype=torch.int32, device="cpu")
    assert input_ids.shape == (1, args.seq_len), input_ids.shape

    print("📝 Traçage TorchScript…")
    wrapper = Phi3Wrapper(base_model, output_dtype=torch.float32)
    traced = torch.jit.trace(wrapper, (input_ids,))

    compute_precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    target_str = args.deployment_target
    def _parse_ios_version(s: str) -> int:
        if s.lower().startswith("ios"):
            return int("".join(ch for ch in s[3:] if ch.isdigit()))
        raise ValueError(f"Cible iOS invalide : {s}")

    try:
        target_version = _parse_ios_version(target_str)
    except ValueError:
        target_version = 0

    if args.quantize_bits == 4 and target_version and target_version < 18:
        print("ℹ️ 4-bit requiert iOS18+. Ajustement automatique du déploiement à iOS18.")
        target_str = "iOS18"
    try:
        deployment_target = getattr(ct.target, target_str)
    except AttributeError as exc:
        raise ValueError(f"Cible inconnue pour minimum_deployment_target : {target_str}") from exc

    output_name = "logits"

    print(f"⚙️ Conversion CoreML (precision={args.precision})…")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_ids", shape=(1, args.seq_len), dtype=np.int32)],
        outputs=[ct.TensorType(name=output_name, dtype=np.float32)],
        convert_to="mlprogram",
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=deployment_target,
    )

    mlmodel.author = "Louison Beranger"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Phi-3 medical assistant (float16 optimisé)" if args.precision == "float16" else "Phi-3 medical assistant"
    mlmodel.version = "1.1"

    print(f"💾 Sauvegarde : {output_path}")
    mlmodel.save(str(output_path))

    if args.quantize_bits is not None and args.palettize_bits is not None:
        raise ValueError("Choisissez soit --quantize-bits soit --palettize-bits, mais pas les deux.")

    if args.quantize_bits is not None:
        if linear_quantize_weights is None:
            print("⚠️ Quantification indisponible : mettez à jour coremltools (>=7.0).")
        else:
            dtype_map = {4: "int4", 8: "int8"}
            dtype = args.quant_dtype or dtype_map.get(args.quantize_bits, "int4")
            block_size = args.quant_block_size if args.quant_granularity == "per_block" else None
            print(
                f"🎛️ Quantification {args.quantize_bits}-bits des poids "
                f"(dtype={dtype}, mode={args.quant_mode}, granularity={args.quant_granularity}, block={block_size})…"
            )
            try:
                from coremltools.optimize.coreml import OptimizationConfig, OpLinearQuantizerConfig

                config = OptimizationConfig(
                    global_config=OpLinearQuantizerConfig(
                        dtype=dtype,
                        mode=args.quant_mode,
                        granularity=args.quant_granularity,
                        block_size=block_size,
                        weight_threshold=args.quant_weight_threshold,
                    )
                )
                quantized_model = linear_quantize_weights(mlmodel, config=config, joint_compression=False)
                quant_output = output_path.with_name(
                    f"{output_path.stem}_q{args.quantize_bits}{output_path.suffix}"
                )
                quantized_model.save(str(quant_output))
                print(f"✅ Modèle quantifié sauvegardé : {quant_output}")
            except Exception as err:  # pragma: no cover
                print(f"⚠️ Échec de la quantification : {err}")
    elif args.palettize_bits is not None:
        if palettize_weights is None:
            print("⚠️ Palettization indisponible : mettez à jour coremltools (>=7.0).")
        else:
            print(
                f"🎨 Palettization {args.palettize_bits}-bits des poids "
                f"(mode={args.palettize_mode}, threshold={args.palettize_weight_threshold})…"
            )
            try:
                from coremltools.optimize.coreml import OptimizationConfig, OpPalettizerConfig

                config = OptimizationConfig(
                    global_config=OpPalettizerConfig(
                        nbits=args.palettize_bits,
                        mode=args.palettize_mode,
                        weight_threshold=args.palettize_weight_threshold,
                    )
                )
                pal_model = palettize_weights(mlmodel, config=config)
                pal_output = output_path.with_name(
                    f"{output_path.stem}_pal{args.palettize_bits}{output_path.suffix}"
                )
                pal_model.save(str(pal_output))
                print(f"✅ Modèle palettisé sauvegardé : {pal_output}")
            except Exception as err:  # pragma: no cover
                print(f"⚠️ Échec de la palettization : {err}")

    print("✅ Conversion terminée !")
    print("ℹ️ Étapes suivantes :")
    print("  1. Supprimer l'ancien MedicalLLM.mlpackage / .mlmodelc du projet.")
    print(f"  2. Copier {output_path.name} dans ios_app/MedicalAssistant/MedicalAssistant/")
    print("  3. Dans Xcode > Build Phases > Copy Bundle Resources, vérifier que le nouveau fichier est inclus.")
    print("  4. Recompiler, puis laisser CoreMLRunner compiler/charger automatiquement le modèle.")


if __name__ == "__main__":
    main()
