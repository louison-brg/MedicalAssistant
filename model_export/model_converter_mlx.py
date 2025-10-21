import os
import numpy as np
import mlx.core as mx
import mlx_lm

MODEL_PATH = "training/models/phi3-medprof-merged"
SAVE_PATH = "model_export/phi3-medprof-mlx_weights.npz"

print("📦 Chargement du modèle MLX...")
model, tokenizer = mlx_lm.load(MODEL_PATH)
print("✅ Modèle MLX chargé avec succès !")

# ====================================================
# 🧩 Fonction récursive universelle et robuste
# ====================================================

def to_numpy_safe(x):
    """Convertit un objet MLX en numpy, quelle que soit la version."""
    try:
        # Cas 1 — Conversion directe via np.array
        return np.array(x)
    except Exception:
        try:
            # Cas 2 — Conversion en float32 si le dtype n’est pas supporté
            return np.array(x.astype(mx.float32))
        except Exception as e:
            print(f"⚠️ Conversion échouée pour {type(x)} : {e}")
            return None

def flatten_params(params, prefix=""):
    """Aplatis les dictionnaires de paramètres MLX en clés/valeurs plates."""
    flat = {}
    for k, v in params.items():
        name = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(flatten_params(v, prefix=name))
        elif isinstance(v, mx.array):
            np_val = to_numpy_safe(v)
            if np_val is not None:
                flat[name] = np_val
    return flat

# ====================================================
# 💾 Conversion + sauvegarde
# ====================================================
params = model.parameters()
params_np = flatten_params(params)

os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
np.savez(SAVE_PATH, **params_np)

print(f"🎉 Poids sauvegardés avec succès dans : {SAVE_PATH}")
print(f"📊 Total de tenseurs exportés : {len(params_np)}")
