import numpy as np
import re
try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except Exception:
    HAS_PLOT = False
from collections import Counter
from datasets import load_from_disk

# ==========================================================
# 🔍 Analyse complète du dataset avant fine-tuning
# ==========================================================

DATA_PATH = "training/data/processed_professor_phi3/tokenized"

print("📦 Chargement du dataset depuis :", DATA_PATH)
dataset = load_from_disk(DATA_PATH)
print(dataset)

# ==========================================================
# 🧩 Étape 1 — Structure du dataset
# ==========================================================
print("\n📂 Colonnes disponibles :", dataset.column_names)

# Affiche un exemple brut
print("\n🧠 Exemple 0 :")
print(dataset[0])

# ==========================================================
# 🧩 Étape 2 — Longueur des séquences
# ==========================================================
if "input_ids" in dataset.column_names:
    lengths = [len(x) for x in dataset["input_ids"]]
    print("\n📏 Analyse des longueurs de séquence (tokens) :")
    print(f"   - Nombre total d’exemples : {len(lengths)}")
    print(f"   - Moyenne : {np.mean(lengths):.1f}")
    print(f"   - Médiane : {np.median(lengths):.1f}")
    print(f"   - Maximum : {np.max(lengths)}")

    if HAS_PLOT:
        plt.hist(lengths, bins=50)
        plt.title("Distribution de la longueur des séquences")
        plt.xlabel("Nombre de tokens")
        plt.ylabel("Fréquence")
        plt.show()
    else:
        print("ℹ️ matplotlib non installé — histogramme ignoré.")
else:
    print("\n⚠️ Pas de colonne 'input_ids' — le dataset n’est pas encore tokenisé.")

empty = []

# ==========================================================
# 🧩 Étape 3 — Détection d’exemples vides
# ==========================================================
if "input_ids" in dataset.column_names:
    empty = [i for i, ex in enumerate(dataset) if len(ex["input_ids"]) == 0]
    print(f"⚠️ Exemples vides : {len(empty)}")
    if len(empty) > 0:
        print("👉 Conseil : supprimer avec dataset.filter(lambda ex: len(ex['input_ids']) > 0)")
else:
    if "text" in dataset.column_names:
        empty = [i for i, ex in enumerate(dataset) if not ex["text"].strip()]
        print(f"⚠️ Exemples textuels vides : {len(empty)}")

# ==========================================================
# 🧩 Étape 4 — Répartition par type d’exemple (si meta_info)
# ==========================================================
if "meta_info" in dataset.column_names:
    print("\n📊 Répartition des types d’exemples (meta_info) :")
    counts = Counter(dataset["meta_info"])
    for k, v in counts.items():
        print(f"   - {k}: {v}")
else:
    print("\nℹ️ Aucune colonne 'meta_info' détectée (pas de catégorisation des exemples).")

# ==========================================================
# 🧩 Étape 5 — Vérification du format de prompt
# ==========================================================
def is_medical_prompt(text):
    return bool(re.search(r'(Patient|Question|Textbook|Doctor|Answer)', text))

if "text" in dataset.column_names:
    valid_ratio = sum(is_medical_prompt(t) for t in dataset["text"]) / len(dataset)
    print(f"\n✅ {valid_ratio*100:.1f}% des exemples ont un format de prompt valide.")
    if valid_ratio < 0.8:
        print("⚠️ Peu d’exemples contiennent un format clair (Patient:, Question:, etc.)")
else:
    print("\n⚠️ Impossible de vérifier les prompts — dataset déjà tokenisé.")

# ==========================================================
# 🧩 Étape 6 — Vérification du vocabulaire médical
# ==========================================================
medical_terms = ["heart", "lung", "infection", "tumor", "diabetes", "fever", "hypertension", "CT", "MRI"]
if "text" in dataset.column_names:
    ratio = sum(any(term in t.lower() for term in medical_terms) for t in dataset["text"]) / len(dataset)
    print(f"🩺 {ratio*100:.1f}% des exemples contiennent du vocabulaire médical.")
    if ratio < 0.4:
        print("⚠️ Le corpus semble peu médical — attention à la pertinence du fine-tuning.")
else:
    print("ℹ️ Dataset tokenisé — vocabulaire non vérifiable directement.")

# ==========================================================
# 🧩 Étape 7 — Rapport final
# ==========================================================
print("\n==================== 📋 RAPPORT FINAL ====================")

if "input_ids" in dataset.column_names:
    print(f"📈 Moyenne de longueur : {np.mean(lengths):.0f} tokens")
    print(f"📉 Médiane : {np.median(lengths):.0f}")
    print(f"📏 Max : {np.max(lengths)}")
    if np.max(lengths) > 1024:
        print("⚠️ Certaines séquences dépassent 1024 tokens (seront tronquées).")

if len(empty) > 0:
    print(f"⚠️ {len(empty)} exemples vides détectés → à filtrer avant entraînement.")

if "meta_info" in dataset.column_names:
    print("📊 Types d’exemples présents :", list(Counter(dataset["meta_info"]).keys()))

print("\n✅ Vérification terminée.")
