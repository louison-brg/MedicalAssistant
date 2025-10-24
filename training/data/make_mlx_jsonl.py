import os, json, random, pathlib

BASE = pathlib.Path(__file__).resolve().parent
OUT_DIR = BASE / "mlx_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUT_DIR / "train.jsonl"
VALID_PATH = OUT_DIR / "valid.jsonl"

random.seed(42)
rows = []

# Exemple fictif : ton dataset doit suivre ce format
# Tu peux remplacer cette partie par ton propre parsing de MedQA / MedDialog / Textbooks
examples = [
    {
        "prompt": "Student: What are the symptoms of pneumonia?\nProfessor:",
        "completion": " Pneumonia typically presents with fever, cough, and difficulty breathing."
    },
    {
        "prompt": "Student: Explain the difference between type 1 and type 2 diabetes.\nProfessor:",
        "completion": " Type 1 is autoimmune and insulin-dependent, while type 2 is linked to insulin resistance."
    },
]

rows.extend(examples)
random.shuffle(rows)

n = len(rows)
train, val = rows[: int(0.9 * n)], rows[int(0.9 * n):]

with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    for r in train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(VALID_PATH, "w", encoding="utf-8") as f:
    for r in val:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Dataset MLX ready: {len(train)} train / {len(val)} valid examples.")
print(f"📂 Saved in {OUT_DIR}")
