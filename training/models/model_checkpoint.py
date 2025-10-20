from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Ton checkpoint
checkpoint_path = "training/models/checkpoints/checkpoint-3700"

# Charger le tokenizer et le modèle
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
model.to("mps")  # ou "cuda" si tu es sur GPU NVIDIA

# Prompt médical
prompt = "Patient: I have been feeling tired and dizzy for 3 days. What should I do?\nDoctor:"

# Tokenisation
inputs = tokenizer(prompt, return_tensors="pt").to("mps")

# Génération
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Affichage
print("\n🩺 Réponse générée :\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
