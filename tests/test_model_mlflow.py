import os
import torch
import mlflow
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
import numpy as np
from tqdm import tqdm

# Configuration
MODEL_PATH = "/Users/louison/Projets/MedicalAssistant/training/models/phi3-medprof_final_full"
MLFLOW_TRACKING_URI = "file:./mlruns"
TEST_PROMPTS = [
    "Explain the typical symptoms and causes of acute otitis media.",
    "What are the main risk factors for type 2 diabetes?",
    "Describe the typical presentation of bacterial pneumonia.",
    "What are the common side effects of ACE inhibitors?"
]

def evaluate_perplexity(model, tokenizer, eval_dataset, device, max_samples=100):
    """Calcule la perplexité sur un dataset d'évaluation"""
    model.eval()
    total_loss = 0
    total_length = 0
    
    # Prendre un sous-ensemble aléatoire si le dataset est trop grand
    if len(eval_dataset) > max_samples:
        eval_indices = np.random.choice(len(eval_dataset), max_samples, replace=False)
        eval_dataset = eval_dataset.select(eval_indices)
    
    for item in tqdm(eval_dataset, desc="Calculating perplexity"):
        with torch.no_grad():
            inputs = tokenizer(item["text"], return_tensors="pt").to(device)
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item() * inputs.input_ids.size(1)
            total_length += inputs.input_ids.size(1)
    
    return torch.exp(torch.tensor(total_loss / total_length))

def generate_response(model, tokenizer, prompt, device):
    """Génère une réponse pour un prompt donné"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False  # Désactiver le cache pour éviter les erreurs
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # Configuration MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("model-evaluation")
    
    # Détection du device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Chargement du modèle et du tokenizer
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map=device,
        dtype=torch.float16,  # Utilisation de dtype au lieu de torch_dtype
        trust_remote_code=True,
        attn_implementation="eager"  # Forcer l'implémentation eager
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    
    try:
        # Charger un petit dataset de test si disponible
        eval_dataset = load_from_disk("training/data/processed/test")
        has_eval_dataset = True
    except:
        print("No evaluation dataset found, skipping perplexity calculation")
        has_eval_dataset = False
    
    with mlflow.start_run(run_name="model_evaluation"):
        # Log des paramètres du modèle
        mlflow.log_params({
            "model_path": MODEL_PATH,
            "device": device,
            "dtype": str(model.dtype)
        })
        
        # Calcul et log de la perplexité si dataset disponible
        if has_eval_dataset:
            print("Calculating perplexity...")
            perplexity = evaluate_perplexity(model, tokenizer, eval_dataset, device)
            mlflow.log_metric("perplexity", perplexity.item())
            print(f"Perplexity: {perplexity:.2f}")
        
        # Test des prompts médicaux
        print("\nTesting medical prompts...")
        for i, prompt in enumerate(TEST_PROMPTS):
            print(f"\nTesting prompt {i+1}: {prompt}")
            response = generate_response(model, tokenizer, prompt, device)
            
            # Log dans MLflow
            mlflow.log_text(response, f"response_{i+1}.txt")
            print(f"Response: {response}\n")
            
            # Log de la longueur de réponse comme métrique
            mlflow.log_metric(f"response_{i+1}_length", len(response.split()))

if __name__ == "__main__":
    main()
