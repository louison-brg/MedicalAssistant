from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

model_path = "/Users/louison/Projets/MedicalAssistant/training/models/phi3-medprof_final_full"

try:
    # Forcer MPS si disponible
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔧 Using device: {device}")
    
    # Chargement avec paramètres optimisés
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.float16,  # Updated from torch_dtype
        trust_remote_code=True,
        attn_implementation="eager",
        use_cache=False  # Désactiver le cache globalement
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Test avec un prompt médical
    prompt = "Explain the typical symptoms and causes of acute otitis media."
    print(f"\n🔍 Testing with prompt: {prompt}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        try:
            # Add debug info
            print(f"Input shape: {inputs.input_ids.shape}")
            print(f"Device: {inputs.input_ids.device}")
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # Désactiver le cache
                return_dict_in_generate=True,
                no_repeat_ngram_size=3  # Éviter les répétitions
            )
            
            response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            print(f"\n📝 Model response:\n{response}")
            
        except Exception as e:
            print(f"❌ Generation error: {str(e)}")
            print(f"\nModel config:\n{model.config}")
            raise  # Re-raise to see full stack trace

except Exception as e:
    print(f"❌ Error: {str(e)}")
    import traceback
    print(f"\nStacktrace:\n{traceback.format_exc()}")