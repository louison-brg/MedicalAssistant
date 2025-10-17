import torch
import coremltools as ct
from transformers import AutoTokenizer, OPTForCausalLM

# ------------------------------------------------
# 1️⃣ Charger le modèle pré-entraîné
# ------------------------------------------------
model_name = "facebook/opt-350m"  # Un bon compromis taille/performance
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = OPTForCausalLM.from_pretrained(model_name)
model.eval()

print("✅ Modèle et tokenizer chargés depuis :", model_name)

# ------------------------------------------------
# 2️⃣ Créer un wrapper simplifié pour TorchScript
# ------------------------------------------------
class OPTWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        self.device = next(model.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Assurer que les tenseurs sont sur le bon device et ont le bon type
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Forcer le type à long pour être compatible avec le modèle
        input_ids = input_ids.long()
        attention_mask = attention_mask.long()
        
        # Forward pass avec paramètres explicites et minimaux
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
        return outputs.logits


wrapped_model = OPTWrapper(model)

# ------------------------------------------------
# 3️⃣ Exemple d'entrée pour traçage
# ------------------------------------------------
example_text = "Patient: I have a cough and fever."
example = tokenizer(
    example_text,
    return_tensors="pt",
    padding="max_length",
    max_length=64,
    truncation=True
)

# Force les tenseurs à être sur CPU et en mode eval
input_ids = example["input_ids"].cpu()
attention_mask = example["attention_mask"].cpu()

print(f"✓ Taille des tenseurs d'entrée :", input_ids.shape)

# ------------------------------------------------
# 4️⃣ Conversion CoreML avec séquence dynamique
# ------------------------------------------------
# Tracer le modèle avec des entrées spécifiques
with torch.no_grad():
    traced_model = torch.jit.trace(
        wrapped_model,
        (input_ids, attention_mask)
    )
mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(name="input_ids",
                      shape=(1, ct.RangeDim(1, 512))),  # min=1, max=512 tokens
        ct.TensorType(name="attention_mask",
                      shape=(1, ct.RangeDim(1, 512)))
    ],
    minimum_deployment_target=ct.target.iOS15,
)

# ------------------------------------------------
# 5️⃣ Sauvegarde
# ------------------------------------------------
output_path = "model_export/MedicalLLM.mlpackage"
mlmodel.save(output_path)

print(f"✅ Conversion terminée avec succès !\n📦 Modèle exporté vers : {output_path}")
