import torch
import numpy as np
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "/Users/louison/Projets/MedicalAssistant/training/models/phi3-medprof_final_full"
SAVE_PATH = "ios_app/MedicalAssistant/MedicalAssistant/MedicalLLM.mlpackage"

print(f"📦 Chargement du modèle depuis : {MODEL_PATH}")

# Création d'un wrapper pour le modèle
class Phi3Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        
        # Force all parameters to float32
        for param in self.model.parameters():
            param.data = param.data.to(dtype=torch.float32)
        
        # Initialize rotary embeddings if needed
        for layer in self.model.model.layers:
            if hasattr(layer.self_attn, 'rotary_emb'):
                rotary_emb = layer.self_attn.rotary_emb
                if rotary_emb.inv_freq is None:
                    dim = rotary_emb.dim
                    max_seq_len = getattr(rotary_emb, 'max_seq_len', 4096)
                    base = getattr(rotary_emb, 'base', 10000.0)
                    device = next(self.model.parameters()).device
                    
                    # Initialize inv_freq properly
                    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
                    rotary_emb.inv_freq = inv_freq.to(dtype=torch.float32)
                    rotary_emb.max_seq_len = max_seq_len
                else:
                    rotary_emb.inv_freq = rotary_emb.inv_freq.to(dtype=torch.float32)
    
    def forward(self, input_ids):
        # Ensure input_ids are int32
        input_ids = input_ids.to(dtype=torch.int32)
        
        with torch.inference_mode():
            # Ensure deterministic behavior
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Pre-process the input
            input_ids = input_ids.to(dtype=torch.int32)
            
            # No need to patch rotary embeddings here as they're initialized in __init__
            
            outputs = self.model(
                input_ids,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True
            )
            
            # Ensure output is float32
            logits = outputs.logits.to(dtype=torch.float32)
            
        return logits

# Chargement du modèle et tokenizer
print("🔄 Initialisation du modèle...")
# Set default tensor type to float32 for consistency
torch.set_default_dtype(torch.float32)

# Set deterministic behavior globally
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Force default dtypes for specific operations
torch.set_default_tensor_type(torch.FloatTensor)  # Force default tensor type
torch._C._jit_set_profiling_executor(False)  # Disable JIT executor profiling

# Disable gradients for conversion
torch.set_grad_enabled(False)

print("ℹ️ Configuration de la quantification dynamique...")
from torch.ao.quantization import quantize_dynamic
from torch.ao.quantization import QConfig, get_default_qconfig
from torch.ao.quantization import quantize_dynamic_jit
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,  # CoreML nécessite float32 pour la conversion
    device_map="cpu",    # Nécessaire pour CoreML Tools
    trust_remote_code=True,
    attn_implementation="eager",
    use_cache=False      # Le cache sera géré par CoreML sur device
)

# Force model to eval mode
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Clear CUDA cache if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Préparation du modèle
model.eval()
wrapped_model = Phi3Wrapper(model)

# Exemple d'entrée pour le traçage avec une taille fixe
print("🔍 Préparation des entrées d'exemple...")
example = tokenizer(
    "Patient: I have a headache. Doctor:",
    return_tensors="pt",
    padding="max_length",
    max_length=64,
    truncation=True
)

# S'assurer que les input_ids sont dans le bon type et taille
input_ids = example["input_ids"].to(dtype=torch.int32, device="cpu")
# Ensure shape matches exactly what we specified for CoreML
input_ids = input_ids[:, :64]  # Truncate to exact size
assert input_ids.shape == (1, 64), f"Expected shape (1, 64), got {input_ids.shape}"

# Traçage du modèle
print("📝 Traçage du modèle...")
traced_model = torch.jit.trace(wrapped_model, (input_ids,))

# Configuration de la conversion CoreML
print("⚙️ Configuration de la conversion CoreML (optimisé pour Neural Engine)...")
mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="input_ids",
            shape=(1, 64),  # Fixed shape for more stable conversion
            dtype=np.int32
        )
    ],
    outputs=[
        ct.TensorType(
            name="logits",
            dtype=np.float32
        )
    ],
    minimum_deployment_target=ct.target.iOS15,
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT32,  # Keep FLOAT32 for stability
    compute_units=ct.ComputeUnit.ALL
)

# Ajout des métadonnées
mlmodel.author = "Louison Beranger"
mlmodel.license = "MIT"
mlmodel.short_description = "Fine-tuned Phi-3 model for medical assistance"
mlmodel.version = "1.0"

# Sauvegarde du modèle
print(f"💾 Sauvegarde du modèle vers : {SAVE_PATH}")
mlmodel.save(SAVE_PATH)

print("✅ Conversion terminée avec succès!")
print("\nℹ️ Vous pouvez maintenant :")
print("1. Ouvrir le projet Xcode")
print("2. Glisser-déposer MedicalLLM.mlpackage dans votre projet")
print("3. Vérifier que le modèle est bien ajouté aux ressources de la cible")
