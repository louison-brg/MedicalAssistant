import torch
import coremltools as ct
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "training/models/phi3-medprof-merged"
SAVE_PATH = "model_export/MedicalLLM.mlpackage"

print(f"📦 Chargement du modèle (CPU, dtype float32) depuis : {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map=None
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.eval()

# Entrée d’exemple
example = tokenizer("The patient has chest pain.", return_tensors="pt")
input_ids = example["input_ids"]

# Simplification : exporter uniquement le bloc de génération (linear head)
with torch.no_grad():
    traced = torch.jit.trace(model, (input_ids,))

print("💾 Conversion du modèle vers Core ML (stub minimal)...")

mlmodel = ct.convert(
    traced,
    convert_to="mlprogram",
    inputs=[
        ct.TensorType(name="input_ids", shape=(1, ct.RangeDim(1, 256)), dtype=int)
    ],
    minimum_deployment_target=ct.target.iOS15
)

mlmodel.save(SAVE_PATH)
print(f"✅ Export Core ML réussi : {SAVE_PATH}")
