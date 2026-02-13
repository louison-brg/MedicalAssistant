
import coremltools as ct
from coremltools.optimize.coreml import (
    linear_quantize_weights,
    OpLinearQuantizerConfig,
    OptimizationConfig,
)
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", help="Path to input .mlpackage", required=True)
    parser.add_argument("--output-path", help="Path to output .mlpackage", required=True)
    args = parser.parse_args()

    print(f"📦 Loading model from {args.model_path}...")
    mlmodel = ct.models.MLModel(args.model_path)
    
    print("🔍 inspecting model ops...")
    # ops = mlmodel.get_spec().description ... 
    # coremltools doesn't expose ops easily via spec for mlprogram, but we can try quantizing.

    print("🎛️ Attempting 4-bit linear quantization...")
    # Try block_size=32 first (should divide 3072 and 8192)
    # Try per_channel if per_block fails?
    
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32 
    )
    config = OptimizationConfig(global_config=op_config)

    quantized_model = linear_quantize_weights(mlmodel, config=config)
    
    print(f"💾 Saving to {args.output_path}...")
    quantized_model.save(args.output_path)
    
    # Check size
    size = os.path.getsize(args.output_path) # This is a dir for mlpackage
    # calculating size of dir
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(args.output_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    
    print(f"✅ Done. Size: {total_size / (1024**3):.2f} GB")

if __name__ == "__main__":
    main()
