import json
from pathlib import Path

import pytest
from safetensors import safe_open


MODEL_DIR = Path("ios_app/MedicalAssistant/MedicalAssistant/Phi3_Medical_Latest")


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Local deployment model is not checked into Git")
def test_deployment_model_matches_swift_runner_contract():
    required_files = {
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert required_files.issubset({path.name for path in MODEL_DIR.iterdir()})

    config = json.loads((MODEL_DIR / "config.json").read_text())
    quantization = config.get("quantization") or config.get("quantization_config")
    assert config["model_type"] == "phi3"
    assert quantization == {"group_size": 64, "bits": 4, "mode": "affine"}
    assert config["hidden_size"] % config["num_attention_heads"] == 0

    with safe_open(MODEL_DIR / "model.safetensors", framework="np") as model:
        keys = set(model.keys())
    expected_tensors = {
        "model.embed_tokens.weight",
        "model.embed_tokens.scales",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.0.self_attn.qkv_proj.scales",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.gate_up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
        "lm_head.weight",
    }
    assert expected_tensors.issubset(keys)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="Local deployment model is not checked into Git")
def test_tokenizer_special_ids_match_generation_config():
    tokenizer = json.loads((MODEL_DIR / "tokenizer.json").read_text())
    generation = json.loads((MODEL_DIR / "generation_config.json").read_text())
    added = {item["content"]: item["id"] for item in tokenizer["added_tokens"]}
    assert added["<|assistant|>"] == 32001
    assert added["<|end|>"] == 32007
    assert added["<|user|>"] == 32010
    assert {32000, 32001, 32007}.issubset(set(generation["eos_token_id"]))
