from argparse import Namespace
from pathlib import Path

from training.models.model_checkpoint import is_mlx_model
from training.models.model_benchmark import answer_option, is_correct, reference_answer
from training.models.model_export import detect_backend, mlx_commands
from training.models.model_training import build_config, parse_args


def training_args(**overrides):
    values = {
        "model_path": "base",
        "data_path": "data",
        "adapter_path": "adapters",
        "seed": 42,
        "num_layers": 16,
        "batch_size": 1,
        "grad_accumulation_steps": 4,
        "iters": 1_000,
        "val_batches": 25,
        "learning_rate": 1e-4,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "steps_per_report": 10,
        "steps_per_eval": 200,
        "save_every": 250,
        "max_seq_length": 2_048,
        "resume_adapter_file": None,
        "lora_rank": 16,
        "lora_scale": 32.0,
        "lora_dropout": 0.05,
        "lora_keys": "",
    }
    values.update(overrides)
    return Namespace(**values)


def test_mlx_config_really_masks_prompt_and_sets_lora_parameters():
    config = build_config(training_args())
    assert config["mask_prompt"] is True
    assert config["grad_accumulation_steps"] == 4
    assert config["optimizer"] == "adamw"
    assert config["lora_parameters"] == {"rank": 16, "scale": 32.0, "dropout": 0.05}


def test_training_defaults_are_conservative_mlx_values():
    args = parse_args([])
    assert args.learning_rate == 1e-5
    assert args.lora_rank == 8
    assert args.lora_scale == 20.0
    assert args.lora_dropout == 0.0


def test_benchmark_accepts_reference_text_or_exact_option():
    target = "In this clinical scenario, Captopril is the best answer."
    prompt = "Options:\nA. Niacin\nB. Captopril\nC. Furosemide"
    reference = reference_answer(target)
    option = answer_option(prompt, reference)

    assert reference == "Captopril"
    assert option == "B"
    assert is_correct("The best treatment here is captopril.", reference, option)
    assert is_correct("Option B", reference, option)
    assert not is_correct("Niacin", reference, option)


def test_mlx_adapter_detection(tmp_path):
    (tmp_path / "adapters.safetensors").touch()
    assert detect_backend(tmp_path) == "mlx"


def test_quantized_mlx_checkpoint_is_routed_to_mlx(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"quantization": {"group_size": 64, "bits": 4}}',
        encoding="utf-8",
    )
    assert is_mlx_model(tmp_path)


def test_mlx_export_fuses_before_quantization():
    args = Namespace(
        python="python",
        no_quantize=False,
        q_bits=4,
        q_group_size=64,
        q_mode="affine",
    )
    commands = mlx_commands(
        args,
        Path("base"),
        Path("adapter"),
        Path("fused"),
        Path("quantized"),
    )
    assert commands[0][3] == "fuse"
    assert commands[1][3] == "convert"
    assert "--quantize" in commands[1]
