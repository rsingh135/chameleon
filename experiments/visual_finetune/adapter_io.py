"""Preserve the saved FP32 LoRA values when adding another resident adapter."""

from pathlib import Path

from .data import sha


def precast_then_load(model, checkpoint, adapter_name, config):
    model.add_adapter(adapter_name, config)
    # PEFT 0.18.1 normally upcasts after copying, which first rounds an FP32 file
    # through the BF16 base dtype. Upcast the destination before that copy.
    model.base_model._cast_adapter_dtype(adapter_name=adapter_name, autocast_adapter_dtype=True)
    model.load_adapter(
        str(checkpoint),
        adapter_name=adapter_name,
        is_trainable=False,
        autocast_adapter_dtype=True,
        local_files_only=True,
    )
    model.set_adapter(adapter_name)


def audit_saved_adapter(model, checkpoint, adapter_name):
    import torch
    from peft.utils.save_and_load import get_peft_model_state_dict
    from safetensors.torch import load_file

    checkpoint = Path(checkpoint)
    saved = load_file(str(checkpoint / "adapter_model.safetensors"), device="cpu")
    live = get_peft_model_state_dict(model, adapter_name=adapter_name, save_embedding_layers=False)
    if set(saved) != set(live):
        raise ValueError("Live and saved LoRA parameter keys differ")
    parameters = {}
    for name, value in live.items():
        actual, expected = value.detach().cpu(), saved[name]
        if actual.shape != expected.shape:
            raise ValueError("Live and saved LoRA shapes differ")
        parameters[name] = {
            "live_dtype": str(actual.dtype),
            "saved_dtype": str(expected.dtype),
            "exact": actual.dtype == expected.dtype == torch.float32
            and torch.equal(actual, expected),
            "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        }
    return {
        "passed": all(row["exact"] for row in parameters.values()),
        "parameter_tensor_count": len(parameters),
        "parameters": parameters,
        "adapter_sha256": sha(checkpoint / "adapter_model.safetensors"),
        "adapter_config_sha256": sha(checkpoint / "adapter_config.json"),
    }
