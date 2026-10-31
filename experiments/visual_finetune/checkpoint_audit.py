"""Inspect the saved FP32 LoRA tensors without importing a GPU framework."""

import array
import json
import math
import struct
import sys
from pathlib import Path

from .data import read_json, sha, write_json


def audit(path):
    path = Path(path)
    model = path / "adapter_model.safetensors"
    receipt, config = (
        read_json(path / "training_receipt.json"),
        read_json(path / "adapter_config.json"),
    )
    if sha(model) != receipt["adapter_sha256"] or config["init_lora_weights"] is not True:
        raise ValueError("Adapter identity or zero-B initialization contract differs")
    tensor_count, parameter_count, b_nonzero, b_norm_squared = 0, 0, 0, 0.0
    with model.open("rb") as stream:
        length = struct.unpack("<Q", stream.read(8))[0]
        if not 0 < length < 2_000_000:
            raise ValueError("Unexpected safetensors header size")
        header = json.loads(stream.read(length))
        base = 8 + length
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if entry["dtype"] != "F32" or "lora_" not in name:
                raise ValueError("This audit expects only FP32 language LoRA tensors")
            begin, end = entry["data_offsets"]
            if not 0 <= begin <= end <= model.stat().st_size - base:
                raise ValueError("Tensor span exceeds saved bytes")
            stream.seek(base + begin)
            values = array.array("f", stream.read(end - begin))
            if sys.byteorder != "little":
                values.byteswap()
            if math.prod(entry["shape"]) != len(values) or not all(
                math.isfinite(v) for v in values
            ):
                raise ValueError("Saved tensor shape or finite-value check failed")
            tensor_count += 1
            parameter_count += len(values)
            if "lora_B" in name:
                b_nonzero += sum(value != 0 for value in values)
                b_norm_squared += sum(value * value for value in values)
    if not b_nonzero or not b_norm_squared > 0:
        raise ValueError("No learned change from the documented zero-B initialization")
    return {
        "adapter_sha256": sha(model),
        "checkpoint_step": receipt["step"],
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "all_tensors_finite": True,
        "nonzero_B_parameters": b_nonzero,
        "B_delta_from_zero_l2": math.sqrt(b_norm_squared),
        "actual_optimizer_steps_recorded": len(receipt["losses"]),
        "dataset_sha256": receipt["dataset_sha256"],
        "quality_approved": False,
    }


if __name__ == "__main__":
    result = audit(sys.argv[1])
    if len(sys.argv) > 2:
        write_json(sys.argv[2], result)
    print(json.dumps(result, indent=2))
