# Offline packaging of a qualified visual adapter

Research checked October 31, 2026. This is a source audit and proposed build
recipe, not a conversion, CPU inference, qualification, or Docker test receipt.
No model was loaded and no large model file was downloaded for this audit.

## Decision

Keep the visual adapter experimental and disabled by default. The current
purpose checkpoint is not qualified: its recorded 48-example comparison moved
from 21 to 23 correct weak labels, while background recall remained zero. Its
existence and successful CUDA execution do not establish useful layout judgment.

For the first faithful offline CPU check, retain Transformers/PEFT and the exact
saved classifier inputs and scoring method. A separate llama.cpp GGUF plus LoRA
route is technically plausible at the project's existing runtime pin, and should
reduce the deployment dependency and weight footprint. It needs a closed-set
logit bridge and a new evaluation before promotion. A normal JSON-generation
provider is a different inference method from this trained classifier.

The stage-one engine must continue preserving element IDs/count, copy, font
families, original asset pixels and strict canvas bounds. Keep the intentional
missing-asset placeholder. The final brief's offline engine is the current
submission priority; Adobe host verification is deferred and the FAQ discrepancy
remains unresolved. See [rules](../RULES.md), [FAQ](../FAQ.md), and
[submission evidence](SUBMISSION_CHECKLIST.md).

## What the actual saved adapter contains

The audited artifact is
`output/visual-finetune/runs/purpose-r4-32-v2/checkpoint-8/adapter_model.safetensors`:

- SHA-256 `8eddb0524f8874dc83e03c843938c5c9b46ed07933dd035cac6edb4b66193ec1`.
- Header-only inspection found 112 FP32 tensors, 802,816 parameters, covering
  q/v projections in language layers 0–27: 56 A/B pairs.
- Adapter configuration: standard LoRA, rank 4, alpha 8, no DoRA, no RS-LoRA,
  no saved replacement modules, no trainable token embeddings, vision frozen.
- The config's `base_model_name_or_path` is an absolute Windows path and its
  `revision` is null. Supply the verified local base explicitly; do not treat
  that path as a portable base identity or rewrite the historical checkpoint.
- The original eight-step save lacks optimizer/RNG state. Preserve it as a
  weights-only artifact; it cannot establish exact training resumption.

The exact original HF base is `Qwen/Qwen3-VL-2B-Instruct`, revision
`89644892e4d85e24eaac8bacfd4f463576704203`; the verified local
`model.safetensors` is 4,255,140,312 bytes, SHA-256
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
An adapter alone does not include its base weights. PEFT documents the separate
adapter tensors and configuration required alongside the original model.
[PEFT checkpoint format](https://huggingface.co/docs/peft/en/developer_guides/checkpoint).

## Pinned llama.cpp compatibility

The project's downloaded Windows CUDA runtime is **b12017**, source commit
`5047748f114564fcf271e6a648f60e2d2eae6264`. Those Windows executables and CUDA DLLs
cannot be copied into the Linux CPU submission; build the Linux CPU runtime from
the same source revision. The recorded CUDA 12.4 archive hashes describe that
Windows smoke runtime only.
[Release b12017](https://github.com/ggml-org/llama.cpp/releases/tag/b12017).

At this exact commit:

1. `Qwen3VLForConditionalGeneration` has a registered text converter with
   `QWEN3VL` architecture, plus a separate vision converter.
   [Qwen3-VL conversion source](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/conversion/qwen3vl.py).
2. The LoRA converter reads PEFT A/B tensors, strips `base_model.model.`, applies
   the base model's tensor filtering, and writes adapter GGUF. It accepts an
   explicit local `--base`, requiring base config rather than the full base
   tensors for adapter conversion. The common base filter strips
   `language_model.`. Thus the actual saved name
   `base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight`
   reaches the standard `model.layers.0.self_attn.q_proj.weight` mapping. This is
   a source-level compatibility finding; no converter run has proved success.
   [LoRA converter](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/convert_lora_to_gguf.py),
   [base tensor filtering](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/conversion/base.py).
3. Qwen3-VL's attention calls `build_qkv`; its separate q/v path uses
   `build_lora_mm`. LoRA application is therefore present in the relevant runtime
   path rather than merely accepted as a CLI option.
   [Qwen3-VL graph](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/src/models/qwen3vl.cpp),
   [q/v graph construction](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/src/llama-graph.cpp).
4. CLI options include separate `--lora`, `--device none`, and
   `--mmproj-device none` / `--no-mmproj-offload`. The projector otherwise
   defaults to GPU offload. Use local `-m` and `--mmproj` paths, never `-hf` in
   offline execution.
   [Pinned CLI options](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/common/arg.cpp),
   [multimodal local-file use](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/docs/multimodal.md).

## Public base candidates and lineage limit

Qwen publishes language and vision GGUF components in its Apache-2.0 repository.
The metadata API returned revision `52d6c8ffea26cc873ac5ad116f8631268d7eb503`:

| File | Bytes | LFS SHA-256 |
| --- | ---: | --- |
| `Qwen3VL-2B-Instruct-F16.gguf` | 3,447,350,304 | `27b80f2e258f8372c9ee3a9b2293bd747cc08b18ed99709b57288cf457c42595` |
| `Qwen3VL-2B-Instruct-Q8_0.gguf` | 1,834,427,424 | `1e8db19207c8ce0733ddd78c2eff8a9e22c27c82f4443df94c25792ed8fe04f2` |
| `Qwen3VL-2B-Instruct-Q4_K_M.gguf` | 1,107,409,952 | `089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae` |
| `mmproj-Qwen3VL-2B-Instruct-F16.gguf` | 819,394,848 | `c3d5afbef5287953acd57b4043d2269456e5761a4eaccb3b71b062996970aea5` |
| `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` | 445,053,216 | `f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82` |

These are verified public **metadata**, not downloaded/hash-verified weight
files. Q8 language plus F16 vision totals 2,653,822,272 bytes before adapter,
runtime, KV cache and temporary buffers. The Qwen card permits mixed component
precisions. Start parity evaluation with F16 vision; Q8 vision is another
numerical change. The inspected card does not bind conversion to the exact HF
revision used for training. Prefer converting our verified base when preserving
exact lineage, or establish that lineage separately before using the public
conversion. Model name equality does not prove identical tensors.
[Pinned Qwen files](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/tree/52d6c8ffea26cc873ac5ad116f8631268d7eb503),
[Qwen metadata API](https://huggingface.co/api/models/Qwen/Qwen3-VL-2B-Instruct-GGUF?blobs=true),
[Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF/blob/52d6c8ffea26cc873ac5ad116f8631268d7eb503/README.md).

## Proposed conversion and CPU smoke commands

These commands have **not been executed**. Run conversion in a separate build
environment after a checkpoint qualifies, with the already verified base and a
new output directory. Preserve the original checkpoint and retain the complete
conversion log, dependency lock, source revision and output hashes.

```bash
CHAMELEON_CPP=/work/llama.cpp-5047748
CHAMELEON_BASE=/work/models/qwen3-vl-2b-8964489
CHAMELEON_ADAPTER=/work/checkpoints/qualified-checkpoint
CHAMELEON_OUT=/work/offline-candidate-v1

git clone https://github.com/ggml-org/llama.cpp.git "$CHAMELEON_CPP"
git -C "$CHAMELEON_CPP" checkout --detach 5047748f114564fcf271e6a648f60e2d2eae6264
mkdir -p "$CHAMELEON_OUT"

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python "$CHAMELEON_CPP/convert_hf_to_gguf.py" "$CHAMELEON_BASE" \
  --outtype f16 --outfile "$CHAMELEON_OUT/base-F16.gguf"
python "$CHAMELEON_CPP/convert_hf_to_gguf.py" "$CHAMELEON_BASE" --mmproj \
  --outtype f16 --outfile "$CHAMELEON_OUT/vision-F16.gguf"
python "$CHAMELEON_CPP/convert_lora_to_gguf.py" "$CHAMELEON_ADAPTER" \
  --base "$CHAMELEON_BASE" --outtype f32 \
  --outfile "$CHAMELEON_OUT/adapter-F32.gguf"

cmake -S "$CHAMELEON_CPP" -B "$CHAMELEON_CPP/build-cpu" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF \
  -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_CONNECT=OFF -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TESTS=OFF
cmake --build "$CHAMELEON_CPP/build-cpu" \
  --target llama-mtmd-cli llama-quantize --parallel 2

"$CHAMELEON_CPP/build-cpu/bin/llama-quantize" \
  "$CHAMELEON_OUT/base-F16.gguf" "$CHAMELEON_OUT/base-Q8_0.gguf" Q8_0
sha256sum "$CHAMELEON_OUT"/*.gguf

"$CHAMELEON_CPP/build-cpu/bin/llama-mtmd-cli" \
  -m "$CHAMELEON_OUT/base-Q8_0.gguf" \
  --mmproj "$CHAMELEON_OUT/mmproj-vision-F16.gguf" \
  --lora "$CHAMELEON_OUT/adapter-F32.gguf" \
  --device none --mmproj-device none --no-mmproj-offload -ngl 0 \
  --image /work/probe/source.png,/work/probe/context.png \
  -p 'Use the exact frozen purpose prompt here for a load-only smoke test.' \
  -c 4096 -t 4 -n 1 --temp 0 --no-warmup
```

The mmproj converter prepends `mmproj-` to its output filename. The prompt above
is explicitly a load smoke, not a scored classification. The source and crop
must be ordered exactly as in training. The pinned CLI accepts a comma-separated
media argument and adds media markers when none are in the prompt. Image budget,
chat template, markers and tokenization still require parity checks.
[Conversion flags](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/convert_hf_to_gguf.py),
[multimodal prompt processing](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/tools/mtmd/mtmd-cli.cpp).

The CPU build uses dynamic CPU variants so it does not assume the build host's
instruction set. Package all required backend/shared libraries and verify them
inside the actual Linux AMD64 image. `GGML_NATIVE=OFF` alone is not a portable-ISA
guarantee. No CPU latency or memory claim is established by these commands.
[CPU build options](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/ggml/CMakeLists.txt),
[CPU backend construction](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/ggml/src/ggml-cpu/CMakeLists.txt).

## Faithful inference and promotion gates

The current worker formats ordered images followed by the frozen prompt using
`AutoProcessor.apply_chat_template(..., add_generation_prompt=True)`. It reads
the next-token logits for the exact saved single-token class aliases and
normalizes within that closed set. It does not generate an explanation or
arbitrary JSON. A grammar-constrained word or a free-text top token must not
silently replace this scoring method.

For GGUF, a small `libmtmd`/llama wrapper can evaluate the same image/prompt
sequence, call `llama_get_logits_ith(context, -1)`, gather the verified alias token
IDs and compute the same closed-set softmax. This is a proposed bridge, not
implemented evidence. Record raw logits and class order; classifier confidence
is not a calibrated probability of design quality.
[Raw-logit C API](https://github.com/ggml-org/llama.cpp/blob/5047748f114564fcf271e6a648f60e2d2eae6264/include/llama.h).

Transformers/PEFT avoids that bridge and conversion initially, preserving the
verified safetensors and original processor. A CPU-only wheel set still needs a
separate inference entry point: the current training worker requires BF16 CUDA.
Freeze CPU dtype, tokenizer, processor/image budget, package versions and thread
limits. BF16 CPU support, peak RAM, startup time and practical throughput on the
reference machine remain unmeasured. Float32 would approximately double raw
parameter storage before other runtime allocations.

Before enabling either deployment route:

1. Compare base and adapted outputs under identical CPU preprocessing, including
   original/blank/replacement controls and source-family-disjoint labels. Run
   quantized evaluation separately; do not inherit a CUDA checkpoint's score.
2. Prove adapter application with actual active-tensor/load receipts and a
   bounded base/adapter score comparison. Successful file loading is insufficient.
3. Exercise analyzer → organizer → verifier on native source/result renders.
   Learned purpose remains advisory; it cannot alone authorize cover/contain,
   crop, copy changes, font substitution or a new group.
4. Bake all runtime libraries, base/projector/adapter weights, config, processor
   and prompts into the image. Run with `--network none`, no user HF cache mount,
   no `-hf` / hub fallback, and repeat the official JSON/native-render contracts.
5. Publish the relied-on adapter at a stable public URL with an immutable revision,
   SHA-256, original adapter config, exact base revision, training/input recipe,
   license/provenance and limitations. The current local checkpoint has no public
   URL verified by this audit; public base weights do not satisfy the adapter rule.
6. Keep model approval, total model-size accounting and exact scoring-machine
   limits listed as unverified organizer details. Report fallback behavior and
   all unpassed checks explicitly.

Preserve original safetensors and full training checkpoints even if GGUF is the
deployment format. Conversion is an export, not a replacement for recoverable
training state.
