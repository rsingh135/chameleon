# Offline visual planner experiment

This provider executes a local generative vision-language model. It is an
experimental adapter, **not a qualified automatic design judge**. A successful
process and a valid plan identifier do not establish visual reasoning or quality.
No model here has been fine-tuned on Chameleon layouts.

## Project-local CPU setup

Run from the repository root using the existing Python environment with Pillow:

```powershell
.\.venv\Scripts\python.exe -m experiments.visual_planner.download_native
```

This explicit setup downloads three public files totaling 660,850,322 bytes into
ignored `output/visual-planner/native`. It verifies hard-coded SHA-256 and byte
sizes before extracting the runtime. It does not install packages, modify Ollama,
start a service, use credentials, or start a cloud machine.

The model is the Apache-2.0
[official SmolVLM-500M GGUF](https://huggingface.co/ggml-org/SmolVLM-500M-Instruct-GGUF/tree/72e986006ef53e37cdd3f6d4241c90b0f01df376)
at revision `72e986006ef53e37cdd3f6d4241c90b0f01df376`, with Q8_0 language
weights and its F16 vision projector. The MIT
[llama.cpp CPU runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b12017)
is pinned to `b12017`. Exact file hashes are in `native.py` and
`download_native.py`; inference rechecks the weights, ZIP, executable and DLLs.
The extracted runtime is project-local; no executable is downloaded at inference.

An explicitly selected second profile uses the Apache-2.0
[bartowski Qwen3.5-0.8B conversion](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF/tree/f36b1ea49a332ede8fe5f389bbf5b3575ef71f48),
based on [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B).
This is a third-party quantization, not an official Qwen or ggml-org conversion.
The pinned Q4_K_M text weights and matching F16 projector add 784,602,944 bytes;
they reuse the verified CPU runtime and remain ignored local artifacts.
The exact quantized revision is `f36b1ea49a332ede8fe5f389bbf5b3575ef71f48`.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Qwen_Qwen3.5-0.8B-Q4_K_M.gguf` | 579,615,840 | `fb044e93939a70469c905781334f5de1e6c8b608ced6cbc8c9249bd4127d9526` |
| `mmproj-Qwen_Qwen3.5-0.8B-f16.gguf` | 204,987,104 | `1dc1351c82e41b48edb55fd6ddfa7ca60fb5a16b3d5abf3ce7054880dd022847` |

Model binaries are not included in the source commit. Base-model attribution is
Qwen/Qwen3.5-0.8B; conversion attribution is bartowski. The distribution metadata
records Apache-2.0 for these weights and MIT for llama.cpp; runtime extraction
retains the archive's license files.

```powershell
.\.venv\Scripts\python.exe -m experiments.visual_planner.download_native --profile qwen35_08b
```

Select `configs/visual_planner/qwen_cpu.example.json` to use that profile. Its
single-turn, no-tools template follows the original Qwen template's
`enable_thinking=False` branch, so a short caption or answer can use the token
budget. The example also requests evidence-first JSON field order: short visible
observations, short design reason, then choice. Schema serialization preserves
this order, and receipts bind the exact schema hash. Qwen uses a fixed physical
microbatch of 64 tokens to reduce its CPU scratch allocation. Evidence-first
prose fields are bounded to 32 characters and the token budget remains 96.
Neither profile is enabled
as an application default by this module; Smol remains the adapter's default
profile. These experiments do not establish challenge-approved model eligibility.

Use the operator config in `configs/visual_planner/native_cpu.example.json`:

```powershell
.\.venv\Scripts\python.exe -m experiments.visual_planner.provider `
  --config configs/visual_planner/native_cpu.example.json `
  --request output/request.json --out output/response.json
```

The native backend uses four CPU threads, zero GPU layers, no projector offload,
at most 96 generated tokens, a 180-second worker deadline, and a 2662 MiB worker
RSS limit. Sustained available system RAM below 256 MiB aborts the owned worker.
The supervisor never terminates other applications. `python_executable` is only
used by the optional Transformers backend, but is present for config compatibility.

## Request and result contract

An optional source-only `mode: "analyze"` now returns bounded, element-grounded
observations and abstention instead of a menu choice. An independent
`analyzer_command` can run that VLM once before a separately configured
organizer, which receives the hash-bound sidecar as advisory data. Both native
and Transformers backends support this mode; invalid analysis never falls back
to letter scoring. See the [two-model setup and evidence contract](../../docs/IMAGE_REASONING_PIPELINE.md).
The existing quality limitations below still apply; the new handoff has not
qualified either model's design judgment.

Python callers use `predict(request, config)` from `provider.py`. CLI callers
can supply the same request through stdin instead of `--request`.

A request contains an unchanged engine `plan_menu`, `source_render` with local
`path` and SHA-256, and optionally the matching `source_digest` and
`target_canvas`. `source_summary` supplies bounded design metadata as data. The
model chooses one letter from at most seven existing menu options. Code maps the
letter to a known `plan_id` and copies the menu digest; model output cannot
provide geometry, alter text, authorize a crop, or create unknown objects.
The engine must independently regenerate and validate its menu before applying
a proposal. A digest check in this adapter is an integrity check, not that
engine validation.

For a critic request, use `mode: "critique"`, two clean menu plans, and a top-level
`plans` copy containing the same metadata plus each plan's `render` reference.
This presents three images: original source, choice A, choice B. Render references
do not change the clean engine menu digest. Source and candidate images remain
whole, proportional views in a 512-pixel white letterbox; the adapter does not
crop or change source assets. Local paths must remain inside the repository.

The result has two separate fields:

- `decision`: closed versioned menu digest, known plan ID, optional short rationale.
- `receipt`: actual execution, raw generation, source and view hashes, target,
  weights/runtime pins, exact command/prompt identity, media-encoding log evidence,
  timing, memory supervision, and provider implementation hashes.

The native CLI receives one comma-separated `--image` argument and explicit
ordered `<__media__>` markers. Its deprecated repeated-image warning is an error.
The receipt requires completed image-encoding log entries; submitted filenames
alone are insufficient evidence. Raw stdout must be a complete JSON document.
Prompt echoes, multiple documents, unknown choices, geometry, duplicate JSON
keys and nonfinite values are rejected. There is no deterministic replacement
for an invalid native model answer.

## What the measured pilots establish

The 2026-10-30 local pilots are diagnostics; they are not acceptance runs against
a globally frozen engine, and no applied-layout improvement is established.
Their ignored artifacts are under `output/visual-planner`.

| Experiment | Observed result | Meaning |
| --- | --- | --- |
| Original Transformers CPU planner, 180 seconds | Imports consumed about 115 seconds; model loaded and inputs prepared, scoring exceeded the remaining budget; peak RSS about 760 MiB | Runtime deadline failure, not an out-of-memory result |
| Native source-only menu controls | About 6–7 seconds, 803 MiB RSS; normal, blank, target-flipped and reversed-menu calls all generated A | Current proposal choices are position-biased; these controls do not qualify useful planning |
| Corrected native three-view Rooftop critic | Three distinct views and six encoded media chunks; 16.2 seconds, 814 MiB RSS; chose the first, visually worse alternative and copied metadata | Real vision execution, failed useful-critique demonstration |
| Swapped three-view critic | Exceeded 180-second deadline; owned worker stopped | No usable decision and no order-consistency claim |
| Simple native grounding probes | Core caption read “SOLSTICE.”; Rooftop caption said “Picture.”; four classification/order probes chose A | Some visible text recognition, inadequate grounding qualification |
| Original-Transformers reference VQA | Aborted during import at 83 seconds after sustained available system RAM fell to 23 MiB; owned worker peak RSS 426 MiB | No caption or native-versus-original comparison; owned process stopped safely |
| Qwen Q4 CPU grounding | About 6–8 seconds and 1.06–1.15 GiB RSS; captions identified a Solstice bottle and people; free category answers matched both posters | Basic visible category recognition, not layout-quality validation |
| Qwen choice-format controls | Explicit JSON B/C instructions and free letter choice worked; semantic JSON category was correct in three of four order controls | Decoder is not hard-wired to A; the choice formulation still affects reliability |
| Qwen evidence-first counterbalanced grounding | Complete original-order decisions: Core B=product, Rooftop A=people. Complete swapped-order decisions: Core A=product, Rooftop C=people; final two tight calls took 6.14/7.14 seconds and 1155 MiB RSS | Basic pixel-grounded category decisions survive reordered choices; layout planning and quality remain pending |
| Fixed Qwen physical microbatch 64 | Runtime logs confirm CPU compute scratch fell from 501.03 to 62.63 MiB; final-code original-order category controls passed in 9.59/9.05 seconds | Smaller owned scratch allocation; total RSS and available system memory remain separate measurements |
| Final-code exact Core production request and critic controls | Core request and both three-view critic orders aborted on sustained low available system RAM; no model decisions were returned | No final-code Core planning or visual-comparison qualification; the guard stopped owned workers |

The initial multi-image run used repeated `--image` flags; its distinct-image
claim was withdrawn and the corrected adapter was rerun. Refer to
`grounding/rooftop-critic.response.json` for the corrected distinct-image run,
`grounding/summary.json` for basic probes and the timeout, and `controls/summary.json`
for the earlier source/menu controls. Each worker retains stdout, stderr and
supervision data. A separate original-Transformers reference VQA is recorded
under `hf-reference` when run; inspect its results and phase journal before
attributing a mismatch to weights or native execution.

Qwen grounding controls are in `qwen-grounding`, `qwen-small-controls`, and
`qwen-semantic-controls`. Evidence-first controls are recorded in
`qwen-evidence-controls` and `qwen-evidence-tight-controls`. The first
evidence-first probe used 64-character prose fields and a 64-token budget; two
Core generations stopped before a complete choice. The final two calls used
32-character fields, completed within 64 tokens, and retained the raw clipped
prose. The production example now uses 32-character fields and 96 tokens;
incomplete JSON remains an error. Read the raw outputs and exact prompts/schemas;
category recognition and copying a layout label are different capabilities.

The later `ub64-validation` run bound the final provider hashes before and after
execution. Its exact saved Core production request aborted at 5.80 seconds with
883 MiB worker RSS after available system RAM fell to 123 MiB. Two original-order
grounding calls completed correctly; both swapped grounding calls and both
diagnostic critic orders subsequently aborted under memory pressure. Earlier
counterbalanced successes used the prior larger-microbatch implementation.
Both histories are retained; neither is silently substituted for the other.

Further qualification needs image-versus-blank and target controls, stable choices
under reordered options, two-candidate visual comparisons in both orders, and
independent inspection of actual compiled renders. Model execution, valid
proposal, admitted candidate, selection, changed layout and improved quality
remain separate facts. Do not enable a default automatic judge on the evidence
above.

## Alternative original-Transformers backend

`download_model.py` explicitly fetches pinned official SmolVLM 500M or 256M
Safetensors plus a fixed processor/tokenizer whitelist. The already existing
Transformers environment must support `Idefics3`; setup does not install one.
Use an explicit local `python_executable` and `model_dir`, and
`backend: "transformers"`. Inference uses `local_files_only=True`,
`trust_remote_code=False`, offline HF settings, CPU BF16 or FP32, and the same
owned-child supervision. `choice_logits` records actual model conditional
letter likelihoods; it does not substitute algorithmic scores for the VLM.
