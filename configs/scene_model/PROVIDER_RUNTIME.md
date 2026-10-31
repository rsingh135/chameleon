# Offline scene-provider runtime

`local-clip.example.json` is a portable configuration with no account paths or
embedded weights. `~` expands to the local user's profile; other relative paths
resolve from the configuration file. Use the already cached checkpoint at
`~/.cache/clip/ViT-B-32.pt`, or copy the configuration outside Git and point it
at an existing local file. The provider
checks the original OpenAI CLIP ViT-B/32 SHA-256 before loading any weights:

```text
40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af
```

No checkpoint or model package is downloaded automatically. Importing the
provider uses only the standard library; CLIP, PyTorch and Pillow are loaded
only when explicitly requested. The normal application environment does not
need these model dependencies. Use a separate existing local inference runtime.

## Verified environment

The actual CPU runs used Python 3.10.19, PyTorch 2.0.1, torchvision 0.15.2,
OpenAI CLIP (installed distribution reports `clip` 1.0), and Pillow 12.1.0.
CUDA was unavailable and no GPU/cloud instance was started. A package name or
version alone does not identify OpenAI's CLIP source: the provider additionally
records and pins its installed model, tokenizer and BPE files. The verified
source-file hashes are:

| File in the OpenAI CLIP package | SHA-256 |
| --- | --- |
| `clip.py` | `9964be5024a30b0268285855e83f2aa14b362488d9797a7ce2aaa09e8f72b966` |
| `model.py` | `d4225196bddfa2950ce0d926b7e3b1d80a37c258b49a7452fe1f5499b974cf67` |
| `simple_tokenizer.py` | `1e40526d4deb22e7ab5fce9a675d139b4cccf8567a88b0b0e400fa33480c96e2` |
| `bpe_simple_vocab_16e6.txt.gz` | `924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a` |

To recreate the environment offline, use trusted local wheels/source archives
for this stack and verify those files. This task did not install packages or
change an existing environment. Do not replace the OpenAI implementation with
an unrelated package merely because its distribution is named `clip`.

## Complete source and asset views

The feature schema is `clip-two-view-letterbox-l2-source-asset-v2`. Both original
source and exact asset are composited on white, resized proportionally into a
224×224 white letterbox, then passed through the pinned CLIP transform. This
retains every image edge and its aspect ratio. Each 512-dimensional image vector
is L2-normalized; the feature order is source first, asset second, for 1024
features. A four-edge portrait/landscape regression checks this framing policy.

The processor identity contains typed transform parameters, fixed function
names, exact view-code and CLIP file hashes, and package versions. It contains
no process-specific object representations or memory addresses. Two independent
actual model loads produced the same processor digest:

```text
d506737341b6c2c8694341fac6d9c96f7900f3ee2bad53450daa2077301862b7
```

The initial center-cropped prototype and later `repr`-based processor identities
are superseded diagnostic artifacts. Their head checkpoints must not be
relabeled or silently loaded under this stable processor identity. A changed
processor requires fresh feature/training evidence.

## Running it

From the repository root, use the Python executable belonging to the inference
runtime:

```text
python -m experiments.scene_model.provider \
  --config configs/scene_model/local-clip.example.json \
  --request output/scene-harness-rooftop/model-request.json \
  --receipt output/scene-model/provider-receipt.json
```

Without `--request`, the provider reads the harness JSON on stdin. Stdout is
only the harness-compatible response. The optional receipt records exact
backbone/processor/head identities, source bindings and runtime limitations.
For the harness, pass this command as its JSON argument array; use
`--model-policy protect` only when intentionally testing the backend's
conservative crop-veto policy. The default review mode changes no layout.

An optional role head must be enabled through an explicit `head_path` and actual
`head_sha256` pair in the configuration. The JSON head must match this feature
schema, backbone hash, processor digest, role ordering, and finite 5×1024 weights
plus 5 biases. There is no automatic latest-checkpoint discovery or promotion.

## Limits

The zero-shot provider combines source and asset semantic similarity; an optional
CPU role head is a learned classifier on frozen CLIP features. Neither is a
general reasoning VLM or a detector. Confidence is an uncalibrated class score.
Protected-region lists are empty because no regions were measured. Model output
does not grant background crop permission, replace assets, edit text, or bypass
the engine's contract. Actual validation results and the pilot's held-out
limitations are recorded in `docs/LOCAL_SCENE_MODEL_PILOT.md`.
