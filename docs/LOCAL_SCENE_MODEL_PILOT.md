# Offline source-scene model pilot

The backend can now consume local image-role predictions before selecting a
resize. `resize_with_state(..., scene_proposal=response)` validates the exact
source, scene and asset hashes, then restricts new product/logo crops. It cannot
grant background cropping, rewrite text, replace images, or invent object boxes.
The harness exposes this through `--model-policy protect`; its default remains
`review`. The ordinary Studio remains deterministic until an explicitly reviewed
model policy is integrated by its owner.

## What runs locally

This laptop has 16GB RAM and no verified CUDA device. The complete cached OpenAI
CLIP ViT-B/32 checkpoint is used on CPU, with no model download or cloud start.
The SHA-256 is
`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
The tested isolated runtime is Python 3.10 with PyTorch 2.0.1 and the installed
OpenAI `clip` package; the application Python remains independent of Torch.
Package versions, CLIP source/tokenizer hashes and transform parameters are
bound in the provider receipt and checkpoint processor identity.

Each prediction sees two complete, aspect-preserving, white-letterboxed views:
the actual native source render and the exact selected source asset. Each view
becomes a normalized 512-dimensional CLIP feature. A learned five-class linear
head consumes their concatenation. CLIP stays frozen; this is task-head training,
not Qwen LoRA, a layout generator, or learned relationship/region detection.
The feature schema is `clip-two-view-letterbox-l2-source-asset-v2`.

The [upstream implementation](https://github.com/openai/CLIP) documents extracting
image features and training a linear classifier on frozen representations. Its
default center crop is unsuitable for complete poster evidence, so this provider
letterboxes both views first. Four-edge portrait/landscape tests verify that the
preprocessing keeps every edge and the image proportions.

## Reproduce the data and training

[Tracked labels](../fixtures/scene_model/pilot-labels.json) record 34 image roles
across10 development sources. All annotations are **agent visual reviews**;
there was no human labeling pass. Sources, selected asset bytes and shared
design families cannot cross splits. There are 11 training and 23 validation
records. Grand Opening has no image elements, so contributes no classifier
record. Advanced's intentionally missing asset is excluded from training.
The ten personal-test reserves are excluded by source ID and source digest.

Rebuild the source-only evidence and checked manifest from tracked originals:

```powershell
.venv\Scripts\python.exe -m experiments.scene_model.prepare_pilot `
  --out output/scene-model-pilot-data
```

Choose an existing offline model Python and the complete cached checkpoint.
The trainer does not install packages or fetch weights. A fresh output directory
is required; every checkpoint is immutable and contains parameters, optimizer
progress, loss history, identities and file hashes.

```powershell
$scenePython = 'C:/path/to/model-runtime/python.exe'
$sceneClip = Join-Path $env:USERPROFILE '.cache/clip/ViT-B-32.pt'
$sceneSha = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
& $scenePython -m experiments.scene_model.train_clip `
  --manifest output/scene-model-pilot-data/manifest.json `
  --clip-checkpoint $sceneClip --clip-sha256 $sceneSha `
  --out output/my-scene-40 --max-steps 40 --save-every 20 `
  --max-seconds 240 --execute
```

Resume in another fresh directory with `--max-steps 80` and
`--resume output/my-scene-40/published/checkpoint-000040`.
This optimizer is class-balanced full-batch gradient descent without momentum;
saved weights and progress are its complete optimizer state. The worker has an
independent wall-clock supervisor. Source pixels, role labels, feature bytes,
base checkpoint, processor and optimizer runtime must match on resume.

## Connect the provider to both resize modes

Set the explicit local checkpoint path in
[the provider configuration](../configs/scene_model/local-clip.example.json).
For a trained head, add `head_path`, its exact `head_sha256`, and the checkpoint's
`processor_digest`. They must match; no silent processor fallback is permitted.
An actual provider command can be passed as a JSON argument array:

```powershell
$sceneCommand = @($scenePython, '-m', 'experiments.scene_model.provider',
  '--config', 'output/scene-model/local-head.json') | ConvertTo-Json -Compress
.venv\Scripts\python.exe -m tools.scene_harness `
  --prepared fixtures/open_posters/prepared/doory_rooftop-event-story `
  --target 540x960 --target 1600x900 `
  --model-command $sceneCommand --model-policy protect `
  --out output/my-connected-scene
```

The model runs once on the source and its predictions are reused for both the
smaller same-aspect resolution and the landscape reflow. Model identity and
runtime provenance are recorded with the provider's optional `--receipt` argument.
The engine state records the exact proposal digest, protected IDs and candidate
veto count. Low-confidence/unknown roles abstain. Confidence is uncalibrated;
no region detector ran, and background permissions granted by the model are zero.

## Admission and remaining work

The actual CPU pilot completed 40 steps, resumed to 80, and was compared with a
fresh uninterrupted 80-step control. The final `head.json` files were byte-identical:
`cf0c9ce35d166a4e85b13664d361c1f13f911195afbb00c28b251151886ab988`.
Reload and resume used the stable processor identity
`d506737341b6c2c8694341fac6d9c96f7900f3ee2bad53450daa2077301862b7`;
no identity check was bypassed. Loss decreased from 1.6097 to 0.8354.
The resumed run took 21.953 seconds including model loading and feature extraction;
warm two-view encoding plus zero-shot comparison averaged 0.327 seconds per record.
These are CPU worker timings, not Studio response-time or memory guarantees.

| Condition | Training correct | Validation correct | Validation macro recall |
| --- | --- | --- | --- |
| Zero-shot CLIP | 3/11 | 9/23 | 0.333 |
| Trained two-view role head | 9/11 | 9/23 | 0.333 |
| Source-view features removed | 11/11 | 9/23 | 0.333 |
| Asset-view features removed | 8/11 | 2/23 | 0.074 |

The trained head predicted 22 backgrounds and one product on validation, with
zero foreground/logo recall there. This failed the semantic improvement gate.
Removing the poster-view features improves training accuracy without improving
validation: global source context is not reliable object-conditioned reasoning
in this tiny linear-head experiment. The ablations expose that limitation; they
are not used to select a supposedly better model on the same validation set.
The head remains an explicit research option and is not enabled in Studio.

Local reproducibility evidence is in
`output/scene-model-clip-stable-80-resume/training-report.json` and
`resume-control-proof.json`. The historical center-cropped prototype and unstable
`repr`-identity run are superseded, retained separately, and never relabeled as
stable checkpoints. The executable training details are in
[CPU training notes](../configs/scene_model/CPU_TRAINING.md).

No experimental head is promoted by a passing structural test. The pilot lacks
an independent positive product family and any unknown-class supervision.
It cannot establish unseen-product recall, calibrated uncertainty, region
protection, or improved design quality. The dense VLM/LoRA path remains a
separate future experiment in the [training plan](SCENE_MODEL_TRAINING_PLAN.md).

Before enabling a model by default, collect independent editable product,
scene-background, inset-photo and ambiguous-role families; obtain human role
and crop reviews; then repeat family-disjoint evaluation, source/asset ablations,
checkpoint reload/resume and both actual resize modes. Keep model confidence
subordinate to the asset contract and source relationships throughout.
