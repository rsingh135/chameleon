# Local CPU scene-role training

This trains a small role classifier on a frozen, locally cached CLIP ViT-B/32.
It does not train Qwen, update CLIP's base weights, predict face boxes or prove
design quality. Source and selected-asset views are both letterboxed in full;
each contributes a normalized 512-value feature vector to the 1024-input head.

The feature schema is `clip-two-view-letterbox-l2-source-asset-v2`. The backbone
SHA-256 is `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`.
The exercised processor digest is
`d506737341b6c2c8694341fac6d9c96f7900f3ee2bad53450daa2077301862b7`.
Processor identity includes typed transforms, code/tokenizer hashes and package
versions. A changed processor, backbone, feature vector or training recipe
invalidates resume; metadata must never be relabeled to bypass that check.

## Prepare and inspect supervision

Use the application's renderer environment to rebuild the separately reviewed
pilot records, then inspect the manifest and original renders:

```powershell
.venv/Scripts/python.exe -m experiments.scene_model.prepare_pilot --out output/scene-model-pilot-data
```

This produces `output/scene-model-pilot-data/manifest.json`. The labels are
explicitly agent visual reviews, not an independent human benchmark. The dataset
loader checks source/render/asset hashes, known image IDs, review/rights flags,
family/source/selected-asset split separation and reserve exclusions. Opaque
IDs and hashes remain audit metadata; they are absent from learned targets.
All ten personal reserve sources remain excluded.

The exercised pilot has 34 image records across six source families: 11 training
and 23 validation records. Only the training split contains a distinct product
family, so no held-out product-recall claim is available. Grand Opening has no
image records. The Advanced source's intentionally missing asset is preserved
in source inspection and excluded from this training manifest.

## Execute bounded local optimization

The tested existing runtime is
`C:/Users/edmun/anaconda3/envs/conceptgraph/python.exe`, with torch 2.0.1,
torchvision 0.15.2, CLIP 1.0, Pillow 12.1.0 and NumPy 1.26.4. Nothing is
downloaded or installed by the commands. Substitute an independently verified
compatible local runtime and existing checkpoint on another machine.

```powershell
$sceneRuntime = 'C:/path/to/model-runtime/python.exe'
$sceneCheckpoint = Join-Path $env:USERPROFILE '.cache/clip/ViT-B-32.pt'
$sceneSha = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
& $sceneRuntime -m experiments.scene_model.train_clip `
  --manifest output/scene-model-pilot-data/manifest.json `
  --clip-checkpoint $sceneCheckpoint --clip-sha256 $sceneSha `
  --out output/scene-model-local-40 --max-steps 40 --save-every 20 --max-seconds 240 --execute
```

Without `--execute`, the command validates the data/recipe and writes a plan;
it does not load CLIP or optimize anything. Actual execution uses class-balanced
full-batch gradient descent with no momentum. The supervised worker has an
external wall-clock deadline as well as the internal step/time bounds. A new,
empty output directory is required; previous generations remain untouched.

Resume into another empty run directory with the same data and stable recipe:

```powershell
& $sceneRuntime -m experiments.scene_model.train_clip `
  --manifest output/scene-model-pilot-data/manifest.json `
  --clip-checkpoint $sceneCheckpoint --clip-sha256 $sceneSha `
  --resume output/scene-model-local-40/published/checkpoint-000040 `
  --out output/scene-model-local-80-resume --max-steps 80 --save-every 20 --max-seconds 240 --execute
```

Each immutable generation contains `head.json`, `training-state.json` and their
checksum manifest. The head has five ordered roles: background, product, logo,
foreground and unknown. Loading it in the provider requires its explicit file
SHA-256, matching backbone, feature schema and processor digest. See
[the provider runtime guide](PROVIDER_RUNTIME.md). Training does not enable it
in the engine automatically.

## Verified execution, October 30

The original processor-repr run is retained as a historical diagnostic because
its identity included a process-specific function address. It is not the
portable checkpoint. After replacing that identity, independent fresh model
loads matched and three new runs completed:

| Artifact directory under `output/` | Optimizer steps | Worker seconds |
| --- | --- | --- |
| `scene-model-clip-stable-40` | 40 from initialization | 21.719 |
| `scene-model-clip-stable-80-resume` | 40 resumed to 80 | 21.953 |
| `scene-model-clip-stable-80-control` | 80 uninterrupted | 35.719 |

The resumed and uninterrupted `head.json` files are byte-identical, SHA-256
`cf0c9ce35d166a4e85b13664d361c1f13f911195afbb00c28b251151886ab988`.
Parameters, loss history, processor and frozen feature identities match.
`scene-model-clip-stable-80-resume/resume-control-proof.json` records that proof.

Training loss fell from 1.6097 to 0.8354. Both the learned head and zero-shot
baseline matched only 9/23 validation labels. The learned head predicted
background for 22/23 cases and product for one. Removing the source view gave
the same validation accuracy; removing the asset view gave 2/23. The full-view
linear context therefore has not demonstrated source-aware generalization.
Every report retains `quality_validated: false`; do not promote these weights
as a quality improvement or tune against this already observed validation set.

The stable 40-step run measured 10.328 seconds for model load plus its first
image pair and 0.338 seconds mean for later pairs. These are local encoder and
zero-shot-comparison timings on the exercised host, not full resize/API latency
or reference-hardware guarantees. CPU optimization used no cloud GPU or model
download. A better next experiment needs more independent product and logo
families, ambiguous-role labels and a separately frozen evaluation set.
