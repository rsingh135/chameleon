# Local visual adapter experiments

This isolated experiment trains language-side LoRA tensors in the real
`Qwen/Qwen3-VL-2B-Instruct` visual model. It is separate from the small geometric
fit policy in `engine.fit_policy`. A completed fit-policy training run is not a
VLM fine-tune.

The base is Apache-2.0, pinned at
`89644892e4d85e24eaac8bacfd4f463576704203`. Its 4,255,140,312-byte
`model.safetensors` has SHA-256
`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`.
All execution uses verified local files, `trust_remote_code=False`, offline
Hugging Face mode, and the existing CUDA environment. No training or serving
weights are downloaded implicitly.

Two separate tasks are supported:

* **Purpose:** full source render plus an element context crop, predicting one
  role word. Aliases bind to the existing semantic taxonomy: `visual` means
  `primary_visual`, `brand` means `brand_mark`, `accent` means `decoration`,
  `support` means `supporting_copy`, and `footer` means `footer_destination`.
* **Critic:** source, candidate A, candidate B, predicting A/B/N. N means neither,
  a tie, or insufficient source evidence. A/B/N examples are balanced within
  each six-example optimizer accumulation group.

The purpose export contains **agent-reviewed weak labels**, not human labels.
Unknown or masked rows are excluded. Added all-blank and source-mismatch
controls have distinct programmatic provenance. Train and validation authored
source families are disjoint, including their transformations; shared font or
asset collections do not establish unseen-brand generalization. Sealed test
records are ignored by this training loader and belong to an independent
evaluator. Validation records never contribute gradients or temperature fitting.

Each run copies verified input images, freezes the dataset and a snapshot of
this package, and binds their hashes in `recipe.json`. Parallel changes to the
viewer or layout engine cannot silently change the experiment. The frozen base
is evaluated before inserting trainable adapters. Training uses full-vocabulary
cross entropy only at the answer token; it does not learn a long repetitive JSON
template. The vision tower and all base weights remain frozen. Checkpoints
record losses, actual gradient norms, sampled labels, trainable names and hashes.
A successful run must demonstrate a finite nonzero adapter delta and reload the
saved adapter into the resident base for a matching-score check. New runs publish
complete atomic generations every four optimizer steps, including optimizer,
scheduler, Python/Torch/CUDA RNG, sampler state and progress. These states support
future continuation; replay remains explicitly unvalidated until tested.

Evaluation files contain actual pixel-conditioned next-token logits, closed-set
class probabilities, raw top token, exact image hashes and image token grids.
Closed-set selection is explicitly reported as scoring, not generated prose.
Report macro F1, per-class recall, choice histogram and pixel-control behavior
alongside accuracy so a constant-B model cannot appear successful. Geometry
execution and independent native-render review remain separate requirements;
this experimental trainer never enables a provider by default.

Run from the repository root with the lightweight application Python:

```powershell
.venv/Scripts/python.exe -m tools.train_visual_adapter `
  --manifest output/semantic-analysis-data-v1/dataset-smoke-v1/manifest.json `
  --task purpose --out output/visual-finetune/runs/new-run `
  --steps 32 --evaluation-limit 48
```

The supervisor launches only its own existing `acestep` Python on CUDA. It
retains a 256 MiB sustained host-availability guard, a run timeout, an explicit
sprint deadline, and an output-local `STOP` signal. On this memory-constrained
Windows host it may trim only the working set behind its own stable worker
handle, with before/after receipts. It never closes or trims user applications,
security processes, or unrelated workers. Failed runs remain on disk and are
never relabeled as successful training.

The first saved purpose adapter is
`output/visual-finetune/runs/purpose-r4-32-v2/checkpoint-8`. It contains 112 finite
FP32 LoRA tensors (802,816 parameters); all 344,064 B entries changed from their
zero initialization. Its SHA-256 is
`8eddb0524f8874dc83e03c843938c5c9b46ed07933dd035cac6edb4b66193ec1`.
The worker performed 13 optimizer steps, then stopped under host-memory pressure;
only the step-8 checkpoint had been saved. Its frozen 48-record base comparison
scored 43.75% accuracy and 0.427 macro F1 against agent weak labels. A separate
saved-checkpoint evaluation reproduced all base logits exactly, then scored
47.92% accuracy and 0.455 macro F1 with the adapter. Original-role accuracy moved
from 17/39 to 18/39; blank abstention moved from 3/6 to 4/6. Background remained
0/6, so the adapter is **not deployment qualified**. Same-prompt image-only
controls change model scores, establishing pixel influence rather than style
quality. Independent sealed evaluation is still pending. This artifact proves
actual local visual-model training; it does not prove improved layouts.

The next input version explicitly identifies the target's real paint rank from
the native renderer's stable z-index order. Exported crops contain surrounding
layers; they are not isolated elements. Source JSON hashes are verified before
reading this metadata. This declared input change requires a new base comparison.
Semantic purpose and placement policy remain distinct: a featured photograph can
be a `primary_visual` that also covers the full background. A role prediction
alone must not decide cover versus contain.

`tools/evaluate_visual_adapter.py` accepts a label-free independent request with
protocol `visual-adapter-label-free-evaluation-v1`, a `task`, and `records` having
`record_id`, `prompt`, and ordered `images: [{path, sha256}]`. It rejects records
containing `label`, `target`, `expected`, or `answer`. Original training dataset
and recipe hashes and both adapter weights and configuration are verified. For
the historical step-8 checkpoint, which predates config-hash receipts, the config
is explicitly bound after training and before evaluation. New checkpoints bind
the configuration when saved. The evaluator does not decide quality or expose
sealed labels to the trainer.
