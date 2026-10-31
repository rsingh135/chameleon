# Experimental purpose-adapter analyzer bridge

`adapter_analysis.py` converts saved-adapter **classification scores** into the
existing source-bound `engine.visual_analysis` contract. It does not generate
free-form visual reasoning, infer attachment/order relationships, choose layout
coordinates, or grant background crop/cover permissions.

This bridge explicitly declares integer `purpose_views=2`: original source then
context crop. Historical missing metadata defaults to two. A recipe/dataset
declaring three views or predictions embedding three images is rejected. The
future source/context/isolated-layer contract requires its actual third image;
it cannot silently reuse this two-view bridge.

The actual trained aliases are `accent`, `action`, `background`, `body`, `brand`,
`footer`, `headline`, `support`, `unknown`, and `visual`. Their mapping is checked
against the training loader. A legacy twelve-letter output is rejected. IDs
ground observations only: the shared training/inference prompt contains type,
normalized box, raw copy, authored paint rank and element count. It contains no
source IDs, group IDs, declared roles, file names, paths or gold annotations.

## Prepare and complete

1. Supply the immutable original source render and trusted native context crops
   to `build_adapter_request(source, source_render, context_images)`. Contexts
   must be original-source pixels with no gold labels/ID overlays. The second
   image is a context crop, not an isolated layer. Missing contexts are recorded
   as explicit abstentions.
2. Write `request_bytes(bundle)` exactly. It is compact sorted finite JSON plus a
   newline, matching the evaluator's request-file SHA convention. The request is
   label-free and contains no target resize, candidate menu or repair feedback.
3. The operator prepares and freezes the inference recipe **before running the
   model**, retains its base/adapter/config/recipe/dataset/code identities, then
   uses the supervised evaluator. The worker owner controls GPU scheduling,
   process lifetime and offline checkpoint loading. This bridge starts no job.
4. Pass exact `external_adapted.json` bytes, `inference_receipt.json`, and that
   independently prepared identity to `complete_adapter_analysis`. Never derive
   the expected identity by copying an untrusted model response or receipt.

The receipt must attest completed purpose scoring, immutable base commit and
weight SHA, adapter weight/config SHA, recipe/data/worker-code SHA, exact request
bytes and exact prediction-file bytes. Every record must bind the source image
first and context crop second and attest both image embeddings. Old evaluation
outputs without this receipt remain unverified and abstain; no successful model
execution is manufactured for them.

The bridge checks the full-vocabulary greedy token as well as closed-class
scores. Nonclass raw tokens, unknown predictions, inconsistent score/probability
distributions and weak normalized candidate scores abstain. The current diagnostic
screens are normalized candidate score ≥0.85 and candidate-score margin ≥0.30;
these are **uncalibrated screens**,
not measured reliability or evidence of aesthetic quality. Only `subject`
observations are emitted, describing independent role proposals. No relationship
is invented from shared class membership. Budget omissions and per-element
abstention reasons remain in provenance.

## Graph composition and deployment

`adapter_roles.create_adapter_roles(...)` returns a `LocalModelRoles` subclass so
the graph retains its existing configuration-identity checks. Only `analyze` is
overridden; organizer and verifier calls delegate unchanged to their separately
configured roles. The purpose classifier is never described as an organizer.

An operator supplies `context_builder` and `score_runner` callbacks. The latter
returns `{prediction_bytes, receipt, prepared_identity}`; it must enforce the
supplied timeout and obtain prepared identity before inference. Its model/code
pins must equal the role's independently configured `model_identity`.

Default `mode="off"` invokes neither crop preparation nor scoring. Explicit
`mode="diagnostic"` permits experimental graph observations while retaining
`qualified=false`, `deployment_enabled=false`, `experimental_only=true`, and no
layout/crop authority. There is currently no qualified/deployment mode. A pinned
execution receipt proves a bound scoring run, not useful or calibrated purpose
recognition. Actual held-out/control qualification and paired render outcomes
must precede any future deployment path.

Tests use visibly synthetic envelopes and fake hashes to check protocols, ID
feature invariance, cold/low-confidence abstention, image order, model pins and
default-off delegation. They launch no model and establish no adapter quality.

## JSONL transport and asynchronous worker ownership

Run `python -m experiments.agentic_graph.adapter_stream --artifact-root OUTPUT`
as a CPU-only JSONL transport. Add `--diagnostic` explicitly to publish unqualified
observations; packets cannot enable it. Each packet supplies an opaque `job_id`.

- `prepare`: fields `operation`, `job_id`, `source`, `source_render`. It verifies
  the native PNG's actual bytes and dimensions, makes unannotated original-pixel
  contexts, and writes `OUTPUT/JOB/adapter-request.json` plus grounding receipts.
  Crops exactly match the inspected training recipe: padding
  `max(8, source_short_axis * 0.025)`, clamped bleed, 256-pixel LANCZOS thumbnail.
  Completely invisible frames abstain. No purpose export or semantic-rule labels
  are opened.
- The operator calls `python -m tools.evaluate_visual_adapter --prepare-only
  --inference-only` with its existing training-run, adapter, pinned base/manifest,
  request and fresh `OUTPUT/JOB/evaluation` paths. Set and freeze the intended
  `recipe.supervision` time/RSS/absolute-deadline guards before the next step.
- `freeze`: fields `operation`, `job_id`, `evaluation_dir`. It verifies the
  label-free recipe, exact prompts/images/classes, base manifest and worker code
  before any worker logs/results exist. Expected identity is stored separately;
  model binaries are verified by the actual worker, not by this CPU transport.
- The operator asynchronously launches the canonical bounded command:
  `python -m tools.run_prepared_visual_adapter --recipe
  OUTPUT/JOB/evaluation/recipe.json --python GPU_ENV_PYTHON`. The transport never
  starts or schedules a GPU process and never prepares the same job again.
- `complete`: fields `operation`, `job_id`. It rechecks frozen inputs, reads exact
  `external_adapted.json` and `inference_receipt.json` bytes, and emits the graph
  analyzer result with the trained adapter/base identities and abstentions.

`trace.source_analysis_digest` instruments the exact advisory context sent to the
organizer. Comparing this digest across actual frozen adapter checkpoints can
show whether trained weights changed the downstream role context. It does not
establish a better layout, learned relationships, or any geometry authority.
