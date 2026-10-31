# Semantic reflow: descriptors, training and acceptance

The source-first inspection harness, rule-derived descriptors and source-anchor
execution are implemented. A fresh bounded local Qwen VLM experiment is owned
by the primary training chat; its qualified outcome remains pending.
Preserve every original element, copy,
asset, font and authored relationship. Learn what an element does and what it
belongs with; let trusted layout code choose valid transformations.

## What exists, and what it cannot establish

- `engine/scene.py` measures physical image evidence. The closed response in
  `engine/scene_model.py` and `experiments/scene_model/schema.py` proposes five
  image roles and protected regions. It has no text/shape purpose or relational
  annotation contract. Protection can veto a crop; a prediction cannot grant it.
- `engine/layout/relationships.py` already provides source-bound units, explicit
  groups, message/brand attachments, axes, clusters and ambiguities.
  `engine/layout/reading_order.py` measures reading relations. Reuse these
  structures rather than infer another inconsistent grouping system.
- `engine/visual_plan.py::_bindings` binds units and graph/resource/code/source
  digests into a target-conditioned menu. The model selects a known ID.
  `visual_pipeline.make_request` supplies the source image and bounded element
  summary; the critic in `visual_selection.py` selects only admitted designs.
- The current `engine/fit_policy.py` uses 28 measured geometry features and a
  small pairwise ranker trained with `synthetic_geometry_teacher_v1` labels.
  Its fit/readability supervision is useful infrastructure, not learned design
  purpose or independent aesthetic judgment. `fitted_columns.py` is a trusted
  geometry executor, not a vision model.
- The earlier hybrid image-role experiment reached 13/23 agent-labeled
  validation records versus 9/23 for its original head. Its eight actual paired
  resize comparisons were unchanged. That evidence does not qualify a semantic
  reflow policy; see [the measured recovery plan](RESIZE_MODEL_ITERATION_PLAN.md).
  The local VLM pilots also remain unqualified for automatic layout judgment;
  see [their recorded controls](../experiments/visual_planner/README.md).

The WeBelieve source is now an explicitly opened **manual acceptance case**.
Its corner logo, footer and headline–cube grouping are exposed development
regressions, so improvements on this case are not independent evidence of
generalization. Exclude it from training, calibration and held-out aggregates.
Choose and freeze future thresholds on separate development families. Keep the
remaining unseen reserves untouched, and never automatically collect this audit
as training data.

## Verified baseline regression

The saved [baseline receipt](../output/semantic-reflow-we-believe/baseline-1600x400/receipt.json)
and [semantic audit](../output/semantic-reflow-we-believe/baseline-1600x400/semantic-audit.json)
bind an actual native [1600 × 400 PNG](../output/semantic-reflow-we-believe/baseline-1600x400/output_png.png).
Corner retention was approximately `1.28e-10` and footer retention `0.01965`,
while explicit-group retention was approximately `1.0` and reading-order
retention `1.0`. Coverage was one corner, one footer, one explicit group and two
reading edges. These diagnostics expose anchor failures despite preserved group
geometry/order; they are not calibrated probabilities or independent visual
approval. The receipt reports no immutable identity differences.

The later [v3 native manual proof](../output/semantic-anchor-manual-proof-v3/report.json)
restores the logo to the top-right and centers the footer. Cube width increases
from approximately 226 to 248 pixels, with the headline/cube group preserved;
corner retention is approximately `1.0`, footer retention `0.9761`, and group
and reading-order retention remain approximately `1.0`. The target is still
1600 × 400, with no recorded hard-contract violations. Native visual readability
remains **partial**: lettering baked into the atomic cube image is still small,
and the footer has about an 11-pixel bottom gap. Live-text font checks cannot
verify embedded raster lettering. The trained fit ranker chooses between two
admitted geometries; this is not learned purpose inference or automatic visual
approval. Neither this exposed development proof nor its weight ablation is a
production acceptance or held-out generalization result.

## Implemented all-element inspection contract

`engine/design_semantics.py` provides `describe_design`,
`semantic_source_digest`, `validate_semantic_annotations` and
`audit_semantic_preservation`. The separate
`tools/semantic_scene_harness.py::inspect_semantic_scene` exports source-first
evidence without calling a model or resize engine. The existing image-role v1
contract remains unchanged. Original descriptions and supplied-result audits
remain separate records.

| Field | Meaning and authority |
| --- | --- |
| `source_digest_protocol`, `source_digest`, `semantic_digest` | `canonical-compact-v1` original JSON and deterministic descriptor identity under `design-semantics-v1`. There is no core `scene_digest` field. The harness separately records actual asset/font bytes, resource digest and implementation hashes. |
| `elements` | Exactly one record per original element ID, including explicit `unknown`/abstained records; never invented or omitted IDs. |
| Rule purpose fields | `purpose`, explanatory `design_purpose`, `purpose_confidence`, `confidence_kind` and `provenance`. Confidence is explicitly `heuristic_uncalibrated`; explanatory prose is not evidence that a trained model inferred intent. |
| Rule geometry/relationships | `source_box`, `visible_box`, `visible_box_provenance`, `normalized_box`, `anchors`, `rigid_group`, `atomic_asset`, `preservation_rules` and per-element `relationships`, plus top-level `relations`. Frame fallback remains distinguishable from measured ink. |
| Proposal `purpose`, `confidence`, `evidence` | Closed labels: `background`, `brand_mark`, `corner_marker`, `headline`, `primary_visual`, `body`, `supporting_copy`, `footer_destination`, `footer_legal`, `action`, `decoration`, `unknown`. Proposals express supported meaning or abstention, not geometry. |
| Proposal `relations` | Known endpoint IDs and a closed type: `attached_to`, `caption_for`, `supports`, `reading_before`, `aligned_with`, or `same_group`; confidence and evidence text. Supplied relations remain soft. |
| Review receipt | Explicit `supervision_origin` and `reviewed_by`, separate from untrusted proposal fields. Rule, model-file, agent and human provenance stay distinct. Every current harness artifact has `training_eligible:false`; future reviewed dataset admission is separate. |

The supplied proposal schema is exactly `schema_version`,
`source_digest_protocol`, `source_digest`, `elements`, `relations`. Element rows
contain exactly `element_id`, `purpose`, `confidence`, `evidence`; relation rows
contain exactly `source_id`, `target_id`, `type`, `confidence`, `evidence`.
It currently binds **source JSON only**. The harness records current resource
hashes alongside it; that does not prove a supplied model file saw those bytes.
A future authority-bearing proposal needs a versioned request/resource binding.
Model output cannot supply target coordinates or declare itself human-reviewed.
The v1 validator accepts no model `anchor_proposal`, `anchors`,
`design_purpose`, `preservation_rules` or geometry fields. Anchors and
preservation rules are currently generated from source evidence by trusted code.
A learned anchor head and any externally supplied anchor proposal require a
future versioned contract, validation and qualification; they are not accepted
v1 inputs.

Use endpoint membership for identities; never train IDs, filenames, hashes or
campaign names as features. Unknown is a real label; an unreviewed field is
masked supervision, not an asserted negative. A footer can be a URL, legal copy
or contact information: source position alone cannot establish its purpose.

The validator rejects duplicate/unknown IDs, stale digests, nonfinite
confidence, unsupported fields, oversized lists, invalid self-edges and cycles
in accepted `reading_before` edges. Bound input to the current viewer's 200
elements and at most 1,024 sparse proposed edges. Report incomplete edge
coverage instead of fabricating absent relationships. Derive a canonical
sidecar digest after validation. Keep source IDs intact in artifacts.

Name digest encodings explicitly: `engine.scene` uses compact canonical JSON
for its source digest, while `visual_plan.source_digest` uses sorted JSON with
default separators. They are different existing protocols. Bind and verify
each with its own helper or add a separately named canonical sidecar digest;
do not compare the two strings as though they had the same encoding.

The ready training export makes both bindings explicit: rich semantic labels
use `source_digest_protocol: canonical-compact-v1` and `source_digest`;
`engine.visual_analysis` source-only observations use
`analysis_source_digest_protocol: sorted-spaced-json-v1` and
`analysis_source_digest`, plus the exact native `source_render_sha256`.
Request/sidecar digests remain separate compact content hashes. Consumers must
use the appropriate stored binding rather than copy one digest into both
protocols. Sparse `analysis_raw` observations refer only to IDs surviving the
bounded source summary; full purpose records still cover every original ID.

Explicit `group_id` and immutable design identity retain their existing hard
authority. A model cannot break such a group, declare an inferred connection
hard, replace text, drop a footer, introduce padding artwork, authorize a crop,
or add an element. Soft proposals can abstain. If the canvas makes an inferred
relation infeasible, record which one was relaxed and why. Preserve the source
design's intentional negative space; maximizing occupancy is not the goal.

## Harness implementation and remaining integration seams

A first sparse source-only observation handoff is now implemented in
`engine/visual_analysis.py` and the shared visual pipeline. Its separate local
analyzer feeds advisory facts to the organizer without target/menu influence.
See [implementation and setup](IMAGE_REASONING_PIPELINE.md). This does not
itself establish per-element semantic quality or the fit-feature changes below.

The source-first harness and deterministic source-anchor execution are
implemented. `engine/layout/semantic_anchors.py` generates source-bound atomic
layouts and `engine/semantic_execution.py` admits/ranks them under native text,
group, bounds, overlap and affinity checks. Purpose authority is source evidence
and deterministic constraints; the fit policy ranks geometry and makes no
learned-purpose claim. The full marketing-pixel profile is not evaluated by that
specialized execution profile. Supplied purpose/relationship proposals remain
soft; the inspection harness exports them without applying them to a design.
Richer learned descriptor/menu features below remain proposed, not a second
parallel pipeline. Coordinate extensions through the shared visual pipeline.

1. `design_semantics.py` implements source-bound descriptors and the closed
   proposal validator. Source anchors are measurements and preferences; explicit
   source groups remain hard. Native ink measurement is separately disclosed by
   provenance, including frame fallback.
2. `inspect_semantic_scene(source, assets_dir, fonts_dir, out, *, design=None,
   model_response=None, supervision_origin='unreviewed', source_id=None,
   manual_acceptance=False, reviewed_by=None)` requires a fresh output directory.
   It exports `source.json`, native `source.png`, `semantic-scene.json`,
   `semantic-model-request.json`, `semantic-review.json`, and a hashed
   `report.json`. Every original ID has a request record and an explicit unknown
   response template. Proportional source-context crops are at most 256 pixels
   per axis and may include neighboring artwork; they are not isolated assets
   or replacement pixels. Request metadata is capped at 1 MiB, excluding
   external PNG files. No provider process is invoked.
3. An optional supplied `design` is rendered at its actual dimensions and
   audited into `semantic-invariants.json`; no resize search runs. Current
   invariants include corner/footer affinity, explicit-group geometry, reading
   order and eligible counts, plus existing hard-contract violations. Richer
   baseline/attachment/ink diagnostics below remain follow-up work. Source,
   supplied design, resource bytes, reserve catalogue identity and implementation
   files are checked before/after; a changed run produces no completed receipt.
   Supplying a validated proposal does not silently relabel rule descriptors or
   apply it to the design. Resource/output overlap, existing output directories,
   oversized resources and invalid source/design geometry fail explicitly.
4. In `visual_plan._bindings`, add the validated descriptor digest and a
   compact semantic summary to the bound menu. Its validator must regenerate
   the same binding. `visual_pipeline.make_request` may expose that summary as
   data with explicit omission counts. Do not append fields to a closed menu
   after calculating its digest. Preserve the existing instruction boundary
   around source artwork and copy.
5. In a **new** fit feature version, add measured anchor/footer/attachment
   preservation features. The engine owner can pass validated descriptors to
   `fit_policy.extract_features` and menu construction. Changing the feature
   schema requires a new model artifact, teacher/review label kind and digest;
   do not relabel the current 28-feature checkpoint or feed it new columns.
   Keep candidate generation/admission and the known-ID decision contract.
6. Extend the harness report and test interface with separate facts:
   `descriptor_executed`, `annotation_valid`, `action_eligible`, `abstained`,
   `constraint_applied`, `candidate_vetoed`, `layout_changed`, and
   `quality_review`. Valid JSON or model execution does not imply improvement.

First land schema and reporting in review-only mode. Then run a reviewed
descriptor ablation before training; it isolates whether better semantics can
help the executor at all. If correct annotations cannot change a bad candidate
set, fix the generic executor or candidate coverage instead of training longer.

## What to measure on actual transformations

Use native text ink/baselines and image alpha-visible bounds where available.
Record any layout-box fallback; do not describe it as visible-ink measurement.
Compare with the rendered source to distinguish an existing defect from a
regression. Report per-element values as well as family aggregates.

| Invariant | Measurement |
| --- | --- |
| Exact inventory and identity | Existing hard contract: every original ID exactly once; no new copy/assets/fonts/colors or unintended appearance changes. All required content remains in frame. |
| Corner affinity | Same accepted horizontal/vertical edges; signed edge-gap drift divided by the canvas short side. Retain source margin policy rather than pinning every small image to a corner. |
| Footer placement and baseline | Accepted footer members retain bottom affinity and source horizontal alignment; measure baseline spread, edge-gap drift, and separation from main content. A box bottom is not a text baseline. |
| Rigid groups | Existing explicit-group deformation, plus residual after one common scale/translation of source member geometry. Separately report text rewrap allowances; do not penalize a permitted flow change as arbitrary movement. |
| Image–text attachment | Preserve the accepted side/order, visible gap relative to unit size, cross-axis alignment, and association with the correct image. A group may move as a unit. |
| Attention and reading order | Source-relative font/prominence ranks and reviewed precedence inversions. Ambiguous pairs are excluded with counts, not guessed into a total order. |
| Proportion and collisions | Image aspect/visible-content retention, group scale consistency, text overflow, visible ink collisions and approved layering. Keep the existing full contract and scoring checks. |
| Space usage | Visible content envelope, source-derived margins and designed negative space, alongside absolute minimum font size and native text-fit checks. High occupancy alone is not a win. |

Predeclare tolerances from development annotations before the held-out run.
An initial anchor diagnostic can flag drift above 2% of the target short side
plus one raster pixel, but that is a proposed development threshold, not a
validated universal design rule. Report raw drift and eligible counts too.

## Small, bounded training experiment

**Current data is ready for an experimental run.**
[READY.json](../output/semantic-analysis-data-v1/READY.json) points to
[the frozen manifest](../output/semantic-analysis-data-v1/dataset-smoke-v1/manifest.json)
with SHA-256 `17d014c14e4271897978513acc72a0994f6235d8dd7037ec680e1c78fc0c6863`.
There are eight authored sources: five train and three validation, with 344
element records comprising 169 eligible reviewed training elements, nine masked
uncertain training elements and 166 validation elements excluded from training.
The 16 source-analysis rows contain original/opaque-ID variants of the same eight
sources; they are not 16 independent scenes. The frozen split hash is
`f231be09489b7d6fa438d772410792b92bced668f3250afb12280eb38cfd549c`.

`analysis.jsonl`, `elements.jsonl` and `relationships.jsonl` are under
`output/semantic-analysis-data-v1/dataset-smoke-v1`. Admission is explicitly
`WEAKSUPERVISED_EXPERIMENT_ONLY`, with agent visual review rather than human
ground truth. The original inspection artifacts remain training-ineligible;
the separate exporter admits only reviewed train records with `review_mask:true`. Validation
labels must not train or calibrate the model. Authored-work separation is not
unseen-brand or unseen-typography evidence: the manifest discloses shared fonts.
Relations and rule-derived anchors are currently exported as
`artifact_only_not_training_features`, not learned relational/anchor targets.
The [optional blank controls](../output/semantic-analysis-data-v1/blank-controls-v1/manifest.json)
are separate synthetic-abstention diagnostics, not original-render parity or
agent-reviewed labels. A trainer must opt in explicitly; they are not consumed
automatically and cannot enlarge the independent-source count.

**Freeze the trainer/evaluator contract before interpreting results.**
The exported [purpose question](../output/semantic-analysis-data-v1/element-purpose-question-v1.json)
uses twelve purpose classes and one-letter answers. The primary
`experiments/visual_finetune/data.py` currently maps purposes to ten word aliases.
These contracts differ. The primary training owner has selected the ten-word
contract for its current purpose pilot; the twelve-letter question remains an
unconsumed alternative. Record the actual label mapping, prompt, tokenizer,
masks and evaluation decoding together in each frozen recipe. Scores from
different mappings are not directly comparable. Eligible reviewed training has
no `corner_marker`, `footer_legal` or `unknown`
examples; those behaviors cannot be claimed as learned from this corpus.

**Expand labels at the right level after the bounded pilot.** The earlier 34
image-role labels cannot supervise text purpose, footers or pairwise connections.
Add licensed or user-owned independent development families covering corner marks,
footers, grouped image/text, editorial columns and ambiguous decorative marks.
For each source, review all element purposes and only the relevant relationships;
include hard negatives such as an unrelated nearby icon and footer-sized body
copy. Review source and render parity before labeling. Save a disagreement and
uncertainty field. Agent labels can support the pilot with disclosure; they
are not independent human quality evidence.

Plan for 20 independent families if available: 12 training, four calibration
and four locked development evaluation families. This is a collection target,
not a claim of sufficient statistical power. Keep campaign variants, source
duplicates, reused assets and all target versions together. A missing class or
relationship in held-out data blocks a claim about that behavior. The existing
reserves and the manual acceptance case remain excluded. Group-aware splitting
prevents group overlap; see [the official GroupKFold contract](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html).

**Primary experiment and ownership.** The bounded local Qwen VLM pilot uses the
primary training chat's exact cached Qwen3-VL-2B base and frozen adapter recipe.
Subsequent compute uses that chat's freshly verified configuration and budget
guard, including the newly authorized Brev direction. Final qualification and a
deployed adapter remain pending. Data readiness or optimizer steps alone
establish no task benefit. Keep source-only analysis, organization and
verification on the existing shared contracts described in
[the pipeline guide](IMAGE_REASONING_PIPELINE.md) and
[the training plan](SCENE_MODEL_TRAINING_PLAN.md). The primary training chat owns
GPU and cloud scheduling, budget/stop verification, checkpoints and adapter-backed
role configuration. Supporting chats provide artifacts, review and verification;
they start no competing model/GPU work. This document creates no new GPU job,
cloud session, download, time allowance or budget authorization. Historical
Brev limits do not authorize a new cloud run.

**Verified early finding.** The saved
[checkpoint-8 training receipt](../output/visual-finetune/runs/purpose-r4-32-v2/checkpoint-8/training_receipt.json)
records eight real optimizer steps and nonzero LoRA updates. The
[same-adapter evaluation receipt](../output/visual-finetune/runs/purpose-checkpoint8-eval-v3/result.json)
binds adapter SHA-256
`8eddb0524f8874dc83e03c843938c5c9b46ed07933dd035cac6edb4b66193ec1`.
On the 48 closed-class cases, the
[public-development analysis](../output/visual-finetune/runs/purpose-checkpoint8-eval-v3/public-validation-analysis.json)
reports base `21/48` versus adapter `23/48`: original-source cases `17/39` to
`18/39`, blank controls `3/6` to `4/6`, and source-swapped cases `1/3` unchanged.
This is a compatibility and weak public-development result under that run's
contract, not qualified design fidelity, generalization or a completed full
validation of all 166 held-out elements. It uses the primary word-label task,
not the alternative twelve-letter question, and establishes no learned relationship/anchor
execution. The separate optional blank-control export is still opt-in.

**Critic training evidence is separate.** The frozen
[critic-v1 manifest](../output/critic-pair-training-v1/manifest.json), SHA-256
`1bba2ce4d27338d8be38e32443c7871fa615f268ae265fd214e78e1fffb729bd`,
contains 15 unique pairs and 30 counterbalanced A/B/NEITHER records from the
same five training works. All 97 output hashes were checked. Each source uses
one hard-valid saved fit and two deliberately invalid native transformations;
labels are algorithmic weak supervision, not designer preferences. The first
negative views are often blank, so this is an easy compatibility curriculum.
Keep its training-only manifest separate from untouched evaluation; do not move
validation families into training to satisfy a loader's two-split requirement.
Harder negatives and source-mismatch controls belong to a new immutable version.

The [frozen critic-v2 manifest](../output/critic-pair-training-v2/manifest.json),
SHA-256 `1c042e89c33c84ccaaa76f0d9a31d3a992e6454e72148cf042a28c752941fc5a`,
retains five train families and 30 balanced records, with 103 verified files.
Its five new misfits retain recognizable artwork/copy, and two real other-train-source
mismatch controls plus their swaps declare false source-render parity. Some
inherited occlusion views remain blank. These are controlled weak comparisons,
not learned taste or independent resizing qualification.

## Three-view perception experiment

The acknowledged input order is **original scene, target context, isolated
original layer**. The [native evidence manifest](../output/semantic-purpose-input-v3-evidence/run-2/manifest.json)
has SHA-256 `001292824831719997a0f8c8ac7abf0cfedd4f00b38044efcc1f04473418fa61`.
It preserves all 344 record identities, labels, splits and masks. There are 335
available isolated views: 169 train and 166 validation; nine uncertain records
remain masked and unrendered. The [separate final audit](../output/semantic-purpose-input-v3-evidence/final-audit.json)
checks all 1,512 declared file hashes and unchanged parent source/context bytes.

Each third view paints only the original element using its original resources,
font face, style, opacity and effects. Native alpha bounds plus fixed padding
produce a proportional thumbnail of at most 512 pixels; a separate transparent
RGBA view and coordinate/raster receipts remain available for audit. Every class
uses the same checker. White glyphs have low contrast on this light checker,
especially at 256 pixels; this representation limit is recorded separately.
An isolated layer is not its design intent: retain the complete source and
neighbor context. Never treat these thumbnails as replacement output assets.

The primary worker owns the explicit three-view loader and prompt. Its planned
matched runs keep the same sample IDs, labels and optimization recipe: R1 uses
the 256-pixel preprocessing budget; R2 changes only that budget to 512. Compare
an unchanged base and adapter under each exact input/prompt contract. Keep new
product/art family expansion as a separate experiment. An unavailable third
view must be reported and excluded consistently or fail the matched run; no
silent two-view fallback is allowed. Blank diagnostics blank all three views;
mismatch diagnostics retain explicit provenance for each view.

Keep saved adapter precision consistent during reload checks. A rounded reload
must not be accepted by weakening the equality threshold. Execution, nonzero
updates and successful reload are compatibility checks, separate from measured
purpose accuracy, visual discrimination and actual resized-render improvement.

**Optional conservative CPU baseline.** Retain the small descriptor-head
experiment as a control, not a prohibition on the authorized VLM work. Freeze
one recipe and one seed if this control is scheduled by the primary owner.
Use normalized source geometry, type, measured alpha/ink evidence, font ranks
and neighbor geometry. Where available, add the already pinned frozen visual
features with an explicit missing-feature mask. Existing image-only cached
features do not cover text/shape crops: new contextual crop/full-scene features
need their own exact pixel and extractor receipts. Geometry alone cannot
establish brand identity or copy meaning; report that limitation and ablation.
Any new visual features need the primary owner's memory and runtime preflight;
do not silently substitute them for the frozen baseline features.

Fit regularized linear purpose and anchor classifiers, plus a small sparse
pairwise relation classifier. Normalize using training data only. Use masked,
class-balanced cross-entropy for one primary purpose and per-axis anchor
classes; use masked binary cross-entropy with logits for the multi-label
relations. These are standard supported objectives, documented by
[PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.CrossEntropyLoss.html)
and [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html).
Average losses per source/family and reviewed field so a dense poster or its
many negative pairs cannot dominate. Unlabeled edges contribute no loss.
Record any hard-negative sampling and the population used for evaluation.

Use at most 200 fixed optimization steps, one CPU worker and a 600-second
external stop deadline after feature preparation. Save the recipe, split and
label hashes, feature version/bytes, normalizer, weights and code identities to
a fresh ignored `output/semantic-reflow/<run-id>` directory. Verify checkpoint
reload predictions exactly. No paid GPU is needed for this optional head control.
This is task-head training on frozen features, not VLM fine-tuning.

Calibrate probabilities on disjoint calibration families when coverage permits;
otherwise retain review-only authority. Confidence thresholds and abstention
must be fixed before locked evaluation. Report reliability/Brier or log loss,
precision at action coverage and unknown recall; raw softmax confidence is not
proof of reliability. Official [calibration guidance](https://scikit-learn.org/stable/modules/calibration.html)
requires data separation and notes failures when a class lacks calibration
examples. Do not lower thresholds after seeing locked failures.

## Paired resize evaluation and promotion

Compare four conditions on the same frozen engine, resources, target matrix and
candidate/check budgets: deterministic baseline, current fit policy, reviewed
descriptor policy, and learned descriptor policy. The reviewed condition is an
oracle diagnostic with label provenance, not a deployable model score.

For each held-out source use proportional reductions at 50% and 25%, then one
wide and one tall target. Record the exact requested/rounded dimensions. At
least four independent held-out families must cover all claimed behavior; if
four families cannot supply the required corner/footer/group cases, add families
or narrow the claim. Do not count the four target variants as four independent
sources. Repeat a small middle-aspect case if needed to expose discontinuous
anchor switches. Baseline and learned condition must produce actual native PNGs.

Promotion requires zero new inventory/identity, out-of-frame, explicit-group or
crop-permission violations; no new severe visible collision or lost footer;
and nonzero useful actions on at least two independent held-out families.
Require improved paired native-render reviews for both proportional and aspect
changes, with corner/footer/group cases represented. Keep ties, losses,
abstentions and failures in the report. Report family-level uncertainty rather
than interpreting a tiny perfect sample as established reliability. A classifier
accuracy increase with unchanged layouts is not a resize improvement.

Use blinded A/B ordering where feasible and inspect swapped ordering for any
automated critic. Human/agent review provenance stays explicit. Preserve all
failed runs and do not tune on the manual acceptance case or reserves. If gates
fail, ship the verified geometry path and keep semantic descriptors visible for
review; do not claim that the model understands the design.

## First implementation sequence

1. Keep the implemented descriptor/harness and source-anchor regression evidence
   frozen. Verify READY, native-image/annotation hashes and the eight-source split
   before the primary trainer consumes the separately admitted weak labels.
2. Resolve and freeze the trainer/evaluator's twelve-letter versus ten-alias
   contract explicitly, including prompt, masks, tokenizer and decoding. Preserve
   validation and manual/reserve exclusions; opt into blank controls separately.
3. Let the primary owner finish the bounded local VLM experiment under its
   existing memory/offload and checkpoint controls. Record actual steps, saved
   versus restorable state and all failed runs; supporting chats start no jobs.
4. Compare checkpoint-backed purpose predictions with unchanged base, majority
   and optional CPU controls using the same frozen contract. Include per-class
   support, source/metadata ablations and abstention. Relationship/anchor learning
   remains future work until those labels are admitted as actual targets.
5. Run source-bound paired native transformations on independent development
   cases through the shared analyzer/organizer/verifier path. Publish invariant
   changes, action/veto counts, PNG/checkpoint hashes, reviewed wins/losses and
   remaining readability gaps. No automatic promotion follows training completion.

Run the new source-first harness directly from the project root. For the
explicitly opened manual case, use a fresh output name:

```powershell
$preparedScene = 'fixtures/open_posters/prepared/opensuse_banner_we_believe'
.venv/Scripts/python.exe -m tools.semantic_scene_harness `
  --source "$preparedScene/source.json" `
  --assets "$preparedScene/resources/assets" `
  --fonts "$preparedScene/resources/fonts" `
  --source-id opensuse_banner_we_believe --manual-acceptance `
  --out output/semantic-reflow/we-believe-source-1
```

Optional flags are `--design`, `--model-response`, `--supervision-origin` and
`--reviewed-by`; a reviewed origin requires both a supplied proposal and an
explicit reviewer. Reserve detection uses source content as well as supplied
IDs; omitting an ID cannot bypass it. Only the two explicitly opened reserve
cases permit manual acceptance. Manual artifacts never become training data.
Plain development cases also need separate reviewed dataset admission.

The existing `tools.scene_harness --help` and `tools.train_fit_policy --help`
remain runnable entry points. The latter accepts only the current fit
feature/label schema; it cannot train the proposed descriptors unchanged.
The primary VLM trainer has its own purpose-data adapter; it must use the frozen
training/evaluation contract above. Additional relation/anchor trainers remain
future work and require implementation and focused tests before use.
