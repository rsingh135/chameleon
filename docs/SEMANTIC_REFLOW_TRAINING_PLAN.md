# Semantic reflow: descriptors, training and acceptance

The source-first inspection harness and rule-derived descriptors are now
implemented. Semantic model training and qualification remain the next work.
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
approval. The receipt reports no immutable identity differences. **After-fix
native-render verification is pending.** This exposed case remains development
regression evidence, excluded from held-out generalization aggregates.

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

The source-first harness is implemented; menu/ranker integration and the
descriptor trainer below remain proposed. Coordinate engine ownership before
making those changes.

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

**Collect labels at the right level.** The existing 34 image-role labels cannot
supervise text purpose, footers or pairwise connections. Start with licensed or
user-owned independent development families covering corner brand marks,
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

**Train a small descriptor policy first.** Freeze one recipe and one seed.
Use normalized source geometry, type, measured alpha/ink evidence, font ranks
and neighbor geometry. Where available, add the already pinned frozen visual
features with an explicit missing-feature mask. Existing image-only cached
features do not cover text/shape crops: new contextual crop/full-scene features
need their own exact pixel and extractor receipts. Geometry alone cannot
establish brand identity or copy meaning; report that limitation and ablation.
Do not download or fine-tune a larger VLM as part of this first experiment.

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
reload predictions exactly. No paid GPU is needed for this head experiment.
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

1. Freeze engine/model ownership; land sidecar validation and identity tests.
2. Export source descriptors and per-target invariants from the harness without
   changing resize authority. Test stale/duplicate IDs, cycles, partial labels,
   source immutability, source/asset changes and absent ink measurements.
3. Review a small development set and run the reviewed-descriptor ablation. Test
   a corner symbol, footer alignment, explicit image/text group and an unrelated
   nearby element; require generic rules independent of names/sample IDs.
4. Freeze split/features/recipe, execute the one bounded CPU head experiment,
   reload, calibrate where supported, and run the paired native matrix.
5. Publish classifier results, invariant changes, actual action/veto counts,
   PNG/checkpoint hashes, reviewed wins/losses and remaining gaps. Default use
   follows evidence, not the successful completion of training.

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
New semantic trainers still require implementation and focused tests before use.
