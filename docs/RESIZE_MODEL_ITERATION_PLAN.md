# Eight-hour resize and model recovery plan

October 30, 2026. Deliver usable batch resizing first, with a measured local
model experiment alongside it. The current learned role head has not improved
held-out recognition or any measured resize. Keep the deterministic method as
the reference and default until a replacement passes the gates below.

The plan was first written without running training. The later explicitly
authorized CPU experiment and native comparison are recorded at the end.
No GPU, weight download, shared-provider change or reserve-artwork use occurred.

## What the existing evidence actually says

The exercised model is frozen CLIP ViT-B/32 plus a five-class linear role head.
It classifies an existing image element; it does not resize elements, identify
attention order, detect object regions, or generate a composition. Its source
and selected-asset views are complete letterboxed images. A task prompt in a
request does not turn this feature classifier into an instruction-following VLM.

| Condition | Training correct | Validation correct | Validation macro recall |
| --- | --- | --- | --- |
| Zero-shot CLIP | 3/11 | 9/23, 39.1% | 0.333 |
| Trained two-view head | 9/11 | 9/23, 39.1% | 0.333 |
| Always predict background | 6/11 | 9/23, 39.1% | 0.333 |
| Trained head, source features removed | 11/11 | 9/23 | 0.333 |
| Trained head, asset features removed | 8/11 | 2/23 | 0.074 |

The trained head labels 22/23 validation images background and one product.
Validation background recall is 1.0; logo and foreground recall are both zero.
There are no validation products or unknowns. The 23 records come from only two
image-bearing family groups, not 23 independent designs. All 34 labels were
agent visual reviews; no independent human labeling pass occurred. Loss falling
from 1.6097 to 0.8354 establishes optimization, not task improvement.

Evidence: [stable training report](../output/scene-model-clip-stable-80-resume/training-report.json),
[byte-identical resume/control proof](../output/scene-model-clip-stable-80-resume/resume-control-proof.json),
and [tracked pilot labels](../fixtures/scene_model/pilot-labels.json).

Two-level actual resizing has already been exercised: ten usable development
sources, two targets per source, deterministic versus model-protect conditions,
for 40 real renders. All 20 paired JSONs and PNGs are identical. The maximum
model confidence was 0.5801, below the existing 0.8 protection gate: zero
protected IDs and zero candidate vetoes. This measures an abstaining no-op,
rather than successful model-assisted layout selection.

Both conditions have 17/20 accepted outputs, hence 34/40 combined. The three
unique rejected targets are Botanical reduced and orientation changes and
Trifold orientation change; their recorded minimum text sizes are 6.75,
8.3625 and 9.8455 px against that audit's 10 px floor. These are readability
failures shared with the baseline. Advanced remains blocked for missing
original asset bytes. Source/resource preservation and no added assets are
separate checks from readability or aesthetic approval.

Evidence: [paired verification summary](../output/scene-model-matrix-stable-80/verification-summary.json)
and [explicitly agent-only visual review](../output/scene-model-matrix-stable-80/visual-review.json).
At this audit, current `engine/layout/backgrounds.py` and `engine/resize.py`
differ from that matrix's recorded implementation. Preserve the old receipts;
rerun against a frozen current implementation before reporting current quality.

## Time-boxed priorities

| Elapsed hour | Work and exit condition |
| --- | --- |
| 0–1 | Freeze engine/provider revision with the other owner; identify supported input families, exact outputs and two target modes. Run existing functional checks. Capture hashes before/after each batch. If code changes mid-run, discard it as fixed-revision comparison evidence. |
| 1–2 | Run current deterministic batch on development sources and the user's authorized inputs. Review full-size renders, especially the three recorded failures, identity, crop, reading order and frame usage. Keep failing cases visible. Begin the focused label pass below. |
| 2–3 | Freeze one local experiment recipe and its family split. Validate current source/resource/feature identities and reserve exclusions. Collect additional labels where available; preserve agent versus human provenance. Do not select settings from validation outcomes. |
| 3–4 | Execute at most one supervised CPU task-head fit, capped at 600 seconds, after execution authorization and the owner handoff. Save parameters, normalizer, feature schema, recipe and identities in a fresh ignored directory. Reload and compare predictions. No GPU or backbone training is needed. |
| 4–5 | Measure zero-shot, old head, new head and constant baselines on the same frozen records. Report confusion, family-macro results, abstention and action coverage. A failed experiment remains a valid result; do not lower the confidence gate to manufacture intervention. |
| 5–6 | Run paired actual resize tests at reduced same-aspect resolution and changed aspect ratio, using identical sources, assets, targets and engine revision. A classifier metric alone cannot pass this stage. |
| 6–7 | Blind A/B review the actual source and both target renders; record ties and neither-acceptable outcomes. Check current export JSON, native PNG size and resource identities. Keep agent review distinct from human approval. |
| 7–8 | Ship the verified deterministic batch path unless every promotion gate passes. Deliver artifacts, failure list, timings and model comparison. Stop experimentation rather than consuming the delivery buffer. |

## One feasible local fine-tune

Use the already cached base, stable head, two-view feature file and existing
34-record manifest. A tractable diagnostic is one regularized hybrid task head
using five frozen role logits plus a small fixed set of source-only geometry
and alpha features: normalized frame area/position/aspect, asset aspect/alpha
coverage, and overlap with live text. No semantic element names, asset filenames,
candidate targets, inferred role labels or validation outcomes enter features.
Fit normalization on training data only. Preserve the 11/23 split for this
diagnostic and explicitly disclose its missing validation classes.

Keep this in a separate experimental module and feature schema; do not edit the
provider or relabel its 1024-feature checkpoint metadata while another batch is
running. Predeclare seed, regularization, optimizer steps and the 600-second CPU
deadline. Compare once, including source/asset removal tests and the constant
baseline. A new provider adapter, if implemented, must validate exact base/head,
normalizer and feature identities before serving its own experimental response.

This hybrid is our engineering hypothesis, not a proven improvement. The
[official CLIP example](https://github.com/openai/CLIP#linear-probe-evaluation)
supports fitting a small classifier on frozen image features; it does not prove
that our geometry features, labels or recipe generalize. More steps on the same
11 examples are not a substitute for independent product supervision.

If the hybrid is not implemented within its hour, execute one bounded existing
head fit on the explicitly reviewed development corpus and report the result
as a controlled optimization diagnostic. Do not describe a repeated old-data
fit as new semantic learning or a layout-quality gain. Do not run a parameter
search against the already observed validation set.

## Labels that matter next

The bottleneck is independent examples and actionable intent, not model size.
Add licensed or user-owned editable source families outside the ten reserves.
Aim for 6–10 new families with multiple product brands, logos in varied slots,
inset photos/decorative foreground, full-scene backgrounds and ambiguous roles.
Reusing one bottle across many canvases does not create independent products.

Each selected asset needs its exact source/render/asset hashes, family, role,
reviewer and uncertainty. Add a separate source-and-target crop review: preserve
the complete visible object versus crop permitted, the authored visible interval,
and any reviewed normalized protected region. Include both safe crops and
near-miss crops that remove a logo, face, product edge or text-support area.
Keep these crop labels in a review sidecar; the current closed model response
does not accept new permission fields. Whole-image role labels are not region
detections. Existing roles can only veto new product/logo crops, not authorize
background cover or solve typography packing.

Obtain human review where available, prioritizing product/foreground versus
background confusion and ambiguous cases; double-review a predetermined 20%
plus disagreements. Agent labels may support the urgent diagnostic with clear
provenance. They cannot be relabeled as human truth or satisfy the final human
benefit gate. Human role records are accepted by `dataset.load_records`; the
`prepare_pilot` helper intentionally rebuilds agent pilot labels only. A human
manifest therefore needs its own correctly attributed records.

Split complete source/template/shared-asset families before derivatives. Keep
train, calibration/validation and final test groups disjoint. With too few
families, mark a class unmeasurable instead of reporting a perfect empty recall.
Group-aware splitting is documented by
[GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html);
our asset-family grouping remains a project-specific requirement. The ten
reserves remain outside training, tuning, prompt selection and this experiment.

For richer source context, investigate a selected-element overlay or bounded
neighborhood view paired with the unchanged exact asset and measured geometry.
This makes the selected instance explicit instead of concatenating one global
poster embedding with an asset embedding. Treat it as a separately versioned
analysis input, never replacement artwork. Compare against the complete two-view
control; do not claim this feature or a dense scene adapter already exists.

## Confidence and admission gates

The following are proposed engineering gates to freeze before a new comparison,
not thresholds established by the tiny pilot or vendor documentation.

| Gate | Required evidence |
| --- | --- |
| Recognition | Validation macro recall at least 0.70 and at least 0.10 above both zero-shot and the constant baseline. Product/logo recall at least 0.90, each with at least ten reviewed positives spanning two held-out families. Missing support means not evaluated, not passed. |
| Useful action | Nonzero protection interventions on reviewed eligible cases in at least two independent held-out families. Report denominator, abstention, precision and crop errors; a zero-action system cannot claim model layout benefit. |
| Confidence | Freeze the decision threshold using separate calibration families or family-wise out-of-fold predictions; retain abstention. Do not reinterpret raw softmax as a calibrated probability or tune on the final test. |
| Hard preservation | Zero new missing/added IDs, changed copy/fonts/assets, forbidden crop/distortion or stale source/target/export identity. Any source with missing required bytes stays explicitly blocked. |
| Paired resizing | No new hard or readability regressions in either target mode; strict reviewed visual wins in at least two independent held-out families, with ties/neither retained. An initial pilot win warrants shadow use only. |
| Default promotion | Independent human benefit over the frozen baseline, with family counts, uncertainty and a prespecified non-tie win criterion; use a 95% confidence interval whose lower bound exceeds 0.5 before claiming reliable superiority. If the sample is too small, remain experimental. |
| Practical deployment | Measured cold/warm CPU latency, package size, memory and actual offline operation inside the batch budget. The old 0.327 s warm feature-pair mean excludes renderer/API work and is not an end-to-end guarantee. |

[Official calibration guidance](https://scikit-learn.org/stable/modules/calibration.html)
requires fitting calibration separately from classifier training and explains
that raw classifier probabilities can be poor estimates. It also warns that
isotonic calibration can overfit small data. With the current absent product and
unknown validation labels, neither reliable calibration nor all-role promotion
can be demonstrated. Do not reduce the current 0.8 protection threshold merely
because the old head's maximum was 0.58.

## Existing framework commands

These are templates for an authorized run after the implementation freeze.
Use a fresh output directory for every condition; no command below was executed
to write this plan. Keep the separate verified CPU model runtime:

```powershell
$sceneRuntime = 'C:/Users/edmun/anaconda3/envs/conceptgraph/python.exe'
$sceneClip = Join-Path $env:USERPROFILE '.cache/clip/ViT-B-32.pt'
$sceneSha = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
```

Validate the existing pilot and write a training plan without loading a model or
optimizing. Substitute a newly reviewed manifest for a genuinely expanded run:

```powershell
.venv/Scripts/python.exe -m experiments.scene_model.train_clip `
  --manifest output/scene-model-source-inspection/agent-reviewed-scene-manifest.json `
  --clip-checkpoint $sceneClip --clip-sha256 $sceneSha `
  --out output/resize-iteration/role-head-plan `
  --max-steps 40 --save-every 20 --max-seconds 240
```

For an authorized existing-head fit, use `$sceneRuntime` instead of application
Python, a new `--out`, and add `--execute`. The current trainer already uses
class-balanced full-batch optimization; do not present class balancing as a new
fix. Its CLI supports one bounded recipe and immutable checkpoints. A hybrid
feature head needs its own implemented runner, not this command unchanged.

Current source harness: run baseline and explicit experimental protection on
the same two targets. The existing stable-head configuration is a research
control, not an approved replacement:

```powershell
.venv/Scripts/python.exe -m tools.scene_harness `
  --prepared fixtures/open_posters/prepared/doory_rooftop-event-story `
  --target 540x960 --target 1200x675 `
  --out output/resize-iteration/rooftop-baseline

$sceneCommand = @($sceneRuntime, '-m', 'experiments.scene_model.provider',
  '--config', 'output/scene-model/local-clip-stable-80.json') | ConvertTo-Json -Compress
.venv/Scripts/python.exe -m tools.scene_harness `
  --prepared fixtures/open_posters/prepared/doory_rooftop-event-story `
  --target 540x960 --target 1200x675 `
  --model-command $sceneCommand --model-policy protect `
  --out output/resize-iteration/rooftop-stable-head
```

The larger paired runner already exists at
`output/scene-model-matrix/run_matrix.py`; it supports frozen plan/config,
explicit cases, at most two workers, artifact hashes and fresh output folders.
It currently requires the existing trained-head provider and is an ignored
development artifact, not a packaged public CLI. Coordinate adapting it to a
hybrid provider with its owner. The source harness is the supported seam for a
separate experimental provider command.

The Studio batch interface is useful for multi-scene deterministic smoke tests;
it does not silently enable these experimental models. For the eight-hour
delivery, success is a reproducible batch, honest failure accounting and a
measured experiment, even when the correct model decision remains “do not
promote.”

## Executed recovery experiment

The authorized fixed hybrid task-head fine-tune completed 200 steps in 8.36
worker seconds using the existing CPU runtime. This was one recipe, seed 2026,
learning rate 0.05 and L2 0.01, with train-only normalization and a 600-second
external supervisor. CLIP and the original head stayed frozen. The new head
uses five frozen role logits plus ten measured source-geometry/alpha features.
All 34 records retained their original agent-review provenance and 11/23 split.
Checkpoint reload reproduced predictions exactly. The validation set has been
observed during development; this is a development signal, not a fresh final test.

| Measured outcome | Original head | Hybrid head |
| --- | --- | --- |
| Training correct | 9/11 | 11/11 |
| Validation correct | 9/23, 39.1% | 13/23, 56.5% |
| Validation macro recall | 0.333 | 0.481 |
| Validation background recall | 9/9 | 9/9 |
| Validation logo recall | 0/9 | 4/9 |
| Validation foreground recall | 0/5 | 0/5 |
| Eligible validation product/logo protections at 0.8 | 0 | 0 |

Product and unknown validation recall remain unmeasurable. Loss fell from
1.6078 to 0.0902, but semantic promotion still fails: macro recall is below
0.70, foreground recall is zero, action coverage is absent on validation,
and there is no independent human benefit evidence. The model is not promoted.

The isolated comparison then used frozen engine revision `2369569` in
`output/gui-preview-f5003cc`, with hashes unchanged before/after. Four sources
were evaluated at both target levels, baseline versus hybrid protection:

| Source | Classifier split | Reduced resolution | Aspect change |
| --- | --- | --- | --- |
| Core | Training | 540×540 | 960×540 |
| Rooftop | Training | 540×960 | 1600×900 |
| Bold Sale | Validation | 540×540 | 960×540 |
| openSUSE Booth | Validation | 576×1440 | 1440×576 |

All eight CLI jobs completed and all sixteen actual PNGs passed their exact
pixel-dimension checks. All sixteen outputs passed the current hard contract;
source/resources remained unchanged. All eight baseline/hybrid pairs have
identical native JSON and PNG hashes. The hybrid actually protected Core's logo
and product at both targets: four protected-ID/target incidences, two unique
image-element IDs in Core. It vetoed zero candidates. Other evaluated sources protected zero
IDs. Thus the classifier improved, but the measured resize outputs did not.

Core's reduced output has 9 px minimum copy in both conditions, below this
plan's 10 px review floor. Passing the hard contract does not erase that issue.
The contact sheets expose the unchanged composition; they are agent review
artifacts, not independent human approval. No reserves were used.

Artifacts:

- [Hybrid training report](../output/scene-role-hybrid-20261030-run1/training-report.json)
- [Actual paired comparison](../output/scene-role-hybrid-eval-20261030-run1/comparison-report.json)
- [Independent PNG-size/source checks](../output/scene-role-hybrid-eval-20261030-run1/png-size-verification.json)
- [Core comparison](../output/scene-role-hybrid-eval-20261030-run1/core_01-comparison.png)
- [Rooftop comparison](../output/scene-role-hybrid-eval-20261030-run1/doory_rooftop-event-story-comparison.png)
- [Bold Sale comparison](../output/scene-role-hybrid-eval-20261030-run1/doory_bold-sale-announcement-comparison.png)
- [Booth comparison](../output/scene-role-hybrid-eval-20261030-run1/opensuse_booth_poster-comparison.png)

The local checkpoint remains ignored at
`output/scene-role-hybrid-20261030-run1/published/checkpoint-000200/model.json`,
SHA-256 `f9fdd57b1933edf4a3ff26b5bc245966e3e4e0b5cf488b89f7db08d6a2f12ae4`.
The isolated provider entry point is implemented in
[tools/scene_role_probe.py](../tools/scene_role_probe.py); it validates explicit
model/dependency identities and source/asset receipts. Its training command is
`train --out <fresh-output-directory> --execute`; its inference command is
`predict --model <model.json> --model-sha256 <exact-sha256>` using JSON stdin.
This executed run already satisfies the one-run budget; these commands are for
reproduction, not an instruction to start another tuning run.
