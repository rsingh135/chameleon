# Testing one method across different scenes

The viewer provides two layout-testing interfaces:

| Interface | Address | Purpose |
| --- | --- | --- |
| Single scene | `http://127.0.0.1:8765/` | Inspect one source and its real native result, compare views, inspect element geometry, and export the selected result. |
| Batch benchmark | `http://127.0.0.1:8765/benchmark.html` | Apply the same method and shared target canvases to multiple scenes, then inspect the source/result matrix and its structural diagnostics. |

The source+A+B annotation tool, normally on port 8770, serves a different job:
collecting grounded human preference labels. It does not train a model merely by
saving feedback, and it is not another layout-testing mode.

## Start and run

From the repository in PowerShell, start the local server if it is not running:

```powershell
.\.venv\Scripts\python.exe -m viewer.server --port 8765
```

1. Open **Batch benchmark** and select several JSON designs or PNG/JPEG images.
   The file picker accepts multiple files. Native JSON may be a source design,
   task, or portable document bundle containing its asset/font resources.
2. Give each scene a meaningful **Source family**. Variants of the same template
   or campaign belong to one family, even if their text or colors differ.
3. Enter one shared target per line, such as `1080x1080`, `1600x900`, and
   `600x1200`. Every scene receives every selected target; an imported task's
   previous target does not override this matrix.
4. Run the benchmark. Jobs run sequentially and completed results appear with
   their source render. **Stop after current job** preserves completed jobs and
   marks the remainder unrun; it does not interrupt the active engine operation.
5. Inspect the real renders and diagnostics, then download the run report and
   useful per-result JSON/PNG files. Retain the original input files beside the
   report. The browser queue is in memory and is not a persistent dataset.

Limits are eight scenes, four target canvases and sixteen jobs per run. For
example, eight scenes by two targets or four scenes by four targets fits the
limit; eight by four does not. Targets are at most 8192 pixels per side and
20 megapixels. Raster uploads are at most 8 MiB; JSON files and submitted
document requests are bounded to 24 MiB.

Editing scenes, family labels, targets or evaluation purpose marks displayed
results as belonging to the previous setup and disables their downloads. Run
again to obtain results for the new setup. A timed-out request may still be
finishing on the local engine; the interface stops submitting further jobs.

## What multiple images test

Editable JSON preserves the individual source elements and their exact content
and resources. It exercises the current layout processor's ability to move and
scale those elements while retaining the source contract.

A PNG/JPEG has no editable scene structure. It becomes **one image element** with
the original pixels and aspect ratio. Its result is a whole-image fitting test.
It cannot demonstrate that a model detected objects, resized internal text,
preserved semantic groups or recomposed the image. Vision-based scene analysis
and editable reconstruction are not enabled by this upload path.

The **Add bundled smoke scenes** button supplies a convenient pipeline check.
Those examples share a campaign and must not be counted as independent visual
families or proof of generalization.

## Same method and reproducibility

The only implemented benchmark method is `deterministic-current`. The vision
analyzer and learned critic are explicitly disabled; running the benchmark does
not download a model, train anything or use Brev GPU credits.

The method receipt fingerprints the engine and renderer Python sources, resize
bridge/harness implementation, Python version and relevant installed package
versions. Each job checks the expected digest and records canonical source and
resource hashes. The server refuses changed implementation files until it is
restarted. Start a new benchmark after changes rather than combine results from
different methods. The receipt is not a container image, dependency lock or
proof that all environmental differences have been eliminated.

The downloaded report records the method receipt, scene/family declarations,
shared targets, evaluation purpose, per-job identities, outcomes, native renders,
diagnostics and timings. Interrupted and deliberately stopped runs retain their
completed evidence and final status. Keep failures and missing jobs visible when
comparing runs; compare matched scene-target jobs rather than only successes.

## How to investigate generalization

1. Assemble an approved corpus with genuinely different template/source
   families: different hierarchy, density, product prominence, typography,
   grouping, background treatment and authored overlap. Add both ordinary and
   extreme target aspect ratios. Keep any reserved challenge material untouched.
2. Split by family before training or prompt tuning. Keep near-duplicate scenes,
   campaign variants and derivatives in the same split. Freeze a holdout manifest
   and record where the inputs came from; the interface's **Declared holdout**
   setting records your declaration and does not verify training exclusion.
3. Freeze the method and target matrix. Use the same inputs and targets for a
   deterministic baseline and each proposed model configuration. Tune on training
   and validation families, then use untouched test families once the recipe is
   fixed.
4. Review source and actual target renders. Judge hierarchy/attention, reading
   relationships, group attachment, relative prominence, readability, source style,
   and useful space distribution. Record constraint failures, infeasible cases,
   fallback behavior and visual regressions. Human A/B reviews should conceal
   method identity and counterbalance the candidate sides.
5. Report outcomes by held-out family and target aspect, including failures and
   unrun jobs. Structural checks passing means the checks passed; it is not a
   designer's approval, an attention measurement or evidence of learned vision.

The current family summary counts rendered jobs and structural-contract passes.
Visual quality is explicitly **unreviewed**. A small local matrix can expose
failures and guide collection; a generalization claim needs enough independent
held-out families and source-aware judgments to support it.

## Planned connected-model experiments

The intended future path is source render plus exact JSON → validated intent
graph → constrained native layout candidates → source-aware critic → hard
revalidation. One pinned vision base can initially use separate analyzer and
critic prompts; separate task adapters are an experiment after appropriate labels
exist. Native assets and geometry remain under the deterministic contract.

For each such experiment, the benchmark recipe must also freeze the base model
revision, adapter hashes, analyzer/critic prompt versions, inference settings,
intent-graph policy and fallback/admission rules. Apply that same recipe to every
scene in the matrix. Compare deterministic-only, prompt-only and fine-tuned
configurations separately. These configurations are planned, not implemented by
the current benchmark. See [the scene-model training plan](SCENE_MODEL_TRAINING_PLAN.md)
for source analysis, data requirements and promotion gates.
