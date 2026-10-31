# Visual model execution and layout evidence

The opt-in pipeline in `engine.visual_pipeline` supplies rendered pixels to a
local vision-language model, compiles its bounded plan choice into candidate
geometry, admits candidates through the engine's existing guards, and can ask
the model to compare two actual rendered alternatives. The model never writes
design JSON, edits asset pixels, executes source text, or bypasses admission.

Execution is not proof of useful visual reasoning. The trace separates a model
call, a valid response, new geometry, admission, final selection, and independently
reviewed appearance. A baseline result is an acceptable fallback and must remain
visible as a no-op.

## What the previous local model did

The previous Studio and scored entry points called the deterministic engine
directly: **zero model calls in those normal paths**. The optional scene harness
could render the source and call a frozen CLIP image encoder plus a small linear
role classifier. Its five asset-role classes could protect sufficiently confident
product/logo images from newly introduced crops. It received no requested target
or candidate renders and could not propose placement, grouping, or reorganization.

The original recorded matrix contained 42 renders / 21 target pairs, 10 optional
model calls and 34 image-role predictions. No prediction reached the crop-veto
threshold; no protection, veto, final design change, or PNG change was observed.
The subsequent hybrid probe's eight checked pairs also retained identical final
designs and PNGs. Those are observations about those reports, not evidence of
visual layout reasoning or a general model-quality claim.

## One shared production path

`resize_configured(source, target, ..., config=None, strict_stage_one=True)` is
the production dispatch. The audit harness calls the same phase implementation,
`evaluate_visual_resize`; it does not maintain a separate model-enabled engine.
Callers must use the configured dispatch for the opt-in path to run. Merely
installing model weights does not activate model calls.

1. Load the original design and its existing assets/fonts. Record exact source,
   asset, font and implementation hashes. Render the **original source first**.
2. Collect the actual deterministic candidate snapshot, including its admitted
   finalists, final baseline design and native PNG.
3. Build a source/resource/relationship/code/target-bound planning menu. The model
   receives the source image, target dimensions, semantic units, and a finite menu
   of supported transformations. Source text is data. Its text summary is bounded
   explicitly; the stored source JSON and rendered image remain complete.
4. Execute the local provider. It selects a known menu ID. Trusted code binds that
   ID to the request and compiles its transformation into a new candidate. Compare
   complete design hashes against the original candidate pool so a renamed or
   duplicate candidate cannot count as new geometry.
5. Re-run the same hard, safety and fidelity admission. A generated candidate can
   be rejected. The report lists both new candidate IDs and admitted new IDs.
6. If two distinct safe alternatives exist, render both. The critic receives three
   images: the original source, alternative A, and alternative B. It may select
   only those admitted IDs. One safe alternative records `no_choice` and does not
   manufacture a critic call. Disabling the critic is recorded separately.
7. Validate the selected result again, save its native PNG and design/state JSON,
   and verify source, resource and code immutability. An invalid selection retains
   the contract-valid baseline. Changing code during inference makes the evidence
   incomplete; production does not return that run as a successful visual result.

The compiler can produce new geometry from a model's supported plan choice; this
is more than reranking a deterministic pool. The scope is still constrained: the
model selects authored transformation families rather than inventing arbitrary
coordinates, semantic edges, replacement artwork, or new copy. Its rationale is
recorded evidence of output, not a verified explanation of its internal process.

Exact-aspect resolution changes use the resolution path and do not ask the model
to reorganize the composition. Shape changes can use the planning and critic
stages. The configured default enables the strict stage-one profile: every rotated
frame must remain inside the target canvas, including source-cropped decoration.

## Activation and evidence

Set `CHAMELEON_VISUAL_PLANNER_CONFIG` to an operator-owned local JSON file, or pass
that configuration explicitly to `resize_configured`. Configuration is not
accepted from a poster or model response. The fixed provider module runs as an
argument array with `shell=False`; its input, output, wall time and owned process
tree are bounded. Normal unconfigured requests import no model library.

The outer configuration accepts `schema_version`, `enabled`, `provider_command`,
`timeout_seconds`, `artifact_root`, and `critic`. The command must have this form,
using existing local absolute executable/configuration paths:

```json
{
  "schema_version": 1,
  "enabled": true,
  "provider_command": [
    "ABSOLUTE_LOCAL_PYTHON",
    "-m",
    "experiments.visual_planner.provider",
    "--config",
    "ABSOLUTE_LOCAL_PROVIDER_JSON"
  ],
  "timeout_seconds": 240,
  "artifact_root": "output/visual-production",
  "critic": true
}
```

The example contains placeholders; use the installed, pinned local runtime and
provider configuration. Relative paths resolve from the outer configuration file;
set `artifact_root` to the project's absolute `output` subdirectory when that
configuration lives elsewhere. Provider configuration/manifest/code/runtime identities
are recorded before dispatch. That lightweight identity explicitly does not claim
to verify model binary bytes; the provider's execution receipt verifies actual
loaded model resources. Model responses are never served from a success cache.
Each configured request creates a distinct evidence directory.

`state.visual_model` and `report.json` distinguish these fields:

| Field | Meaning |
|---|---|
| `model_executed` | A returned execution receipt attested actual model execution. |
| `proposal_valid` | The planner returned a closed, known, request-bound choice. |
| `new_candidates_generated` | At least one complete design was absent from the original pool. |
| `admitted_new_candidate_ids` | New geometry survived the actual admission rules. |
| `model_candidate_selected` | The final selected design is one of those new designs. |
| `layout_changed` | Final JSON differs from the deterministic baseline design. |
| `critic_changed_selection` | A valid critic changed the model-augmented pool's winner. |
| `same_png` | Native baseline/final PNG bytes match. |
| `quality_improved` | Explicit image-bound review verdict; `null` until reviewed. |
| `fallback_reason` | Why a result retained the baseline or a stage could not run. |

Execution without a returned receipt is not claimed as proven model execution.
An output can be contract-valid while still looking poor. Proxy scores and valid
JSON do not set `quality_improved` automatically.

The CLI `python -m tools.visual_resize_harness --help` supports original task JSON,
prepared editable posters, and imported editable bundles, explicit target sizes,
an explicit local provider argv, and an evidence directory. It uses the existing
`scene_harness.load_input` resource path. The ten personal-test reserve posters
are excluded before loading their pixels. Use development/training examples for
iterative model experiments.

After viewing source, baseline and final PNGs, `review_existing` can attach a
separate `human_review` or `agent_visual_review`. A review must bind the exact
source/target and both design/PNG hashes. It cannot establish improvement from
identical PNGs or approve mutable-source evidence.

## Acceptance and current limits

The initial real-engine activation smoke used an explicitly marked **test stub**
on Core at 1600×900. It generated one new design, admitted zero new designs, and
retained the baseline design and PNG. The critic correctly skipped its sole
safe choice. This proves the integration/admission trace, not model usefulness.

As of the first native CPU pilot, the real 500M VLM can execute in roughly 6–13
seconds at about 803–813 MiB in that runtime. Initial planning controls all chose
the first letter, including a blank-image and target-swap control; reversing the
menu changed the selected plan ID. Therefore useful image-conditioned planning
was **not established**. Early three-image CLI binding also required correction
and repeat validation. Consult each run's actual receipt and controls rather than
treating provider installation or these timing measurements as quality approval.
The provider remains experimental and off by default.

Before claiming the model helps resizing, run and review all of these:

- Development sources: Core, Rooftop, Sale, and a complex multi-product/panel
  source. For each, test exact-aspect resolution scaling and a changed aspect.
- Actual receipts binding the original image, target, code, model resources, and
  both candidate images whenever the critic executes. No stub receipts qualify.
- Source-image ablation, target swap, and counterbalanced candidate order, using
  the harness helpers. Compare selected **IDs**, not letters. A change alone is
  not success; the response must make visual sense for the changed input.
- At least one actual model-generated, admitted, selected design that differs
  from the baseline, without forcing a different winner to inflate contribution.
- Original/source/baseline/final inspection of product prominence, background
  coverage, copy order, semantic grouping, badges, readability and visible ink
  collisions. Record regressions and no-ops as well as improvements.
- Fixed source/assets/fonts/copy/IDs, unchanged image proportions, all frames
  in bounds, and final native rendering at the exact target dimensions.

The challenge requires intentional adaptation across formats; it does not make a
model call itself the quality criterion. A local model is optional in the supplied
brief. If a submission relies on adapted weights, confirm the approved model/size
rules and provide the required public reproducible weights. Offline scoring also
requires the correct Linux runtime, dependencies, provider and weights to be
baked into the image at build time. A working Windows local pilot does not verify
that deployment, the official judge, or native Adobe-host behavior.
