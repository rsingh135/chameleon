# Local source-scene inspection

`tools/scene_harness.py` records the deterministic scene analysis when a design
is loaded, renders its actual source, and optionally renders requested resize
targets. It does not start a server, download models, alter author originals,
or enable a model in production selection.

An operator can now explicitly connect a source-bound local prediction to the
resize backend with `--model-policy protect`. This only vetoes new crops of
products/logos classified with confidence at least 0.8; it never creates a
background cover permission. The ordinary viewer and default CLI remain on
their existing deterministic policy.

Run from the repository root:

```powershell
.venv\Scripts\python.exe -m tools.scene_harness `
  --prepared fixtures/open_posters/prepared/doory_rooftop-event-story `
  --target 540x960 --target 1600x900 `
  --out output/scene-rooftop
```

Other supported inputs are a source JSON with explicit resource directories,
or a portable **JSON** bundle containing source, assets, fonts and document
metadata. The bundle path reuses `viewer.document_bridge.prepare_document`,
including its exact font, image byte, size and path validation. ZIP files are
not extracted. A source task wrapper with a `source` field is also accepted.

```powershell
.venv\Scripts\python.exe -m tools.scene_harness `
  --source input/chameleon_sample_level2_core.json `
  --assets-dir brand_kit/assets --fonts-dir brand_kit/fonts `
  --out output/scene-product

.venv\Scripts\python.exe -m tools.scene_harness `
  --bundle fixtures/open_posters/prepared/doory_rooftop-event-story/bundle.json `
  --out output/scene-rooftop-bundle
```

Source-only inspection is the default: no target means no resize call. This is
also the mode to use for untouched reserve examples. Targets must be explicitly
requested, at most six per run, within the local backend's 8192px/20MP limits.
Output must remain separate from the original input directory/files.

## Evidence and authority

`engine.scene.analyze_scene` owns classification and crop permission. Its
versioned result binds the source digest, individual image IDs and asset hashes,
roles, confidence, physical source evidence, support IDs and `allow_cover`.
An independently verified fade can additionally have role `background_support`
and `allow_support_reflow:true`, while its `allow_cover` remains false. This
permission belongs to the deterministic source proof; it allows the complete
original gradient to continue supporting live copy at the new canvas geometry.
It does not authorize stretching unrelated artwork. The profile is exported
intact, including these separate support fields.
The harness writes:

- `source.json` and `source.png`: the source and its real local native render.
- `scene.json`: deterministic source analysis, including its scene digest.
- `model-request.json`: source render path/hash, original element metadata,
  local asset paths, deterministic profile and review response contract.
- `model-review.json`: optional validated suggestions or the rejection reason.
- `WIDTHxHEIGHT.plan.json` and `.png`: the deterministic engine result and its
  actual local render for each requested target.
- `report.json`: source/resource hashes, source immutability, selected states,
  contract results, render hashes and the model sidecar status. It hashes all
  `engine/`, `renderer/converter/` Python files and this harness at the start and
  verifies them again at the end. `implementation_unchanged:false` means the run
  is incomplete as fixed-revision evidence and causes a nonzero CLI exit.

The source dictionary and resource bytes are checked before/after inspection.
A failed resize is recorded explicitly; no substitute preview is generated.
`contract_valid` does not mean a design looks good. The tool never assigns
visual approval and never claims an Adobe Express host round trip.

## Optional local model proposals

`--model-policy review` is the default. With `--model-policy protect`, a validated
response is passed to `engine.resize.resize_with_state(..., scene_proposal=...)`.
Every target state records the proposal digest, protected IDs and candidate veto
count. Low confidence and unknown roles abstain; background predictions only
corroborate the independent source/pixel policy. Model inputs have a separate
result-cache key and are rebound to current asset bytes before cache access.
Invalid responses fall back to the ordinary deterministic path in the harness.
The public resize API rejects invalid/stale supplied proposals explicitly.

The protection retains each product/logo's authored crop, image proportions,
and image position while allowing uniform scaling and cluster translation.
Protected-region suggestions remain review evidence: the current classifier
does not measure object boxes. Confidence is a pilot score, not a calibrated
probability or proof of semantic correctness. See the local scene-model pilot
instructions for the actual provider, training and evaluation status.

No model is installed or started automatically. A review response can be loaded
with `--model-response response.json`, or an explicitly chosen local program can
receive the same request on stdin and return JSON on stdout:

```powershell
.venv\Scripts\python.exe -m tools.scene_harness `
  --prepared fixtures/open_posters/prepared/doory_rooftop-event-story `
  --model-command '["C:/local-model/python.exe", "C:/local-model/inspect_scene.py"]' `
  --model-timeout 60 --out output/scene-local-model
```

The command is a JSON argument array with no shell expansion. It is never built
from poster text or model output. The subprocess receives offline Hugging Face
flags, has a maximum 60-second timeout and 256KiB stdout limit; stderr is not
retained. The harness itself uses no network client. These controls do not
sandbox an executable chosen by the operator; use a trusted offline program.

The response is a closed review schema:

```json
{
  "schema_version": 1,
  "source_digest": "copy exact value from model-request.json",
  "scene_digest": "copy exact value from model-request.json",
  "images": [{
    "element_id": "an existing source image ID",
    "asset_sha256": "the exact bound asset SHA-256",
    "proposed_role": "background",
    "confidence": 0.9,
    "protected_regions": [{
      "kind": "face",
      "box": [0.2, 0.3, 0.4, 0.2],
      "confidence": 0.8
    }],
    "rationale": "Optional short explanation for human review."
  }]
}
```

Roles are `background`, `product`, `logo`, `foreground`, or `unknown`. Region
kinds are `face`, `person`, `product`, `logo`, or `other`. Boxes are normalized
`[x,y,width,height]` coordinates inside that exact source image asset. They do
not refer to canvas coordinates or the original uncropped photo. Each image has
at most 32 regions and a response at most 256 in total. Unknown IDs, changed
asset hashes, duplicate images, nonfinite coordinates, escaping boxes and
unexpected fields are rejected. A model cannot return `allow_cover`,
`allow_support_reflow`, a command,
design geometry, source text edits or replacement assets.

Poster text is carried as quoted source data, never instructions. Even a valid
high-confidence proposal is **review only**: it is not fed into scene analysis,
crop/support permissions, resize generation, admission, ranking or native writeback.
A malformed/unavailable model leaves the deterministic baseline intact and is
recorded as rejected. Model suggestions have not thereby become verified face
detections or aesthetic assessments.

## Rooftop audit and native crop boundary

The source Rooftop photo fills its 1080×1920 canvas, is opaque, and supports live
copy over an authored fade. Its prepared image is a source-cropped bitmap with
`fit:fill`. The earlier media-column/fade-stack states kept that entire portrait
bitmap at its aspect, leaving a 506.25×900 image on 1600×900 and a 607.5×1080 image
on 1080×1080. Recognizing `image_underlay` in the graph alone did not change that
placement behavior.

A verified-background policy must distinguish this scene support from a
foreground product, logo or transparent cutout. A target-filling proportional
crop must preserve source image bytes, protect the significant subject region,
and independently check photo/fade attachment and text contrast. Covering the
canvas does not prove that faces survive or that copy remains readable.

The prepared bitmap is 1080×1920; the retained original photo is 2000×1410.
Recovering content outside the authored portrait slot would require an explicit
original-media/crop mapping. The harness does not silently replace the prepared
asset with that original or label a newly flattened poster as editable.

The initial default-center crop clipped the central person's hairline in the
wide target. The current source analysis can use a measured quiet upper region
to adjust the normalized crop position. This is physical pixel evidence, not
face or subject detection; no semantic protection is claimed.

The demonstrated 540×960 run selects `source-guides/source-fit` and keeps a
uniform 0.5 geometry/type scale. The 1600×900 run selects
`background/cover-caption-0.28`, fills the complete background frame, and uses
`image_position:[0.5,0.4420357560568086]`. Its native render shows central hair
headroom, readable lower copy and no sidebars. Both runs pass the current
contract with unchanged source/resources and fixed implementation hashes.
These observations apply to the exact inspected PNGs, not every photo or aspect.
The local model was not requested; this is deterministic scene analysis.

The v0.9 image schema has `contain`, `cover`, and `fill`; the local renderer
implements proportional center-cropping for `cover`. The backend's verified
background policy, rather than model confidence alone, must authorize any
departure from its otherwise frozen source fit/aspect rules. A source-derived
semantic protected-region offset would need separate admission and evaluation.
The current native `image_position` extension already shares an explicit crop
transform across local painting, geometry diagnostics and output JSON. Its
source-pixel headroom proxy is not a learned protected-region detector.

Adobe represents a crop as a native media rectangle and mask. Its documentation
describes changing their geometry for cropping; this is a useful future mapping,
not proof of compatibility with our current add-on. The current adapter refuses
cropped image/mask bounds and writes only whole-root similarity transforms.
Native crop support needs explicit import, write-plan, verification and rollback
tests before being described as host verified. See Adobe's
[MediaContainerNode](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/media-container-node)
and [ImageRectangleNode](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/image-rectangle-node)
references.
