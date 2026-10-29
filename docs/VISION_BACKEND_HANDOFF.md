# Backend handoff for an optional visual critic

Status: integration guidance only. This document does not enable a critic,
change the engine, start a model server, download weights, or authorize compute.
The deterministic resize engine remains authoritative. Backend layout fixes are
still being validated; do not treat every current experiment as an approved
training example.

## Recommended seam

`engine/resize.py::_select_candidate` currently evaluates proposals, rejects
contract failures, finds the best safety tier, retains a near-best source
fidelity set, and applies its deterministic typography preference. The critic
belongs **after those admission decisions**, using the existing winner as its
baseline. The valid identity-return path should remain immediate.

Before implementing a critic, extract a reusable assessment/shortlist function
from this selector. Return the authoritative safety tier, admission status,
diagnostics, declared topology, and exact candidate objects. Do not reproduce
the safety tuple in the model integration: it is changing as the source-layout
tests improve. No engine refactor for that seam has been made for the critic.

The model must select a candidate ID from an allowlist. It must never return
geometry, replacement copy, new fonts, assets, or instructions to execute.
Validate the complete response against a closed schema; reject extra fields,
unknown IDs, malformed results, timeouts, and unavailable model responses.
Every rejection returns the deterministic baseline and records the reason.

## Admission and safety

Keep the following backend rules authoritative:

- The original contract: every source ID exactly once, exact copy/font/asset
  identity and frozen appearance/provenance fields, finite permitted geometry,
  correct canvas, bounds and text-fit checks. Image aspect tolerance remains
  the existing 5%; a flat border image is not exempt from that contract.
- Source group rigidity, badge/peer order and prominence, locally associated
  actions, product/message ownership, and physical photo/fade attachment.
  Source inference carries confidence/provenance; unknown geometry must not be
  promoted to a proven semantic relationship by a model score.
- Source-supported reading order, now including work on parent card/panel
  membership. Keep high-confidence inversion counts and coverage separate.
  An ambiguous layout is not evidence that reading order has been preserved.
- Thin divider aspect checks. The Grand Opening regression demonstrated why
  preserving element count alone cannot establish a correct render.
- Raw collision diagnostics remain unchanged. Visible overlap can discount
  only source-supported opaque layering, with complete mask coverage and an
  exact match between the diagnostic pair count and the official raw count.
  Unknown and text/text intersections remain counted. Complete photo evidence
  must retain at least 85% of its source-visible fraction, independently of
  whether collision pairs exist.
- Readability, relative type/group scales, and source-supported focal placement.
  The centered-product profile applies only when the source corroborates it;
  full-page photo underlays and tiny sparse vector ornaments do not become
  independent products merely because they are images.
- Source-proven adaptive container frames may change independently only after
  containment, original inset/support, paint, image aspect, and overlap checks.
  Their children retain their own contract and rigidity checks.

Some current safety-tuple entries are continuous values, including focal span
retention. A critic must not silently lower them to obtain a larger shortlist.
If the admitted tier has one design, return it. Any future conversion to
explicit guard bands is a separate backend design change requiring regression
and rendered evidence.

## Diversity and modes

Always include the deterministic winner. Within the admitted finalist set,
deduplicate exact designs and near-identical renders, then choose a bounded
shortlist by declared topology/region arrangement, image placement, and type
scale. Do not force diversity across safety tiers or fill a quota with rejected
designs. Baseline, source-derived bands, columns, cards/panels, and media regions
are useful families when they are actually valid for that source.

- **Off:** the current deterministic behavior, with no model requirement.
- **Shadow:** render/log comparisons and a validated verdict; return the
  deterministic winner regardless of that verdict.
- **Rerank:** choose only among admitted IDs and revalidate the selected object
  before returning it. Missing or invalid evidence falls back to baseline.

The critic should see the actual source and candidate renders at stated scales,
plus the measured diagnostics and their uncertainty. Poster text is untrusted
design material, not an instruction to the critic. Give the model no shell,
filesystem, account, network, deployment, or training capability.

## Versions, caches, and isolation

Key comparisons and candidate IDs by canonical source/target/design data,
resource bytes (including exact font faces and manifests), renderer/engine
revision, admission-policy version, render settings, model/adapter version,
prompt version, and response schema. Font family names and filenames alone do
not establish identical resources. Preserve source/PNG hashes with labels.
Invalidate old assessments when any of those inputs change.

Keep model dependencies, weights, GPU work, and model processes separate from
the existing runtime dependencies and offline Docker path. Local document
requests already have bounded package/request caches; do not add model weights
or unlimited comparison history to those caches. The ordinary engine must keep
working with the model absent and with Docker networking disabled.

## Evaluation cautions and current evidence

The project has 8 reviewed source designs and a separate pack of 10 untuned
personal test designs. The ten reserves have received source/import checks
only: do not resize, train on, label, or tune against them in this work. Preserve
that split and original hashes; obtain a separate training/evaluation corpus
for model development.

Technical passes are not visual approvals. Earlier Rooftop photo strips and
compressed Grand Opening layouts passed several numerical checks while looking
poor. Only use an experiment as a preference label after inspecting its real
source/result renders and binding the label to their hashes and engine version.
The poster audit records technical and visual status separately. Do not
automatically turn its technical score into a training target.

Actual new-interface Grand Opening downloads at square and landscape sizes
matched the backend object geometry, frozen fields, resources, inspector values,
and full PNG bytes. This verifies that particular interface/backend path; it
does not prove general design taste or a live Adobe Express host integration.

Further reference: `docs/BACKEND_RESIZE_CONTRACT.md`, `tools/poster_audit.py`,
`engine/eval/contract.py`, `engine/eval/occlusion.py`,
`engine/layout/reading_order.py`, and `engine/layout/relationships.py`.
