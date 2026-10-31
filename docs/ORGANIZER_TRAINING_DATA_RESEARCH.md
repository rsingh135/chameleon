# Organizer training data: corresponding layouts and admission gates

Primary-source research for the current collection sprint. No downloads,
crawling, new dataset, rendering, model calls or training were performed for
this note. The canonical collection manifest remains owned by the main task.

The strongest verified correspondence is authored responsive HTML: one document
has different layouts at its breakpoints. This research did not establish a
ready-made permissively licensed corpus of same-content editable poster/SVG
retargeting pairs. Similar themes and different canvas dimensions do not prove
that two records are transformations of the same work.

## Candidate sources

| Source and status | Actual pair evidence | Preservation and limits |
| --- | --- | --- |
| **MDN — accept for a small reflow pilot, pending normal admission checks.** [Exact HTML/CSS](https://github.com/mdn/css-examples/blob/main/learn/rwd/type-rwd.html); [CC0 license](https://raw.githubusercontent.com/mdn/css-examples/main/LICENSE). | The same DOM changes at 600 px from stacked blocks to a 1:2 grid; its heading changes from 2rem to 4rem. These are author-written alternate viewport states, not an existing screenshot-pair dataset. | Copy, font-family declarations and the two column wrappers persist. No image, script or downloaded-font dependencies appear in this file. Actual resolved font bytes still need pinning. Useful for wrapping, hierarchy and block reflow; little rich visual variety. |
| **Foundation for Emails — accept its text-only grid as a structural pilot.** [Official example and small/large sizing](https://get.foundation/emails/docs/grid.html); [MIT license](https://raw.githubusercontent.com/foundation/foundation-emails/develop/LICENSE.md); [grid implementation](https://raw.githubusercontent.com/foundation/foundation-emails/develop/scss/grid/_grid.scss). | The documented One / Two / Three example uses the same columns with `small-12 large-4` sizing: stacked mobile content versus three desktop columns. Row/column nesting gives explicit lineage. | Retain all three strings and their semantic row/column membership; distinguish generated empty expander cells from semantic content. This example needs no image assets. Full marketing templates are not automatically admitted: [their source](https://raw.githubusercontent.com/foundation/foundation-emails/develop/templates/marketing.html) references external placeholder images. Retain the MIT notice. |
| **LICA — conditional source analysis; not verified organizer pairs.** [Repository README](https://github.com/lica-world/lica-dataset); [CC BY 4.0 license](https://raw.githubusercontent.com/lica-world/lica-dataset/main/LICENSE). | The authors describe 1,183 layouts with ordered components and template-level themes producing multiple slide variations. `template_id`, layout index and width/height support candidate discovery, but do not establish same-content resize correspondence. | The documented TEXT, IMAGE and GROUP records expose typography, asset URLs, transforms and clip paths. The examples do not prove shared element identity, identical copy, bundled font bytes or original asset rights across layouts. Credit, license link and change notices are required by the stated license. |

At inspection time, LICA's [official Hugging Face viewer](https://huggingface.co/datasets/lica-world/lica-dataset)
reported dataset generation failure with a missing image. This is an availability
problem, not evidence that every record is absent or unusable. Do not bypass it
by treating metadata or a template theme as a visually reviewed pair.

## Correspondence and rights gates

These are proposed project admission checks, not claims that any candidate has
already passed them:

1. **Pin provenance.** Record an immutable source revision, exact selected files,
   author/source URL, license text/hash, required attribution and intended use.
   A repository or curator license alone does not establish the rights to every
   linked image, original artwork, trademark or font.
2. **Prove one work at two layouts.** Require unchanged canonical content and a
   bijective semantic element map. For HTML, stable DOM paths or added stable
   identifiers must come from one frozen source, not unrelated templates.
   For LICA, shared theme, layout index, visual resemblance or a same-size image
   is insufficient: verify the actual copy, assets and component correspondence.
3. **Bind resources separately.** Record every image/font byte hash and its
   applicable permission. Preserve font family/style/weight and resolved faces;
   do not silently replace missing fonts or external images. Preserve original
   copy, asset references/pixels, stacking, opacity, groups and intentional clips.
4. **Verify visibility and frame coverage.** Both views must retain every required
   semantic element. Reject disappearance through media queries, overflow,
   off-canvas placement, clipping, crop changes or an unrecorded hidden state.
   Keep implementation-only cells separate from document content.
5. **Bind visual review.** Store the exact original/result render hashes,
   dimensions, renderer/runtime, resource receipt, source/target geometry and
   review origin. Agent review and algorithmic teachers remain weak supervision;
   uncertain intent is masked rather than learned as a negative label.
6. **Split by work and family.** All viewport variants, template siblings and
   derived negative examples of one work stay in one split. Hold out original
   families and use source-aware human review before claiming taste improvement.

## CSS reflow is a different target from fixed-canvas retargeting

Responsive documents can grow vertically when columns stack. MDN/Foundation
can therefore supervise authored reading order, wrapping and group arrangement,
but do not establish that all content fits a chosen fixed-height poster/banner.
For a future pilot, record natural full-document bounds separately from viewport
dimensions; a clipped viewport screenshot cannot serve as a complete target.
Pairs requiring scroll height belong to a reflow task until independently
verified against the fixed-canvas contract.

## Crello is not an automatic rights shortcut

The [official curator card's licensing section](https://huggingface.co/datasets/cyberagent/crello#licensing-information)
states that the distributor does not own the original template copyrights,
requires agreement to VistaCreate's license agreements, and does not redistribute
the original files because of the terms. The dataset's CDLA-Permissive-2.0 label
does not by itself clear the underlying artwork for this collection. The same
card also warns that supplied renders need not reproduce the original designs.
Crello remains outside the immediate v2 paired-source pilot pending explicit
rights and fidelity review; no license conclusion is inferred from our renderer.

## Current critic evidence and a future authored-reflow collection

The current critic-v1 data uses **five existing licensed training works**, with
algorithmic fit candidates and controlled negative transformations. These are
not designer-authored before/after pairs. They can test failure discrimination
and source-preservation priors; they do not establish supervised design taste,
human preferences or unseen-family generalization.

The next nonblank comparison evidence uses the project's three existing
synthetic TRAIN document families: Materials Brief, Guild Bulletin and Garden
Plan. Their narrow source-frame variants preserve frame similarity but boost
text size; they are not pure uniform scaling. New native fitted-column views
can be compared under an explicit minimum-font/fit/relationship weak rule.
Operations Digest remains validation and is excluded from this collection.
The project has no public license file for these originals: record internal
user-authorized synthetic training provenance separately from the actual
Poppins/OFL font permission, rather than inventing a public redistribution
license. Keep historical HEX documents unchanged; any equivalent numeric-RGBA
derived source needs separate parent/derived identities and an exact color-only
transformation audit.

For a separate authored-reflow pilot, first propose only
**two canonical documents / four viewport states**:
the MDN file and Foundation's text-only grid, with one width on each side of
their authored breakpoint. Admit through the existing canonical manifest after
the gates above; do not create a parallel dataset or bulk-fetch a repository.
Keep this pilot explicitly structural and variable-height where appropriate.
This proposal is separate from critic-v2, which diversifies weak negatives from
the existing five frozen training works; it does not rename or enlarge that corpus.

The other collection chat owns a separate, frozen-family rare-role pilot of
agent-authored originals using individually verified CC0 artwork/photographs
and OFL fonts. Those are new authored training sources, not designer resize
pairs. Keep purpose separate from placement: an inset or full-bleed photograph
may be the primary visual; full-bleed geometry alone does not establish a
background purpose. Review any visually indistinguishable conflicting role
labels as ambiguous instead of treating authored intent as visible evidence.

Then seek directly licensed, designer-authored fixed-canvas versions with the
same copy, assets, fonts and semantic groups. Use LICA only for a bounded
candidate inspection after availability/resource checks, and admit a pair only
when actual lineage and unchanged content are demonstrated. If no pair passes,
retain it as source-analysis evidence rather than inventing organizer targets.
