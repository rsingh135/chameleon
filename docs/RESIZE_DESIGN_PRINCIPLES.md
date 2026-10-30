# Source-preserving resizing: principles, evidence and batch assertions

Research checked October 30, 2026. This note combines primary-source design
guidance with a read-only review of existing local artifacts. The proposed
assertions and candidate experiment below are engineering recommendations, not
claims that a paper validated this engine or that they have already passed.
No engine, renderer, source or reserve was modified or rerendered for this note.

## What the research supports

[Adobe's layout guidance](https://www.adobe.com/learn/illustrator/web/layout-basics)
describes negative space, grouping, repetition, contrast, alignment, focal emphasis
and hierarchy as ways to communicate a message. Related event details should
remain understandable as a unit. Space can separate and balance content; it is
not a defect merely because it is empty. These are design principles, not
published numerical acceptance thresholds.

[O'Donovan, Agarwala and Hertzmann's author page](https://www.dgp.toronto.edu/~donovan/layout/)
describes analysis of perceived importance, alignment and hierarchical grouping,
followed by an energy model and layout optimization, including retargeting.
This supports separating source analysis from constrained layout search. It does
not establish that our inferred groups or model judgments are correct. The
linked full paper exceeded the browsing service's size limit in this session;
the claims here are limited to the accessible author-page description.

[Scout's author-submitted paper abstract](https://arxiv.org/abs/2001.05424)
maps semantic structure, emphasis and order into spatial constraints for layout
exploration. Its interface-design setting differs from editable poster resizing,
but it supports expressing intent as relations instead of unconstrained model
coordinates. [DesignScape's author page](https://www.dgp.toronto.edu/~donovan/design/)
also distinguishes conservative refinement from alternatives that change style.
For this project, source-preserving retargeting must not silently become a style
redesign. These applications are our inferences from the sources.

## Evidence actually inspected

I viewed the original/imported-source/landscape/square/story contact sheets for
all eight **reviewed** sources: Sale, Botanical Invitation, Grand Opening, Night
Bloom, Rooftop, Trifold, Booth Poster and Cheatsheet. I additionally viewed the
full saved Night Bloom landscape, Grand Opening landscape, Trifold story and
Booth square renders. No reserve render or source was opened.

The evidence lives under [`output/poster-audit/`](../output/poster-audit/).
[A receipt](../output/design-visual-review-20261030/evidence.json) binds five
examined targets to their existing file hashes and numeric frame/font evidence.
It records the saved implementation digest, not the concurrently changing
checkout. The saved matrix reports 24 technical passes. Prior `visual approved`
labels are earlier agent judgments, not new human ground truth.

| Examined target | Concrete observation | Interpretation and limit |
| --- | --- | --- |
| Night Bloom, 1600×900 | Text-frame envelope is `y=238.5…661.5`, about 47% of target height; source is `96…2163.2` on 2304 px, about 90%. All source font scales are approximately `0.4304`. | Strong change in frame usage, not broken font hierarchy. The two columns can be width-limited; height occupancy alone cannot prove a failure or justify enlarging everything. Compare alternative framing and whole-group packing. |
| Booth Poster, 1080×1080 | Native result retains a large leading logo/title/URL group and five attached icon-description pairs, with substantial upper/lower negative space. Minimum live font is 18.07 px. | A useful packing experiment, not a demonstrated readability failure. MicroOS icon-to-copy space is already tight in the source: 9 source px becomes about 2 target px under the same `0.2259` scale. Do not mislabel this inherited gap as a new collision. |
| Cheatsheet, 1600×900 | Minimum live font is 11.45 px; the saved configured floor is 10.8 px. The render remains visually dense, with unused space beside shorter columns. | Passing that numeric floor is not approval for screen reading. Compare whole-card packing and declared display-scale readability; preserve command grouping and avoid independently inflating labels. The original is dense too. |
| Grand Opening, 1600×900 | All font multipliers are approximately `1.1919`. Cafe header center is `x=800`; the Grand/Opening cluster centers near `403`. Both were centered near `408` in the source. | An explicit topology/anchor choice, not proven damage: the earlier review intentionally treats the cafe name as a global page header. A stricter shared-anchor preference needs an intent label, not an assumed regression assertion. |
| Trifold, 1080×1920 | Communication sits top-left, Desktop top-right and MicroOS below Communication. The source has Communication/MicroOS/Desktop in three horizontal panels. All font multipliers are approximately `1.3007`. | Panel traversal changes. Independent brochure panels do not automatically imply a total reading order; preserve declared/strongly supported edges and expose ambiguity. The white lower-right area is not sufficient evidence to stretch or duplicate content. |
| Rooftop, 1600×900 and 1080×1080 | The saved renders retain the complete date row, caption hierarchy and continuous fade with a filled photo frame. | Useful positive controls. The prepared portrait-photo crop is a disclosed input limitation; resizing cannot recover absent lateral pixels. A newly invented background would violate the current source contract. |

Two priorities follow: **better source-aware packing/readability tradeoffs**, and
**explicit, reviewable group/attention/anchor intent**. Neither is solved by
maximizing foreground area. The saved examples do not show a general font-ratio
collapse or missing-element defect.

## Two resizing levels

**Proportional resolution change.** When the aspect ratio is exactly equal and
the contract permits it, apply one scale `s=Wt/Ws=Ht/Hs` to all pixel geometry,
font sizes, strokes and other scaled geometry fields. Keep rotations, copy,
styles, resources and element identity. This changes resolution without deciding
new reading order or topology. Current `engine/resize.py:_resolution_state`
provides a `resolution-scale` route and declines it when its safety/contract
requirements fail. This note did not execute or newly validate that route.

**Aspect-ratio reflow.** Retain rigid internal group proportions and source
hierarchy, then explore row/column/stack arrangements, supported text wrapping,
inter-group gaps and permitted background adaptation. A uniform fit remains a
useful baseline, with disclosed empty margins. Exact global proportions, no
cropping and complete frame filling cannot all hold under one uniform scale
when source and target aspect ratios differ. A square fitted into 1920×1080
occupies 1080×1080; filling the remaining width requires reflow, permitted crop,
distortion or new pixels. The last two are not acceptable shortcuts here.

## Principle → mechanism → concrete batch assertion

These are proposed assertions for a frozen native-render batch. Some components
already report related diagnostics; the complete assertion set is not yet a
verified suite. Tolerances below are initial engineering choices to calibrate on
validation data, not published design laws.

| Principle and observed risk | Existing seam | Concrete assertion or review |
| --- | --- | --- |
| Preserve original identity, including styled copy and layers. A visually tidy replacement is not a resize. | `check_contract`, immutable collection/resource hashes, document bridge. | Every mode: exact multiset of IDs/count; exact literal copy, asset bytes/references, font families/weights, colors, opacity and z-order; exact target canvas; source/resources unchanged after processing. Permitted geometry/type sizes may change. Deliberate duplicate/missing IDs, copy edits and swapped assets must fail. |
| Proportional reduction must preserve composition. | `_resolution_state`, `_transform`, contract. | Same-aspect supported fixtures at `s=1, 0.5, 0.25`: each permitted pixel field equals `s×source` within floating-point tolerance; angles and source-relative center distances unchanged; all font-size ratios preserved. No new clipping/overflow. Track readability separately rather than alter hierarchy silently. |
| Group related content and preserve intentional layering. Booth icon-to-copy gaps and event date/support plates need source evidence. | `build_relationship_graph`, source containers, native alpha/text masks, occlusion analysis. | For explicit rigid groups, fit one similarity transform and bound residuals; attachment IDs/ownership must remain. Synthetic label/plate and icon/caption mutations must be rejected or flagged. Check actual visible-ink collisions; source box intersection alone never grants blanket overlap permission. |
| Preserve hierarchy and focal relationships. Native font ranks are already stable in the inspected wide examples. | Style hierarchy, focal/image-prominence and proportions diagnostics. | For unambiguous authored role pairs, preserve relative font rank and declared prominence bands. Track visible image silhouette, not transparent box area. Include a deliberate tiny-product/oversized-badge negative control and require a correctly grounded warning. An ambiguous pair may abstain. |
| Preserve supported order and anchors; do not fabricate intent for independent panels. Grand global header and Trifold traversal need judgment. | Reading-order partial relations, relationship ambiguities, corner affinity. | A reviewed source profile declares only required `before`, attachment, common-anchor and corner edges. Require zero newly violated locked edges and no new cycles. Include two synthetic sources with identical geometry but different explicit ordering intent; results must follow the chosen profile. Do not assert that every source x/y ordering is sacred. |
| Respect designed negative space and source-dependent density. Night and Booth have occupancy opportunities, not area-maximization mandates. | Source composition span in `analyze_proportions`, candidate topology/packing. | Record foreground ink/alpha spans, source-relative prominence and normalized gap distribution while excluding proven full-bleed backgrounds. Compare the valid fit baseline with at least two feasible group arrangements. Treat occupancy drift as a review flag; reject any "improvement" that gains area through distortion, collision or weakened hierarchy. |
| Readability must match the intended viewing scale. Cheatsheet's 11.45 px passes its old floor but remains small. | Native text measurer, overflow checks, readability diagnostics. | Declare the viewing policy before the run; report minimum and lower-tail body type at that scale, wrapping/ink clipping and infeasibility. Require no new overflow. If keep-all/group/readability constraints conflict, record the conflict and fallback rather than remove, rewrite or independently inflate fine copy. |

These geometry assertions are insufficient for design approval. Pair each
accepted structural result with a source-aware render review of hierarchy,
grouping, intentional space, image visibility and attention sequence.

## Bounded next experiment, without source-specific branches

After the other agent's frozen baseline matrix completes, try one generic
candidate-family change: enumerate a small number of **order-respecting cuts
between source-owned bands/cards**, including the current fit baseline. Keep
explicit groups and native text-line ownership intact. Compare feasible
one-/two-/three-column or stack arrangements and bounded gap choices, using
existing `_source_cut_allowed`, `_packed`, `_balanced_columns` and source-container
evidence. Do not use poster IDs, fixed coordinates or a scene-specific aspect
switch.

Select only after the hard contract, ownership, layering and readable text
checks. Among survivors, compare source-relative visible prominence and gap
relationships together with readability and latency. Occupancy is a diagnostic
and bounded preference, never the sole reward. A wide Night Bloom with genuine
width-limited groups may legitimately retain larger vertical margins.

First use synthetic hierarchy/ownership cases plus the already reviewed
development families. Hold out a separately approved, newly created family
whose role/group graph and typography differ; do not consume the ten reserves.
Freeze the candidate recipe before that held-out family is rendered. Require no
hard regression and source-aware review of any apparent visual gain.

## Training labels and promotion

Collect separate labels for source intent and realized A/B quality. Intent labels
cover required groups, importance tiers, partial order, anchor scope and explicit
unknowns. Comparison labels include ties, neither and abstentions, plus grounded
candidate/element issues. Do not convert the numeric occupancy warning or old
agent approvals into automatic human preference truth.

Use a frozen matrix across independent source families and target aspects, with
mirrored A/B and source-swapped controls. Keep one model/prompt/adapter recipe per
run. The previous tiny synthetic LoRA run improved JSON validity but collapsed
to a constant judgment; it does not justify deploying a design critic. A new
fine-tuning session needs reviewed diverse labels, source/side-sensitivity gates,
verified local backups and a newly authorized bounded GPU session. See the
[training plan](VISION_DESIGN_TRAINING_PLAN.md) and
[batch workflow](SCENE_BENCHMARK.md).
