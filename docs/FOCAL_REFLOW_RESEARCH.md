# Preserve the focal point during reflow

Research and source review: October 28, 2026. This note supplements
[composition reflow research](COMPOSITION_REFLOW_RESEARCH.md). Primary-source
guidance alone cannot establish the quality of an automatic result; implementation
and actual rendered evidence are recorded below.

## The brief determines what should lead

The user's current direction is specific: the Solstice product is the central
hero. A landscape result should retain its central placement, prominence and
original visual order, while the headline and action support it and smaller
copy has more freedom to fit. This is stronger evidence of intent than a
generic template, an inferred role, or a predicted attention map. It must not
be generalized to center every product or every source design.

The inspected Core source supports that direction. In
`input/chameleon_sample_level2_core.json`, the 1080-square canvas places the
logo, product, headline, body and action on the same horizontal center at
x=540. The product precedes the headline/body/action vertically; the small
flavor badge sits near the upper-left corner. Its actual source render has a
bright product against a dark field, a quieter brand above, and supporting
copy/action below. The analogous Advanced source also centers its product and
main message, but has much more copy, a badge row, a ribbon and a legal footer.
These observations are a profile of these sources, not universal layout rules.

The product asset's transparent padding matters. The supplied
`brand_kit/assets/solstice_bottle.png` is 800x800 pixels, with nonzero-alpha bounds
`(230, 130, 571, 681)`. An element-frame measurement therefore overstates its
visible width and height. A product can have a large frame while its visible
silhouette is small. The normalized visible center is close to, but not exactly,
the frame center. Preserve the original pixels and measure their placement;
do not crop or replace the asset to improve a score.

## What primary sources support

### Importance is a relationship between elements

Adobe's introductory Express guidance defines emphasis as drawing attention to
a focal element and hierarchy as ordering importance. It explains that scale
changes the relationships among elements; color, contrast, texture and position
also contribute. It explicitly supports both symmetric and asymmetric balance.
This supports preserving the intended importance relationship, rather than
giving every element the same growth rate. It supplies no numeric product-size
threshold or automatic reflow algorithm.
[Adobe Learn: graphic design principles](https://www.adobe.com/learn/express/web/graphic-design-basics).

Adobe's hierarchy examples show that visual weight can overcome a low position
on the page: a large title may remain the focal point even near the bottom.
Position alone is consequently insufficient to preserve hierarchy. The source
does not prescribe one correct arrangement for all designs.
[Adobe Express: visual hierarchy strategies](https://www.adobe.com/express/learn/blog/the-key-to-better-graphic-design-and-five-design-strategies).

Canva describes a deliberate first, second and final destination for attention.
Its guidance combines size, contrast, typography, proximity and composition;
spacing can isolate a focal point while grouping related information. It warns
that making too many elements prominent erases hierarchy. Its off-center
composition examples also make clear that centering is an option rather than
a general requirement. These are design heuristics, not a controlled account of
every viewer's eye movement. Its F/Z reading patterns should not be imposed as
an automatic law on this product poster.
[Canva: visual hierarchy](https://www.canva.com/learn/visual-hierarchy/).

### Balance is perceived weight, not equal box areas

Adobe Lightroom Academy distinguishes formal symmetric balance, asymmetric
balance using counterweights or negative space, and radial balance. Its
examples use tone, color, texture, shape and foreground/background relationships,
not equal rectangle widths. This is photography instruction, so its application
to a layered poster is an informed analogy. It does not prove a formula for
optical balance or require a geometrically centered subject.
[Adobe Lightroom Academy: symmetry and balance](https://lightroom.adobe.com/academy/principles/composition/symmetry).

There is a concrete Adobe product precedent for distinguishing mathematical
edges from visible edges: InDesign's optical margin alignment lets punctuation
and angled letters extend outside text margins to make the visual edge appear
cleaner. That is a typography feature, not a poster-balancing algorithm. It
supports inspecting rendered ink rather than assuming frame alignment alone
establishes visual alignment.
[Adobe InDesign: text formatting and optical margin alignment](https://helpx.adobe.com/indesign/desktop/format-and-style-text/composition-and-text-wrapping/apply-and-manage-text-formatting.html).

For product imagery, Adobe advises directing attention to the product and
limiting competing background distractions. Its ecommerce guide includes
centered individual shots and hero shots, alongside detail, lifestyle and group
shots. This supports a prominent single-product treatment when the brief calls
for one; a product-catalog photograph does not determine the composition of all
advertising layouts. Advice to alter lighting or backgrounds is outside our
immutable-asset contract.
[Adobe: product photography](https://blog.adobe.com/en/publish/2016/07/13/8-tips-for-product-photography),
[Adobe for Business: product photography types and composition](https://business.adobe.com/blog/basics/guide-effective-product-photography).

### Space carries meaning and need not scale uniformly

Spectrum explains proximity as a cue for relationships and separation, with
negative space helping focus attention. It distinguishes spacing within a
component from spacing among larger regions, and says these can scale at
different rates. This is an interface design system, not a poster recipe. Our
transfer is limited: preserve meaningful relationships while allowing different
levels of the hierarchy to adapt separately.
[Adobe Spectrum: spacing overview](https://spectrum.adobe.com/foundations/layout-and-structure/spacing/spacing-overview).

Canva Engineering's actual transformation work preserves implicit composites
whose configuration carries meaning. It reports that whitespace-only grouping
becomes noisy in dense or decorative designs, and uses visible geometry and
element information together. This supports retaining product/message ownership
and rigid action/badge internals; it does not establish that the largest
proximity cluster is the focal object or that a whole campaign must share one
scale.
[Canva Engineering: groups in design](https://www.canva.dev/blog/engineering/how-we-see-groups-in-design/).

### Retargeting research directly measures relative importance

Fosco et al. (MIT/Adobe, UIST 2020) use a predicted importance map to rank design
elements, then map them to equally ranked slots in authored templates matching
element count and target aspect ratio. On 17 designs, UMSI-assisted reflow was
preferred to the compared alternatives by 43 retained crowdworkers and six
professionals. The study is bounded to that tool, template set and design set.
The authors distinguish importance from eye-fixation saliency, and explicitly
say their importance-adjustment tool does not optimize balance or symmetry and
cannot guarantee good design. Their discussion recommends combining importance
with other aesthetic constraints. This is direct evidence for a separate
importance objective, not proof that importing a saliency network will solve
Chameleon's composition.
[Predicting Visual Importance Across Graphic Design Types, sections 6.1-6.2](https://arxiv.org/html/2008.02912v1).

The earlier Bylinskii et al. (UIST 2017) work learns importance from human
annotations and demonstrates retargeting/thumbnailing applications. Its stated
limits include training-data bias toward text and partial-element selection
that can cut off text. Its cropping and seam-carving applications are not
permitted substitutes for preserving this challenge's content and asset pixels.
Use an attention predictor as evidence with failure modes, not as authority
over an explicit product-hero brief.
[Learning Visual Importance for Graphic Designs and Data Visualizations, sections 6-7](https://arxiv.org/html/1708.02660v1).

## Operational recommendations for Chameleon

The following are proposed engineering decisions derived from the brief and
the evidence above. They are not claims about Adobe or Canva's implementation.

### Capture the focal policy independently of groups

Store the immutable source, group/ownership graph and a separate focal policy.
A rigid button group says its label and plate move together; it does not say
that button must be as important as the product. A campaign parent records
belonging; it need not force identical scaling of every sentence, legal line,
product and decorative accent.

For this explicit Solstice brief, record the product as the primary focal
element with a central horizontal anchor. Retain the brand/product association
and the headline-before-body-before-action relationship. Preserve the original
relative order of meaningful peer groups; do not reverse product/message
ownership or badge ordering while fitting a new aspect ratio. Keep intended
vertical relationships where feasible. When a target requires topology change,
record exactly which directional relationship changed and inspect the resulting
reading sequence; a graph membership check alone cannot approve that change.

The central anchor must refer to the target canvas, not merely to the center of
an off-center campaign column. Measure the visible silhouette as well as its
frame. Small optical corrections may be justified by the source pixels, but
must not become permission to slide the hero to a side column because that
allows larger copy. Use this policy only for sources with explicit user intent
or sufficiently strong source evidence. Multiple-product campaigns, asymmetric
art direction and text-led posters require different policies.

### Fit around a protected hero, with bounded supporting flexibility

Generate states that reserve the hero's central placement and a useful visible
size before allocating the remaining regions to supporting content. Try
source-aligned arrangements first, then only structurally explicit alternatives
that preserve the user's order requirement. A center-hero state can have
asymmetric supporting content; its total box layout need not be mirrored.
Neither a fixed three-column template nor automatic rule-of-thirds placement
follows from this research.

Fit supporting text through permitted width, wrapping, spacing and size changes.
Keep the headline/body/action coherent and the headline above its supporting
body in importance. Do not inflate all text to maximize the smallest font, or
shrink the product until the text is the dominant mass. Allow a legible legal
footer or long body paragraph to depart from the campaign's uniform scale,
while recording that sizing exception. Action labels must remain usable and
attached to their plates. Identity, copy, fonts, colors, rotation and original
asset pixels remain governed by the existing contract.

Use a constrained priority order:

1. Preserve required identity, bounds, valid text ranges, usable content and
   essential ownership/order. Apply explicitly declared readability floors for
   the intended viewing context.
2. Among feasible candidates, preserve the user-declared central hero and its
   importance relative to headline, action and accents. A larger legal/body
   font above its floor is not a reason to demote the product.
3. Refine source-derived spacing, optical balance, supporting typography and
   proportion fidelity, allowing documented flexibility where necessary.

Readability floors are project decisions to test at the intended output size;
none of the sources establishes one universal pixel threshold. If every
candidate breaks a required floor or order/hero constraint, report the conflict
and retain alternatives for review. Do not hide infeasibility behind a weighted
average, omitted copy, clipped content or unreadable legal text.

### Preserve useful space instead of maximizing occupancy

Keep a quiet region around the visible product so the headline, CTA or a bright
badge does not crowd or visually merge with it. Preserve short internal gaps
within message/action groups and larger gaps separating independent regions.
Scale or repack those gaps at the parent level without splitting a composite.

Judge negative space by its job: does it frame the product, separate meanings,
and support the intended reading route? Blank area is not automatically a
defect, and extra width need not enlarge supporting text. Conversely, a tiny
hero stranded in a large empty field does not become prominent just because it
is centered. Measuring total empty area alone misses both cases.

## Evaluation that can falsify the proposed behavior

Save the original and selected full-resolution renders with the exact asset and
font context. Report actual selected states, not the best diagnostic from an
unselected candidate. Compare at target pixel size and a useful review scale.

| Question | Proposed diagnostic | Practical limit |
| --- | --- | --- |
| Is the product still centered? | Visible-silhouette center error `abs(cx / canvas_width - 0.5)`, plus frame-center error and source-anchor evidence. | A centered transparent frame can conceal an off-center silhouette; this tests placement, not dominance. |
| Is it still the hero? | Visible product height relative to canvas height, silhouette area relative to content area, and ratios to headline ink and CTA area; compare source and candidate. | Area and size are proxies. Color, contrast, shape and semantic content also matter. No universal ratio is justified here. |
| Does the viewer read the same message? | Directed source-order/ownership checks and a rendered review of the first, second and final attention destinations. | Geometry does not prove actual eye movements. A brief human review is evidence, not an eye-tracking study. |
| Is the composition optically balanced? | Compare rendered mass on each side of the protected hero, using foreground alpha/contrast and text ink rather than empty frames. | A brightness or mass centroid can be fooled by decoration. Preserve intended asymmetry rather than forcing zero imbalance. |
| Does space preserve meaning? | Within-group versus between-group gap ratios, hero clearance, canvas margins and source-derived corner affiliations. | A numeric clearance does not prove that the negative space is well composed. |
| Is smaller text actually usable? | Role-specific minimum size, rendered line/ink fit, absence of clipping, and native-size inspection of body, legal and action text. | A passing minimum size is necessary only within its declared context, not sufficient for legibility. |
| Were content and groups preserved? | Contract, element correspondence, original asset/font checks, rigid label/plate ratios, per-product ownership and peer-order checks. | Membership alone cannot certify focal hierarchy. |

At minimum, compare the same Solstice source at 1600x400, 913x517 and 1080x1920,
plus the extreme 728x90 case. Use the tiny banner to expose incompatible
requirements, not to declare success from in-bounds boxes. Include an asymmetric
source and a multi-product source as counterexamples: the new central-hero
policy should not silently take over their art direction.

Use a brief blurred/squint review to check which large masses remain prominent,
then inspect the sharp render for hierarchy, copy fit, product/message meaning,
and the actual footer. This complements numeric diagnostics. A source-specific
review can support a concrete improvement claim; it cannot certify universal
style preservation or guarantee that a learned attention model matches this
user's intent.

## Implemented focal priority

`engine/layout/focal.py` captures an immutable profile only when a single,
unambiguous campaign has a real visible hero image near the source horizontal
center and at least two centered logo/headline/action companions. Missing images,
transparent assets, asymmetric sources and multiple offers decline this policy.
It imposes no vertical centering: Advanced's source product sits above mid-height.
Opaque IDs and transparent padding are covered by the profile tests.

`engine/layout/focal_states.py` proposes up to four landscape adaptations: a
centered brand/product spine with either all supporting content on one side,
or a headline on the left and body/badges/action on the right. Source ownership
and directed order remain checked. The latter deliberately releases the original
headline/body shared alignment; its relationship warning remains in the report.
Optional 0.85 scaling applies only to standalone supporting text and is declared
in the selected state's sizing exceptions. Product/headline/action scale ratios
and complete control internals retain their shared source scale.

Selection first protects the contract, unwanted collisions, essential ownership
and order, readable text, and rigid controls. For eligible sources it then favors
a hero within 2.5% of its horizontal center anchor, retention of source visible
height share, and an occupied-mass centroid within 8% of the source reference.
These bands are engineering tolerances. The mass estimate counts each complete
foreground unit once using visible bounds; it is an approximate layout diagnostic,
not a model of human perception or a universal symmetry rule.

The source-matching prior renders are frozen in `output/grouped-final-verified/`.
On the reviewed 1600x400 selections, Core's visible hero moves from x=0.321 to
x=0.500 and Advanced's from x=0.191 to x=0.500. Both retain 1.0 product-scale
ratios relative to headline and CTA, with zero measured collisions. The selected
Core occupied-mass drift is 0.0377 and Advanced's is 0.0052. Both choose the
flanking-copy arrangement. Exact selected plans and PNGs are in
`output/focal-selected/`; unselected proposals are separately labeled in
`output/focal-proposals/`. Reproduce using:

```powershell
.venv\Scripts\python.exe -m tools.focal_audit --out output/focal-selected
.venv\Scripts\python.exe -m tools.focal_audit --proposals --out output/focal-proposals
```

The extreme immutable-copy banner and Advanced's inherited missing image remain
limitations. These examples establish the inspected improvement, not general
poster compatibility or universal aesthetic quality.
