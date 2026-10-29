# Preserving composition through relational reflow

Research and audit date: October 28, 2026. Baseline inspected: `5606d92`.
This document separates observed failures, documented product/research behavior,
and the proposed Chameleon architecture. A relational prototype is underway;
this research does not establish its endpoint quality or universal semantic
recognition.

## The failure is compositional

Preserving every element is necessary for the challenge, but insufficient for
preserving a design. A product, its brand, an explanatory sentence and an action
form a message through their arrangement. Their original colors and text can
survive while that message becomes disconnected.

The baseline Core banner illustrates this gap. Its source logo, product,
headline, body and CTA share center x=540 on a 1080-square canvas. After resizing
to 1600x400, the logo centers at x=800, the product at x=204.16 and the message at
x=993.16. The logo/product axis separates by approximately 596 px. The local
source-style score is still **0.941683 with no warnings**. A global brand header
could be a valid alternative, but our model neither records that interpretation
nor compares it with a brand/product cluster.

Two controlled counterexamples preserve the baseline score, structural validity
and zero detected collisions: move the body above its headline, or shift the
complete CTA 460 px left, detaching it from the message. The first escapes an
undirected proximity check; the second escapes because CTA ownership is absent.
These are local audit results, not Adobe evaluator results. Evidence and renders
were saved under ignored `output/research/`, including
`semantic-audit-data.json`, `counterexample_reversed_reading.png` and
`counterexample_detached_action.png`.

The architectural cause is visible in [source inference](../engine/layout/model.py)
and [candidate generation](../engine/layout/candidates.py): atomic units receive
roles, then role buckets receive independent slots. There is no persistent
parent joining brand/product, no message/action ownership edge, and no explicit
directed reading sequence. A high score over the measured properties cannot
certify relationships the model never measured.

## What real applications actually represent

### Canva: meaning can belong to an implicit group

Canva Engineering describes preserving diagrams, charts and tables assembled
from ordinary elements during Magic Layouts/Switch transformations. Whitespace
alone misgroups dense/decorative designs. Its grouping subsystem combines a
rasterized design with element information, detects implicit groups using a
fine-tuned vision transformer, then maps detections to constituent elements.
Visible-pixel polygons improve membership detection when element rectangles
contain padding. Training uses annotated Canva templates. This is unusually
specific evidence for a production grouping subsystem; it is not a disclosure
of the complete resize optimizer. Our inference is to combine visible geometry
with structural/semantic evidence, not to assume its proprietary data or model
is available. [Canva Engineering, June 2024](https://www.canva.dev/blog/engineering/how-we-see-groups-in-design/).

### Figma: adaptation policies belong to containers

Auto Layout represents direction, spacing, padding, alignment, wrapping and
sizing behavior within frames. Content-hugging, filling and fixed dimensions
express different dependencies. Positional constraints retain frame-relative
edge/center relationships, while variants store authored component alternatives.
These are explicit relationships and states, not proof that Figma automatically
understands an arbitrary poster. Our inference is to give clusters stable local
coordinates and allowed behaviors, then distinguish a continuous size change
from a discrete topology change. [Auto Layout](https://help.figma.com/hc/en-us/articles/360040451373-Guide-to-auto-layout-in-Figma),
[constraints](https://help.figma.com/hc/en-us/articles/360039957734-Apply-constraints-to-define-how-layers-resize),
[variants](https://help.figma.com/hc/en-us/articles/360056440594-Create-and-use-variants).

### InDesign: hierarchy, rules and alternate layouts coexist

Flex Layout provides nested parent/child containers with direction, wrapping,
spacing, alignment and fixed/content-driven/fill sizing. Adobe documents sizing
conflicts, so a production system still needs dependency-resolution policies.
Liquid rules distinguish scaling, recentering, guide-based and object-based
adaptation. Alternate layouts preserve distinct sizes/orientations from source
pages and can link stories. These support separate content identity and layout
states. Some options modify text breaks/styles; those freedoms must not be
copied into this challenge. [Flex Layout](https://helpx.adobe.com/indesign/desktop/layout-and-grid-tools/apply-layout-adjustments/flex-layout-overview.html),
[conflicts](https://helpx.adobe.com/indesign/desktop/layout-and-grid-tools/apply-layout-adjustments/flex-layout-conflicts.html),
[liquid rules](https://helpx.adobe.com/indesign/desktop/layout-and-grid-tools/apply-layout-adjustments/liquid-page-rules-overview.html),
[alternate layouts](https://helpx.adobe.com/indesign/desktop/layout-and-grid-tools/apply-layout-adjustments/create-alternate-layouts.html).

Express also exposes resized variations to keep/discard; its documentation does
not disclose the optimizer. Optional generative expansion is incompatible with
our immutable-asset requirement. Reviewable alternatives are the useful lesson.
[Express resize documentation](https://helpx.adobe.com/express/web/arrange-layers-and-pages/change-page-size.html).

## Research that changes the engineering decision

The following primary sources support particular mechanisms. Their benchmarks
do not collectively establish professional-quality source retargeting.

| Work | What was demonstrated | Implication and limit for Chameleon |
| --- | --- | --- |
| [Learning Layouts for Single-Page Graphic Designs, TVCG 2014](https://www.dgp.toronto.edu/~donovan/layout/designLayout.pdf) | Infers importance, alignment groups and hierarchical regions; energy-based retargeting compares previous positions, relative heights, importance and overlap. | Closest direct precedent. Preserve a source relationship model, rather than only appearance fields. It used learned parameters from paired designs, reported about 40-minute optimization, and did not consistently match professional retargets. |
| [DesignScape, CHI 2015](https://www.dgp.toronto.edu/~donovan/design/designscape.pdf) | Separates nearby refinement from style-changing brainstorming; supports inferred/explicit constraints, locks, retargeting and saved layouts. | Keep conservative adaptation separate from exploration. Its interactive GPU implementation and novice studies do not establish our CPU runtime or autonomous fidelity. |
| [Cassowary, TOCHI 2001](https://people.eng.unimelb.edu.au/pstuckey/papers/cassowary-tochi.pdf) | Incremental linear equalities/inequalities with required constraints and weaker preferences. | Useful inside a chosen state for alignment, gaps and containment. A solver does not discover semantics, choose topology or solve text wrapping by itself. |
| [ORC Layout, CHI 2019](https://duruofei.com/papers/Jiang_ORCLayout-AdaptiveGUILayoutWithOR-Constraints_CHI2019.pdf) | OR alternatives combine grid/flow behavior and constraints across hierarchy; resizing can reuse previous solutions. | Represent structural alternatives explicitly. Its GUI-domain objective does not supply poster aesthetics; bounded state enumeration may be simpler here. |
| [Constraint-Based Breakpoints, TVCG 2024/2025](https://arxiv.org/html/2409.01339v1) | Selects the first feasible view from an authored stack using size, overlap and aspect constraints; maps valid states over width/height. | Choose states using measured feasibility, not one universal width threshold. The system requires designed views and concerns visualization; some alternatives alter information, which our contract forbids. |
| [Neural Design Network, ECCV 2020](https://arxiv.org/html/1912.09421v2) | Components are nodes; directed location/size edges express constraints; relation completion precedes box generation and refinement. | Relationships deserve first-class representation, but a graph need not be neural. Generation from categories/constraints does not establish fidelity to a supplied source's specific narrative. |
| [LayoutDM, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Inoue_LayoutDM_Discrete_Diffusion_Model_for_Controllable_Layout_Generation_CVPR_2023_paper.html) | Models structured layouts with modality-wise discrete diffusion and supports optional generation constraints. | A learned proposal prior may expand candidate coverage. Denoising steps are not guaranteed valid responsive canvas states; conditional generation does not establish original message ownership, exact typography or source-style fidelity. |
| [PosterLayout, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Hsu_PosterLayout_A_New_Benchmark_and_Approach_for_Content-Aware_Visual-Textual_Presentation_CVPR_2023_paper.html) | Models both inter-element and layout/background relationships, including avoidance of salient content. | Measure visible product/text interactions, not just rectangle collisions. Generating text/logo/underlay boxes on a background is not exact-font, exact-copy retargeting of every source element. |
| [RALF, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Horita_Retrieval-Augmented_Layout_Transformer_for_Content-Aware_Layout_Generation_CVPR_2024_paper.html) | Retrieved layouts augment content-aware generation, including constrained tasks. | Approved master examples can supply better structural proposals. A visually similar reference still needs source-element/relationship remapping and validation. Retrieval alone does not preserve our semantics. |
| [LayoutRectifier, CGF 2025](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70273) | Alternates discrete grid search/snapping and continuous geometry repair, including containment and saliency constraints, while limiting deviation. | A useful repair architecture after topology selection. It acknowledges unsuitable grids, blank-space tradeoffs and remaining flaws; repairing geometry does not discover missing meaning. |

## Alternatives considered

This assessment is our synthesis, not a claim about undocumented product internals.

- **Uniform scaling** preserves exact source geometry and is a valuable baseline.
  Across extreme ratios it creates tiny text or unused space; it cannot by itself
  produce a strong target composition.
- **Anchors and global role templates** are inexpensive but underrepresent intent.
  Pinning a logo globally can detach it from the product; placing every action in
  one action region can destroy multiple independent messages.
- **Distance springs between every pair** overconstrain orientation changes and
  confuse proximity with ownership. Use sparse typed relations, local normalized
  gaps and confidence instead.
- **Hierarchical constraint graphs** preserve composites while permitting controlled
  reflow. Their main risk is wrong inference: confidently preserving the wrong
  grouping still produces the wrong meaning.
- **Authored or inferred layout states** encode stacked, split and overlay
  arrangements with distinct allowed geometry. Authored masters are stronger
  evidence; inferred states need explicit feasibility and rendered review.
- **Multiple exemplars/retrieval** can capture professional alternatives that cannot
  be inferred from one square design. Matching should consider cluster structure,
  copy load and visible mass, not just background similarity.
- **Learned graphs, diffusion and layout-language models** can diversify proposals.
  Their learned distribution is not the source's identity contract; reject invalid
  proposals and retain deterministic repair. Model permission, training data,
  licenses and CPU/memory budgets must be checked before deployment.
- **Vision plus element semantics** can distinguish a diagram from scattered shapes
  and a caption from an independent slogan. Geometry and raster analysis complement
  text/image interpretation; neither proves arbitrary ownership. A local model
  should propose confidence-tagged edges, not rewrite source pixels or copy.
- **Human meaning hints** such as “this action belongs to this message,” rigid
  lockups and an approved landscape master resolve ambiguity cheaply. They are
  useful corrections and evaluation references, even if stage-one automation must
  operate without interactive input.
- **Intermediate frames** can warm-start optimization and stabilize transitions.
  They do not repair a bad endpoint. Interpolating unrelated atomic positions can
  introduce collisions; repeatedly resizing the last result compounds inference
  and font shrinkage.

## Proposed source graph and layout states

The implementation direction below is an engineering proposal informed by the
research. It is not a reproduction of Canva, Figma or Adobe's internal algorithm.

### Capture intent once

Build an immutable source snapshot plus a separate graph. Keep explicit groups,
inferred label/shape containment, visible alpha support, measured text lines,
source geometry and typography preferences. Opaque IDs establish correspondence;
sample names and hard-coded copy must not determine behavior.

Clusters need local coordinates and policies: rigid lockups, uniformly scalable
groups, reflowable ordered content, peer rows and edge-affiliated accents. A
button's label/plate is a rigid relationship; a headline/body/action cluster may
adapt width and wrapping while preserving membership and order. The campaign's
parent flow can change from vertical to horizontal without scattering those
children or changing frozen glyph rotation.

Store typed edges such as `contains`, `labels`, `precedes`, `associated_with`,
`aligns`, `near`, `more_prominent_than` and `intentional_overlap`. Record source
measurement, confidence, evidence, priority and permitted transformations on
each edge. Explicit grouping outranks uncertain geometric association. A shared
axis supports a brand/product hypothesis; it does not prove exclusive ownership.
Retain competing interpretations when evidence is ambiguous.

Distance constraints should use source ranges normalized by local line height,
visible object size or cluster extent. Preserve the contrast between small
within-cluster gaps and larger between-cluster gaps. Preserve designed negative
space; distributing spare space evenly between all children can sever a sentence.
Include decorative/hero affiliations and intentional layering where supported,
because a foreground-only graph misses part of the visual rhythm.

### Choose topology, then fit

Enumerate a bounded set of coherent cluster states: conservative source topology,
stacked campaign, brand/product beside message, or an explicit global-header
alternative. Each state declares parent flow, child reading order, alignment,
protected regions and which soft relation may transform. Change above/below to
left/right between parent clusters when appropriate; do not silently reverse the
headline/body sequence.

Within a state, fit translation, scale, width and local gaps using shared cluster
parameters. Measure real text wrapping and visible image support. Linear solvers
can handle many geometric constraints, but wrapping and non-overlap alternatives
need discrete choices or a bounded solve/measure/repair loop. A continuous
optimizer alone should not be expected to invent a new topology.

Apply hard content/asset identity and bounds first, then essential ownership and
ordering, then readability, prominence, alignment and spacing preferences. Do not
let a slightly larger minimum font outweigh a detached action. Keep an explicit
worst-essential-edge diagnostic and a small Pareto set across fidelity,
readability and target composition. Several high local scores must not conceal
one catastrophic relationship failure.

If no state satisfies the important conditions, expose the conflicts and show
reviewable alternatives. Dense immutable copy may have no attractive realization
in a tiny strip. The allowed output contract and the quality diagnosis must stay
separate; no element deletion, copy shortening, font-family replacement or asset
generation is authorized to make a metric pass.

### Preserve original intent across states

Store accepted states with the source revision, canvas, topology, relation
assignments, transforms and diagnostics. Coordinates alone do not explain why a
state works. Reuse the original graph and font preferences for every frame; a
previous result contributes only a warm start or an explicitly accepted master.

Within a valid topology, solve nearby sizes continuously. When it becomes
infeasible, select another coherent topology. A transition margin/hysteresis can
reduce repeated switching near boundaries; this is our proposed policy, not an
established feature of the cited poster systems. Only after endpoint layouts
look good should a path objective reduce jumps, with extra intermediate frames
rendered and checked.

## Delivery constraints and current scope

The first prototype should remain a bounded offline CPU method using the
existing renderer and unchanged source content/assets/fonts. Three sample
designs cannot train or validate universal semantic inference. Canva's annotated
template data is proprietary; academic datasets and pretrained weights require
separate suitability and licensing review. For example, the official
[CLG-LO implementation](https://github.com/ktrk115/const_layout) uses a trained
PyTorch/PyG stack and AGPL licensing. The supplied
[RALF cache](https://github.com/CyberAgentAILab/RALF) includes data/features/results
as well as checkpoints; its 13 GB bundle is not a model-size requirement or a
promise of lightweight CPU use.

A bounded first relational family now participates in default selection for one
inferred brand/product cluster, one message cluster and upper-corner accents.
Unsupported structures decline this family and retain other candidates; the
graph diagnostics still expose modeled relationship losses. This is a narrower
correction, not arbitrary human meaning recognition. Confidence values are
hand-set evidence weights, not calibrated probabilities. Existing continuation
results in [REFLOW_EXPERIMENT.md](REFLOW_EXPERIMENT.md) concern an earlier endpoint
and must not be reused as evidence for this new representation.

## First implementation and observed results

`engine/layout/relationships.py` snapshots the original source into frozen units,
clusters and typed edges. `appearance.py` separates occupied content from frames.
`composition.py` realizes bounded column/stacked states whose internal brand and
message axes stay coherent. Each state retains a source digest and returns a
fresh design copy. Explicit-group/reading-order failures cannot be compensated
by a larger minimum font; softer inferred associations remain below readability
in selection. The studio reports relationship losses and no longer equates a
clean local score with preserved visual meaning.

The actual Core 1600x400 render was inspected against its saved previous result.
Its roughly 596 px logo/product axis separation becomes zero in visible geometry;
headline, body and action remain an ordered aligned block. The smallest font is
13.33 px. Original assets, copy, fonts, colors and stacking remain unchanged.
Tests also reproduce reversed supporting copy and a detached action that the
previous style score missed, plus opaque IDs, rotated groups, full accent bounds
and two-product structures that must decline this limited realizer.

Validation: 461 tests pass with one expected skip; lint/format and JavaScript
syntax checks pass. A 17-case sample/shape audit has valid contracts, deterministic
results, unchanged sources and zero detected foreground collisions. Five cases
still raise relationship warnings and six contain text below 10 px. Approximate
visible rectangles and recognized low-opacity underlays do not establish full
ink-level safety. Advanced badge rows, multiple product/message ownership and
natural-language/visual semantics remain open work.

That first checkpoint has since been extended with ordered badge rows, local
campaign ownership and source-evidenced sibling alignment. The later
[Advanced layout checkpoint](ADVANCED_LAYOUTS.md) records 523 tests, 12 actual
selected renders and the remaining tiny-format/semantic-inference limits;
the earlier measurements above are retained as the research baseline.

Rendered review also rejected a ranking that put the aggregate relationship
score ahead of typography: exact source gaps could select a visibly worse
portrait. Selection now retains a near-best joint style/relationship set (0.01
local-proxy tolerance) before comparing source typography scale. The tolerance
is a heuristic, not a perceptual just-noticeable-difference claim.

Reproduce the state artifacts with:

```powershell
.\.venv\Scripts\python.exe -m tools.composition_demo --task input/chameleon_sample_level2_core.json --out output/composition-core
```

The tool saves `graph.json`, `states.json`, proper design JSONs, full-resolution
PNGs and `contact_sheet.png`. An optional `--baseline` takes a saved design JSON;
without it, the baseline is explicitly the current default. These are generated
candidate states, not human-approved masters. A source digest is provenance,
not a semantic explanation; edge evidence and state assignments supply that
additional context.

## Acceptance and ablation matrix

Every row needs original/output renders and per-relation diagnostics, in addition
to the hard-contract suite. Compare aesthetic preference and message fidelity
separately in rendered A/B review; neither is certified by one scalar score.

| Case or ablation | Required evidence |
| --- | --- |
| Baseline versus graph-aware placement | Detached CTA and reversed-body counterexamples incur relation failures; improvement is visible without diagnostic overlays. |
| Geometry-only versus typed graph | Ownership/order survives even when unrelated objects are closer; ambiguous inference remains labeled. |
| Fixed topology versus state search | A parent cluster changes axis while label lockups and message order remain coherent. |
| Padded rectangles versus visible support | Transparent assets and oversized text frames do not falsify membership, prominence or distance. |
| One versus two product/message pairs | Each caption/action stays with its own product; global role buckets cannot merge them. |
| Authored hints/master versus inference alone | Record which ambiguity the hint resolves and whether fidelity/readability improve. |
| Graph versus graph plus visual-content costs | Text avoids salient product pixels while intentional layering remains recognizable. |
| Independent sizes versus continuation | Same immutable graph/preferences persist; extra sampled frames pass checks and transitions avoid gratuitous jumps. |
| Unseen copy lengths, opaque IDs and peer-row wrapping | No sample-specific behavior; order, peer prominence and group membership survive. |
| Extreme or conflicting target | Report limiting relations/readability, bounded runtime and remaining defects; do not hide failure behind zero structural violations. |

Measure inference confidence, essential-edge loss, local gap hierarchy, alignment
drift, reading order, type contrast, visible prominence and actual overlaps.
Map the valid/failed state regions across width and height, including nearby
sizes around transitions. Promotion requires a bounded regression set that
improves composition without merely moving failures into unseen formats.
