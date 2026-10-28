# Preserve the source's visual language

## Read the original

Use the supplied source JSON, assets, fonts, and its actual rendered appearance
as the reference. Read the current challenge rules and schema for the contract.
Treat previous resized outputs as experiments rather than the style authority.

Record a compact source profile: typography hierarchy, alignment anchors,
logo/product relationship, group geometry, badge and action prominence, corner
affinity, spacing rhythm, and designed negative space. Derive it from each source;
keep the algorithm independent of sample IDs and fixed compositions.

## Keep identity and relationships

Preserve the identity required by the task: original copy, font families, colors,
asset pixels and references, grouping, opacity, and stacking order. Adapt the
permitted geometry and type sizes to the canvas while retaining that identity.

Move labels and their supporting shapes as a unit. Preserve their internal
alignment and the relative prominence of badges and actions. A small corner
badge should retain its corner association and supporting role as space changes.
Keep adjacent headline/body/action content recognizable as a related cluster.

Distinguish source defects from regressions. Inspect the rendered source before
concluding that overlapping boxes represent an intentional visual overlap.

## Measure visible content

Measure image alpha bounds and text ink or rendered line extents alongside
layout boxes. Transparent image padding can accommodate nearby content when the
visible pixels remain separated. Product prominence depends on the visible
silhouette, not only the dimensions of its padded asset box.

Preserve actual intentional layering. Check text against text and busy image
pixels even when those element boxes overlapped in the source. A source pair's
box intersection does not authorize arbitrary later ink collisions.

Use source-derived font ranks, spacing ranges, alignment, and group proportions
as candidate constraints or penalties. Record absolute readability limits as
additional diagnostics. An in-bounds design with very small text can still fail
visually; a large minimum font alone cannot establish faithfulness.

## Improve gradually and verify

Change one demonstrated cause at a time, beginning with a representative source
and target. Keep the original profile as the reference across iterations.
Intermediate aspect ratios can provide warm starts for a continuation optimizer;
measure frame-to-frame movement and relationship changes so drift is visible.
Release a soft relationship when the new canvas requires it, and inspect the
resulting tradeoff.

Inspect real source/result renders at useful display scales and check the export
at its requested pixel dimensions. Compare hierarchy, visible product size,
badge position, copy grouping, spacing, and collisions. Then run relevant
contract tests and a bounded set of other sources and aspect ratios.

Report concrete visual evidence and remaining failures. Structural checks and
proxy scores are diagnostics; describe style preservation only for the cases
actually inspected. Skills guide judgment and cannot guarantee design quality.
