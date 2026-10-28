# Keep the local studio useful

Treat the viewer as a working product for loading a design, choosing a canvas,
comparing real results, and exporting them. Keep its established controls and
layout appropriate to that task. Use existing typography, spacing, colors, and
component conventions; make scoped improvements to demonstrated problems.

## Make the result trustworthy

Use the engine's real source and output previews. Keep optional element-box
annotations visually separate from those previews. Studio styling and engine
composition are separate decisions: changing the surrounding interface does not
improve the exported design.

Tie a result and its downloads to the exact source and target that produced it.
When either changes, label any retained preview as previous and invalidate its
exports and success status. Keep the inspector's selected element and geometry
consistent with the displayed result.

Provide clear loading, error, timeout, and recovery states. Handle cancelled file
selection, invalid imports, stale async responses, and image decoding failures.
Release object URLs after their useful lifetime. Preserve a prior valid import
only when its identity and state remain clear.

## Keep controls accessible and predictable

Use conventional labeled controls, coherent focus order, visible focus, and
keyboard access. Provide an element-list alternative to clicking a canvas.
Communicate status and errors through appropriate live regions, with text as
well as color. Check contrast and responsive behavior using existing tokens.
Keep any motion tied to feedback and respect reduced-motion preferences.

## Verify the changed path in a browser

Exercise the affected flow with real data: source loading, a catalogue preset,
custom dimensions, invalid input, retry, inspection, and current-result export
as relevant. Check the interface at laptop and narrow widths. Parse downloaded
JSON and inspect the PNG's dimensions and contents against the selected canvas.
Keep dependencies and services consistent with the existing local workflow.

Use product-interface guidance when consulting additional taste skills. The
installed design-taste-frontend skill targets landing pages and portfolios;
its marketing composition and motion defaults do not describe this studio.
