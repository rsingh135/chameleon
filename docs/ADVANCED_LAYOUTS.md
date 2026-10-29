# Advanced badge rows and local product/message ownership

This checkpoint extends the composition model described in
[the research decision](COMPOSITION_REFLOW_RESEARCH.md). It addresses two
demonstrated failures: a badge row moving below its own action/footer, and two
products' messages/actions being combined into global role buckets.

## What changed

The original source graph now records ordered peer rows and disjoint local
campaigns. Product/message associations use local alignment, spacing and section
evidence with a separation margin; ties remain explicit ambiguities. Product
candidates are not restricted to the single globally largest image. Layer
creation order and element/group names do not decide campaign reading order.

A peer row may wrap while keeping its ordered members, plate/label geometry and
relative prominence. Its attachment to its own body and action is a separate
relationship: moving an intact row below the action or into another offer is a
failure even when the row itself remains neat. Peer-row members are not treated
as independent canvas-corner accents.

Advanced states reserve corner promotions, legal copy and a side accessory
without detaching the main message. The realizer fits typography and balanced
peer wrapping together, retains a visible brand/product block, and rejects
collapsed or out-of-bounds states. Badge labels may reach a readable minimum
only when measured text still fits their unchanged frames. Original copy,
families, colors, assets, opacity, grouping, rotation and stacking remain fixed.

Multi-campaign states arrange complete offers as columns, stacks or a wrapped
grid. Local product/message arrangements can change axis while each offer keeps
its own content. Typography scaling is shared across offers so the longest copy
does not silently shrink only one offer. An additional comparison-row state is
available when the source actually shares heading, product and action anchors;
it is declined when maintaining those rows would break local attachment.

`resize()` retains the original design-only API. `resize_with_state()` also
returns the selected name, topology, source digest and declared cluster flows.
Selection, API diagnostics and experiment reports use those same declarations;
they do not guess orientation from the resulting rectangles. Returned plans and
designs are independent copies. The studio warns when source ownership inference
is ambiguous.

## Actual rendered evidence

The audit uses the supplied Advanced source and two authored two-offer sources
with unequal copy, local badge peers and opaque IDs. Both fixtures reuse the
original supplied assets. They test local ownership, not general product-image
recognition. All three sources are rendered at 728x90, 1600x400, 913x517 and
1080x1920.

| Evidence across 12 selected outputs | Saved baseline | Current selection |
| --- | ---: | ---: |
| Reading-order violation instances | 4 | 0 |
| Detached product/message/action instances | 12 | 0 |
| Cases with detached instances | 8 | 0 |
| Render/contract failures | 0 | 0 |

The relationship counts come from authored membership/order checks independent
of the inference graph. They are diagnostics, not an aesthetic certification.
Source JSON and graph digests remain unchanged throughout the audit.

Native-size review confirms that Advanced's wide/custom outputs keep a coherent
headline/body, badge row and CTA. Its portrait output wraps the badges before
the CTA and retains the legal footer. The authored wide comparison selects the
aligned state with common product, heading and CTA anchors, while keeping each
offer separate. Narrower and originally stacked sources retain other states.

| Source/target | Smallest output text |
| --- | ---: |
| Advanced, 1600x400 | 10.0 px |
| Advanced, 913x517 | 10.0 px |
| Advanced, 1080x1920 | 14.3 px |
| Two offers from columns, 1600x400 | 13.2 px |
| Two offers, 913x517 | 11.2 px |

Validation: 523 tests pass with one expected skip; Ruff lint/format and JavaScript
syntax checks pass. CI requires the declared `jsonschema` development dependency
for the fixture audit. It is not added to the submission's runtime dependencies.
The running studio passes 17 HTTP checks, including retained state/source-digest
validation. The actual Advanced custom-size browser preview was checked. The
two-offer fixture was imported through the browser file picker and rendered at
1600x400; both JSON and PNG export links were exercised. The downloaded JSON
matches the verified API design and the PNG matches its full-resolution pixels.
Advanced custom-size and two-offer API exports also match the official renderer
pixel for pixel. The browser viewport override was reset after responsive checks.

## Reproduce and inspect

```powershell
.\.venv\Scripts\python.exe -m tools.composition_cases --out output/advanced-audit
```

This writes proper design JSONs, full-resolution PNGs, source/selected contact
sheets and per-case reports. To compare with the locally captured pre-change
artifacts, add `--baseline-dir output/advanced-prechange`. Those generated
baselines are ignored by Git; without them the command evaluates current
sources/results rather than inventing a historical comparison.

## Remaining limits

The 728x90 cases still have roughly 3.5–3.9 px minimum text and are not readable
deliverables. The full original copy cannot be treated as successfully adapted
merely because the structure and ownership checks pass. Readability warnings
remain visible.

Advanced deliberately references a missing asset. Its warned placeholder remains
visible; it is not a newly rendered watermark. Some custom two-offer wordmarks
are small, and portrait arrangements can retain generous inter-offer whitespace.
Inference is still geometric and uncertain, approximate visible rectangles are
not complete pixel-level collision analysis, and arbitrary diagrams or ambiguous
multi-product artwork may require authored ownership hints or further work.
