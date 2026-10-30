# Backend resize integration

The local studio uses the same backend as the portable document bridge. The
frontend can change independently of the layout engine; the returned objects
and rendered PNG must continue to describe the same result.

## Requests and results

- `POST /api/resize` accepts `source` and `target_canvas` with the bundled assets.
- `POST /api/document/resize` accepts the portable source/resource bundle, plus
  `target_canvas`. Preserve its `assets`, `fonts`, `document`, and `attribution`
  when exporting another editable bundle. The poster catalogue supplies working
  bundles through `/api/posters/<id>/bundle`.
- Both return `design`, `source_png`, `output_png`, `report`, `warnings`, and
  `elapsed_ms`. Inspector geometry and exported objects must come from `design`.
  The PNG is rendered from that same design at the requested canvas dimensions.

The engine keeps source IDs, copy, font faces, assets, paint, stacking order, and
source provenance. Permitted geometry and type sizes change during reflow.
Native effect descriptors preserve proportional tracking, gradients, outlines,
and baseline behavior without rewriting copy or flattening live text. SVG
non-text art retains its source provenance; supported rectangular containers
can remain editable shapes.

Changing a target invalidates the previous result. Keep exports disabled until
the replacement request succeeds. Selecting an object or changing the studio
view must not rewrite the result or its download data.

## Additive diagnostics

Existing `report.metrics` fields retain their meaning. In particular,
`metrics.raw_collisions` is the original geometric diagnostic and is unchanged.

- `report.reading_order.high_confidence_inversions` counts violations of
  source-supported reading order. Uncertain source relationships remain explicit
  in its coverage and diagnostics.
- `report.rules.distorted_rules` identifies stretched thin source dividers.
- `report.relationships` includes source-supported ownership, peer rows, and
  `overlay_attachment`. A photo fade is attached to its actual source underlay;
  it is not an independent product. `ornament_attachment` preserves a supported
  ornament's size and gap relative to its heading. `caption_hierarchy` compares
  source caption and calendar type ratios. Proven text-panel frames retain their
  enclosure role instead of acquiring unrelated product/message ownership.
- `report.occlusion` explains visible overlap separately from raw geometry.
  `visible_pair_count` may discount only source-supported opaque layering.
  Treat that count as verified only when `coverage.complete` is true and
  `raw_pair_count` equals `metrics.raw_collisions`. Otherwise retain the raw
  count. Unknown and text/text intersections remain counted.
- `occlusion.visibility_retention` compares the visible fraction of each photo
  with its source. Complete evidence below `0.85` fails the independent photo
  visibility guard, even if there are no collision pairs.
- `report.layout_state` identifies the selected proposal and declared flow axes.
- `report.proportions.containers` records source-proven card/panel ownership.
  Only accepted container frames may adapt independently of their contents.
  Text and image children retain their own rigidity and contract checks; an
  image frame still has to meet the original image aspect-ratio contract.

These are diagnostics, not automatic visual approval. The poster audit records
technical checks separately from reviews of actual source/result renders.

## Verification

Focused backend checks:

```powershell
python -m pytest tests/test_viewer_server.py tests/test_document_bridge.py tests/test_selection.py tests/test_reading_order.py tests/test_overlays.py tests/test_occlusion.py
```

The actual new-interface verification downloaded Grand Opening JSON and PNG
results at 1080×1080 and 1600×900, then compared them with fresh requests to
`/api/document/resize`. All 16 objects changed geometry, all 11 text sizes were
adjusted, all 3 font files and frozen fields were preserved, the inspector
matched the exported title, and the full PNG bytes matched the API. Reading
order and divider checks passed. Local evidence is in
`output/interface-object-verification/report.json`.

The same interface check also covers the 1080×1080 Rooftop result: all 17
objects changed geometry, all 13 text sizes changed, and both image assets and
both font files stayed exact. Inspector values, editable JSON, and full PNG
bytes match a fresh API response. Body/calendar type remains in the source
30:32 ratio (25.875:27.6 px), with zero reading inversions or relationship
warnings. The image-prominence warnings remain visible: the full portrait
photo keeps its aspect ratio and uses the existing dark matte on a square.

Local validation completed with 1,435 passing tests and four platform/fixture
skips. After the final layout fixes, 232 focused selection, snapshot, native
panel, media, relationship, and focal regressions passed again. The supplied
renderer and vendor files remain unchanged.

Generate the reviewed render matrix and the separate personal test pack:

```powershell
python -m tools.poster_audit --out output/poster-audit
python -m tools.poster_pack --out output/poster-audit/reserve-10.zip
```

The ten reserve designs receive source/import checks only. The audit does not
resize them or use them to tune selection. Native Adobe Express host behavior
still requires a separate live host test; local adapter tests do not establish
that result.
