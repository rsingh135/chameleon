# Source-backed backgrounds and two resize scales

The source scene is analyzed before candidate generation by `engine.scene`.
The profile records the source digest, exact image IDs and asset hashes, roles,
confidence and physical evidence. It is cached by source and local resource
identity; callers receive a fresh copy. This implementation is deterministic
geometry/pixel reasoning. It does not claim face detection or learned semantic
understanding. `tools.scene_harness` supplies a separate local-model review seam.

## Crop authority

A full-bleed opaque imported photo can become a background only when an
independently verified lower fade supports multiple live captions. Explicit
product/hero/logo roles, cutouts, rotations, groups, unknown art and missing
resources decline the new policy. Source `fill` must match the asset aspect
before the engine may switch it to proportional `cover` on an in-bounds canvas
viewport. Original image bytes, asset reference, paint, stacking and IDs remain.

The supporting layer is a different case: a native vector **rectangle** that
was rasterized at import, proven locally to be horizontally invariant with a
monotone alpha ramp. It retains `fill` and its entire gradient extent when the
rectangle is resized. This narrowly scoped shape-paint exception to our local
image-frame aspect check prevents a cover crop from cutting away the transparent
edge and introducing a dark horizontal seam. It does not permit deforming
photographs, illustrations, borders or arbitrary raster art. The JSON remains
explicit about the photo's `cover` and the gradient rectangle's `fill`.

Caption support is tested against the original alpha ramp at every supported
text's upper edge, with full caption-frame enclosure. Changed overlay/frame
ratios remain reported. A relationship exception applies only when the verified
background viewport and actual caption support both pass. Invalid geometry,
off-canvas elements, dropped IDs, changed copy, fonts, assets and unknown crops
still fail the existing hard checks. Official judge behavior is not inferred
from these local checks.

## Layout and verification

The candidate family preserves complete source-owned header and caption rows,
copy/font-size ratios, and each calendar/control unit. Background viewport
coverage participates in source-aware selection alongside the existing hard
contract, visibility, collision and reading-order constraints. It does not
replace those checks or infer that a cropped subject is fully preserved.

Test two distinct changes through the same method:

1. **Resolution change:** Rooftop 1080x1920 to 540x960. Preserve the composition
   with a uniform scale; a new aspect/layout topology is unnecessary.
2. **Orientation change:** Rooftop 1080x1920 to 1600x900. Fill the background with
   a proportional crop and adapt the separate live caption/header regions.

The local harness writes the original render, source/asset receipts, profile,
each target's native render and JSON, contract results and selected state. Its
model-review sidecar cannot mutate source data, grant crop permission or change
the selected output. The existing candidate critic and a future first-load
scene analyzer are separate tasks; neither is currently a validated semantic
scene model in production.

The ten reserve posters remain outside resize tuning and model training.
Inspect actual image crops and text support, not just successful contract
checks. Adobe Express crop writeback still needs a separately verified host
round trip; the existing uniform-transform adapter does not establish it.
