# Source-backed backgrounds and two resize scales

The source scene is analyzed before candidate generation by `engine.scene`.
The profile records the source digest, exact image IDs and asset hashes, roles,
confidence and physical evidence. It is cached by source and local resource
identity; callers receive a fresh copy. This implementation is deterministic
geometry/pixel reasoning. It does not claim face detection or learned semantic
understanding. `tools.scene_harness` supplies a separate local-model review seam.

The two resize paths are explicit. An exact-aspect request first tries one
uniform scale of all permitted geometry, type, stroke and corner fields; it
preserves the source composition instead of promoting individual elements.
The normal contract, model crop veto, cache and snapshot paths still apply.
Unsafe source image cropping or an invalid text fit declines that route and
returns to the ordinary constrained search. Readability diagnostics remain
visible when the requested resolution makes original fine print too small.
The background family proposes alternate compositions when the aspect changes.

Background caption search includes its existing bounded height samples and,
where it falls between those samples, the source-derived minimum-readable type
scale. This closes a demonstrated search gap at 800×500: a smaller sample made
the header too small, while the next sample's fade hid too much photo. The new
boundary candidate retains the existing 85% relative photo-visibility guard,
caption-support checks and one common scale for all source typography.

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

Manifest-backed native documents can also use `image_position: [x, y]`, with
finite values in `[0,1]`, to align a proportional cover crop inside its frame.
This is an optional local native-document geometry extension; the original
schema permits extra element properties but does not define this field. The
owned native adapter paints it from the original asset. The vendored renderer
and ordinary brand rendering remain unchanged. Missing positioning retains the
original center crop. A contract check also rejects unauthorized removal or
changes to an authored position.

For a verified background with a bright, quiet upper field, the analyzer records
the onset of substantial central image content. If a center crop would remove
that authored headroom, the native viewport shifts enough to retain an upper
margin. This is a local pixel proxy, not recognition of heads or semantic ROIs.
Sources without that evidence retain a center crop. Model-proposed protected
regions remain review-only in the harness.

## Layout and verification

The candidate family preserves complete source-owned header and caption rows,
copy/font-size ratios, and each calendar/control unit. Background viewport
coverage participates in source-aware selection alongside the existing hard
contract, visibility, collision and reading-order constraints. It does not
replace those checks or infer that a cropped subject is fully preserved.
The proportion checker excludes these verified adaptive viewport/paint frames
from rigid-block deformation, while retaining the photo crop, source-pixel,
caption-support and foreground grouping checks. A changed viewport aspect is
not reported as distortion of the pictured subject.

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

Rooftop's prepared photo is already a baked portrait-slot crop of a wider
retained original photograph. This change uses the exact prepared asset rather
than silently replacing it with pixels outside that source. Recovering the full
wide composition requires a separate importer change preserving the original
photo and its exact authored affine viewport, followed by source-render parity
verification. Crop positioning cannot recover pixels absent from an input asset.

The ten reserve posters remain outside resize tuning and model training.
Inspect actual image crops and text support, not just successful contract
checks. Adobe Express crop writeback still needs a separately verified host
round trip; the existing uniform-transform adapter does not establish it.
