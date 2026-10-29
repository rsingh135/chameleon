# Source-guided reflow of original editable posters

Audit date: October 29, 2026. This note records the first two reviewed original
poster failures and the bounded candidate family added in response. It supplements
[the source catalogue](OPEN_POSTER_SOURCES.md) and
[composition research](COMPOSITION_REFLOW_RESEARCH.md). These are local engine
observations, not an Adobe host validation or a claim that all catalogue examples
have passed visual review. Reserve examples have not been resized for this work.

## What passed mechanically but failed visually

The Doory Grand Opening original has a small business name above a large script
title, two fine rules accompanying an invitation, and one coherent event row:
time, day/date/month, address and two vertical dividers. The original exact copy,
typefaces, tracking and element data are retained in its prepared source.

The first selected baseline under
`output/poster-first-baseline/doory_grand-opening-flyer/` passed the local
technical checks while reversing the business-name/title relationship, scattering
event information and thickening fine rules. Membership retention did not measure
these failures. The implementation explains the observed behavior:

- Legacy stacked flow sorts inferred roles before source position. Font-based
  role inference can consequently put a smaller authored brand after its headline.
- Standalone rules receive independent rectangular slots and anisotropic scaling.
  Their original width/thickness relationship is lost.
- The legacy uniform proposal uses a font multiplier that can reach 1.3 times
  its geometry scale, then fits individual text separately. This is not one
  foreground similarity transform. Frozen absolute tracking can cause a further
  isolated font reduction when a frame shrinks.

A direct 1600 by 900 legacy candidate repro found role-flow proposals reversing
the business/title order and an approximately 85.5% rule aspect-ratio deviation.
Its nominally uniform proposal retained rule proportions but reduced minimum text
to approximately 6.50 pixels; matching font and geometry scales instead gives
12.78 pixels for this original. These numbers describe candidate generation, not
the scores or results selected by the integrated engine.

The openSUSE booth original has a source intro followed by five icon/description
rows. Its first baseline under `output/poster-booth-baseline/` retained the pairs
but rendered the foreground as a narrow, small central column on a landscape
canvas. Pair ownership alone did not ensure useful target-space occupation.
Its imported live text also retains generous frame tails below the final line.
Connecting full frames by y overlap merges the intro, URL and first description
into one long band even though their occupied lines are separated.

## What the primary sources justify

Canva Engineering describes implicit composites whose configuration carries
meaning and must survive a transformation. It also documents why whitespace-only
grouping fails on complex designs, and why element frames containing empty padding
can disagree with painted content. Its production method uses both rendered
evidence and element data. This supports the direction of our rigid rows and
occupied-support affinity, but does not validate our geometric heuristic as
semantic recognition. [Canva Engineering: How we see groups in design](https://www.canva.dev/blog/engineering/how-we-see-groups-in-design/).

Adobe teaches that changing one element's scale changes its relationships to the
rest of a composition, and that hierarchy concerns the intended order of
importance. Alignment, balance and negative space contribute to a unified whole.
This supports checking source order, relative prominence, fine ornament geometry
and actual composition. It does not prescribe one numerical scale policy or say
every poster should be centered. [Adobe Learn: Understanding the Basic Principles
of Graphic Design](https://www.adobe.com/learn/express/web/graphic-design-basics).

Spectrum distinguishes spacing within a component from spacing among larger
regions, with proximity communicating relationships. Applying that interface
principle to posters is an engineering analogy: preserve tight local relationships
while allowing larger inter-section gaps to adapt. It does not imply that all
negative space is disposable. [Adobe Spectrum: Spacing overview](https://spectrum.adobe.com/foundations/layout-and-structure/spacing/spacing-overview).

## The initial bounded proposal family

`engine/layout/source_guides.py` adds proposals, not a new unconditional layout
policy. Existing selection still has to compare readability, focal intent,
relationships and quality with the other candidate families.

The canonical proposal preserves the foreground's original canvas center and
applies exactly one scale to positions, dimensions, fonts, rules and images.
Backgrounds cover the new canvas. It includes photographs and intentional
overlaps; it does not independently clamp or fit leaves. A standalone text frame
may widen about its original text-alignment anchor when immutable tracking needs
more room. Its font is unchanged relative to the common scale, and that frame
exception is recorded. If the text cannot fit faithfully, that proposal is omitted.

Band proposals start with explicit source units and connect their occupied
y support. Complete frames and internal geometry remain rigid during placement;
occupied support only informs affinity and meaningful inter-band cuts. Bands keep
their source order. Three bounded vertical gap policies and at most two
source-whitespace column cuts provide alternatives to the conservative fit.
A spanning-first-band alternative is also offered. All foreground bands retain
one common font/geometry scale; only singleton text frames may widen. There are
at most eight returned states, deduplicated by design contents.

Arbitrary photographs are excluded from band detachment. An imported image can
participate only when explicit `source_art.kind = "svg-nontext"` provenance
identifies retained original vector artwork. This is a provenance-based capability,
not recognition of its semantic meaning. Exact asset pixels and the original SVG
remain retained. Grouped text frames do not widen through their plates or peers.

No source IDs, poster titles, brand names or copy strings enter the generator.
The optional relationship graph is accepted without mutation. These source-band
states do not pretend to infer or rewrite the existing graph's cluster axes.

## Initial local verification and limits

Five focused regression tests cover the actual Grand Opening source's business
order, all four rule proportions and the seven-member event-row transform;
canonical font/geometry equality with a photo; tracking without isolated font
shrink; explicit vector eligibility; and deterministic immutable outputs.
The tested Grand proposals have no local contract violations. Ruff passes.

Full 1600 by 900 native renders were produced through the same exact-font-alias
path used by the viewer. A first attempt through the bundled converter bypassed
document aliases; those PNGs were replaced and were not accepted as visual
evidence. The two full-resolution proposals inspected after replacement were:

- `output/source-guides-proposals/doory_grand-opening-flyer/source-guides-bands-0.35.png`:
  the business name remains above Grand; the original thin rules and complete
  event row remain coherent. Minimum text is approximately 16.86 pixels.
- `output/source-guides-proposals/opensuse_booth_poster/source-guides-columns-4.png`:
  the source intro occupies the left section and all five original paired rows
  occupy the right. Minimum text is approximately 22.07 pixels, compared with
  12.5 in canonical source fit. For this initial prototype render, vector markers
  were projected from the original import-report mapping; the importer owns their
  permanent materialization.

These are inspected proposals, not claims that the integrated selector chose
them. The other generated column arrangements can still be weaker: an arbitrary
cut can split a repeated list between sections or create excess whitespace.
Selection must not prefer larger minimum text blindly. Very dense artwork,
intentional overlaps across rows, diagonal layouts, ambiguous implicit groups,
multi-font runs and exact glyph ink are beyond this initial row heuristic.
Occupied rectangles are approximations, not pixel collision proofs. The same
family must continue to be compared against the source and full native renders;
the central-product Solstice brief remains a separate source-specific priority.

## Doory text-coordinate corrections before target review

Independent author thumbnails exposed source-import defects before any target
layout could be judged: the Anton headline in the YouTube template was roughly
35 pixels too low at a 720-pixel-wide reference, Poppins sale headings were about
11 pixels low, and the night-bloom venue/date block was 14–16 pixels low at a
720-pixel-high reference. These were text baseline conventions, not evidence
that the resized layout itself had moved the source frames.

The pinned Doory commit
`2fb4d0297ec4c6e417b5225215502f8217c2a4dc` resolves Fabric 7.4.0 in its
[frontend lockfile](https://github.com/shaongitbd/doory/blob/2fb4d0297ec4c6e417b5225215502f8217c2a4dc/frontend/package-lock.json).
The [exact Fabric archive](https://registry.npmjs.org/fabric/-/fabric-7.4.0.tgz)
was checked against that lockfile's SHA-512 integrity before its source was read;
no upstream application code was executed. `Text.ts` uses a line height of
`fontSize * 1.13`, then subtracts that height times `0.222` in `_renderChars`.
Consequently the first baseline is `fontSize * 1.13 * (1 - 0.222)`, or
`0.87914em`. Relevant SHA-256 records are:

- Doory lockfile: `616869ccfa4243cce334379bf34a62e14a2876bc87ab405b07989294e2f4a9c0`.
- Extracted Fabric `Text.ts`: `0c48f494e1ad1b0ee3b0621531bb0ec4940baac260cf3504be20e38ee65dad0b`.
- Extracted `Text/constants.ts`: `e46108f3750960540f3a3b899a8bf124150e69fb2904c9fa24a9b4cd37cead53`.

The importer now declares this baseline in each Doory text element's frozen
`native_text_effects` metadata. Live glyph painting applies it at the current
fractional font size. Original copy and source-frame coordinates remain intact;
ordinary Adobe, SVG and bundled-brand text keep their existing baseline path.
The same descriptor retains live gradient/outline paints and exact PostScript
font aliases for mixed-face runs. It does not contain a flattened text image.

The source-only Grand body also wrapped differently because the bundled converter
omitted tracking when deciding line breaks. The document-only
`engine/native_typography.py` now measures the actual run faces and signed
tracking for both fitting and rendering. It replaces existing spaces with
newlines at stable source indices, preserving style-run ranges and original copy.
Trailing whitespace cannot eject an otherwise fitting word. The corrected Grand
first line ends after “PASTRIES”, matching the author reference.

The focused importer/effect/native-typography suite passes 41 tests, including
independent author ink-position checks for real Anton, Poppins and Oswald text.
Poppins headline ink spans
rows 260–305 versus reference rows 260–306 at the reference scale. Fresh source
comparisons are recorded under `output/doory-source-fidelity-final/`; all ten
selected pages have source-only evidence, including reserves. This establishes a
more faithful input for the target matrix, not visual approval of its reflows.
References are compressed author thumbnails, and historical upstream font-build
byte identity remains unknown; exact pinned named-face bytes and
licenses are retained and disclosed.

The first selected matrix exposed a separate unit error: Doory's `charSpacing`
is measured in thousandths of an em, but the initial adapter materialized it as
immutable pixel tracking. At landscape scale, `PAPER TIGERS` consequently painted
739.05 pixels of ink over a 581.25-pixel lavender plate. The outer black letters
became invisible against the black page even though its enlarged text frame
passed containment checks. Doory now preserves native `tracking_em` in the frozen
descriptor and keeps ordinary `letter_spacing` zero; fitting and painting both
evaluate tracking against the current font size. SVG and ordinary Adobe pixel
tracking retain their existing units. Source-size pixel equality and an actual
half-size tracked-glyph/plate test guard this correction. This failure also shows
why a frame-only containment pass cannot establish foreground/support fidelity.
