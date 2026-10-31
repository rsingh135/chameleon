# Open editable design sources

Curated October 29, 2026. The frozen source catalogue is
[`fixtures/open_posters/catalogue.json`](../fixtures/open_posters/catalogue.json).
It contains **18 distinct author-published works**: eight source-screened review
cases and ten untuned reserves. No paper-size, language, or color variant is
counted as a new work. The collection contains posters, flyers and adjacent
design formats; it must not be described as 18 printed posters or 18 independent
designers.

These are original editable SVG or Doory JSON documents downloaded verbatim.
They are not screenshots reconstructed as editable documents, newly invented
posters, or restricted Adobe practice files. Source screening covers provenance,
license scope, structure and font availability. It is **not** acceptance of an
import, a successful resize, or a claim that every source feature is supported.

## Sources and rights

### Doory: ten original starter designs

Source revision: `2fb4d0297ec4c6e417b5225215502f8217c2a4dc` in
[`shaongitbd/doory`](https://github.com/shaongitbd/doory/tree/2fb4d0297ec4c6e417b5225215502f8217c2a4dc/backend/data/templates).
The publisher identifies its starter JSON files as original compositions,
shapes and sample copy. These are publisher-authored starter designs, not
evidence of a commissioned real-world event. The JSON contains independently
editable text, shapes, paths, groups and images. We downloaded the existing
documents; we did not execute their generator or invent new designs from it.
[Asset scope and per-content licenses](https://github.com/shaongitbd/doory/blob/2fb4d0297ec4c6e417b5225215502f8217c2a4dc/THIRD_PARTY.md).

The templates are under the project's **Attribution Assurance License**, recorded
as `LicenseRef-Doory-AAL`. They are not MIT licensed. The signed license permits
modification and redistribution subject to its attribution and notice
requirements. Preserve its signed `LICENSE`, `NOTICE`, authors and third-party
notices; show the required author credit prominently when these examples are
exposed in the running app. The catalogue's `attribution_text` is:

> Doory by Siam, creator and maintainer of Doory, https://github.com/shaongitbd/doory

The complete controlling text is retained under
[`licenses/doory-LICENSE`](../fixtures/open_posters/licenses/doory-LICENSE) and
[`licenses/doory-NOTICE`](../fixtures/open_posters/licenses/doory-NOTICE).
[Original signed license](https://github.com/shaongitbd/doory/blob/2fb4d0297ec4c6e417b5225215502f8217c2a4dc/LICENSE).

Five associated photos were copied from the same pinned repository with their
pixels unchanged. Doory records the StockSnap URLs and photographers: Porapak
Apichodilok, Helena Lopes, Fuse Brussels, Direct Media and Matt Moloney.
Their own license remains CC0-1.0, rather than the template's AAL. StockSnap
expressly publishes its photos under CC0. The original image references are
mapped to the local bytes in each entry's `asset_requirements`; no stock image
was replaced by a placeholder.
[Photo provenance](https://github.com/shaongitbd/doory/blob/2fb4d0297ec4c6e417b5225215502f8217c2a4dc/THIRD_PARTY.md#starter-template-photos),
[StockSnap license statement](https://stocksnap.io/license).

Doory's author-provided native-app reference thumbnails are retained separately
in `originals/doory_reference/`, with URLs, hashes and dimensions in its
`manifest.json`. They are independent of Chameleon's importer, but have a
720-pixel long side. They are not full-resolution originals or proof of all
pages. The business card and pitch deck cases select page zero explicitly;
their remaining original pages are retained and are not counted separately.

### openSUSE: six marketing works

Source revision: `f9ef4460f8bac3e29ed7fc7e1ac6a1320a765dfb` in
[`openSUSE/artwork`](https://github.com/openSUSE/artwork/tree/f9ef4460f8bac3e29ed7fc7e1ac6a1320a765dfb).
Its README applies CC-BY-SA-3.0 to repository content unless otherwise stated.
The selected SVGs preserve text and vector/group structure; some lettering or
logos are already paths. The three reserve banners are different authored
compositions and copy, not dimensions or translations of one file. Individual
designers are not identified in these source files, so attribution names the
openSUSE Artwork contributors and links to the repository.
[License scope](https://github.com/openSUSE/artwork/blob/f9ef4460f8bac3e29ed7fc7e1ac6a1320a765dfb/README.md#license),
[license text](https://github.com/openSUSE/artwork/blob/f9ef4460f8bac3e29ed7fc7e1ac6a1320a765dfb/LICENSE.md).

Retain the CC attribution/license and identify later adaptations when publishing
them. This copyright license does not imply endorsement by openSUSE or transfer
its trademarks. Native outlined lettering stays disclosed as vector artwork;
it must not be described as editable text merely because its outlines can move.

### blaketyro: two educational posters

Source revision: `7a3efb1ec83f7b5611372c9d42f0a6352411ae8f` in
[`blaketyro/printables`](https://github.com/blaketyro/printables/tree/7a3efb1ec83f7b5611372c9d42f0a6352411ae8f/Project).
The author explicitly licenses both the repository and its printables under
MIT and publishes SVG exports alongside Affinity Designer originals. We selected
one poster-size SVG for each of two distinct topics. Preserve the author's
copyright and MIT license. Neither another paper size nor another export format
is counted as a separate work.
[Scope statement](https://github.com/blaketyro/printables/blob/7a3efb1ec83f7b5611372c9d42f0a6352411ae8f/README.md#license),
[MIT license](https://github.com/blaketyro/printables/blob/7a3efb1ec83f7b5611372c9d42f0a6352411ae8f/LICENSE.md).

Their SVGs request proprietary Helvetica/Consolas faces. Author originals remain
byte-identical in `originals/`; separate, explicitly labeled open-font
adaptations in `adapted/` replace those requests with DejaVu Sans/Mono before
establishing a portable baseline. The Asymptotic Notation adaptation also changes
the exported Cantarell Regular/Oblique weight request from 500 to the supplied
400 face. Each exact replacement and occurrence count is recorded in
`adaptations`, with `prepared_reference_path` and its hash. This is changed
typography, not an assertion that the author used DejaVu or that original font
metrics were preserved. No resized result was used to choose these changes.

## Frozen split

“Reviewed” below identifies the source-screened development set, not rendered
quality acceptance. “Reserve” identifies the separate personal-test pool; no
reserve resize was run during source curation. Prepared packages and actual
render reviews must report their own acceptance or failure.

| ID | Authored format | Split |
| --- | --- | --- |
| `doory_bold-sale-announcement` | Sale advertising post | Reviewed |
| `doory_botanical-celebration-invitation` | Wedding invitation | Reviewed |
| `doory_grand-opening-flyer` | Cafe opening flyer | Reviewed |
| `doory_night-bloom-concert-poster` | Concert/tour poster | Reviewed |
| `doory_rooftop-event-story` | Event story/promotion | Reviewed |
| `opensuse_trifold_front` | Trifold flyer, front | Reviewed |
| `opensuse_booth_poster` | Booth poster | Reviewed |
| `opensuse_cheatsheet` | Command-reference flyer | Reviewed |
| `doory_bold-youtube-thumbnail` | Video thumbnail | Reserve |
| `doory_clean-business-card` | Business card, first page | Reserve |
| `doory_modern-pitch-deck` | Presentation cover, first page | Reserve |
| `doory_modern-resume` | Resume | Reserve |
| `doory_serif-quote-card` | Quote card | Reserve |
| `blaketyro_uml_arrows` | Educational poster; disclosed font adaptation | Reserve |
| `blaketyro_asymptotic_notation` | Educational poster; disclosed font adaptation | Reserve |
| `opensuse_banner_we_believe` | Values/community promotional banner | Reserve |
| `opensuse_banner_makers_choice` | Promotional banner | Reserve |
| `opensuse_banner_leap` | Release celebration banner | Reserve |

The reviewed set includes photographic, text-led, invitation, dense reference,
brand and concert treatments. Source diversity is limited to three collections;
five reviewed works share the Doory publisher. This is a useful bounded test
set, not a representative sample of professional design or an unseen benchmark.

## Strict validation replacement and retained stress original

Fresh source parsing rejected the Connect banner: its visible legacy Inkscape
`flowRoot` captions, “LINUX” and “FOR OPEN MINDS”, were omitted by the native
reference renderer. Its requested Open Sans Condensed 300 face was also absent
from the original resource inventory. It is no longer counted among the selected
18 or the ten usable reserves. The catalogue retains its complete record under
`excluded_examples`, including the reason, original URL/hash, prepared path and
license; its original SVG and existing files are unchanged. This is an excluded
parser stress source, not an accepted import.

The replacement is the distinct author-published
[We Believe banner](https://github.com/openSUSE/artwork/blob/f9ef4460f8bac3e29ed7fc7e1ac6a1320a765dfb/banners/vector/banner-we-believe.svg)
from the same licensed revision. Its unmodified original SHA-256 is
`c3e021b36f08bc29924538f70606e0a98bbd7819f936fcd9e086006b45ba7590`.
It uses existing exact Open Sans 400 and Source Sans Pro 300 faces, with no font
addition or font adaptation. Source Sans Pro's legacy family name is “Source Sans
Pro Light”; the source's weight-300 request resolves to the pinned
`SourceSansPro-Light` face through an explicitly recorded provider alias.

Strict preparation produces five objects: two live text elements, one native
background rectangle and two disclosed rigid artwork assets. The illustration's
already outlined lettering remains artwork, not editable text. The full native
and imported source renders visibly contain the headline, URL, logo and cube in
matching positions; their normalized mean RGB error is approximately 0.000496.
Validation also fixed a generic render-only CSS style join that had incorrectly
hidden the headline in the first reference render. The original SVG bytes were
never changed. No reserve resize was run. Native source raster agreement does
not establish historical author-machine font-byte identity or resize quality.

## Exact resources and remaining qualifications

[`fonts/source-manifest.json`](../fixtures/open_posters/fonts/source-manifest.json)
records 31 faces with named family, weight/style, PostScript name, source URL,
SHA-256 and license. Fontsource originals are retained as WOFF2 alongside TTF
conversions for the renderer. Conversion changes the container, not the named
family or requested face. No unavailable face is silently substituted and no
bold face is simulated from a regular one.

The Doory build resolves Fontsource package versions dynamically and does not
save its font hashes. We pin the retrieved 5.3.0 family packages, but cannot
establish historical byte equality with its reference thumbnails. Consequently,
native-reference comparison must account for this known limitation rather than
claim pixel-exact original typography. The upstream fetch contract is preserved
as source text under `licenses/doory-frontend_scripts_templates_fonts.ts` and
`fontPalette.ts`; it was inspected, not executed.
[Pinned upstream font fetching policy](https://github.com/shaongitbd/doory/blob/2fb4d0297ec4c6e417b5225215502f8217c2a4dc/frontend/scripts/templates/fonts.ts).

The font licenses travel with the actual faces. Doory's selected faces are
OFL-1.1. The legacy Open Sans Condensed 700 file comes from Fontsource 4.5.5;
its font name table links to Apache-2.0 and credits Google. Its npm wrapper's
MIT label does not replace the font license. DejaVu fonts are original TTFs
from the official 2.37 release, with the archive/member provenance and
Bitstream/DejaVu license retained.
[DejaVu official release](https://github.com/dejavu-fonts/dejavu-fonts/releases/tag/version_2_37),
[DejaVu font license](https://dejavu-fonts.github.io/License.html).

The Asymptotic Notation source contains encoded U+FB00/U+FB01 ligatures. The
Latin-subset Cantarell files omit those codepoints, so that case selects full,
unmodified upstream Cantarell TTFs instead, pinned at Google Fonts revision
`9862abc066c9b83bf84c3e8ed1fc2d8d48f9bbf0`. The original copy is unchanged;
there is no compatibility normalization of ligatures to different characters.
The superseded subset-source records remain in the font manifest for provenance.
[Full Cantarell faces and OFL](https://github.com/google/fonts/tree/9862abc066c9b83bf84c3e8ed1fc2d8d48f9bbf0/ofl/cantarell).

Two registration details need explicit handling:

- SVGs may request PostScript-style family aliases such as Cantarell-Regular or
  Inconsolata-Bold. The catalogue records the authored stack and supplied face
  separately; names must be checked against the actual font, not inferred from
  a filename.
- Fontsource's static SUSE 700 has OS/2 weight 700 and no variable axis, but its
  legacy family is “SUSE Thin” and its PostScript name is “SUSEThin-Bold”. Its
  provider identifies the face as SUSE 700. An importer must explicitly support
  that evidenced alias or reject it; it must not quietly fall back to Open Sans.

An authored generic `sans-serif` does not identify a historical font. The
catalogue declares a portable Open Sans policy as `explicit_generic_policy`,
separate from exact named-face resolution. This choice must be visible in
prepared provenance. It does not authorize replacing a named source face.

Doory uses Fabric-style geometry, including centered origins, local group
transforms, text scaling, gradients, outline text, photo slots and custom lines.
The OpenSUSE/educational SVGs include nested transforms, clip paths, symbols,
tspans and embedded images. A feature inventory is not a successful import.
Preserve these semantics or return a specific unsupported-feature result;
dropping a clip, flattening all text, guessing a gradient or omitting a photo
would not be faithful compatibility. Read-only helper references for mask,
slot and line semantics were saved under ignored
`output/open-poster-research/doory-helpers/` with pinned provenance; no upstream
application code was executed or merged.

## Reproduction and verification boundary

Each catalogue entry gives its pinned original URL, source commit, local path
and SHA-256. Download bytes from that URL and compare the hash before parsing.
Do not overwrite an author original with normalized XML or a prepared task.
`prepared/<id>` is a separate, per-source output location for an importer; the
catalogue alone does not certify that location exists or is accepted.

At source freeze, all 18 original checksums matched, all referenced font files
and licenses existed, and the split was eight reviewed plus ten reserves.
Prepared compatibility, source-render fidelity, resize validity and visual
quality are subsequent checks owned by their respective reports. In particular,
an unsupported source is not counted as a successful engine test.

Other leads were excluded rather than relabeled: Adobe tutorial samples have
restricted practice-file terms; sampled GNOME marketing files lacked a verified
specific license in their file/repository; several well-licensed FSFE posters
publish Scribus originals that the current SVG/Doory import path does not
consume. Public availability alone was not treated as redistribution permission,
and a flattened PDF or screenshot was not counted as an editable original.
