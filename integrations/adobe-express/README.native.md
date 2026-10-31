# Native editable development integration

This experimental adapter is separate from the distributable stage-one
[Design Viewer](README.md). Use `npm run start:native` and
`manifest.native.json` to test it locally. The submission ZIP excludes its
sandbox and UI code.

This add-on reads editable native elements from the current Adobe Express page,
sends a schema document plus its own image/font resources to the local Chameleon
engine, and creates a **new editable page** from native clones. It never replaces
the original page and never turns a whole poster into a screenshot.

The SDK boundary is checked against Adobe's published **1.40.0** TypeScript
declarations. Adapter tests and compilation are local checks, **not evidence of
execution inside Adobe Express**. A real Express document must still be captured,
reflowed, inspected and exported in the host before claiming end-to-end support.

## Build and check

```powershell
Set-Location 'D:\My Projects\chameleon\integrations\adobe-express'
npm ci --ignore-scripts
npm test
```

`npm test` compiles the UI and sandbox against the official SDK and runs the pure
adapter/roundtrip tests. `dist/manifest.native.json`, `dist/native-ui/` and `dist/sandbox/` form
the add-on. The no-bundle output uses relative ES modules, which the document
sandbox supports. `src/sdk-types.d.ts` redirects runtime module names to Adobe's
public package type exports; it does not widen or invent host APIs.

## Load it in Adobe Express

Adobe's documented local workflow uses an **HTTPS development server**, normally
`https://localhost:5241`, and **Test your local add-on** in Express. ZIP upload is
the separate private/public distribution workflow. This experimental build is
for local development; it must not be submitted or distributed while it depends
on experimental APIs.

This repository does not install or trust a certificate. If a trusted localhost
certificate/key already exists, supply those paths explicitly:

```powershell
$env:CHAMELEON_ADDON_TLS_CERT = 'C:\path\to\localhost-cert.pem'
$env:CHAMELEON_ADDON_TLS_KEY = 'C:\path\to\localhost-key.pem'
npm run start:native
```

Then open [Adobe's add-on development entry point](https://www.adobe.com/go/addon-cli),
enable development/testing if needed, and connect the local server. Review any
Adobe developer terms and browser local-network permission yourself. Creating or
trusting a certificate is a separate machine-level action; the server does not
do it. Existing Adobe CLI development certificates may be supplied if appropriate.

Start the Chameleon viewer/engine separately. The default local engine URL is
`http://127.0.0.1:8765`. The engine must allow this explicit add-on development
origin in its CORS policy. There are no secrets or remote asset URLs in this add-on.

## Native-to-schema mapping

| Native content | Schema mapping and editable writeback |
| --- | --- |
| Standalone auto-width/auto-height text | Literal `fullContent.text`, exact PostScript face alias, font size, tracking, alignment and per-run colors. Preserve font family/style metadata. Clone native text, never reset its content or character-style array. |
| Image `MediaContainerNode` | Actual image resource bytes, subject to `canExportMedia()`. Keep the native container editable. Initial support requires an uncropped rectangular, unrounded mask. No image-rendition fallback. |
| Solid rectangle / ellipse | Editable native clone, solid fill, opacity and uniform rectangle corner radius. Stroked shapes are refused because the current local renderer omits their strokes. |
| Plain local-horizontal line | Solid stroke and width, with native/ancestor rotation retained. Other endpoint geometry, arrowheads and dashes are refused. |
| Nested native group | Keep the complete tree in `document.group_tree`; flatten only the engine's leaf representation. All descendant leaves use the **outermost** native group as `group_id`. Native writeback keeps that group and every nested group intact. |
| Native transforms | Compose local matrices to obtain a world center, dimensions and rotation. Only orientation-preserving similarity transforms are representable. Reject reflection, shear or nonuniform ancestor scale. |
| Artboard fill | Explicit source background rectangle; copy the original native artboard fill to the new page. |

Express character ranges use UTF-16 code units, while the Python engine uses
Unicode code points. The adapter converts boundaries and rejects split surrogate
pairs, out-of-content ranges and incomplete style coverage.

The schema cannot represent mixed font faces, mixed sizes or mixed tracking
inside a text node. Such nodes stop import. A face such as `SourceSans3-Semibold`
is a distinct exact font alias, not a guessed conversion to regular/bold. The
matching font file must be supplied before the engine is called; the backend must
verify that its internal PostScript name matches. Native fonts are never exported
by this add-on: the SDK exposes font names, not licensed font-file bytes.

The SDK documents font sizes in **points** and geometry in **pixels**. The adapter
records the explicit `96/72` point-to-pixel convention in
`document.font_pixels_per_point`; native writeback uses dimensionless size ratios.
Adobe's default font leading is not exposed as an absolute pixel value. The
schema uses `1.2 × native lineSpacing` as its preview convention. Exact font files
prevent substitution but do not prove raster parity between Adobe and the local
text renderer. Compare the actual host output before treating a preview as exact.

The SDK returns image **resource** bytes and currently exposes no general image
filter-inspection API. Native clones preserve existing image edits, but resource
preview parity for externally filtered images remains unverified. The UI states
this limitation. Such filtered documents are outside the verified subset; no
claim of their exact preview is made.

Other explicitly refused features include unknown node types, video/linked media,
threaded/area/circular/magic-fit text, text effects, unavailable fonts, links,
underline, baseline shifts, justified text, paragraph lists/spacing, gradients,
unequal corners, non-default blends, group masks and nontrivial group opacity.
Group opacity cannot be replaced with individual child opacity without changing
overlap compositing.

## Local engine contract

The iframe sends `POST /api/document/resize` with:

```json
{
  "source": {"canvas": {"width": 600, "height": 800}, "elements": []},
  "target_canvas": {"width": 1080, "height": 1920},
  "assets": [{"asset_id": "document-image.png", "mime_type": "image/png", "data_base64": "..."}],
  "fonts": [{"font_family": "SourceSans3-Semibold", "postscript_name": "SourceSans3-Semibold", "weight": "regular", "style": "normal", "data_base64": "..."}],
  "document": {
    "snapshot_id": "opaque-capture-identity",
    "page_id": "native-page-id",
    "artboard_id": "native-artboard-id",
    "node_bindings": [],
    "group_tree": [],
    "font_pixels_per_point": 1.3333333333333333
  }
}
```

The abbreviated arrays above describe the transport shape, not an executable
empty source. Each source element appears exactly once. The backend owns
per-document asset/font isolation and must refuse missing or mismatched resources
instead of consulting a sample brand kit or fetching arbitrary URLs.

The response contains `{design, document: {snapshot_id}, source_png?, output_png?, report?, warnings?}`.
`preview_url` and `render_url` are accepted aliases for `output_png`; previews may
be bounded PNG base64 data URLs or URLs on the configured local engine origin.
The echoed snapshot ID and returned
canvas must match the request; a stale response is refused. Error responses use `{error}`.

The backend checks image pixel decoding and exact font PostScript names before
materializing a package. Native end-exclusive color ranges are adapted on a
render-only copy for the legacy converter; returned source/design ranges remain
unchanged. Resource transport is capped at 24 MiB, decoded bytes at 16 MiB and
each file at 8 MiB. The immutable local cache accepts at most 64 packages or
256 MiB and then reports an error; it never removes an existing document.

Target/font/engine edits invalidate the previous result. An in-flight result is
discarded if these inputs change. Before writing, the sandbox re-reads source
geometry/styles and re-exports image resources to detect replaced images. It then
checks that every original native root admits **one common scale and translation**.
Internal group rearrangement, font-only scaling, changed text/styles, dropped or
duplicated members, and changed stacking are refused rather than applied.

Writeback uses the real SDK methods `cloneInPlace()`,
`editor.documentRoot.pages.addPage({width,height})`, `children.append()`,
`resize({behavior: constants.ResizeBehavior.proportional, ...})`, and
`setPositionInParent()`. It re-reads every cloned leaf and verifies actual geometry,
font sizes and style fields. Any mismatch removes the cloned nodes and newly
created page through `removeFromParent()` / `pages.remove()`. Original native
objects are never resized or restyled. A failed grouped layout can still be
inspected in the local engine; it cannot be written as an editable native group.

## Official API evidence

- [Add-on architecture](https://developer.adobe.com/express/add-ons/docs/guides/learn/platform-concepts/architecture)
- [Project structure and ES module sandbox](https://developer.adobe.com/express/add-ons/docs/guides/getting-started/addon-project-anatomy)
- [Local loading workflow](https://developer.adobe.com/express/add-ons/docs/guides/getting-started/hello-world)
- [Manifest requirements](https://developer.adobe.com/express/add-ons/docs/references/manifest/)
- [Text and styling](https://developer.adobe.com/express/add-ons/docs/guides/learn/how-to/use-text)
- [Positioning and local registration points](https://developer.adobe.com/express/add-ons/docs/guides/learn/how-to/position-elements)
- [Proportional resize](https://developer.adobe.com/express/add-ons/docs/guides/learn/how-to/resize-elements)
- [Native node cloning](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/node)
- [PageList.addPage and asynchronous page access](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/page-list)
- [Image export entitlement and experimental bitmap access](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/image-rectangle-node)
- [Bitmap byte access](https://developer.adobe.com/express/add-ons/docs/references/document-sandbox/document-apis/classes/bitmap-image)

The adapter tests use synthetic native snapshots. They do not impersonate a
downloaded Adobe template or claim live Adobe rendering.
