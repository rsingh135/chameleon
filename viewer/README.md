# Local studio and file inspector

A dependency-free local inspector for development. The final challenge PDF
requires the engine for stage one and a full Adobe Express add-on for the
finale. This inspector is not an installed Express add-on.

Run `python -m viewer.server --port 8765` from the repository root and open
[http://127.0.0.1:8765](http://127.0.0.1:8765) for the live studio. Choose a sample
or import task/source JSON, select a preset or custom canvas, resize, inspect
element properties, and download the resulting JSON or full-resolution PNG.
All image references use the bundled `brand_kit/assets/` directory; binary asset
uploads are not implemented. Missing assets are displayed as labeled placeholders.

## Studio and comparison workspaces

[Docked studio](http://127.0.0.1:8765/) is the default GUI: source and target
setup on the left, artwork in the center, and elements/properties on the right.
Choose **Compare** to switch to the Review desk layout, with a horizontal setup
ribbon, paired original/result previews, and inspection below. Choose **Result**
or **Original** to return to the docked layout. The comparison view can be opened
directly at [/?view=compare](http://127.0.0.1:8765/?view=compare).

Fold the setup panel with its heading chevron, the left rail's panel button, or
**Setup** in the header. The header button remains available in comparison mode
and on narrow screens. Escape inside setup folds it and returns focus to the
opener. Source/canvas shortcuts reopen the panel and focus the relevant control.
Setup visibility and appearance are remembered in local browser storage when
available. View changes preserve the document, dimensions, selection, appearance,
and export state; they do not trigger a resize.

No frontend build step or new dependency is required. `app.js` owns the document,
rendering, imports, and exports; `workspace.js` and `workspace.css` own the frame.
`startup-art.js`, `startup.js`, and `startup.css` provide a five-second, procedural
Canvas introduction: a chameleon walks in, catches a color swatch, and changes
its printed colors. Its halftone palette and ink registration take inspiration
from [Kevin Ngo's creative-coding animation](https://x.com/kevin_t_ngo/status/2099477219877978289).
The intro plays once per tab session while the engine loads independently.
**Skip intro** or Escape opens the studio immediately; **Replay intro** in the
footer plays it again. Reduced-motion preferences skip it entirely. It also
dismisses when the tab is hidden and fails open if Canvas is unavailable.
The former trial URLs redirect to these production views. The full three-option
experiment is captured in
[checkpoint e9d651b](https://github.com/rsingh135/chameleon/tree/e9d651b), with
reference screenshots still in `viewer/assets/trials/`.

The server binds only to loopback and accepts at most 2 MiB per request, 200
elements, and 20 million pixels per canvas with a maximum side of 8192 px.
`python -m tools.verify_local` checks the running API and writes
`output/local-verification.json`.

## Workspace design

The studio uses an independent, Spectrum-inspired application frame: charcoal
panels, a neutral artboard background, a red Chameleon identity, and exports next
to the current document. A single Result canvas is shown by default;
Compare and Original are explicit options. Design review notes, elements and properties are
collapsible; canvas setup can be folded at every width. A light appearance is
also available. Changing appearance or preview view does not change exported
artwork.

References: [Adobe's app frame](https://spectrum.adobe.com/page/application-frame/),
[Emil's Sonner launch on X](https://x.com/emilkowalski/status/1628742238548250624)
and [his explanation of its interaction details](https://emilkowal.ski/ui/building-a-toast-component).
The X post is linked from the author's accessible article; direct X access was
blocked during research. Frequent actions remain immediate; press feedback uses
a 160 ms transform transition, loading feedback stays in the canvas, and reduced
motion disables animation.

Source Sans 3 regular and semibold WOFF2 files are bundled in `assets/fonts/`
under Adobe's original SIL OFL license. They are unchanged files from
[Adobe's release commit](https://github.com/adobe-fonts/source-sans/tree/87b37a2daaed80fcb8e8ccb0085c4d72ddade12e),
verified against upstream Git blob hashes. No external font service or frontend
build step is required. The supplied `brand_kit` assets and fonts remain intact.

For standalone file inspection, open `viewer/inspector.html` (no server needed),
then load:

1. a task JSON from `input/` (source design + target canvas),
2. the matching `*_output.json` from `output/`,
3. optionally the rendered `*_output.png`, which is drawn under the output's
   element boxes so you can check the JSON against what the renderer drew.

Element boxes are colour-coded by type, labelled with their `id`, and show
their full properties in the inspector on hover. Text is drawn with system
fonts as a rough preview only; trust the PNG, not the canvas text.
