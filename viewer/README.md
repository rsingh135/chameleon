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

The server binds only to loopback and accepts at most 2 MiB per request, 200
elements, and 20 million pixels per canvas with a maximum side of 8192 px.
`python -m tools.verify_local` checks the running API and writes
`output/local-verification.json`.

## Workspace design

The studio uses an independent, Spectrum-inspired application frame: charcoal
panels, a neutral artboard background, a red Chameleon identity, and exports next
to the current document. A single Result canvas is shown by default;
Compare and Original are explicit options. Design review notes, elements and properties are
collapsible; canvas setup can be opened on narrow screens. A light appearance is
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
