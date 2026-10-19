# Viewer

A dependency-free stand-in for the "viewer add-on" the FAQ lists as a
required-but-unscored stage-one deliverable, until the Express add-on is
built for the finale.

Open `viewer/index.html` in a browser (no server needed), then load:

1. a task JSON from `input/` (source design + target canvas),
2. the matching `*_output.json` from `output/`,
3. optionally the rendered `*_output.png`, which is drawn under the output's
   element boxes so you can check the JSON against what the renderer drew.

Element boxes are colour-coded by type, labelled with their `id`, and show
their full properties in the inspector on hover. Text is drawn with system
fonts as a rough preview only; trust the PNG, not the canvas text.
