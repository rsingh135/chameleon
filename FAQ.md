# CHAMELEON — Frequently Asked Questions

*Adobe North America Hackathon 2026 — Pilot Edition*

---

## Input & Output

**Q1. What is the input?**

The engine receives a single JSON object with two parts:

- **`source`** — the design you're resizing, at its original size. This is a `canvas` (width/height) plus a list of `elements`. Every element is one of three types — `text`, `image`, or `shape` — and carries a position, size, and type-specific properties (font, color, asset reference, etc.). There are no semantic labels telling you which element is "the headline" or "the logo" — you get `id`, `type`, and geometry, same as the actual engine will at evaluation time.
- **`target_canvas`** — the width and height you need to resize the design into.

Your engine's job is to return the same elements — same `id`s, same count, nothing added or removed — with updated position/size (and, for text, optionally font size) so the design fits `target_canvas`.

The full field-by-field spec is in the design schema (`chameleon_design_schema_v0.9.json`). Three worked examples, one per difficulty level, ship in the starter repo's **`input/`** folder:

| File | Elements | Resize |
|---|---|---|
| `chameleon_sample_level1_warmup.json` | 4 | 1080×1080 → 1080×1350 |
| `chameleon_sample_level2_core.json` | 11 | 1080×1080 → 1600×400 |
| `chameleon_sample_level3_advanced.json` | 17 | 1080×1350 → 728×90 |

Open one of these alongside the schema — it's the fastest way to see the shape of the data before writing any code against it.

**Q2. Why can't we give the JSON in Crello's format, since you're using cr-renderer to render it?**

Because Crello's format is a research dataset's internal storage layout, not something built for a hackathon spec to read:

- Coordinates are normalized `[0,1]` fractions of the canvas, not pixels.
- Categorical fields (element type, font, etc.) are stored as integer indices — you can't read them without the dataset's `features` object to decode them.
- Images are embedded, pre-rendered pixel crops baked into the data, not a reference to an asset file.
- Rotation is stored in units of π, not degrees.

That's plumbing specific to how CyberAgent happens to store the Crello dataset — it has nothing to do with the actual resize problem, and shouldn't cost you time figuring it out. Our schema keeps pixel coordinates, plain field names, and simple `asset_id` references so the *design problem* is what you're reading, not a dataset format.

cr-renderer is still doing the real rendering work under the hood — we convert our schema into Crello's format right before the render call. That conversion is on us, not something you need to touch.

---

## Rendering & local preview

**Q3. How do I render my output to see what it actually looks like?**

Run `python -m engine.main` from the repository root after installing the
runtime dependencies. The converter is already vendored under `renderer/` and
uses `cr-renderer` to produce real previews in `output/*_output.png`.

For comparisons across target formats, run `python -m tools.harness --targets
all --out output/evaluation`. It writes quality reports and contact sheets.

It's already been verified end-to-end against all three sample designs from Q1, including the level-3 file's intentionally-missing asset — that case renders as a labeled placeholder instead of crashing, so you can see exactly what happens if your own output ever references an asset that isn't there.

What it handles for you automatically:
- Text wrapping at real glyph widths, using the actual brand fonts — not an estimate.
- Rotated elements (e.g. a ribbon at an angle).
- Shape elements like rounded-rectangle buttons — Crello's renderer has no native shape primitive, so these get rasterized before rendering.
- Missing assets — shown as a placeholder, not a crash.

Requires Python 3.10+. If your system Python is older (e.g. 3.9), use the repo's own virtual environment as shown above rather than your system interpreter — cr-renderer won't install otherwise.

The optional `viewer/index.html` is a local JSON/PNG inspector for development.
The final challenge PDF places the full Express add-on in the finale; stage one
requires the engine. Neither the viewer nor local quality metrics replace the
official scoring pipeline.

**Q4. Where do I find the actual brand assets and fonts?**

They ship in the starter repo at:

- **`brand_kit/assets/`** — the image files referenced by `asset_id` in the sample JSONs: backgrounds, logo, product art.
- **`brand_kit/fonts/`** — the font files matching the `font_family` values used in the schema (`Solstice Serif`, `Solstice Sans`).

Point `transformer-convertor` (Q3) at `brand_kit/assets/` when previewing your output, and make sure any asset/font-loading code in your own engine or viewer resolves paths against these same two folders.

One deliberate exception: the Level 3 sample's `holiday_watermark` element references `solstice_holiday_badge_2025.png`, which is **not** in `brand_kit/assets/` — that's the intentional missing-asset case from Q1, not a packaging gap. Don't add it.

---

## Using the starter kit & submitting

**Q5. How do I use the starter kit, and how do I submit?**

**Using it:**

1. Unzip the starter repo. Install dependencies: `pip install -r requirements.txt`.
2. The public entry point is **`engine/resize.py`**. Layout generation and source
   analysis live under `engine/layout/`; local validation and quality metrics
   live under `engine/eval/`. Input/output and rendering are wired into
   `engine/main.py`.
3. Run it locally: `python -m engine.main`. This processes every task file already sitting in `input/` (the same 3 sample designs from Q1) and writes `<task>_output.json` + `<task>_output.png` into `output/` for each. Check the PNGs first — that's your fastest feedback loop on whether your approach is actually working.
4. The real renderer is already vendored. A missing runtime dependency is now
   reported as an error rather than a box-and-label preview.
5. Before you consider yourself done, confirm it works **in Docker**, exactly the way scoring will run it:
   ```
   docker build --platform linux/amd64 -t my-chameleon-engine .
   docker run --rm --network none \
     -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output \
     my-chameleon-engine
   ```
   `--network none` matters — if your engine only works with internet access, it will fail at scoring time, not just look slower.

**Submitting:**

The final challenge PDF requests the engine, `Dockerfile`, and a short read-me
explaining the approach and run instructions for stage one. The full Express
add-on, short pitch and demo video belong to the finale. This updates the older
FAQ wording that described a viewer add-on as a stage-one requirement. Submit
before code freeze and confirm the final checklist with organizers.

Where to send it, and any submission form, will be announced separately ahead of the deadline — keep an eye out for that announcement. In the meantime, make sure your repo builds and runs cleanly in Docker (see step 5 above) so you're not doing that work last-minute once submission opens.
