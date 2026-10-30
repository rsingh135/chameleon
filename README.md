# CHAMELEON — Any Design. Every Canvas.

An offline engine that takes a finished design and a new canvas size and
returns the same design re-laid-out for that canvas: same elements, same copy,
same fonts and assets, new positions, sizes and type scale. Built for the Adobe
University Hackathon 2026 Chameleon Challenge.

```
python -m engine.main          # input/*.json  ->  output/<task>_output.json + .png
```

## The idea in one paragraph

A naive resizer stretches the picture. Ours moves the furniture, then checks
its own work. For every task the engine reads the source design as a designer
would: which pieces are a unit (a button and its label), which overlaps were
intentional, what the hero is, what reads first. It then proposes a bounded set
of complete layouts for the target shape (preserved, stacked, split columns,
focal, grouped, and single-band strips for extreme ratios), measures every text
box with the *official renderer's own font engine* so nothing overflows, rejects
any proposal that breaks the structural contract, and ranks the rest with
source-aware quality metrics: collisions that did not exist in the source,
readability, font hierarchy, group integrity, product prominence, alignment,
balance and margins. If every proposal fails, an always-valid fallback ships.
The result is deterministic, runs in about a second per task on a laptop CPU,
and needs no network and no model.

## What makes it different

- **Closed loop, not one shot.** Candidates are generated *and scored*. The
  scorer (`engine/eval/`) is the same one our regression harness uses, so what
  wins in search is what we measured to be good across hundreds of cases.
- **Renderer-exact text.** Wrapping and overflow are measured through the
  vendored `cr-renderer` Skia path, not an approximation. If we say it fits, the
  PNG agrees.
- **Source-aware rules.** A collision is only an overlap that the designer did
  not make. A badge stays a badge. A legal line never outgrows the headline.
  Corner elements stay in their corner unless the shape makes that impossible.
- **Never zero.** No path in the engine ends without an output. Pipeline
  errors and contract failures degrade to a contain-scaled, text-refitted copy
  of the source that always passes the structural rules.
- **Honest about extremes.** A 320×50 banner cannot hold a poster. For strips
  and towers the engine switches to a one-band layout, product at full height,
  and ranks readability above preserved adjacency.

## How it works

1. **Context** — `engine/layout/model.py`, `engine/layout/relationships.py`:
   units from explicit groups and inferred containment; roles (logo, hero,
   headline, body, badge, action, legal, support) from geometry and typography,
   never from element ids; reading order, ownership, intentional overlaps.
2. **Proposals** — `engine/layout/candidates.py` and siblings: preserved,
   stacked, split, grouped, focal, source-guided, background-crop and strip
   layouts; 8–30 complete designs per task.
3. **Text fitting** — `engine/layout/text.py`: largest size that keeps every
   line inside its box, using the renderer's glyph advances; unbreakable words
   are detected because the renderer will not hyphenate.
4. **Contract** — `engine/eval/contract.py`: ids preserved, nothing off-canvas,
   images in proportion, text inside its box, identity fields unchanged.
   Violations disqualify.
5. **Selection** — `engine/resize.py`: photo visibility and collisions first,
   then hierarchy inversions, readability, reading order, relationship warnings
   and composition fidelity, then typography closest to the source scale.
6. **Fallback** — `engine/layout/fallback.py`: used only when 4 rejects
   everything or 2–5 raise.

## Run it

Python 3.11+. From the repo root:

```bash
make setup            # .venv with runtime + dev deps (installs cr-renderer)
make run              # resize input/*.json -> output/
make check            # ruff + pytest, same as CI
make eval             # samples + 20 synthetic tasks -> eval_out/report.md + contact sheets
```

Windows PowerShell: `python -m venv .venv`, then
`.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt` and
`.\.venv\Scripts\python.exe -m engine.main`.

### Docker (how stage one is scored)

```bash
docker build --platform linux/amd64 -t chameleon-engine .
docker run --rm --network none \
  -v "$PWD/input:/app/input" -v "$PWD/output:/app/output" chameleon-engine
```

Everything installs at build time; nothing is fetched at run time. CI builds
this image and runs all three samples with `--network none`, asserting both
artifacts exist for every task.

## Evidence

Measured with `tools/harness.py` against our source-aware scorer (0–1, higher
is better). The scorer is a local proxy for the rubric, not Adobe's evaluator.

| Set | Rows | Valid | Crashes | Mean score |
| --- | --- | --- | --- | --- |
| 3 samples × 26 catalogue formats | 78 | 78 | 0 | 0.90 |
| 40 synthetic designs, 6 layout families | 43 | 43 | 0 | 0.90 |
| held-out families and formats only | 13 | 13 | 0 | 0.88 |
| 60 fuzzed inputs (stripped fields, newlines, rotation, 8 extra badges, missing assets, opaque ids) | 60 | 60 | 0 | — |

`resize()` takes 0.2–2.1 s per task on an M-series laptop. Weakest formats are
the strips (728×90, 320×50) at about 0.78; everything else scores 0.85–0.95.

Contact sheets for every task land in `eval_out/`; the source renders first,
each target after it with its score and any violation codes in the header.

## Evaluate and inspect

```bash
python -m tools.harness --targets all             # samples x whole catalogue
python -m tools.harness --synth 40 --seed 0       # synthetic + held-out cases
python -m tools.harness --no-render               # geometry only, fast
python -m viewer.server --port 8765               # local studio at 127.0.0.1:8765
```

The studio (`viewer/`) resizes any task against a preset or custom canvas, shows
the real render, the element inspector, contract and style warnings, and exports
JSON and full-resolution PNG. `viewer/inspector.html` is a dependency-free file
inspector. `python -m tools.verify_local` checks a running studio over HTTP.

`tools/synth.py` generates designs from six layout families with random copy,
counts, ids, groups, rotation and missing assets. Two families and four formats
are held out so sample-specific tuning shows up as a seen-vs-holdout gap.

## Adobe Express add-on (experimental, finale groundwork)

`integrations/adobe-express/` is a development adapter compiled against the
official SDK 1.40.0. It captures editable text, images, shapes and groups,
sends original bytes and exact font files to `POST /api/document/resize`, and
writes a verified editable clone to a new page, with rollback on failure.
`npm test --prefix integrations/adobe-express` runs compilation and adapter
tests. Live host execution still needs Adobe's HTTPS add-on workflow; see the
adapter README for the supported subset and limits.

## Project map

| Path | Purpose |
| --- | --- |
| `engine/resize.py` | Public API, candidate pool, contract gate, selection, fallback |
| `engine/layout/` | Source analysis, relationship graph, proposals, text fitting, strips, fallback |
| `engine/eval/` | Contract checks, source-aware metrics, in-memory rendering, canvas catalogue |
| `engine/main.py` | Batch runner used by the Docker entrypoint |
| `renderer/` | Vendored schema converter wrapping `cr-renderer` |
| `tools/` | Regression harness, synthetic generator, audits and demos |
| `tests/` | ~1,200 tests: contract, metrics, selection, layouts, CLI, viewer |
| `viewer/`, `integrations/` | Local studio, file inspector, Express adapter |
| `docs/` | Design research and decision records |
| `brand_kit/` | Supplied assets and fonts |

## Limits

Extreme formats keep every element, so a 320×50 banner with a long headline,
body, several badges and legal copy is valid but not comfortably readable; the
engine never drops elements or rewrites copy (stage-one rules). Role and
relationship inference is geometric, not semantic. Pixel contrast and busyness
under text are reported but do not yet drive selection. Missing assets render as
labelled placeholders, as the brief specifies. The canvas catalogue in
`engine/eval/catalogue.py` is ours; the organizers' list, model limits and
hardware budget are not confirmed.

See `RULES.md`, `chameleon_design_schema_v0.9.json`, `CONTRIBUTING.md` and
`PLAN.md` for the contract, workflow and outstanding work.
