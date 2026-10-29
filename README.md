# CHAMELEON - Any Design. Every Canvas.

An offline design reflow engine for Adobe University Hackathon 2026. It adapts
an existing design to a new canvas by rearranging its elements, fitting text
with the supplied fonts, and keeping images in proportion.

The public entry point is `engine.resize.resize(source, target_canvas)`. It
returns a new design in the supplied v0.9 schema without mutating either input.
Text, font families, assets, colors, style runs, opacity, IDs and grouping are
preserved. No image content is generated and no network is used at runtime.

## Run locally

Use Python 3.11 or newer. From the repository root on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m engine.main
```

On macOS/Linux:

```bash
make setup
make run
```

Each task in `input/*.json` contains `source` and `target_canvas`. A successful
run writes `output/<task>_output.json` and `output/<task>_output.png`. The PNG
uses the vendored converter and `cr-renderer`, with real assets and brand fonts.
A missing asset is rendered as a labeled placeholder; the advanced sample
deliberately references one missing asset.

The CLI checks the local output contract before rendering, continues processing
other tasks after an individual failure, and exits nonzero if any task fails.
Missing rendering dependencies are an error. An unsuccessful rerun removes old
artifacts for that task so stale previews cannot appear to be new results.

## How the engine works

1. Capture an immutable source hierarchy using explicit groups, text/shape
   containment, message ownership, image alpha bounds and conservative proximity.
2. Generate bounded parent layouts: preserve the source arrangement, stack
   related campaigns, or place them in columns. Fit rigid leaves with one shared
   scale; keep badge rows and product/message associations together.
3. Measure text with the document's actual fonts. Preserve source size ratios
   across related products and controls, and align sibling rows when the source
   provides evidence. Declared legal-text exceptions remain visible in reports.
4. Reject candidates that violate the local contract, then rank valid candidates
   using source-aware collision, readability, hierarchy, grouping, prominence,
   alignment and spacing metrics.

For a single product with a corroborated source centerline, preserve its
horizontal focal anchor before softer alignment preferences. Landscape proposals
keep the brand/product spine centered and fit headline and supporting copy into
adjacent regions. Standalone supporting text has a declared optional scale
adjustment; product/headline/action size ratios and atomic controls stay intact.
Asymmetric and multiple-product sources do not receive this centering policy.
See [focal composition research](docs/FOCAL_REFLOW_RESEARCH.md) for the design
principles, evidence, measurements and limits.

Output is deterministic. A bounded cache speeds up repeated requests, and each
caller receives an independent copy. Valid designs requested at the same size
retain their composition.

## Open the local studio

```powershell
.\.venv\Scripts\python.exe -m viewer.server --port 8765
```

Open [http://127.0.0.1:8765](http://127.0.0.1:8765). Choose a bundled sample or
import a task JSON, select a target preset or enter custom dimensions, then
resize. Real previews, element inspection, validation and asset warnings, and
JSON/full-resolution PNG downloads are available. The server binds to loopback
and uses the existing offline engine with no additional packages.

Verify a running instance with `python -m tools.verify_local`. Results are saved
to `output/local-verification.json`. The standalone file inspector remains at
`viewer/inspector.html`.

## Adobe Express development add-on

The [native Express adapter](integrations/adobe-express/README.md) captures
supported editable text, images, shapes and nested groups through Adobe's
document SDK. It sends original image bytes and supplied exact font files to
`POST /api/document/resize`, then creates a separate editable page using native
clones. Missing fonts and unsupported effects stop the operation explicitly.

The engine isolates each document's assets and fonts in a content-addressed
package. The 24 MiB transport permits at most 16 MiB of decoded resources and
8 MiB per file. The cache stops accepting new packages at 64 packages or 256 MiB;
existing packages remain available. Only the development origin `https://localhost:5241`
has cross-origin access to this route; existing studio endpoints remain local
and same-origin. No remote asset URLs or filesystem paths are accepted.

Run `npm test --prefix integrations/adobe-express` for official SDK compilation
and adapter/writeback tests. Loading inside Express requires an HTTPS development
certificate and Adobe's local add-on workflow. Offline adapter tests establish
the supported mapping; live host execution and rendered parity require separate
verification. See the adapter README for the precise supported subset.

## Verify and compare

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\ruff.exe format --check .
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m tools.harness --targets all --out output/evaluation
.\.venv\Scripts\python.exe -m tools.harness --synth 24 --seed 218 --no-render --out output/holdout
```

The harness writes `report.json`, `report.md`, and rendered contact sheets. Use
`--no-render` for geometry and typography checks without PNG generation. The
catalogue in `engine/eval/catalogue.py` contains 26 representative platform,
advertising, print and presentation formats; it is our test catalogue, not a
confirmed organizer list. Metrics are local quality proxies, not Adobe's scorer.

Use the local studio for resizing, or open `viewer/inspector.html` and load a
task, its output JSON, and its rendered PNG for manual inspection.
The final challenge PDF requests the engine for stage one and a full Adobe
Express add-on, pitch and demo video for the finale. The experimental development
add-on is included above; the pitch, demo video and host verification remain
separate finale work.

An isolated orientation-continuation trial is documented in
[`docs/REFLOW_EXPERIMENT.md`](docs/REFLOW_EXPERIMENT.md). It compares intermediate
layouts while holding the endpoint fixed. It remains outside the default engine:
the first trial reduced the worst position jump, but substantial jumps and
unverified motion between anchors remain.

## Preserve source style

The project-local [Chameleon design taste skill](.agents/skills/chameleon-design-taste/SKILL.md)
guides future engine and studio changes through `AGENTS.md`. It separates the
supplied design's visual language from the surrounding viewer interface.

Runtime checks compare each result with its original source: font hierarchy,
relative badge/action proportions, adjacent headline/body spacing, corner
anchors, and visible image prominence. Transparent image padding is excluded
using cached alpha bounds where assets are available. Copy, font families,
colors, assets and grouping remain protected by the existing hard contract.

Selection favors source relationships among candidates with equally separated
content and readable text. Additional candidates preserve source control
proportions and corner anchors. The studio exposes the measured differences as
style warnings, separately from structural checks; `report.style` contains the
metrics, score and affected element IDs. These are bounded heuristics, so inspect
the real render, especially when a new aspect ratio forces a tradeoff.

An immutable source relationship graph also records directed reading order,
message/action ownership, shared axes and inferred brand/product associations.
Bounded compound states arrange related units together; ambiguous or unsupported
structures retain other candidates. `report.relationships` exposes the modeled
losses and evidence. This remains geometric inference, not general semantic
understanding. See the [research and implementation decision](docs/COMPOSITION_REFLOW_RESEARCH.md)
for real-application evidence, alternative approaches, results and limits.

Run `python -m tools.composition_demo --out output/composition-core` to save the
source graph, candidate states, full-resolution renders and comparison sheet.

Advanced badge rows and multiple product/message groups now have bounded layout
families. `resize_with_state()` retains the selected flow declarations alongside
the unchanged `resize()` design API. Run
`python -m tools.composition_cases --out output/advanced-audit` for actual
sample/authored-case renders and independent ownership/order checks. See
[Advanced layout evidence and remaining limits](docs/ADVANCED_LAYOUTS.md).

## Docker submission

Docker Desktop must be running with Linux containers enabled. Build once with
network access, then run with networking disabled:

```powershell
docker build --platform linux/amd64 -t chameleon-engine .
docker run --rm --network none -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" chameleon-engine
```

All dependencies install at build time. CI builds the Linux AMD64 image and
runs all three samples with `--network none`, checking that both artifacts exist.

## Project map

| Path | Purpose |
| --- | --- |
| `engine/resize.py` | Public API, validation and candidate selection |
| `engine/layout/` | Source analysis, composition generation and text fitting |
| `engine/eval/` | Contract checks and source-aware quality metrics |
| `engine/main.py` | Batch JSON/PNG runner |
| `renderer/` | Vendored schema converter and rendering integration |
| `tools/` | Deterministic synthetic fixtures and evaluation harness |
| `brand_kit/` | Supplied image assets and fonts |
| `viewer/` | Local studio, HTTP API and standalone file inspector |
| `PLAN.md` | Build plan, acceptance criteria and outstanding work |

## Current limits

Extreme formats can require very small text when all original copy and elements
must remain. A structurally valid 320x50 banner containing a long headline,
body, several badges and legal copy is not necessarily readable. The engine
does not drop elements or shorten copy to improve its score. Candidate search
uses geometric metrics; pixel contrast and busyness are available in rendered
evaluation reports but do not currently drive selection. Missing assets remain
visible placeholders. Model limits, the final canvas catalogue, hardware budget
and code-freeze/submission details require organizer confirmation.

See `RULES.md`, `chameleon_design_schema_v0.9.json`, and `CONTRIBUTING.md` for
the supplied contract and contribution workflow.
