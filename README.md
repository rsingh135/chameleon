# Chameleon

**Any design. Every canvas.**

Chameleon adapts a finished composition to a new format while preserving its
original content and visual identity. It combines composition-aware layout
search, a lightweight trained fit policy, and real font measurement to produce
editable design JSON and a native rendered PNG.

Built for the Adobe North America Hackathon 2026, the engine runs offline on
CPU. Assets, fonts, and the default model are supplied locally; execution needs
no API key, model download, or network connection.

## What makes it different

- **Preserve the design.** Keep every original element ID, copy, font family,
  asset reference, color, opacity, stacking order, and group identity. Adapt
  geometry and font size without generating replacement imagery.
- **Reflow relationships.** Use source groups, reading order, brand/product
  associations, corner anchors, and footer relationships to organize new layouts.
  Generate compositions around related text, artwork, and controls.
- **Fit actual typography.** Measure glyphs with the supplied fonts, rewrap text,
  and compare hierarchy, spacing, readability, and image proportions.
- **Check both deliverables.** Validate the returned design and render that same
  JSON through the vendored converter and `cr-renderer`. Exports reflect the
  actual requested canvas.
- **Inspect the result.** The full repository includes a local Studio with source
  selection, custom target sizes, side-by-side comparison, element inspection,
  and JSON/PNG downloads.

## Engine quickstart

These commands work in both the engine submission package and the full
repository. Use Python 3.11 or newer.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m engine.main
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m engine.main
```

Each file in `input/` contains a `source` design (canvas and elements) and a
`target_canvas` (width and height). Use the three supplied task files for
complete examples. A successful run writes `output/<task>_output.json` and
`output/<task>_output.png`. The runner checks the output contract and strict
canvas bounds before rendering, and returns a nonzero exit code if any task
fails. The advanced sample's deliberately missing asset remains a labeled
placeholder.

## Run the submission offline

Build the Linux AMD64 image with Docker, then run with networking disabled.
The same commands apply to the engine package and the repository.

**Windows PowerShell**

```powershell
docker build --platform linux/amd64 -t chameleon-engine .
docker run --rm --network none -v "${PWD}/input:/app/input:ro" -v "${PWD}/output:/app/output" chameleon-engine
```

**macOS / Linux**

```bash
docker build --platform linux/amd64 -t chameleon-engine .
docker run --rm --network none \
  -v "$(pwd)/input:/app/input:ro" \
  -v "$(pwd)/output:/app/output" chameleon-engine
```

Dependencies install at build time. The image includes the engine, native
rendering integration, brand assets and fonts, schema, and trained fit-policy
weights. Input and output directories are mounted at runtime.

For a fresh build and offline verification, run the included host-side verifier
with a Python environment that has `jsonschema` installed:

```bash
python -m pip install "jsonschema>=4.26,<5"
python -m tools.verify_docker --out output/docker-check
```

It checks the JSON contract, strict bounds, PNG dimensions, native-render pixel
parity, and packaged file identities. CI runs the offline Docker check alongside
the repository's tests.

## Python API

```python
import json
from pathlib import Path

from engine.resize import resize

task = json.loads(Path("input/chameleon_sample_level2_core.json").read_text())
result = resize(task["source"], task["target_canvas"], strict_bounds=True)
```

`resize()` returns a new design in the supplied v0.9 schema and leaves both
arguments unchanged. Pass `assets_dir` and `fonts_dir` for a document's local
resources. `resize_with_state()` also returns the selected layout's execution
metadata. Repeated requests use bounded caches and return independent results.

## How layout and models work together

1. **Understand the source structure.** Build a relationship graph from groups,
   containment, visible image bounds, text hierarchy, alignment, and reading order.
2. **Generate compositions.** Preserve proportional layouts where appropriate;
   otherwise explore grouped flows, fitted document columns, anchored layouts,
   and source-derived arrangements.
3. **Fit and validate.** Use native font metrics to fit text, maintain image
   proportions, and reject layouts that break protected source fields or output
   constraints.
4. **Select and render.** Combine source-aware scoring with the trained fit
   policy for admitted layout choices, then render the selected JSON with the
   original resources.

The bundled model in `models/fit_policy/default.json` is a trained pairwise
ranker over 28 geometry and typography features. Its supervision comes from
synthetic fit/readability comparisons. It chooses among known valid layout
candidates; the executor controls the geometry and contract checks. Inference
uses the Python standard library and runs on CPU.

The repository also contains an **opt-in experimental local vision pipeline**
for source analysis, structured layout proposals, and rendered-alternative
critique. It uses a separate local model configuration and passes proposals
through validation and the layout executor. The submission Docker image uses
the bundled fit policy; vision providers and their adapters are configured
separately for development.

## Local Studio · full repository

Start the local server using your virtual environment:

**Windows PowerShell**

```powershell
.\.venv\Scripts\python.exe -m viewer.server --port 8765
```

**macOS / Linux**

```bash
.venv/bin/python -m viewer.server --port 8765
```

Open [localhost:8765](http://127.0.0.1:8765). Choose a sample or poster, import a
task JSON, select a target preset or enter dimensions, and resize. Compare the
original and result, inspect elements and validation feedback, and download the
editable JSON or full-resolution PNG. The setup panel folds away to give the
canvas more room; dark and light themes are available.

The server binds to loopback and uses the same offline engine as the CLI.
`viewer/inspector.html` provides standalone file inspection. An Adobe Express
viewer ZIP is also provided under `integrations/adobe-express/submission/`.

## Development checks · full repository

With your virtual environment active:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m tools.harness --targets all --out output/evaluation
```

The harness compares designs across target formats and writes structured
measurements and native contact sheets for visual review. For a running Studio,
use `python -m tools.verify_local --url http://127.0.0.1:8765`.

## Project map

| Path | Role |
| --- | --- |
| `engine/resize.py` | Public reflow API and candidate selection |
| `engine/layout/` | Source relationships, composition families, text fitting |
| `engine/fit_policy.py` | Trained fit-policy inference and artifact validation |
| `models/fit_policy/` | Bundled default model weights |
| `engine/eval/` | Contract checks, native rendering, source-aware measurements |
| `engine/main.py` | Offline batch JSON/PNG entry point |
| `renderer/` | Vendored converter and native rendering integration |
| `brand_kit/` | Supplied original assets, fonts, and notices |
| `viewer/` | Local Studio and inspector in the full repository |
| `experiments/` | Optional local model and fine-tuning development |
| `tools/` | Verification, evaluation, and development utilities |

The supplied contract is defined by `RULES.md`, `FAQ.md`, and
`chameleon_design_schema_v0.9.json`. Asset, font, and dependency notices stay
with their respective resources.
