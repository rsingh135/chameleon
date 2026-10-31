# Source image reasoning, then organization

The optional visual pipeline now separates two responsibilities. A local VLM
first inspects the original image, without seeing target dimensions or layout
choices. A separately configured organizer then receives that analysis and the
original image to choose a source-bound layout. Trusted code compiles geometry,
admits candidates and checks the final export. This is implemented plumbing;
better visual reasoning and improved layout quality still require evaluation.

The analyzer records sparse observations about visible subjects, hierarchy,
attachments, reading order, negative space, alignment, layering and style. Each
fact names known source element IDs. It can explicitly abstain. The versioned
sidecar binds the original design digest and exact rendered PNG hash; the
organizer receipt also binds the supplied analysis and complete request.
Neither model can add/remove objects, change copy/fonts/assets, supply arbitrary
coordinates, grant crop permissions or relax backend admission. Analysis remains
advisory model evidence rather than a measured or human-approved fact.

## Configure the two roles

Existing operator configurations continue working. Add `analyzer_command` to
enable source analysis before `provider_command` (the organizer). Both commands
use the existing pinned offline provider; their JSON configurations and model
identities are recorded independently. They run sequentially. The optional
rendered-alternative critic uses the organizer configuration.

This Python setup, run from the repository root, creates portable local configs
with resolved paths. It uses the already supported Qwen and Smol profiles to
demonstrate distinct models; this pairing is not a quality recommendation.
Weights must already have been explicitly installed. Setup does not download
weights, invoke inference or change application defaults.

```python
import json
import sys
from pathlib import Path

root = Path.cwd().resolve()
out = root / "output/two-model-config"
out.mkdir(parents=True, exist_ok=True)
for name, template, tokens in (
    ("analyzer", "qwen_cpu.example.json", 128),
    ("organizer", "native_cpu.example.json", 96),
):
    config = json.loads((root / "configs/visual_planner" / template).read_text())
    config.update(
        python_executable=sys.executable,
        model_dir=str(root / "output/visual-planner/native"),
        max_new_tokens=tokens,
    )
    (out / f"{name}.json").write_text(json.dumps(config, indent=2))


def command(role):
    return [
        sys.executable,
        "-m",
        "experiments.visual_planner.provider",
        "--config",
        str(out / f"{role}.json"),
    ]


operator = {
    "schema_version": 1,
    "enabled": True,
    "analyzer_command": command("analyzer"),
    "provider_command": command("organizer"),
    "timeout_seconds": 240,
    "artifact_root": str(root / "output/two-model-runs"),
    "critic": True,
}
(out / "operator.json").write_text(json.dumps(operator, indent=2))
```

For an explicit run, pass that file as `config` to
`engine.visual_pipeline.resize_configured`. To select it for an independently
started Studio/headless process in PowerShell:

```powershell
$env:CHAMELEON_VISUAL_PLANNER_CONFIG = (Resolve-Path 'output/two-model-config/operator.json').Path
.\.venv\Scripts\python.exe -m viewer.server --port 8765
```

The audit CLI also accepts `--analyzer-command` alongside `--model-command`;
each takes a JSON argument array. No shell command is built from poster text.

## Evidence and fallback

Analysis executes once per source and is reused across requested targets.
Artifacts include `analysis-request.json`, `analysis-response.json`,
`analysis-trace.json` and, for valid analysis, `source-analysis.json`. Each
target's planner/critic request contains the validated sidecar and its hash,
outside the unchanged geometry-authorizing menu. Their receipts attest exactly
which sidecar they consumed. The report distinguishes execution, valid analysis,
abstention, valid layout proposal, changed geometry and unreviewed quality.

Invalid JSON, unknown IDs, stale image/request/analysis hashes, unavailable
weights, worker failure and abstention retain the deterministic layout before
calling the organizer. Reusing an audit directory clears the four owned analysis
artifacts first, so a failed run cannot leave an earlier valid sidecar behind.
No analysis result is silently treated as a successful trained judgment.

The bounded source summary explicitly reports omitted elements on complex
designs; observations may reference only IDs supplied in that summary. Whole
image views remain proportional 512-pixel letterboxes. Small text, complex
layering and truncated generations can therefore remain unresolved. Existing
worker time, token, memory, CPU and CUDA limits are unchanged. A declared model
identity and a valid observation do not prove pixel grounding or source intent.

Before qualification, compare image-versus-blank/replacement controls, reordered
organizer menus, source-specific relations and actual source/result renders on
independent development families. The existing training corpus acceptance and
held-out reserve rules remain applicable. Keep source analysis, organization,
and trusted geometry compilation separate; evaluate alternate orchestration
through the exact evidence it consumes and its native output renders.

## Local verification notes

Rendered source profiles inspected for this change:

- Warmup: centered sun/brand above a large serif headline and smaller orange
  action copy; purple atmospheric artwork and substantial negative space.
- Core: top-left orange badge, centered brand/product spine, a two-color serif
  headline followed by supporting copy and an orange action plate. The original
  product pixels, typography, badge association and action grouping are the
  reference, rather than a generic fixed role order.

An explicitly injected unavailable-analyzer probe rendered Warmup and Core at
960x540 and 540x960. All four outputs passed the contract and matched their
baseline PNGs byte-for-byte. The inspected Core renders retain its product spine,
corner badge and headline/body/action cluster; this demonstrates unchanged
fallback behavior, not improved model judgment. The live checkout changed during
that probe, so its reports correctly record `implementation_unchanged=false` and
must not be treated as fixed-implementation acceptance evidence. Files are under
`output/visual-handoff-fallback-verification`.

The final visual/provider regression suite runs separately from a frozen local
code/resource snapshot under `output/handoff-validation-snapshot`; its
`snapshot.json` and `validation-tests.xml` record the tested files and results.
All **181 tests passed** in 155 seconds; changed-file Ruff lint and format checks
also passed.
No VLM was executed in either verification. The snapshot includes the existing
small fit-policy artifact, without downloading or training weights.
