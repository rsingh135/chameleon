# Internal cross-scene regression tools

The public application is the [Design studio](http://127.0.0.1:8765/).
Use its source picker, canvas controls and Original/Result comparison for normal
work. The former batch page is retired: its address redirects to Studio, and its
JavaScript and stylesheet are not served. There is no second product workflow.

Cross-scene evaluation remains internal development tooling. It helps check the
same implementation against different original designs and target sizes; it
does not establish visual quality or unseen-scene generalization.

## Internal API and Python runner

The loopback server retains `GET /api/benchmark/method` and
`POST /api/benchmark/run` for regression tooling. Freeze the method receipt, then
include its digest as `expected_method_digest` in every request. Supply original
native JSON designs or portable resource bundles, shared `targets`, and a
`scenes` list whose entries contain `id`, `family` and `payload`.

The equivalent Python entry point is `viewer.benchmark.run_benchmark(payload)`;
`viewer.benchmark.method_receipt()` supplies the current implementation identity.
Limits remain eight scenes, four targets and sixteen jobs per request. Target
canvases must fit the 8192-pixel-per-side and 20-megapixel bounds. Resource bytes,
source identity, method identity, structural checks and per-job failures remain
part of the evidence. Always resize directly from each original, never from a
previously resized output.

Read configured analyzer/critic and request-policy fields from the receipt.
Models are disabled by default; enabling an optional local provider is an
explicit operator configuration, not a consequence of running a regression.
This tooling does not start training, download models or deploy cloud resources.

For a deterministic CLI inspection using bundled sample resources:

```powershell
.\.venv\Scripts\python.exe -m tools.scene_harness `
  --source input/chameleon_sample_level1_warmup.json `
  --assets-dir brand_kit/assets --fonts-dir brand_kit/fonts `
  --out output/internal-scene-regression-warmup `
  --target 800x500 --target 500x800
```

Keep evidence separate from original inputs. Split approved sources by family,
retain failures and unrun cases, and keep reserved sources untouched. Passing
structural checks is not visual approval. Agent labels and tiny development
splits do not substitute for independent human review or a final test set.

## Retained legacy fixtures

`viewer/benchmark.html`, `viewer/benchmark.js` and `viewer/benchmark.css` remain
on disk only as legacy regression fixtures. They are absent from product
navigation and are not public pages/assets. Existing Node suites read these
files directly; API tests continue to exercise source/resource isolation,
identity validation, bounded requests and failure recovery.

The relevant focused checks are `tests/test_studio_navigation.py`,
`tests/test_viewer_benchmark.py`, `tests/test_viewer_http_stress.py`,
`tests/benchmark_ui.test.cjs` and `tests/benchmark_second_pass.test.cjs`.
