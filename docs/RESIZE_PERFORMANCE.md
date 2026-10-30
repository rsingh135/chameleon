# Resize performance and exact-output reuse

The local Viewer reuses completed work while preserving the native layout,
rendering and diagnostic pipeline. Candidate counts, scoring resolution, glyph
sizes and validation checks are unchanged. These changes preserve measured
quality; they do not improve composition or prove generalization.

## Measured results

The real `viewer.server.perform_resize` path resized the prepared openSUSE
Trifold, with 101 elements and 88 live text nodes, to **800×500**. It used the
source's exact native assets and five font aliases.

| Combined Viewer caches | Wall seconds | Process CPU seconds |
| --- | --- | --- |
| First request in a fresh process | 30.32 | 11.31 |
| Identical same-process repeat | 0.119 | 0.125 |

Both requests produced the same design JSON, source PNG, target PNG and complete
diagnostic report as the captured original implementation. The target PNGs are
800×500, with zero hard-contract violations. Source/resource/code identity
remained fixed within the run. Timing and cache-hit fields in the response
envelope naturally differ; the native result and quality report do not.

The earlier **134.83-second cold result was cProfile-instrumented**, under host
memory pressure. It is hotspot evidence, not an unprofiled speedup denominator.
Two uncached plain attempts were aborted and retained as incomplete receipts.
Do not derive an overall speedup factor from those timings or promise a cold
latency SLA from this single source and host.

Evidence:

- [Combined plain cold/warm summary](../output/performance-20261030/combined-viewer-cache/opensuse_trifold_front-plain/summary.json)
- [Original profiled artifacts](../output/performance-20261030/profiles/opensuse_trifold_front-profiled/summary.json)
- [Typeface-only output comparison](../output/performance-20261030/typeface-cache-comparison.json)

A separate bounded font-setup microbenchmark used all five actual Trifold
aliases, exact font bytes, fractional sizes and source text. Four 500-call blocks
ran in ABBA order; each cached block started empty. All glyph-advance arrays were
exactly equal. Across 1,000 calls per condition, Typeface parsing fell from
**1,000 to 10: 99% fewer parses**. Legacy blocks took 5.059/2.366 seconds wall;
cached blocks took 0.024/0.086 seconds. The loop finished in 7.65 seconds after
imports, within its 30-second budget. Short cached CPU timings approach the
clock's approximately 15.6 ms quantum. These are font setup and glyph measurement
costs, not full resize times.
[Microbenchmark receipt](../output/performance-20261030/font-setup-microbenchmark.json).

## What is cached

| Layer | Identity and bounds | Preserved behavior |
| --- | --- | --- |
| Native Typeface | Exact immutable font bytes and TTC index; 32 entries and 32 MiB of font bytes | A fresh mutable `skia.Font` is still created at the exact requested size with the same linear metrics. Failed parses and oversized faces are not retained. |
| Source PNG | Target-independent source/resource/attribution/execution identity; 16 entries and 16 MiB | Reuses the original render across targets; keeps original warning text. |
| Complete resize result | Source **and target**, resources, attribution and execution identity; 16 entries and 16 MiB | Reuses only completed, structurally valid results and returns a newly decoded value. |

Source/result entries contain compressed serialized JSON, not native renderer,
font-manager or image objects. Each cache also bounds the uncompressed entry
size to limit allocations when decoding. Their retained encoded payloads total
at most 32 MiB; that is not a bound on total process RSS or Skia's internal caches.
Typeface bounds likewise describe retained font bytes, not total native memory.

[Skia documents Typeface objects as immutable and shareable](https://api.skia.org/classSkTypeface.html).
The implementation therefore shares the face, not mutable Font instances.
It never rounds sizes or estimates another size's advances by rescaling a
unit-size measurement.

## Identity, failures and concurrency

Request identity includes canonical source and target data, actual asset/font
SHA-256 values and resource directories, PNG attribution, and a frozen-checkout
implementation/runtime fingerprint. Resource bytes are rebound before reuse and
after processing. Same-name changed bytes are not accepted through a stale
path/mtime key; changed loaded resources require a restart or a new immutable
package. Restart after code edits: fingerprinting checkout files does not
hot-reload Python functions.

Legacy missing-image inputs may still render their existing placeholder and
warnings, but incomplete packages enter neither cache. Failed, interrupted,
malformed or contract-invalid complete results are not admitted. Requests using
an enabled configured visual model bypass the complete-result cache; invalid
explicit model configuration also bypasses it, so a prior deterministic hit
cannot hide a requested model or its configuration error. Source-only rendering
may still be reused when its independent identity is valid.

Verified result hits bypass the engine/renderer lock. A miss takes that lock and
checks again: identical concurrent successful misses reuse the first completed
result instead of performing the same solve twice. Native rendering remains
serialized; renderer and mutable Skia caches are not shared between concurrent
solves.

The batch UI retains stable scene/result DOM cells and verified JSON/PNG export
URLs. A completed job creates its expensive DOM/export resources once rather
than recreating every previous job after each completion: allocation work grows
with the number of jobs, O(N). This does not claim every small report-array
lookup is O(1). Setup changes invalidate stale exports; disposal/new runs release
their URLs. Resource-bearing JSON exports retain their exact original bundle.

## Runtime and timeout boundaries

Use one renderer process on this 16 GB host, which has sometimes had only about
1.5 GB free during local model work. The wall/CPU gap in the measurements shows
the effect of paging and contention. A two-worker process pool is only a future
experiment after at least 4 GB is free and measured per-worker memory makes it
safe. It is not implemented or promised as a throughput improvement.

Default per-request browser deadlines remain **60 seconds in Studio** and
**120 seconds in batch**. For an enabled visual model, the advertised deadline
allows the configured planner allowance, optional critic allowance, and another
150 seconds for native work and transport; it never shrinks the defaults.
The current policy uses the configured provider timeout for each model call.
This is a waiting budget, not a latency guarantee. Browser cancellation can
leave the server finishing an active request. These changes start no GPU,
download, training session or paid service.

## Reproduce and validate

The profiling scripts and raw artifacts are local ignored evidence, not a
packaged benchmark dependency. With that evidence directory present, run from
the repository root and choose a fresh output directory:

```powershell
.venv/Scripts/python.exe output/performance-20261030/profiles/profile_resize.py `
  --case opensuse_trifold_front --mode plain --timeout 90 `
  --out output/performance-20261030/fresh-plain-run
```

The script targets the recorded isolated preview worktree; record its current
code identity or deliberately prepare an equivalent frozen worktree before
comparison. Plain cold/warm measurement comes first. cProfile is explicit;
never compare a traced run directly with an untraced run as a speedup claim.
The watchdog terminates its own Windows venv/interpreter tree on expiry and
retains an aborted receipt rather than claiming a completed baseline.

For an optional fresh font-only measurement, the local
`profiles/font_setup_microbenchmark.py` script uses a 30-second after-import
budget and 90-second external cap. Its existing output file is immutable; use
a separate evidence copy/output path rather than overwriting the recorded run.

Relevant tracked checks cover exact advances/PNG parity, mutable Font isolation,
same-alias changed bytes, byte/count eviction, failed faces, source/result key
binding, cache admission, configured-model bypass, concurrent identical misses
and batch export/DOM lifetime. Final aggregate test counts are recorded by the
release check, not inferred here:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_native_typeface_cache.py `
  tests/test_viewer_resize_cache.py tests/test_viewer_server.py tests/test_viewer_benchmark.py
node --test tests/benchmark_ui.test.cjs tests/benchmark_second_pass.test.cjs
```
