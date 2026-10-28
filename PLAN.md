# Chameleon build plan

## Goal and scope

Build a deterministic offline engine that accepts a source design and target
canvas, then emits the same design elements as schema-valid JSON and a real
rendered PNG. Optimize composition and readability across unseen designs and
aspect ratios, preserving copy, fonts, assets, identity and grouping.

The attached **Final Chameleon Challenge** PDF is the current project brief.
Its section 5 asks for the engine, Dockerfile and approach/run instructions for
stage one. The Express add-on, pitch and demo video belong to the finale.
The older repository FAQ's stage-one viewer requirement conflicts with that
final brief. Keep the local inspector for development and confirm any changed
submission checklist with organizers.

## 1. Integrate the existing work

- [x] Review all 14 PDF pages, repository rules, schema and contributor workflow.
- [x] Fetch `rsingh135/chameleon` and preserve existing layout work on a new
  `codex/hackathon-ready` integration branch.
- [x] Merge current `origin/main` evaluation/harness work and the teammate's
  `thornyhead/contract-geometry` shape-scaling and scoring fixes.
- [x] Establish the baseline: 276 tests passing, lint/format passing, and all
  three supplied tasks producing real JSON/PNG outputs.

## 2. Make candidate selection obey the contract

- Reject candidates that drop/duplicate elements, alter identity, distort
  non-cover images, overflow text, or miss the requested canvas.
- Rank valid candidates using the integrated source-aware quality metrics.
- Preserve already valid source composition when the target size is unchanged.
- Keep the public API deterministic and nonmutating; return clear failures when
  no candidate satisfies the contract.

Acceptance: selection regressions prove an invalid high-scoring candidate cannot
beat a valid candidate. Existing sample, grouping, text-fitting, rotation and
determinism tests remain green.

## 3. Improve actual composition

- Give the main product the available space in wide media columns instead of
  limiting it to 42% of column height.
- Transform rotated labels and plates together with a uniform scale so their
  geometry stays consistent.
- Inspect actual sample outputs and run deterministic held-out synthetic designs
  across nearby, portrait, landscape and extreme display-ad canvases.

Acceptance: image proportions and rotated visual bounds remain valid; a wide
composition can retain a prominent product; output content and style are intact.
Record unreadable tiny-format cases as quality limitations even when geometry
passes. Do not hide, delete or rewrite content to make a proxy score improve.

## 4. Make failures visible and reproducible

- Require the real renderer for batch submission previews; remove silent box
  fallback behavior.
- Validate each task before writing artifacts. Continue the batch after a bad
  task and return a nonzero exit code for any failed task.
- Remove stale previews on reruns, reject nonfinite JSON output, and support
  UTF-8 input/output.
- Make evaluation reports and process exit codes reflect render errors, invalid
  layouts and crashes; handle an empty task directory explicitly.

Acceptance: fast CLI/harness regression tests cover malformed input, missing
renderer, invalid output, partial preview failures and mixed successful/failed
batches. Re-running the sample batch produces JSON plus real PNG for each task.

## 5. Validate packaging and deliver

- Run Ruff lint/format checks and the full test suite after integration.
- Run sample renders, representative catalogue cases and held-out synthetic
  fixtures; save reports and contact sheets under ignored `output/` paths.
- Build Linux AMD64 Docker image and run samples with `--network none` when
  Docker is available. Current local blocker: Docker Desktop's Linux daemon is
  not running; the installed CLI alone cannot build or run the image.
- Update README, viewer instructions and FAQ to describe current behavior and
  separate implemented features from finale work.
- Keep generated reports, previews and virtual environments out of commits.

## Follow-on improvements

1. Add a bounded second selection pass using rendered background plates to
   penalize low text contrast and busy image regions.
2. Improve horizontal allocation for dense badge rows, long copy and extreme
   strips without sacrificing source content or brand identity.
3. Build the full Express add-on for the finale: source import, target presets,
   engine invocation, preview selection and placement into the document. Confirm
   the supported SDK and document APIs against official Adobe documentation
   before implementation.

## Organizer details still needed

Code freeze, submission destination, exact CPU/RAM/model limits, approved open
model list, final target catalogue and official evaluator details are not fully
specified in the supplied materials. No model service or network dependency is
required by the current approach.

## Validation checkpoint - October 28, 2026

- Contract-first selection, same-size preservation, rotated-group geometry,
  prominent wide-format products, truthful CLI errors and harness failure
  reporting are implemented with regression coverage.
- Full suite: 303 passed, one expected skip for a sample with no shapes. Ruff
  lint and format checks pass.
- Independent audit passed 45/45 cases: three sample tasks, 24 deterministic
  synthetic designs, and 18 sample/target combinations. Checks include the
  supplied JSON schema, output contract, rotated bounds, deterministic repeated
  requests and immutability of both inputs.
- All three supplied tasks produce JSON plus real PNGs. Final rendered reports
  and contact sheets are in ignored `output/final/`; the independent audit is
  saved at `output/audit-final.json`.
- Local Docker validation remains unavailable after attempting to start Docker
  Desktop: the Linux-engine pipe is absent. The draft PR's existing CI workflow
  builds the image and runs the samples without networking.
- Remaining product work: improve tiny-banner readability and pixel-aware
  selection, then build the Express add-on for the finale.

## Gradual quality checkpoint

- Added and tested a loopback studio around the existing engine. The supplied
  Core style-run range is now accepted without changing its content; JSON and
  full-resolution PNG exports were verified in the actual browser.
- Corrected CTA/badge inference and packed related headline/body copy. The Core
  headline/body box gap decreased from 103 px to 10 px. This is a narrow coherence
  fix, not a claim of solved visual quality.
- Added a separate orientation-continuation experiment. Its guided seven-anchor
  trial reduces the worst measured center jump from 0.6282 to 0.4647, with the
  same final design. The default engine does not use the experiment.
- Current validation: 348 tests passed, one expected skip, Ruff clean; 17 live
  HTTP checks passed. See `docs/REFLOW_EXPERIMENT.md` for results, provenance,
  rejected trials, limitations and the next geometry/ink-aware work.
