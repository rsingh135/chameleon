# Chameleon implementation plan

Build an offline design reflow engine that preserves the supplied elements,
assets, copy and fonts while adapting composition to the requested canvas.

## Ownership and integration

- Codex: `engine/layout/`, `engine/resize.py`, runtime integration, this plan.
- Thornyhead: `engine/eval/`, evaluation tools, regression fixtures, Docker/CI.
- Coordinate through the AgentCouch `chameleon` room and pushed commit SHAs.
- Use separate branches. Keep main buildable; push each tested checkpoint.
- Any shared-file changes are coordinated before integration.

## Checkpoints

1. **Plan and interfaces.** Publish this plan and agree on the layout/scoring
   interface. Done when both agents know their file ownership and branches.
2. **Working layout core.** Infer source relationships and roles without relying
   on sample IDs; generate preserved-composition, stacked and horizontal layouts;
   fit text using the renderer's fonts/wrapping; preserve grouped labels/shapes
   and image proportions. Done when all samples satisfy the structural contract
   across square, portrait, landscape and strip targets, with deterministic output.
3. **Evaluation and candidate selection.** Integrate source-aware validation,
   visual proxy metrics and a render/contact-sheet harness. Reject invalid
   candidates before ranking. Done when synthetic held-out cases and all samples
   pass, and rendered outputs have been inspected and corrected.
4. **Packaging and viewer.** Verify installation, the CLI, offline Docker runs,
   and a minimal viewer for loading tasks and comparing source/output previews.
   Confirm the Express add-on requirements against available organizer materials;
   distinguish a working local viewer from an installed Adobe Express add-on.
5. **Delivery.** Update usage and limitations, run lint/tests and available
   integration checks, push the final checkpoint and share the results with both
   humans. Report unverified environment-dependent checks explicitly.

## Interfaces

`resize(source, target_canvas) -> design` remains the public entry point and
does not mutate its inputs. Layout candidates use the same design schema:

```python
generate_candidates(source, target_canvas) -> list[dict]
score(source, candidate, context=None, rendered=None) -> {
    "violations": list,
    "metrics": dict,
}
```

Structural failures are distinct from quality warnings. Source context records
intentional containment/layering, source image proportions and typography.
Text measurement reuses the vendored renderer when installed and has an offline
fallback for dependency-free contract tests. Production visual verification uses
the real renderer. All render assets and fonts resolve relative to the repository.

## Quality policy

- Preserve each input ID exactly once and preserve copy, font family, asset ID,
  opacity, style runs and type. Preserve image proportions except existing cover
  backgrounds. Do not synthesize imagery or hide content to win a proxy score.
- Keep rotated visual bounds on the target canvas. Keep grouped labels contained
  by their associated shapes and retain hierarchy when reflowing content.
- Score text wrapping, long words and box height using renderer-compatible font
  metrics. Inspect extreme ratios rather than treating a passing JSON check as
  proof of readability.
- Start with bounded deterministic candidates. Add background pixel scoring after
  the working baseline; defer model training and dataset ingestion.

## Outstanding organizer information

Code freeze, exact CPU/RAM/model limits, the canvas catalogue, evaluator details,
reduced-visibility policy and the viewer/add-on requirement need confirmation.
Implementation proceeds without assuming permissive answers to these questions.

## Progress

- [x] Reviewed repository rules, schema, renderer and brainstorm.
- [x] Asked thornyhead to begin Track B and exchanged integration contracts.
- [ ] Publish layout core and inspect sample renders.
- [ ] Integrate evaluation and improve candidate selection.
- [ ] Verify packaging and viewer.
- [ ] Push final validated delivery.
