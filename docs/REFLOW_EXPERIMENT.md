# Gradual reflow experiment

## Why the first result is not good enough

Work remains in the team's existing `rsingh135/chameleon` checkout. On October
28, 2026, its `brand_kit/`, `renderer/`, `input/`, design schema, Dockerfile and
requirements were compared with the official `jhaaj08/chameleon` starter. All
six Git tree/blob hashes matched exactly. The engine has been extended; the
local studio is an adapter around it. Passing schema and contract tests
establishes structural validity, not composition quality.

The Core square-to-banner sample exposed three concrete problems:

- The CTA's 22 px label missed a 22.4 px heuristic threshold and was classified
  as a badge. The actual badge and CTA were then treated as interchangeable.
- The vertical flow distributed unused space between related headline/body
  boxes, separating a copy cluster that should remain close.
- The product asset has transparent padding. A 292 px element box produces a
  visible bottle approximately 124 x 201 px, so a large box does not establish
  a prominent product. Future placement needs visible pixel bounds as well as
  element bounds.

## Algorithmic interpretation of intermediate frames

Treat a frame as a design JSON at a particular canvas shape. Keep one immutable
source context: element identity, groups, content, original typography and image
proportions. For a path of canvas sizes, generate and validate several layouts
at every size, then choose a sequence that balances composition quality and
movement between neighboring sizes.

```text
original source and relationships
    -> square / intermediate / landscape candidate layouts
    -> reject invalid candidates at each size
    -> search for a good sequence with small step-to-step changes
    -> render and inspect the anchors
    -> repair every additional intermediate state before using it
```

Use logarithmic width/height interpolation to distribute proportional canvas
changes evenly. Retain multiple arrangements through the search: a vertical
stack can become two columns when the available space requires it. Penalize
unnecessary movement, broken grouping, unreadable text and unwanted collisions.
Refit text from the original preferred sizes so early shrinkage does not become
permanent.

Simply calling `resize(previous_result, next_canvas)` would repeatedly reinterpret
the design and can shrink fonts at every step. Simply interpolating two layouts
can move elements through one another. Neither is sufficient for reliable
automatic reflow.

## Small checkpoints

1. Correct CTA/badge inference and keep headline/body copy packed together. Add
   regressions for the observed failures; change no global score weights.
2. Run an isolated discrete-anchor prototype on the Core sample. Save the source,
   default result, proposed endpoint, anchor frames and movement diagnostics.
3. Review endpoint quality and jumps between anchors. A valid path is not proof
   of good visual transitions. Keep the prototype out of the default resize API
   until it improves the result on additional designs and shapes.
4. Add local constraint repair and visible-asset/ink geometry before interpolating
   arbitrary frames or presenting continuous automatic adjustment.

## Research context

Constraint-based responsive breakpoints are an established way to determine when
a layout needs structural change rather than more scaling:
[Constraint-Based Breakpoints for Responsive Visualization Design and Development](https://arxiv.org/abs/2409.01339).

Optimization can also refine alignment, overlap and containment while minimizing
deviation from an initial graphic layout:
[LayoutRectifier](https://onlinelibrary.wiley.com/doi/10.1111/cgf.70273).

The staged continuation/search approach above is our proposed engineering
experiment, not a claim that either paper implements this exact engine.

## First controlled trial

Run the separate experiment from the repository root:

```powershell
.\.venv\Scripts\python.exe -m tools.continuation_demo --task input/chameleon_sample_level2_core.json --out output/continuation-guided --steps 7
```

The final anchor is pinned to the exact default result. This isolates transition
quality from endpoint quality: neither a smoother path nor a proxy tolerance can
silently substitute a smaller, less readable final composition.

| Trial | Largest normalized unit-center jump | Endpoint |
| --- | ---: | --- |
| Independent default resizes | 0.6282 | Current default |
| First continuation beam, rejected | 0.7989 | Same default |
| Endpoint-guided minimax beam | 0.4647 | Same default |

The first trial postponed almost all movement until the last anchor. The guided
trial adds endpoint-based intermediate seeds, a lower bound on remaining motion,
and a search priority that minimizes the worst jump before aggregate cost. Each
intermediate must stay within 0.06 of its independent baseline's local quality
proxy. Fonts are fitted from original preferences, not prior fitted sizes.

The guided trial reduced the worst measured jump by approximately 26%. Its seven
anchors have zero local hard-contract violations and zero unexpected box
collisions detected by the current metric. The final design and rendered score
are identical to the default. A large 0.4647 jump remains, and the metric still
exempts some source overlaps. These results do not establish ink-level collision
safety or good continuous animation.

`output/continuation-guided/path.png` shows the checked anchors;
`report.json` contains the diagnostics. The rejected trial is preserved under
`output/continuation-core/`. Generated artifacts are ignored by Git. The public
`resize()` entry point does not call this experiment.

Validation at this checkpoint: 348 tests passed with one expected skip; Ruff
lint/format checks passed; the running local studio passed 17 HTTP checks.
Browser checks and decoded JSON/PNG exports are recorded under
`output/ui-verification.json`.
