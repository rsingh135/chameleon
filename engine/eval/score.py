"""Aggregate: contract first, then weighted proxy metrics.

`score()` is the interface agreed between the layout track and the eval
track. A candidate with violations gets `total = 0.0` and is never chosen
over a valid one; among valid candidates `total` is a weighted mean of the
metrics in `WEIGHTS` that were actually computed.
"""

from __future__ import annotations

from PIL import Image

from engine.eval.context import Context, build_context
from engine.eval.contract import check_contract
from engine.eval.metrics import compute_metrics
from engine.resources import using_resources

WEIGHTS: dict[str, float] = {
    "collision_free": 3.0,
    "text_size": 2.0,
    "text_contrast": 2.0,
    "text_busyness": 1.0,
    "text_lines_stable": 0.5,
    "hierarchy": 1.0,
    "containment": 1.5,
    "group_integrity": 1.5,
    # Product prominence and use of the new canvas must outweigh the appeal
    # of shrinking the entire old composition into a small central island.
    "hero": 3.0,
    "scale_consistency": 1.5,
    "position_affinity": 1.0,
    "alignment": 0.75,
    "balance": 0.5,
    "margin": 0.75,
    "coverage": 1.0,
}


def score(
    source: dict,
    candidate: dict,
    context: Context | None = None,
    rendered: Image.Image | None = None,
    target_canvas: dict | None = None,
) -> dict:
    """Evaluate one candidate.

    `rendered`, when given, must be the *plate* (candidate rendered with text
    removed, see `engine.eval.render.render_plate`) so pixel metrics measure
    what sits under the text. Pass nothing to get JSON-only metrics.
    """
    context = context or build_context(source)
    with using_resources(context.assets_dir, context.fonts_dir):
        violations = check_contract(source, candidate, target_canvas, context)
        if violations:
            return {"violations": violations, "metrics": {}, "total": 0.0}

        metrics = compute_metrics(source, candidate, context, plate=rendered)
        weighted = [(WEIGHTS[k], v) for k, v in metrics.items() if k in WEIGHTS]
        total = sum(w * v for w, v in weighted) / sum(w for w, _ in weighted) if weighted else 0.0
        return {"violations": [], "metrics": metrics, "total": round(total, 4)}
