"""Source-aware evaluation of a resized design (Track B).

Public entry points:

- `build_context(source, assets_dir, fonts_dir)` — one-time analysis of the
  source design: intentional overlaps, containment, groups, text layouts,
  asset statistics. Cache it per task and pass to `score()`.
- `check_contract(source, candidate, target_canvas, context)` — hard
  requirements from RULES.md. Any violation disqualifies a candidate.
- `score(source, candidate, context=None, rendered=None)` — contract check
  plus proxy quality metrics, returning `{"violations", "metrics", "total"}`.

Nothing here mutates its inputs. Nothing here imports `engine.resize`.
"""

from engine.eval.context import Context, build_context
from engine.eval.contract import Violation, check_contract
from engine.eval.score import score

__all__ = ["Context", "Violation", "build_context", "check_contract", "score"]
