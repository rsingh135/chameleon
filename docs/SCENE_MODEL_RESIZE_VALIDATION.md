# Source-scene model and two-mode resize verification

October 30, 2026. These results bind the engine implementation committed as
`236956963b43f73a39929dbbe059ee9116fe3e58`. They are a bounded development audit,
not evidence that every design or target is solved. All visual reviews were
performed by agents; no independent human acceptance is claimed.

## What ran

Ten development sources (eight reviewed external works plus Warmup/Core) were
rendered at a smaller resolution and an orientation-changing target. Each target
ran through both the deterministic backend and the actual trained local model's
opt-in crop-veto path. Rooftop additionally ran at 800×500 in both conditions:
**42 native renders, 21 baseline/model pairs**. Advanced's untouched original was
reported separately as blocked by its intentionally missing asset.

The source model ran once per source, with the same response reused for its
targets. The frozen CLIP base, stable processor, trained 80-step head, provider,
configuration and engine/harness bytes were checked before/after the run.
All source dictionaries, prepared resources and the ten reserve examples stayed
unchanged. The reserve ZIP SHA-256 remains
`257d4aee7a4f817e76846e08a469e39da5f520e5e11afa0f6ff48f219e31f5c6`.

The [local gallery](http://127.0.0.1:8765/posters/scene-model/index.html) shows
originals and both resize modes. Its [bound review](http://127.0.0.1:8765/posters/scene-model/review.html)
keeps actual failures and style observations visible. Full evidence is retained
locally in `output/scene-model-matrix-final/`, including `report.json`,
`verification-summary.json`, `visual-review.json` and per-source contact sheets.

## Fixed after actual render inspection

| Demonstrated issue | Result |
| --- | --- |
| Core 540×540 enlarged the product, badge and type instead of simply changing resolution | Exact-aspect, contract-valid requests now preserve one uniform source scale. Core and Rooftop half-size geometry error is 0.0; all 16 mathematically exact-aspect audit outputs retain geometry/type/radius/stroke within 1e-9. |
| Rooftop 800×500 chose photo-left/copy-right with tiny header text | A source-font-floor-derived caption sample fills the background, keeps all 17 elements and 13 type ranks, with 10px minimum type and 85.279% relative photo visibility. Existing photo/support guards remain enforced. |
| Replacing an asset while retaining its byte length and timestamp could reuse an old scene | Scene and result caches bind actual image bytes. Stale supplied predictions are rejected before a cache hit can be returned. |
| A protected off-canvas product could exchange or lose its visible edge | Crop vetoes require containment of each source-visible normalized interval, alongside unchanged source crop, aspect and image position. |

The four rounded-integer Trifold/Cheatsheet outputs are explicitly outside the
mathematically exact-aspect claim. Supporting small letterbox tolerances is a
separate follow-up; it must retain contract checks and source proportions.

## Remaining failures

All 42 renders executed and passed their hard contract. **34 met the complete
audit acceptance checks;8 condition records failed readability**, representing
four unique source/target cases. There were no model-induced layout changes:
all 21 paired design and PNG hashes match.

| Source/target case | Minimum type | Interpretation |
| --- | --- | --- |
| Core half resolution | 9px | Original 18px body at exact half scale; retain the resolution/readability tradeoff explicitly. |
| Botanical reduced resolution | 6.75px | Original 9px fine print was already small. |
| Botanical orientation change | 8.36254px | Fine print remains below the audit's 10px floor. |
| Trifold portrait | 9.84545px | Actual poor packing: narrow columns remain crowded above large empty lower bands. This needs another composition topology. |

Rooftop reduced/orientation/custom exports keep all 13 text sizes at one shared
factor, preserve reading order and full background coverage. Its 800×500 native
result was inspected directly; it retains clear headroom, smooth fade, complete
date row and the selected-day pill. This observation applies to those inspected
renders, not unmeasured face or object-region detection.

## Model outcome

The local role-head trained, reloaded and resumed successfully. Resumed and
uninterrupted 80-step heads are byte-identical. Validation stayed **9/23**, equal
to zero-shot CLIP, without an independent product-positive validation family.
Predictions in the resize audit all remained below the 0.8 crop-protection
threshold, so the model abstained and did not improve any layouts. The model
remains experimental and is not automatically enabled in Studio.

See [training results and reproduction](LOCAL_SCENE_MODEL_PILOT.md) and
[additional product-source research](SCENE_PRODUCT_DATA_SOURCES.md). Adding more
examples requires genuine product/asset diversity and verified rights; the ten
personal-test reserves remain outside training and tuning.

The source picker is separately committed as `0fee77e`: visible above the
canvas in Result and Compare, retaining target dimensions and source-bound
exports, with poster/brand URL restoration. Backend render quality and picker
functionality are verified separately.

These are local contract/render checks; official judging and an Adobe Express
host round trip are not established by this audit.
