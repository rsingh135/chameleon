# Model validation, 31 October 2026

The production resize engine uses the trained compact fit policy and native
layout constraints. The vision adapters below are experimental and do not have
default layout authority. A purpose prediction describes an element's visual
role; it does not independently authorize cropping, image replacement, grouping,
or geometry changes. The separate A/B/N critic chooses between admitted layout
candidates or abstains. Neither classifier is a trained coordinate generator.

## Changes that were implemented

- Freeze a family-interleaved coverage sampler into each run and checkpoint its
  full state. The corrected R3 run saw all 259 eligible training records,
  including all 14 supporting-copy examples across six families.
- Exclude eight ambiguous training targets without changing canonical source
  labels or pixels. Pin the same broader supporting-copy definition in training
  and inference. Add licensed, agent-authored scenes with explicit provenance.
- Test source typography as an observation-only input: font-size rank, relative
  size, weight, style and alignment. Element IDs and semantic labels are not
  model features.
- Add an explicit native-alpha contrast-matte policy for isolated model views.
  Original design assets and exported pixels are unchanged. Admission verifies
  the native receipt, alpha bytes, compositor implementation and resulting RGB
  bytes; the existing checker policy remains the default.
- Bind both base and adapted evaluation predictions to their actual file hashes.
  Preserve exact FP32 adapter tensors during PEFT reload and fail on a mismatch.
- Add an opt-in image-role template that labels each source/candidate immediately
  before its image. Preserve the legacy template for older runs.

## Measured development results

These are weak, agent-reviewed labels on reused development families, not a
sealed final test or a human aesthetic study. Each model is compared with the
pretrained model using the same input contract. R3 changes data, definitions,
coverage and training length relative to R2, so that comparison is not an isolated
ablation.

| Purpose run | Pretrained / 175 | Adapted / 175 | Adapted macro F1 | Support recall |
| --- | ---: | ---: | ---: | ---: |
| R1, three views at 256 | 40 | 76 | 0.408 | 0/6 |
| R2, three views at 512 | 81 | 109 | 0.383 | 0/6 |
| R3, cleaned data and coverage, 128 steps | 105 | 139 | 0.545 | 0/6 |
| R4, R3 plus observed typography | 87 | 139 | 0.549 | 0/6 |
| R5, R3 with isolated contrast matte | 99 | 138 | 0.466 | 0/6 |
| R6, R3 plus 44 technical training elements | 105 | 140 | 0.424 | 0/6 |

R3 and R4 each classify 137/166 original elements correctly, exceeding the
106/166 always-body baseline. Both detect all 18 development backgrounds.
Both still miss all six supporting-copy examples and recognize only 2/9
blank/mismatched controls. R4 fixes four R3 errors and breaks four; it is not a
clear winner. Their input images, training draws and settings were independently
checked for the matched typography comparison.

The matte makes pale isolated elements easier to see, but R5's aggregate model
score regresses and supporting copy remains unresolved. Better input contrast
does not by itself establish better classification or layout quality.

R6 ties R3/R4 on original elements (137/166); its additional correct prediction
comes from a control. It improves body accuracy but loses the single primary
visual and all four footer cases, so the higher overall count is not a balanced
quality improvement.

## Independent synthetic tuning

Three new, independently authored native-layout families supply 42 original
targets and nine blank/replaced-image controls. Their source intentions and gold
were frozen before predictions. Matching pretrained and adapted outputs are
bound to the requests, pixels, weights, code and both prediction-file hashes.
There are no missing, invalid or fallback predictions in these results.

| Run | Pretrained originals | Adapted originals | TRAIN-only metadata baseline | Support recall | Controls |
| --- | ---: | ---: | ---: | ---: | ---: |
| R3 | 16/42 | 23/42 | 23/42 | 0/9 | 1/9 |
| R4 | 18/42 | 23/42 | 23/42 | 0/9 | 0/9 |
| R5 | 20/42 | 25/42 | 23/42 | 0/9 | 0/9 |
| R6 | 16/42 | 25/42 | 26/42 | 1/9 | 1/9 |

All four candidates fail the frozen promotion thresholds. R5's accuracy gain over
its metadata baseline does not repair supporting copy or control handling. R6
trails the metadata model trained on the same expanded training set. The separate
final purpose and critic sets remain sealed and unused.

The critic's V3 run scores 7/12 on its small public development gold set and gets
both candidate orders right on 1/4 physical comparisons. Increasing only training
length from 32 to 128 steps (C4) collapses to B on all 20 probes. Adding only
interleaved image labels at 128 steps (C5) breaks that collapse but still scores
4/12 and gets both orders right on 0/4 comparisons. Its abstention on 7/8
unlabeled missing-source probes is a behavioral observation, not accuracy.
These results do not justify enabling the critic.

The critic training set contains only six distinct unordered candidate pairs
across three synthetic families. Its neither-candidate examples cover source
mismatches rather than same-source ties or two poor designs. C5 reaches very low
late training losses while failing development preferences; training loss is not
evidence of generalization. A later dataset needs broader native preference
conditions and reviewed design judgments before another promotion attempt.

## Preservation and deployment

Runs save every four optimizer steps, including adapter/configuration,
optimizer/scheduler/RNG state, sampling coverage, frozen code, data and hash
manifests. Completed runs are copied off the VM and every saved generation is
verified. Exact optimizer-state replay has not been validated; file preservation
does not establish exact resumability.

The independent backup audit protects 232 checkpoint generations across eleven
audited cloud runs, including every generation from the six 128-step runs.
All training and tuning outputs were copied locally. The external guard
confirmed the Brev instance **STOPPED at 04:58:20 UTC**, before the fixed
05:01 UTC deadline. No additional training run is queued.

The independently authored synthetic evaluation separates tuning and final
families. Gold stays outside GPU input packages. The personal-test reserve
designs remain unused. Synthetic results must be described as weak synthetic
generalization, not real-poster or Adobe-host verification.

Failed adapters remain experimental. More optimizer steps alone did not repair
the critic, and typography alone did not repair supporting-copy classification.
Promotion requires the predeclared per-class, control, preference-order and
baseline checks, together with native-render validation of the resulting edits.
