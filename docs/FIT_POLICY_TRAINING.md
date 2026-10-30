# Trained local layout-fit policy

The fit policy is a compact, trained pairwise logistic ranker over 28 measured
layout features. It chooses actual generated candidate geometry. Trusted layout
code owns element edits, font measurement, copy preservation and contract gates.
The training process uses the standard library and does not load Torch.

This training updates the fit-policy weights. It does not update Qwen or CLIP
weights, and its input is layout geometry rather than RGB embeddings. The
optional local VLM adapter remains a separate experiment.

## Data and splits

`experiments/fit_policy_dataset.py` uses an explicit development allowlist:
Core and the catalogue's exact eight reviewed authored works. All ten personal
reserve sources are excluded before source or asset loading. Modern Resume is
a reserve and remains manual acceptance material, outside training and validation.
The catalogue's reviewed designation concerns source screening; it does not
provide human preference labels for resized designs.

| Cohort | Training families | Held-out families |
| --- | --- | --- |
| Authored development | Core, Grand Opening, Rooftop, Night Bloom, openSUSE Booth, openSUSE Trifold | Bold Sale, Botanical Invitation, openSUSE Cheatsheet |
| Original synthetic columns | Materials Field Report, Harbor Guild Bulletin, Civic Garden Plan | Orchard Operations Digest |

Every target, variant and candidate from one source family stays in its assigned
split. Shared publishers and fonts across authored families limit the independence
of this small development evaluation.

The four synthetic documents contain original fictional report/bulletin/planning
copy, a wide heading band, an authored column separator and eight typographic
sections. They vary source canvas, column widths, gaps and font sizes, using
existing brand-kit fonts and no image assets. Their geometry and copy are not
copied from Modern Resume or any reserve. Generated original source JSON is
saved alongside the dataset for inspection.

Authored targets are 800×500 and 900×1600. Synthetic-column targets are 1080×1080
and 1600×900, with at most two per source. Exact-ratio resolution requests bypass
policy training. The fitted-column generator also requires at least 35% widening;
the letter-sized bulletin's square target falls outside that contract and is
explicitly skipped.

## Actual geometry and synthetic supervision

Candidates come from the real deterministic, grouped and fitted-column generators.
Synthetic-column groups compare actual fitted-column operations with the two
source-preserved baseline variants. IDs bind complete candidate geometry; no
model-call stubs, invented preferred candidate outputs or hand-written model
weights are used.

Each training candidate passes `check_contract` and strict rotated-frame
containment using the document's native font resources. Only source-classified
background decorations may receive the existing strict containment adjustment.
Source graph units are serialized as `unit_id`, `element_ids`, `role` and a
normalized `box`; the same units must be passed to policy feature extraction
at runtime. Source IDs and candidate IDs are audit metadata, not learned inputs.

The labels are explicitly `synthetic_geometry_teacher_v1`. They distill a
disclosed measurement rule, not human aesthetic judgments:

```text
occupancy = .7*sqrt(clamp(foreground_bbox_area)) + .3*clamp(foreground_area)
readability = .5*clamp(minimum_font/1.8) + .5*clamp(median_font/2.8)
group_style = mean(font_scale_consistency, hierarchy_retention,
                   unit_scale_consistency, unit_attachment)
crowding = max(0, text_frame_fill - 1.3)

teacher = 3*occupancy + 1.8*readability + group_style
          + .8*reading_order_retention + .7*image_proportion_retention
          - .7*crowding - 1.2*overflow_fraction
```

`clamp` means a bound to `[0,1]`. Font features express font size as a percentage
of the target's short side. Occupancy uses element frames and excludes detected
background content; it does not measure actual alpha silhouettes or text ink.
Style, reading-order and image-proportion terms come from source relationships.
These are transparent proxies with known limits: a large bounding box alone
does not establish a good composition or readable rendered copy.

Each source/target group needs at least two distinct hard-valid geometries and
a teacher score gap of at least `.01`. The builder retains both lower-scoring
and fitted candidates, records actual designs and label terms, and fingerprints
the generator/feature implementation before and after generation.

## Reproduce generation and execute training

For a fresh complete dataset:

```powershell
.\.venv\Scripts\python.exe -m experiments.fit_policy_dataset `
  --out output/fit-policy-training-expanded
```

An existing authored-only dataset can be reused to avoid regenerating it:

```powershell
.\.venv\Scripts\python.exe -m experiments.fit_policy_dataset `
  --base-dataset output/fit-policy-training/dataset.json `
  --out output/fit-policy-training-expanded
```

Reuse checks the source and design digests, current extracted features, teacher
scores, family split, reserve exclusions and hard-valid provenance. New synthetic
column candidates still run through the actual generator and native contract gates.

Execute actual weight optimization and publish the checkpoint atomically:

```powershell
.\.venv\Scripts\python.exe -m tools.train_fit_policy `
  --dataset output/fit-policy-training-expanded/dataset.json `
  --out models/fit_policy/default.json --epochs 200
```

The recipe uses pairwise logistic SGD, learning rate `.08`, L2 `.02` and seed
`1729`. Feature means/scales are fitted on training records only. The artifact
stores learned weights, normalization, dataset and implementation digests,
source-family membership, optimization losses and held-out metrics. Training
and validation metrics are also separated by `source_kind`, so synthetic-column
agreement is distinguishable from authored-development agreement.

## Measured evidence and acceptance

The expanded run actually optimized 314 pair comparisons for 200 epochs in
0.500 seconds. It contains 146 hard-valid candidate records across 24 target
groups and 13 source families: 105 training records from nine families and 41
held-out records from four families. The added column cohort contains 28 actual
fitted-column operations and 14 uniform/source-preserved alternatives across
seven supported targets. Pairwise loss fell from `.28357` to `.09984`.

| Held-out domain | Candidate records / groups | Teacher pairwise agreement | Teacher winner agreement |
| --- | --- | --- | --- |
| Authored development | 29 / 6 | 90.28% on 72 comparisons | 5 / 6 |
| Original synthetic columns | 12 / 2 | 100% on 30 comparisons | 2 / 2 |
| Combined | 41 / 8 | 93.14% on 102 comparisons | 7 / 8 |

The zero-weight comparison matches two of eight combined teacher winners.
Synthetic-column validation is one held-out fictional template family with two
targets; this small result is not a broad generalization estimate.

The published artifact is `models/fit_policy/default.json`, model digest
`ec03b22d69e323651e6c1b48d54416ac157cd6ace767ea5916db63775c4fdee8`,
file SHA-256
`0178c80ffb46b79cc84cf93369bce151f8fe8d9493b33e352dbbaa0e5c56684e`.
Its expanded dataset has canonical digest
`e2d068f3fa2f432a2382bf2770d6d9a6adc964c6a2486fcd9383df1edfc2cdbc`
and file SHA-256
`fc2bf3c72b19d01bc00f9ef04ddce3e625c85440ae8adb83039bde87acb09230`.
The generation receipt confirms implementation bytes remained unchanged during
the successful expanded build. Detailed records, actual candidate JSON and
training results are under `output/fit-policy-training-expanded`.

The first authored-only optimization completed 200 epochs over 239 comparisons
in 0.422 seconds. It used 104 candidates in 17 source/target groups: 75 training
records across six families and 29 held-out records across three families.
Pairwise loss fell from `.33838` to `.09365`. Held-out teacher pairwise agreement
was 90.28%, and the trained winner matched the teacher in five of six target
groups versus two of six for zero weights. This initial run contained no new
fitted-column states; its checkpoint and data history are retained separately
from the expanded column-training experiment.

These metrics measure agreement with the synthetic geometry teacher. Native
rendered review must assess actual text, section order, group cohesion, empty
space, source style and clipping. Root's independent Modern Resume acceptance
can test the new fitted-column operation but must not be added to training or
described as an untouched benchmark after user exposure.

For later training, collect source-family-separated pair preferences from real
source/candidate renders, record whether a person or an agent supplied each
label, and compare against this synthetic baseline. Keep the ten personal
reserves out of fitting, tuning and automatic example selection. A separate VLM
fine-tuning run would require actual multimodal optimizer steps, saved/reloaded
weights, and its own held-out visual evaluation; it has not occurred here.
