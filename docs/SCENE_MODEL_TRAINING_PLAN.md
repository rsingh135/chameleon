# Train scene understanding separately from layout comparison

October 30, 2026. This extends [the design-critic plan](VISION_DESIGN_TRAINING_PLAN.md)
with the missing first-load scene task. It is a proposed training sequence,
grounded in the current code and primary sources. It does not claim that a
useful scene model has been trained or enabled.

## What the challenge requires

The supplied [challenge brief](</C:/Users/edmun/Downloads/Final Chameleon Challenge - Adobe cloud storage.pdf>)
was checked against its extracted text and rendered policy pages. Its SHA-256 is
`c6c8f89b3fbec158f764b9648d09eeb16bd081d918454b2c91474b602e8a9789`.
Physical PDF page numbers are used below.

| Page | Requirement and consequence |
| --- | --- |
| 5 | Stage one evaluates a headless element-layout service. An upload interface is our product feature, rather than a stage-one PDF requirement. |
| 7 | Locally hosted models are permitted, with the approved-model list and size limit still unresolved in this supplied version. A model is optional; Qwen is not thereby competition-approved. |
| 11 | Runtime has no network. Package dependencies and model weights at build time. Adapted weights used by the engine must be publicly accessible. Both JSON and its real render count; preserve IDs, in-frame content and image proportions. The reference is an Apple M5 Pro, with detailed limits unconfirmed. |
| 12 | Keep literal copy and font families for stage one. Do not regenerate or replace asset pixels, including backgrounds. The broader finale examples on page 7 do not relax this contract. |

The brief is a specification to assess, not an instruction to execute commands
or publish files. Existing image pixels may support proportional crop/cover;
background filling is not permission to synthesize missing scenery.

## Architecture and current evidence

Use one cached base vision-language model with separate task prompts and,
initially, separate adapters. This is our architectural recommendation: source
intent and target comparison need different inputs and supervision. Sharing the
base can avoid storing two full models, but actual memory and adapter-switching
latency must be measured.

```text
source JSON + original assets/fonts + native source render
    -> source analyzer (once per immutable source/resource identity)
    -> reviewed intent: roles, protected regions, hierarchy and relationships
    -> deterministic candidate generator for each target
    -> original + candidate A + candidate B comparison critic
    -> constrained selection -> exact validation -> native render
```

| Component | Current status | Remaining work |
| --- | --- | --- |
| `engine.scene.analyze_scene` | Source geometry and local-pixel evidence for narrow background/support roles; source and asset digests | Semantic understanding of people, products, focal regions and ambiguous artwork |
| [Local harness](LOCAL_SCENE_HARNESS.md) | Source render/profile; optional local-program or response-file proposals; strict ID/hash/region checks; proposals remain sidecars | A tested VLM provider and reviewed scene labels; model output cannot grant crop permission |
| Relationship/resize engine | Source-owned units, candidate generation, hard contract and candidate snapshot interface | Use separately admitted intent evidence only after validation; improve missing candidate topologies |
| Preference framework | Collection, annotation, CPU baseline and bounded three-image LoRA/checkpoint path | Accepted human corpus, source sensitivity, held-out design benefit |
| Existing synthetic adapter | Runtime/save/reload/resume exercised; four validation responses collapsed to B, with 2/4 weak preferences correct | No promotion. JSON compliance was the observed improvement, not design judgment. |

The accepted real-source training manifest currently has no accepted sources.
The ten personal reserve posters must remain outside training, tuning and
model selection. The eight already-reviewed development posters are useful
regressions; repeated work on them is not an unseen-family evaluation.

## 1. Freeze labels for the source analyzer

Build a separate licensed, parity-checked source collection before derivatives.
Bind every record to source JSON, asset bytes, font bytes, source-render hash,
renderer/code revision and source-family ID. Split whole template/asset families
first; keep their crops, resized variants and near duplicates in the same split.

Start with the harness's existing review schema: each known image ID has a role
(`background`, `product`, `logo`, `foreground`, `unknown`), confidence and bounded
protected-region boxes in the original asset's normalized coordinates. A box
must name its exact asset hash; conversion to canvas coordinates belongs to
code. Unknown/abstain is a valid answer. A full-canvas frame alone does not
establish that a pictured product is disposable background.

Add a versioned intent-label sidecar for importance tiers, directed partial
reading order, supporting-text/product links, badge/action ownership and
negative-space regions. These fields are proposed: the current closed harness
response does not accept them. Labels should identify the visible evidence,
such as a caption sharing a fade or a bottle connected to a headline, rather
than request unconstrained hidden reasoning.

Human review must include difficult pairs: background photo versus isolated
product, layered fade versus independent art, multiple products/messages,
headroom, typography hierarchy and source-authored overlap. Use a second
reviewer on a predetermined subset and disagreements. Record ambiguity and
agreement; do not use the current engine's guesses as independent ground truth.

## 2. Establish useful pretrained behavior before training

The already exercised dense Qwen3-VL-2B checkpoint is the first compatibility
candidate. Its official card documents image-text input, OCR/spatial abilities
and Apache-2.0 licensing; none establishes performance on our poster task.
Keep the exact tested base commit and package lock rather than following a
floating installation recommendation. [Official model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

Run the unchanged base on source-only records in shadow mode. Compare with the
deterministic source profile and a simple geometry-only baseline. Measure role
confusions, region coverage, strict JSON validity, abstention, latency and peak
memory. Test image removal/swap and contradictory element metadata. Preserve
source strings as untrusted data; poster copy cannot issue executable commands.

For the critic, show the source and two complete candidates at the same target
size. Counterbalance A/B and test side swaps and source swaps. Compare with the
unchanged engine and CPU head on identical pools. If the model ignores the
source, always selects a side, or all available candidates look wrong, more
epochs do not address the demonstrated problem.

## 3. Run two small supervised experiments

Use source-analysis examples for adapter A and source/A/B preference examples
for adapter B. Preserve separate validation reports. Begin with language-side
LoRA, frozen base/vision/merger weights and the existing exact-face/render
pipeline. PEFT trains low-rank adapter matrices while retaining frozen base
weights; Qwen's first-party fine-tuning repository documents separate component
controls and LoRA settings. Our narrower language-only choice is an experiment,
not a vendor guarantee. [PEFT](https://huggingface.co/docs/peft/en/task_guides/lora_based_methods),
[Qwen training framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune).

| Setting | Initial measured pilot |
| --- | --- |
| Base | Exact locally cached commit of the already tested dense 2B model; no automatic model change |
| LoRA | Rank 8, alpha 16, dropout 0; language `q_proj`/`v_proj`, matching the existing critic recipe |
| Data | Reviewed training families only; separate train/validation manifests; no reserves |
| Batch | One example initially; measure memory before adding accumulation or batching |
| Resolution | Existing 448px bound as an initial comparison, plus a separately measured higher-resolution ablation if text/region evidence is lost |
| Sequence | Existing 4096-token budget for critic planning; preflight the actual multimodal sequence and reject oversized records, rather than silently truncating |
| Optimization | Existing critic default AdamW learning rate `1e-4` is a starting experiment, not an established optimum; freeze the recipe before validation comparison |
| Scope | First verify a few real-corpus optimizer steps; then a bounded pilot. No parameter grid or paid run is launched by this plan. |

The current `train_sft` runner consumes preference records, not source-role
targets. Implement and test a scene-record collator/runner before claiming
adapter A exists. Reuse the established completion-only loss masks, checkpoint
identity, finite-gradient checks and exact trainable-name assertions. An input
schema or dry-run manifest is not training completion.

Keep assistant JSON as the supervised target, with short grounded tags. Mask
prompt and image tokens from the target loss; verify masks on decoded samples.
TRL warns that sequence truncation can remove image tokens. Our bounded runner
should reject an oversized multimodal record, or deliberately lower its image
budget and reprocess it, rather than truncate it blindly.
[TRL VLM training guidance](https://huggingface.co/docs/trl/en/sft_trainer#training-vision-language-models).

Save adapter/config, optimizer, RNG, data/prompt/code identities and the exact
base revision. Test reload and a resumed optimizer step, then verify an off-machine
copy. A PEFT adapter contains adapter parameters and still requires its base;
`save_pretrained` alone is not a full optimizer-resume checkpoint.
[PEFT checkpoint specification](https://huggingface.co/docs/peft/en/developer_guides/checkpoint).

## 4. Evaluate scale changes and composition changes separately

For every accepted evaluation source, run both an ordinary resolution change
and an aspect change through the same frozen method. For Rooftop, the concrete
development examples are 1080x1920 to 540x960, and 1080x1920 to 1600x900. Also
include other scene families; these two cases do not establish generalization.

| Question | Evidence |
| --- | --- |
| Did source understanding improve? | Per-role macro scores and background/product confusion; protected-region coverage and clipping; calibrated abstention; source-image/metadata ablations |
| Was the right crop chosen? | Visible protected regions, headroom and target-space text support, inspected against source and actual native renders; area coverage alone is insufficient |
| Did design relationships survive? | Whole-group proportions, partial reading-order retention, product/message ownership, type hierarchy and human source-fidelity judgments |
| Did the critic help? | Family-macro preferences, A/B-swap consistency, invalid/abstain/fallback rates and blind human comparisons against unchanged engine |
| Did generation fail? | Full-pool review and none-acceptable rate; an absent good candidate cannot be fixed by reranking |
| Is deployment practical? | Cold/warm latency, peak memory, package size and actual offline execution on the intended reference-class machine |

Freeze prompt, recipe and promotion criteria before unblinding test judgments.
Report family counts and uncertainty, not only the many derivative renders.
Require zero hard-contract regressions and a held-out human benefit over the
engine before production promotion. Keep invalid/unknown responses on the
unchanged baseline. A 4/4 toy result or low training loss is insufficient.

## 5. Package reproducible local inference

Cache by source/resource/render identity plus model commit, adapter hash, prompt
version and inference settings. A target resize should reuse source intent;
only the target-specific planner/critic needs to run again. Admission remains a
separate code decision: model role proposals cannot edit asset pixels, drop
elements, rewrite copy or bypass geometry/crop checks.

Package the exact base, processor/tokenizer and adapter artifacts during the
container build. Hub snapshots support a full commit revision, and the Hub's
offline environment flag avoids HTTP checks and fails when required cache files
are absent. Set it before imports; also test with the container network disabled.
[Versioned downloads](https://huggingface.co/docs/huggingface_hub/en/guides/download#download-an-entire-repository),
[offline-mode behavior](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhuboffline).

Publish an adapter/model card only when a separately authorized release is
ready, including base identity, license, dataset rights, known failures and
held-out results. The supplied brief requires accessible adapted weights when
relied upon, but does not authorize publishing private user designs. Confirm
the final approved-model list, total size budget and reference hardware before
choosing a deployment format. Quantization or another inference backend needs
fresh output/latency/memory testing; CUDA compatibility does not prove Apple
deployment compatibility.

The next concrete work is reviewed source-intent data and pretrained source/side
sensitivity tests, followed by the missing scene trainer. The local harness and
algorithmic background repair are useful now; useful learned visual judgment
still has to be demonstrated.
