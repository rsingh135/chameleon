# Experimental saved-adapter critic bridge

This bridge transports the trained A/B/N classifier's **actual closed-class
scores** into the graph verifier contract. It does not generate visual reasoning,
issue descriptions, layouts, or geometry. It is off by default, unqualified, and
supports explicit diagnostic use only. Passing protocol tests is not evidence of
model quality. The separately trained critic adapter still needs independent
evaluation; no result from that evaluation is asserted here.

`adapter_critic.py` prepares one label-free worker record with the frozen training
`CRITIC_PROMPT` and three real native views in this order: ORIGINAL source,
candidate A, candidate B. Candidate IDs and paths bind the transport; they never
enter the model prompt as features. The caller supplies the existing native
`build_visual_critic_menu(snapshot)` and its actual rendered PNGs, not an external
model-authored menu. Only two distinct admitted designs can be compared.

## Prepare and score

```python
from experiments.agentic_graph.adapter_critic import (
    build_adapter_critic_request,
    request_bytes,
    verify_render_files,
)

bundle = build_adapter_critic_request(source, source_png, critic_menu, native_renders)
verify_render_files(bundle)
request_path.write_bytes(request_bytes(bundle))
```

Freeze the trusted prepared identity **before** inference. The operator owns the
asynchronous cloud/GPU job and its memory, time, checkpoint-loading, and process
guards; this bridge starts no model job. The existing label-free evaluator uses:

```text
python -m tools.evaluate_visual_adapter --request REQUEST_JSON
  --training-run IMMUTABLE_RUN --adapter SAVED_CRITIC_ADAPTER --out NEW_EVALUATION_DIR
  --base VERIFIED_BASE --base-manifest BASE_MANIFEST --prepare-only --inference-only

python -m tools.run_prepared_visual_adapter --recipe EVALUATION_DIR/recipe.json
  --python GPU_ENVIRONMENT_PYTHON
```

Arguments above are schematic paths. Preserve the exact prepared `recipe.json`,
dataset, request bytes, worker code snapshot, and model identity. Completion takes
the **raw bytes** of `external_adapted.json` and its critic-specific
`inference_receipt.json`; a purpose adapter receipt is rejected. The expected
identity includes base repository/revision/weight SHA, adapter weight/config SHA,
recipe SHA, dataset SHA, and copied worker code SHAs. The receipt must bind the
exact request and prediction-file bytes. It remains experimental and cannot claim
quality approval. The score file must attest all three ordered image embeddings.

The optional `base_prediction_file_sha256` receipt field accepts null or an exact
lowercase SHA-256. If non-null, completion requires the actual base score file
bytes through `base_prediction_bytes` and checks their hash and byte limit.
Operator scoring callbacks can return those optional bytes alongside the existing
three fields. Old receipts remain valid without it. Base scores are provenance
only; they never replace the adapted A/B/N preference or alter native gates.

## Graph integration

```python
from experiments.agentic_graph.adapter_critic_roles import create_adapter_critic_roles

roles = create_adapter_critic_roles(
    existing_roles,
    score_runner=operator_score_runner,
    model_identity=preselected_critic_model_and_code_pins,
    driver_identity=operator_driver_identity,
    mode="diagnostic",  # omit for off; there is no qualified/production mode
)
```

`operator_score_runner(bundle, timeout_seconds=...)` returns exactly
`prediction_bytes`, `receipt`, and `prepared_identity`. The static base, adapter,
adapter configuration, and worker code pins must match the factory's identity;
the dynamic recipe/dataset pins must be frozen by trusted preparation before the
worker runs. The callback is responsible for using the real saved adapter. An
injected test callback stays identified as a protocol test. Only an explicitly
supervised driver with supervised delegated roles can report that execution
scope; this does not qualify the model or enable deployment.

The wrapper replaces only `verify`. It delegates `analyze` and `organize`
unchanged, checks the source and both candidate PNG bytes/dimensions before and
after scoring, enforces the graph's remaining time budget, and rejects changed
source/menu/request/model context. Optional source analysis is validated and
logged, but is **not** fed into this classifier: the frozen critic prompt was
trained on three image views, not analyzer prose.

An A or B output maps to that exact menu position's known candidate ID. The
decision contains only `schema_version`, `menu_digest`, and `plan_id`; no learned
rationale or issues are invented. The current graph consumes a valid verifier by
immediately indexing its decision. Therefore N, weak scores, a cold or disagreeing
full-vocabulary token, and disabled mode return **no decision** and graph
`valid=False`. A well-formed abstaining score separately has
`prediction_valid=True`. The graph's existing deterministic fallback/stop behavior
remains in control.

The diagnostic screens require normalized closed-class softmax at least 0.85 and
margin at least 0.30. These are conservative protocol screens, **not calibrated
confidence**, learned quality thresholds, or evidence of useful visual judgment.
N always abstains, including a strong N score. Missing/extra classes, invalid
numeric distributions, logits inconsistent with softmax, mismatched ordered
hashes, extra sealed labels, and unknown record IDs fail closed. A verified model
receipt may truthfully record `executed=True` even when its malformed or weak
output cannot be used.

## Native geometry evidence

`context_digest` binds source JSON, source PNG, target, ordered neutral critic
menu/IDs, actual PNG references/hashes, prompt, and exact worker request bytes.
`choice_digest` joins that context, the actual score-file SHA, saved adapter SHA,
and the bounded decision. These hashes instrument which neural preference was
processed; they do not certify design quality.

For direct diagnostics, `complete_and_apply_adapter_critic(...)` reruns completion
from the raw score evidence, checks actual image files, then calls the existing
`engine.visual_selection.apply_visual_selection` with the trusted snapshot and
critic menu. The executor rechecks admission, full-design candidate identities,
source/target, the hard content contract, and strict bounds. The returned
`native_execution` trace joins the context/choice digests to the actual applied
design SHA and native selection trace. `quality_improved` remains `None`.

The graph already uses that same native executor after `roles.verify`; it stores
the role's context/choice trace and selected design independently. The bridge
does not widen the candidate pool, manufacture admission, replace native
verification, or infer cover/contain authority from classifier scores. A native
producer must create the PNGs from those candidate designs; image hashes alone
are not a substitute for that trusted render producer.

## Verification

`tests/test_adapter_critic_bridge.py` uses synthetic score receipts and tiny real
native shape renders. It tests known-ID mapping and A/B order changes, exact
three-view binding, N/cold/weak abstention, request/model/source/pixel mismatch,
disabled behavior, deadline enforcement, and application through the real native
executor. Its causal fixture makes A and B choose different actual x coordinates
and records the resulting geometry hashes. This proves transport and execution
wiring only, not actual trained-adapter judgment, generalization, or aesthetics.

The source-purpose two-view bridge is separate. No purpose receipt, label, hidden
role rule, or future three-view isolated-layer contract is reused here.
