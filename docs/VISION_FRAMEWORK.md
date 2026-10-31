# Run the local vision-critic framework

The framework now collects rendered candidate evidence, serves human A/B review,
exports validated preferences, trains/evaluates a CPU ranking baseline, and
prepares an offline SFT plan. It does not change production selection.
A supervised synthetic GPU compatibility session is recorded below. No
task-trained critic, real human preference corpus or learned design-quality
improvement is claimed. Follow the [backup and run-guard workflow](BREV_DATA_PROTECTION.md)
for run safeguards. Record instance state, checkpoint backups and the verified
stop receipt separately for each session; this guide does not establish live state.

## Local environment

Run from the repository root in PowerShell. The existing project environment
already needs the real renderer for collection. CPU framework additions are
NumPy `>=1.26,<3` and Pillow, without Torch/Transformers:

```powershell
$py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $py -m pip install -r requirements-vision-cpu.txt
```

Alternatively create a separate environment with `python -m venv .venv-vision-cpu`,
install `requirements.txt` and `requirements-vision-cpu.txt` there, and point
`$py` at its Python executable. Preserve the successful dependency versions.

## Collect real evidence

`collect` accepts an input manifest with explicit source/template-family splits:

```json
{
  "schema_version": 1,
  "purpose": "research",
  "sources": [{
    "source_id": "owned-design-01",
    "family_id": "owned-template-family-01",
    "split": "train",
    "source": "source.json",
    "assets_dir": "assets",
    "fonts_dir": "fonts",
    "targets": [{"width": 640, "height": 360}],
    "source_parity_verified": false,
    "training_rights_reviewed": false,
    "rights_reference": ""
  }]
}
```

Paths resolve relative to the manifest; omit resource directories only to use
the bundled resources. Supply design JSON, not a task wrapper. Compare the source
PNG with the original before recording parity approval; record the actual
rights review separately. Editing a flag is not a review. Freeze family-disjoint
train/validation/test sources **before** generating targets or crops. The existing
ten reserves must not be resized, labelled, trained on or tuned against; obtain
a separate accepted corpus. The collector guards known reserve IDs, paths and
fingerprints.

```powershell
& $py -m experiments.vision_critic.collect --manifest inputs.json --out output/collection-01
```

The default pool contains backend-admitted finalists and includes its baseline.
Use `--pool shadow` explicitly for broader contract-valid comparisons; this
does not admit them for production selection. A single finalist is valid evidence
but supplies no A/B pair.

New collection destinations must be absent. Collection output binds canonical
design hashes, actual PNG-byte hashes, source/resources/code provenance and
backend admission. Native renders retain exact canvas dimensions and aspect
ratios. Failed or changed evidence is not published. Collection and exported
dataset directories are immutable; choose fresh output paths for new runs.

## Practice locally without inventing training data

This creates original shape/text fixtures using the bundled Poppins/OFL font
under its actual engine family, `Solstice Sans`. Purpose remains `synthetic`,
source parity remains false, and no labels are generated:

```powershell
$demo = "output/vision-practice-$(Get-Date -Format yyyyMMdd-HHmmss)"
$labels = "$demo/practice-labels.jsonl"
& $py -m experiments.vision_critic.demo --out $demo
& $py -m experiments.vision_critic.label_pairs --collection "$demo/collection" --labels $labels --annotator practice-reviewer --port 8770 --allow-synthetic
```

Open [the local review UI](http://127.0.0.1:8770). It shows original/A/B PNGs
without baseline or layout-family hints. Open images at full size when needed.
Choose A/B, about equal, neither, or abstain; optional observations can identify
candidate elements. Keys **A/B/T/N/X** select a judgment, and **Save and continue**
persists it. Nothing is selected automatically. Complete enough train and
validation comparisons with A/B/tie judgments for the CPU practice run.

The labels path must be outside the collection. Labels append with integrity
checks and fsync; resume skips this annotator's completed physical pairs, even
if seed changes their orientation. `--seed N` is deterministic. Duplicate or
conflicting submissions do not overwrite feedback. A stale `.lock` requires
checking that the prior server stopped before manual recovery.

The annotation queue defaults to train/validation only and reports the count of
concealed test cases. Add `--include-test` deliberately only for held-out review
after freezing the training recipe; evaluation uses the same explicit unblinding
flag. Synthetic practice does not bypass this default.

Normal research requires source parity. Synthetic practice requires the explicit
flag and stays visibly synthetic; it never establishes task-training readiness.
After reviewing, stop the server with Ctrl+C. There are no automatic labels.

## Export, train and evaluate the CPU baseline

```powershell
& $py -m experiments.vision_critic.dataset --collection "$demo/collection" --labels $labels --out "$demo/export" --allow-synthetic
& $py -m experiments.vision_critic.train_ranker --collection "$demo/collection" --labels $labels --out "$demo/cpu-head" --steps 200 --max-seconds 30 --allow-synthetic
& $py -m experiments.vision_critic.evaluate --collection "$demo/collection" --labels $labels --model "$demo/cpu-head/model.json" --out "$demo/cpu-evaluation.json" --allow-synthetic
```

Export produces `dataset.json`, `sft.jsonl` and `pair-features.jsonl` with matching
hashes. Invalid/stale labels, source/family leakage and conflicting identities
are rejected. The current CPU model is **handcrafted rendered-image features
plus diagnostics and a pairwise logistic head**, not SigLIP or another pretrained
vision encoder. Ties use a soft target; neither/abstain are excluded from fitting.
Only training records fit normalization/weights. Evaluation conceals test results
by default; `--include-test` deliberately unblinds them after model/recipe freeze.
Synthetic results demonstrate plumbing, not design judgment.

## SFT planning and readiness

With a separately reviewed exact 40-hex base-model commit, a dry run prepares
the recipe without loading/downloading model weights:

```powershell
$baseRevision = Read-Host 'Reviewed Qwen3-VL-2B base commit (40 hex)'
& $py -m experiments.vision_critic.train_sft --collection "$demo/collection" --labels $labels --out "$demo/sft-plan" --base-revision $baseRevision --dry-run --allow-synthetic
& $py tools/vision_preflight.py --collection "$demo/collection" --labels $labels --allow-synthetic --out "$demo/preflight.json"
```

`--execute` is separate, requires real accepted data and later GPU compatibility;
synthetic execution is refused. Default model loading is offline. Qwen3.5 needs
a separate compatible profile; the pinned Qwen3-VL smoke environment does not
validate it.

Before another short GPU compatibility session, finish local framework tests,
freeze code/configs, verify backup/stop handling and reconcile actual billing.
Before paid **task training**, additionally obtain reviewed parity/rights,
human preferences, independent splits and passing artifact-aware preflight with
`--require-training-ready`. The accepted-source manifest is currently empty.
Synthetic data can never satisfy that gate.

Keep the VM stopped. The original two-hour guard deadline is stale and must not
be reused for a future session; it is not a native credit cap. Follow the cloud
status document for a fresh supervised session and reserve.

## Supervised synthetic L40S findings — October 30, 2026

The pinned Qwen3-VL-2B environment completed source/A/B image conditioning,
assistant-only LoRA forward/backward, complete adapter/optimizer checkpoints,
adapter reload and a resumed optimizer step on one L40S. The three-step smoke
took 6.77 seconds and peaked at 5.169 GiB of Torch-allocated VRAM. Reload loss
matched exactly (0.561053872). Its generated JSON was invalid and correctly
required fallback. Off-VM checkpoint backup and restore integrity were verified.
The initial setup exposed missing Python venv support and Python development
headers; bootstrap now checks these prerequisites without installing them.

A separate **synthetic weak-label experiment** completed 24 steps in 15.55
seconds, peaked at 6.535 GiB of allocated VRAM, and verified adapter reload.
It used four training and four validation comparisons from separate template
families; test judgments stayed concealed. Strict validation JSON increased
from **0/4 to 4/4**, but the trained model agreed with only **2/4 weak choices**.
The CPU head and untrained reference similarity each agreed with **4/4**.
Training loss decreased across a changing sequence of examples; that is not a
held-out quality result. These observations establish runtime/checkpoint
compatibility and some output-format fitting, **not improved design judgment**.
Every synthetic artifact retains `task_model_trained=false`.

Next, test A/B swaps while preserving the same physical choices; use source
ablation (remove or replace the source image) to check whether predictions
depend on it; and inspect invalid, ungrounded or unknown-element issue JSON.
Keep those probes separate from held-out scoring. Obtain a larger accepted,
human-reviewed corpus with family-disjoint splits and compare against the CPU
and reference baselines before spending on longer training. More steps on this
small programmatic curriculum are not justified by the current result.

Local verification before the GPU session: **266 framework tests passed**, plus **30 snapshot/selection
regression tests**. Three Linux-only checks were skipped on Windows and checked
separately in WSL. Native synthetic collection, CPU training/reload, SFT dry-run
and browser save/resume were exercised. See the local
[integration report](../output/vision-framework-v3/integration-report.json),
[test report](../output/vision-framework-v3/local-tests.xml), and
[practice workspace](http://127.0.0.1:8770). This is infrastructure evidence;
The supervised run above adds GPU infrastructure evidence; real design-quality
learning remains unvalidated.
