> October 30 update: the authorized supervised compatibility session is complete,
> checkpoints are verified/restored locally, and the L40S is STOPPED. GPU runtime
> compatibility passed; synthetic judgment collapsed. Actual disk is about124GiB.
> Session status and verification receipts are local run artifacts. Verify the
> actual instance identity, state and stop receipt before any new session.
> Historical quotes/preparation steps below are not instructions to start compute.

# Brev L40S setup for local vision experiments

Checked October 29, 2026. The quote below was originally inspected before any
launch. After explicit user approval, the CLI created an L40S instance;
the user then requested framework-first work. Brev confirmed it
**STOPPED**. No remote environment install, model download, or training started.
Do not restart compute as part of following this preparation guide. Follow the
[backup and run-guard workflow](BREV_DATA_PROTECTION.md) before a new session.

Local setup/checkpoint/controller code now exists and has offline tests. Actual
multimodal GPU compatibility, task-specific training, and reviewed preference
labels remain unvalidated or unimplemented. No billing or access settings changed.

## The selected instance and the $60 budget

The user's [L40S configuration link](https://brev.nvidia.com/environment/new?gpu=L40S&instanceType=l40s-48gb.1x&storage=256&location=any)
was inspected in the live console without pressing Deploy. The displayed quote
was:

| Item | Observed configuration |
| --- | --- |
| Provider / instance | CRUSOE / `l40s-48gb.1x`, marked pre-release |
| GPU | One NVIDIA L40S, 48 GB VRAM |
| Host | x86_64, 147 GB RAM, 8 CPUs |
| Storage | 256 GiB provisioned disk |
| Software selection | VM Mode with Jupyter |
| Compute | $1.74/hour |
| Storage | $0.03/hour, including while stopped |
| Running total | $1.77/hour |
| Lifecycle | Can stop and restart without losing disk data |

These are the displayed, rounded prices at review time, not a reserved rate.
Recheck the actual selected provider, storage, total, and stoppable status before
Deploy. The URL uses `location=any`, so future availability can change the choice.
The [public instance card](https://brev.nvidia.com/environment/new/public?instanceType=l40s-48gb.1x)
also exposes this configuration, although a plain HTTP fetch may return only a
JavaScript application shell.

At the observed $1.77/hour, **$60 buys approximately 33.9 running hours** if there
is no other organization usage, stopped-storage time, tax, or additional charge.
This is an estimate from the displayed rate, not a spending cap or a promise
that the account stops at $60.

| Running time | Approximate cost |
| --- | ---: |
| 2 hours | $3.54 |
| 4 hours | $7.08 |
| 8 hours | $14.16 |
| 16 hours | $28.32 |
| 24 hours | $42.48 |
| 32 hours | $56.64 |

The displayed stopped-storage charge adds about **$0.72/day** or **$5.04/week**
for this disk. Check Billing for shared organization usage and whether auto
recharge is enabled; the observed quote alone does not answer that.
[Brev billing documentation](https://docs.nvidia.com/brev/guides/console-reference#billing--usage).

## What is sensible to train on one 48 GB L40S

[NVIDIA specifies](https://www.nvidia.com/en-us/data-center/l40s/) 48 GB GDDR6 ECC
memory and BF16/FP16 tensor support. This is substantially more useful for our
experiments than the laptop's integrated graphics, but image resolution, number
of images, token length and batch size still determine actual training memory.

| Experiment | Recommendation |
| --- | --- |
| Frozen SigLIP2 image features plus a small pairwise head | First training baseline once preferences have been reviewed; cache the expensive features |
| Qwen3.5-0.8B or SmolVLM-500M LoRA | Good small multimodal compatibility experiment; start with a tiny batch and frozen vision tower |
| Qwen3.5-2B / Qwen3-VL-2B LoRA or QLoRA | Practical next candidates on this GPU, subject to a real multi-image memory test |
| Qwen3.5-4B QLoRA | Plausible controlled comparison if 2B failure analysis warrants it; not the first spend |
| Full fine-tuning or training a VLM from scratch | Poor starting use of these credits; do not infer fit from inference weight size |

Those are engineering judgments, not completed training results. Training stores
gradients, optimizer state and activations in addition to weights.
[Hugging Face's memory explanation](https://huggingface.co/docs/transformers/model_memory_anatomy)
shows why a model that fits for inference can fail during a training step. For
model capabilities, licensing, exact checkpoint names and compatibility caveats,
see [the model research](LOCAL_VISION_MODEL_RESEARCH.md).

Prepare source-family splits, rendered examples, label review, and frozen-model
evaluation before paying for a long run. The initial seed examples exercise a
pipeline; they are not an independently reviewed training or test benchmark.
There is no training-duration estimate until a timed pilot measures actual
examples/second with our three-image inputs.

## Launch and connect when ready

1. In the configuration page, verify the L40S quote and choose a recognizable
   name, such as `chameleon-l40s`. The commands below assume that name; substitute
   the actual name if different. Keep the selected stoppable card. Deploy is the
   step that provisions the paid resource and accepts the displayed provider
   data-sharing terms; it was not performed during research.
2. Wait for Running and setup completion. The selected card estimates up to
   seven minutes. Open its Access tab for connection details.
3. On Windows, use an Ubuntu WSL terminal for the official Brev CLI. Follow
   [NVIDIA's installation instructions](https://docs.nvidia.com/brev/cli/getting-started)
   rather than installing a similarly named package from an unrelated source.
   The official Linux/WSL installer is linked there. This guide does not install
   WSL, the CLI, or credentials automatically.
4. Authenticate as yourself, sync the console-created instance, and connect:

```bash
brev login
brev refresh
brev list
brev shell chameleon-l40s
```

`brev refresh` configures the CLI's managed SSH entries. Brev documents both
`brev shell chameleon-l40s` and `ssh chameleon-l40s`; after a restart, refresh if
the address changed. Do not use `brev create` merely to connect, since that can
provision another instance.
[CLI connection guidance](https://docs.nvidia.com/brev/cli/connectivity).

## Keep the working environment reproducible

On the remote instance, first inspect the environment:

```bash
nvidia-smi
python3 --version
df -h /home/ubuntu/workspace
```

Expect an L40S with approximately 48 GB device memory. `nvidia-smi` reporting a
CUDA version establishes driver capability; the Python environment must still
have a CUDA-enabled PyTorch build that works with that driver.

Keep code and experiment files under `/home/ubuntu/workspace`, which Brev
documents as persistent across stop/start. Use the verified repository remote:

```bash
cd /home/ubuntu/workspace
git clone --branch codex/hackathon-ready https://github.com/rsingh135/chameleon.git
cd chameleon
python3 -m venv .venv-vision
source .venv-vision/bin/activate
```

For an existing checkout, enter it instead of cloning again. Check out the
reviewed experiment commit and record its full SHA before a run. Keep the
training environment separate from the engine environment; do not upgrade the
production renderer's dependencies to make a research notebook run.

Install the matching CUDA PyTorch/torchvision pair using the command produced by
the [official PyTorch selector](https://pytorch.org/get-started/locally/). Then
resolve the Hugging Face stack for the exact model/recipe. This initial resolution
is a **compatibility spike**, not a tested locked recipe. Qwen3.5 support and TRL
vision-loss handling depend on versions; do not combine an old model tutorial
with arbitrary current packages and call it reproducible.

The new setup helper is currently in the local working tree. Until that helper
has been pushed to the selected Git revision, copy it from the local WSL terminal
after the remote clone. Run this example from the local repository root:

```bash
CHAMELEON_LOCAL_ROOT="$(pwd)"
brev copy "$CHAMELEON_LOCAL_ROOT/tools/vision_gpu_probe.py" \
  chameleon-l40s:/home/ubuntu/workspace/chameleon/tools/vision_gpu_probe.py
```

Run that transfer locally, not inside the remote SSH shell.

Once PyTorch is installed, the repository probe provides a bounded first check:

```bash
mkdir -p output/vision_critic
python tools/vision_gpu_probe.py --out output/vision_critic/gpu-probe.json
```

It checks CUDA access, device/free memory, package versions, GPU arithmetic and
BF16 support. A passing probe is not proof that the model's image processor,
quantization, LoRA targets, or backward pass works.

Next run a small multimodal forward/backward and adapter-save/reload test with
the selected model, processor and training stack. Freeze the successful
environment only after that test:

```bash
mkdir -p output/vision_critic/environment
python -m pip check
python -m pip freeze --all > output/vision_critic/environment/requirements.lock.txt
python -m torch.utils.collect_env > output/vision_critic/environment/torch-env.txt
git rev-parse HEAD > output/vision_critic/environment/code-revision.txt
```

Also retain the CUDA wheel index/install command, Python version, VM/base-image
identity, exact model and processor revision hashes, adapter configuration,
dataset/split hashes, image/token limits, seed and training arguments. `pip
freeze` alone does not record the NVIDIA driver or model revision. Recreate the
locked environment and rerun the smoke test before calling it portable.

Model caches, weights, adapters, optimizer states and training outputs should
stay in explicit ignored locations or outside the Git checkout. Commit small
configs, code and provenance manifests. Copy important artifacts off Brev; a
Git push does not preserve files excluded by Git.

## End each paid session explicitly

Save the chosen adapter/head, logs, evaluation results and lock/provenance files.
From the local WSL terminal, for example, copy experiment outputs back:

```bash
brev copy chameleon-l40s:/home/ubuntu/workspace/chameleon/output/vision_critic ./brev-vision-results
```

Copy any separately stored chosen adapter as well, and confirm the local files
open before discarding remote data. Brev's
[file-transfer instructions](https://docs.nvidia.com/brev/guides/development-tools/file-transfer-scp)
document `brev copy` and managed SCP paths.

For a short break, stop this specific instance from the local terminal:

```bash
brev stop chameleon-l40s
brev list
```

Confirm Stopped in the console. Compute stops; the quoted storage charge
continues. Closing SSH, Jupyter or the laptop does not perform that operation.
For a long break, deleting the instance ends its instance charges but permanently
deletes its data; export and verify everything needed first. Some cheaper cards
do not support stop/start at all, so their lifecycle differs.
[Brev lifecycle documentation](https://docs.nvidia.com/brev/concepts/gpu-instances),
[instance commands](https://docs.nvidia.com/brev/cli/instance-management).

Do not use credit exhaustion as a shutdown mechanism: Brev documents possible
deletion of non-stoppable instances and eventual deletion of remaining resources
when credits run out. No automatic shutdown, billing cap, or recharge setting is
configured by this guide. Review the console burn rate during the pilot and
choose subsequent run lengths from actual throughput and remaining credits.
