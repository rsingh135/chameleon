# Adroit setup for short local vision-critic training experiments

Researched October 29, 2026 against Princeton Research Computing's public documentation. This is a proposed workflow, not a record of an executed cluster session. No SSH connection, account change, model download, environment installation, or job submission was performed. The VLM trainer and its model-specific configuration remain to be implemented separately.

## Recommended first experiment

Use one scheduled GPU: first a 15-minute environment and real-model LoRA smoke test, then a two-hour allocation with periodic checkpoints. Prefer an A40 or an 80 GB A100 for the first VLM run. Keep a conservative 20 GB configuration available if the scheduler assigns a MIG device. A frozen image encoder and small ranking head are the lower-memory alternative. These are engineering recommendations, not measured throughput or fit guarantees.

### Verified Adroit resources and scheduling

| Publicly listed GPU allocation | Count | Device memory |
| --- | --- | --- |
| A100 | 4 | 80 GB each |
| A100 MIG | 8 slices | 20 GB each |
| A40 | 2 | 48 GB each |

Request `--gres=gpu:1`. Published constraints are `a100`, `a40`, and `gpu80`; `a100` alone does not specify 80 GB. No dedicated 20 GB constraint was verified. Published GPU scheduling categories allow 15-minute tests, four-hour short jobs, 24-hour medium jobs, and two-day long jobs. **Do not set `--qos`**: Adroit routes jobs by requested resources. Live limits and availability can differ; inspect `snodes`, `gfree`, and `qos`. [Official Adroit system documentation](https://researchcomputing.princeton.edu/systems/adroit).

The general GPU overview still lists a 40 GB Adroit A100 configuration, whereas the specific Adroit page describes its conversion into 20 GB MIG slices. Treat the live scheduler/node inventory as the deciding evidence. [General GPU overview](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing).

Neither an account name nor a partition is invented in the launcher below. If the user's account needs an additional site setting, obtain it from their allocation or Princeton support.

## Where each operation belongs

Use the login node for lightweight shell work, source preparation, environment setup, downloads, and Slurm submission. Do not train or load a large model there. Princeton's login policy restricts compute and memory-intensive processes. GPU login nodes are also preparation nodes. Visualization nodes permit short GPU tests and downloads, but are shared and should not be used as a substitute for scheduled training. [Login/GPU/visualization-node policy](https://researchcomputing.princeton.edu/get-started/dos-and-donts).

Adroit's compute nodes have no internet access. Download model, processor/tokenizer, and dataset files before submitting. Put the HF cache on `/scratch/network/$USER`, and load the downloaded model by local path in training. [Princeton Hugging Face guide](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face).

| Location | Proposed Chameleon use |
| --- | --- |
| `/home/$USER` | Small source checkout, scripts, configuration, split manifests and labels; optionally the Conda environment if quota permits |
| `/scratch/network/$USER/chameleon` | Model files, image corpus, HF/package caches, training input, checkpoints and logs |
| Compute-node `/tmp` | Optional disposable job-local staging; never the only checkpoint location |
| Accessible `/projects` allocation, another approved durable store, or the user's workstation | Final adapters, manifests, labels and benchmark reports |

Scratch is not backed up. `/projects` access is not automatic. Check the user's actual quota and available durable storage rather than assuming a research-group path. [Data storage](https://researchcomputing.princeton.edu/support/knowledge-base/data-storage), [quota inspection](https://researchcomputing.princeton.edu/support/knowledge-base/checkquota).

## 1. Inspect access and prepare directories

These are commands for the user to run later, not commands executed by this research task. Replace `YOUR_NETID` in connection/transfer examples. Off-campus SSH requires Princeton VPN. [Adroit connection instructions](https://researchcomputing.princeton.edu/systems/adroit).

```bash
ssh YOUR_NETID@adroit.princeton.edu
checkquota
snodes
gfree
qos
module avail anaconda3

export CHAMELEON_ROOT="/scratch/network/$USER/chameleon"
mkdir -p "$CHAMELEON_ROOT"/{models,data,runs,cache,env-records}
export HF_HOME="$CHAMELEON_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$CHAMELEON_ROOT/cache/datasets"
export PIP_CACHE_DIR="$CHAMELEON_ROOT/cache/pip"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$PIP_CACHE_DIR"
```

The current public Python documentation uses `anaconda3/2026.7`. Verify its presence with `module avail`; do not assume a date copied from an older job script is installed. Princeton supports named environments and prefix environments, including prefixes on Adroit's network scratch. [Python environment guide](https://researchcomputing.princeton.edu/support/knowledge-base/python), [module guide](https://researchcomputing.princeton.edu/support/knowledge-base/modules).

## 2. Build the environment before scheduling

For GPU package setup, Princeton's PyTorch instructions explicitly include `adroit-vis`. The following is an adaptation for the project's proposed Python 3.11 environment, not a tested lockfile:

```bash
ssh YOUR_NETID@adroit-vis.princeton.edu
module purge
module load anaconda3/2026.7
export CHAMELEON_ROOT="/scratch/network/$USER/chameleon"
export PIP_CACHE_DIR="$CHAMELEON_ROOT/cache/pip"
mkdir -p "$PIP_CACHE_DIR"
conda create --name chameleon-vlm python=3.11 pip -y
conda activate chameleon-vlm
```

Princeton's current PyTorch page shows this GPU wheel installation command:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132
```

It also documents a CUDA-12 Conda alternative. Select one consistent installation, then verify it against the actual allocated node and chosen model stack. The `cu132` example is a current documentation value, not proof that every existing environment/driver/model combination supports it. Do not combine arbitrary CUDA toolkits, PyTorch wheels, and compiled extensions. [Princeton PyTorch installation instructions](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch).

For an exploratory Transformers/PEFT/TRL implementation, install its model-compatible dependencies into that environment. An initial package list is:

```bash
python -m pip install transformers peft trl datasets accelerate pillow huggingface_hub
python -m pip check
python -m pip freeze > "$CHAMELEON_ROOT/env-records/pip-freeze.txt"
conda env export > "$CHAMELEON_ROOT/env-records/conda-environment.yml"
```

This unpinned list is for discovering a working stack. After the smoke test, preserve exact versions, wheel origin, Python, driver, GPU, model revision and processor version. Reuse that lock for the two-hour experiment. Start with PyTorch's supported attention implementation; a custom FlashAttention build is unnecessary for the first smoke test. If it is later needed, Princeton specifies `MAX_JOBS=4` for installation. [Princeton HF installation note](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face).

If home quota is insufficient, create the environment with `conda create --prefix "$CHAMELEON_ROOT/envs/chameleon-vlm" python=3.11 pip -y` and activate that exact prefix in the job. It can be rebuilt from its saved environment record; it is not backed up simply because it is a Conda environment.

## 3. Stage a pinned model and immutable input data

On a permitted internet-connected preparation node, set the chosen model and its immutable repository commit, then download once. This document intentionally does not select a model ID or invent a revision:

```bash
export CHAMELEON_ROOT="/scratch/network/$USER/chameleon"
export HF_HOME="$CHAMELEON_ROOT/cache/huggingface"
: "${CHAMELEON_MODEL_ID:?Set the chosen model repository ID}"
: "${CHAMELEON_MODEL_REVISION:?Set its immutable commit revision}"
hf download "$CHAMELEON_MODEL_ID" \
  --revision "$CHAMELEON_MODEL_REVISION" \
  --local-dir "$CHAMELEON_ROOT/models/base"
```

Authenticate interactively first if that model requires access. Do not put tokens in Slurm scripts or log them. The compute job must use the local directory and `local_files_only=True`; enable `HF_HUB_OFFLINE=1`. [Princeton staging guidance](https://researchcomputing.princeton.edu/support/knowledge-base/hugging-face), [Transformers offline loading](https://huggingface.co/docs/transformers/installation#offline-mode).

The prepared corpus should contain the rendered original, candidate renders, element JSON, resource hashes, human labels, and a frozen source/template-group split. Keep training and validation manifests separate. Neither dataset preparation nor model warm-up should silently download missing files inside a scheduled training run.

For small transfers, Princeton documents `scp`; for large collections use its Globus workflow. Run the following PowerShell example from the local repository root. It copies an already prepared bundle rather than the repository's virtual environment or model caches:

```powershell
$chameleonNetId = 'YOUR_NETID'
scp -r (Join-Path (Get-Location) 'output/vision-training-bundle') "${chameleonNetId}@adroit.princeton.edu:/scratch/network/${chameleonNetId}/chameleon/data/"
```

If the local SSH client has the documented Windows MAC negotiation issue, Princeton shows `-o MACs=hmac-sha2-512` as an additional `scp` option. Do not alter machine-wide SSH configuration for this transfer. [Princeton data transfer instructions](https://researchcomputing.princeton.edu/support/knowledge-base/data-transfer).

## 4. Portable single-GPU Slurm launcher

The new probe files are currently local working-tree files. A fresh remote clone
may not have them yet. Clone the reviewed project branch on an internet-connected
preparation node and make the destination directory:

```bash
git clone --branch codex/hackathon-ready https://github.com/rsingh135/chameleon.git "$CHAMELEON_ROOT/repo"
mkdir -p "$CHAMELEON_ROOT/repo/tools/slurm"
```

Then copy these small helpers from the local PowerShell terminal, at the repository root:

```powershell
$chameleonNetId = 'YOUR_NETID'
scp (Join-Path (Get-Location) 'tools/vision_gpu_probe.py') "${chameleonNetId}@adroit.princeton.edu:/scratch/network/${chameleonNetId}/chameleon/repo/tools/"
scp (Join-Path (Get-Location) 'tools/slurm/vision_gpu_probe.sbatch') "${chameleonNetId}@adroit.princeton.edu:/scratch/network/${chameleonNetId}/chameleon/repo/tools/slurm/"
```

Submit from `"$CHAMELEON_ROOT/repo"` after the transfer. The `.sbatch` file must
retain LF line endings. Once a Git revision includes these helpers, cloning or
updating to that revision replaces this temporary manual transfer.

The repository's first executable GPU check is `tools/vision_gpu_probe.py`, submitted with `tools/slurm/vision_gpu_probe.sbatch`. From the repository root on Adroit, after loading the selected Anaconda module and activating the intended environment:

```bash
mkdir -p output/vision_critic
export CHAMELEON_VISION_PYTHON="$(command -v python)"
sbatch tools/slurm/vision_gpu_probe.sbatch
# Or choose a published hardware constraint for the same probe:
sbatch --constraint=a40 tools/slurm/vision_gpu_probe.sbatch
# Alternative: --constraint=gpu80
```

Submit one of these alternatives, not all three. That launcher requests one GPU, four CPU cores, 32 GB host RAM and 15 minutes. It uses the caller's configured Python environment and does not set QOS. The probe records device information in JSON; its `--allow-cpu` and optional `--min-vram-gib` switches are useful for separate local checks. A GPU probe is not evidence of real-model training or memory fit. Do not use `--allow-cpu` to disguise a failed GPU allocation.

The larger launcher below is a proposed template for the later trainer, separate from that executable probe.

Save the following on Adroit as `adroit-onegpu.slurm`. It deliberately accepts a trainer command as arguments, so it does not pretend this repository already contains a VLM training entrypoint. The example resources, cache layout, profile thresholds and environment variables are our proposed launcher conventions. Basic job directives, submission and monitoring follow Princeton's Slurm guide. [Slurm guide](https://researchcomputing.princeton.edu/support/knowledge-base/slurm).

```bash
#!/bin/bash
#SBATCH --job-name=chameleon-vlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
module purge
module load anaconda3/2026.7
conda activate chameleon-vlm

export CHAMELEON_ROOT="/scratch/network/$USER/chameleon"
export CHAMELEON_RUN_ROOT="$CHAMELEON_ROOT/runs/$SLURM_JOB_ID"
export HF_HOME="$CHAMELEON_ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$CHAMELEON_ROOT/cache/datasets"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
mkdir -p "$CHAMELEON_RUN_ROOT"
cd "$SLURM_SUBMIT_DIR"

# CUDA_VISIBLE_DEVICES is assigned by Slurm; do not overwrite it.
nvidia-smi
CHAMELEON_GPU_PROFILE="$(python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable in this scheduled environment')
p = torch.cuda.get_device_properties(0)
gib = p.total_memory / (1024 ** 3)
print(f'Allocated device: {p.name}; {gib:.1f} GiB', file=sys.stderr)
if gib >= 70:
    print('80gb')
elif gib >= 40:
    print('48gb')
elif gib >= 18:
    print('20gb')
else:
    raise SystemExit('Allocated device is smaller than the planned profiles')
PY
)"
export CHAMELEON_GPU_PROFILE
printf 'Selected configuration class: %s\n' "$CHAMELEON_GPU_PROFILE"

if [ "$#" -eq 0 ]; then
  printf '%s\n' 'Pass the real trainer command and its arguments to sbatch.' >&2
  exit 2
fi
srun "$@"
```

`--mem=64G` requests host RAM, not GPU memory. The trainer must read `CHAMELEON_GPU_PROFILE` or receive an explicit matching configuration; exporting the variable alone does not make a model fit. The preflight measures the device visible to PyTorch, which matters for MIG. Use this same measured profile when comparing results with Brev.

Before submission, create `logs` **in the submission directory**. Slurm opens the output files before the script runs:

```bash
mkdir -p logs
# A runnable CUDA/environment check; this is not a model-training test.
sbatch --time=00:15:00 --constraint=a40 adroit-onegpu.slurm \
  python -c 'import torch; x=torch.randn(256,256,device="cuda"); (x@x).sum().item(); print(torch.cuda.get_device_name(0))'
```

For an 80 GB allocation replace `--constraint=a40` with `--constraint=gpu80`. Omitting the constraint accepts any available compatible GPU and therefore requires the 20 GB profile. `--constraint=a100` requests the A100 family, not a verified particular MIG slice. These commands do not set QOS, partition, account, or a GPU memory flag.

## 5. Real LoRA smoke test, then the two-hour run

The actual trainer is pending. Once it exists, place its complete smoke command in a Bash array and pass that array to the launcher. This is the submission pattern, with no guessed trainer filename or unsupported model flags:

```bash
# Populate this array with the completed trainer's command and real arguments.
# Example shape: python -m <implemented module> --config <existing config>
: "${CHAMELEON_TRAIN_SCRIPT:?Set the absolute path of the completed Python trainer}"
: "${CHAMELEON_SMOKE_CONFIG:?Set the existing 5-to-10-step smoke configuration}"
: "${CHAMELEON_TRAIN_CONFIG:?Set the existing short-training configuration}"
test -f "$CHAMELEON_TRAIN_SCRIPT"
test -f "$CHAMELEON_SMOKE_CONFIG"
test -f "$CHAMELEON_TRAIN_CONFIG"

# --config is the proposed trainer interface, to implement with the framework.
CHAMELEON_SMOKE_COMMAND=(python "$CHAMELEON_TRAIN_SCRIPT" --config "$CHAMELEON_SMOKE_CONFIG")
sbatch --time=00:15:00 --constraint=a40 adroit-onegpu.slurm "${CHAMELEON_SMOKE_COMMAND[@]}"

# Submit this separately after inspecting the completed smoke test.
CHAMELEON_TRAIN_COMMAND=(python "$CHAMELEON_TRAIN_SCRIPT" --config "$CHAMELEON_TRAIN_CONFIG")
sbatch --time=02:00:00 --constraint=a40 adroit-onegpu.slurm "${CHAMELEON_TRAIN_COMMAND[@]}"
```

The smoke configuration should load the real local checkpoint and processor, freeze the intended base/vision parameters, run 5–10 actual multimodal optimizer steps on training examples, save an adapter to `CHAMELEON_RUN_ROOT`, reload it, and evaluate one validation batch. Record allocated/reserved/peak device memory, finite loss/gradients, trainable parameter count and seconds per step. Check that validation and training source groups do not overlap. A successful import or CUDA multiplication is insufficient evidence that LoRA training works.

Suggested starting profiles below are deliberately conservative experimental settings, not tested memory claims:

| Allocated memory class | Initial configuration |
| --- | --- |
| 20 GB MIG | Small chosen VLM, microbatch 1, low initial image-token budget, frozen vision tower, gradient checkpointing, low-rank adapter; consider 4-bit base weights only if the selected backend is verified |
| 48 GB A40 | Same model and dataset, microbatch 1 initially, frozen vision tower and gradient checkpointing; measure before increasing pixels or batch size |
| 80 GB A100 | Start with the same 48 GB settings for a comparable baseline; spend additional memory on image detail or batches only after measuring |

The model-specific processor must bound image tokens explicitly. Do not truncate through multimodal image tokens merely to reach a text sequence limit. Keep source/candidate images at faithful aspect ratios, and retain crops needed to judge small text. Use gradient accumulation to maintain a comparable effective batch size across devices; it does not remove the per-example activation-memory requirement. The SFT implementation must be tested against its selected processor and adapter target modules. [TRL's multimodal SFT guidance](https://huggingface.co/docs/trl/en/sft_trainer).

For the two-hour allocation, derive the step budget from smoke timing, leave time for validation and saving, and checkpoint periodically. Do not promise a particular epoch count from GPU memory alone. Resume from the last valid checkpoint if the job reaches its wall limit; an unhandled Slurm termination is not a completed experiment.

## 6. Inspect results and preserve the useful artifacts

```bash
squeue --me
squeue --me --start
# Replace 123456 with the returned job ID.
jobstats 123456
```

`jobstats` reports the cluster's memory/CPU/GPU observations; combine it with trainer-side CUDA memory and step timing. Keep scheduler wait time separate from execution time when deciding whether to use Adroit or cloud credits. [Princeton Slurm monitoring commands](https://researchcomputing.princeton.edu/support/knowledge-base/slurm), [Jobstats](https://researchcomputing.princeton.edu/support/knowledge-base/jobstats).

Copy the final adapter, processor settings, environment lock, split/label hashes, model revision, GPU/runtime metadata, and evaluation report to durable storage available to the user. A small adapter bundle is a better transfer artifact than duplicating the full base model for each run. Model availability, license/weight-sharing requirements and the hackathon inference hardware still need to be resolved with the chosen model and official brief; access to a training GPU does not establish submission compatibility.

## Remaining uncertainties to settle on the actual account

- Live GPU features, queue state, quota, environment modules and any account-specific restrictions have not been inspected remotely.
- The specific Adroit hardware page and general GPU table differ about the older 40 GB A100 configuration; inspect the live allocation.
- No published MIG-20-only constraint was confirmed. Do not invent `gpu20` or a partition name.
- A working PyTorch/model/processor/PEFT/TRL combination, LoRA target modules and 20/48/80 GB fit have not been benchmarked.
- The training entrypoint and configuration files shown as required inputs above do not yet exist as part of this document.
- The first two-hour run is a feasibility/learning experiment. It does not establish model generalization from the small poster catalogue or justify replacing the algorithmic baseline.
