#!/usr/bin/env bash
# Run after copying the reviewed tools and requirements into the persistent root.
# This prepares an environment; it does not launch training or stop GPU billing.
set -euo pipefail
umask 077

CHAMELEON_ROOT="${CHAMELEON_ROOT:-$HOME/workspace/chameleon}"
INSTALL_TIMEOUT="${CHAMELEON_INSTALL_TIMEOUT:-20m}"
cd "$CHAMELEON_ROOT"
test -f requirements-vision.txt
test -f tools/vision_gpu_probe.py
test -f tools/vision_lora_smoke.py
command -v timeout >/dev/null
command -v nvidia-smi >/dev/null
mkdir -p local_models/huggingface training_runs output/vision_critic
export HF_HOME="$CHAMELEON_ROOT/local_models/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export PIP_DISABLE_PIP_VERSION_CHECK=1

python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >=3.10 is required; select a supported VM image.")
PY

require_venv_support() {
  python3 - <<'PY'
import sys
try:
    import ensurepip
    import venv
except ImportError as error:
    package = f"python{sys.version_info.major}.{sys.version_info.minor}-venv"
    print(f"Missing Python venv/ensurepip support: {error}", file=sys.stderr)
    print(f"On Ubuntu, install the matching package (sudo apt-get install {package}), "
          "then rerun this bootstrap. This script did not install system packages.", file=sys.stderr)
    raise SystemExit(2)
PY
}

# Reuse a working vendor CUDA stack when available, without modifying it.
# HF packages install into this venv. Record whether system packages are visible.
if test ! -x .venv-vision/bin/python; then
  require_venv_support
  if python3 - <<'PY'
import torch
import torchvision
assert tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6)
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.ones(1, device="cuda").item() == 1
print("Reusing image CUDA stack:", torch.__version__, torchvision.__version__)
PY
  then
    python3 -m venv --system-site-packages .venv-vision
    printf '%s\n' 'vendor torch/torchvision inherited through system-site-packages' \
      > .venv-vision/chameleon-torch-origin.txt
  else
    python3 -m venv .venv-vision
  fi
fi

# A failed `python -m venv` can leave a Python executable with no pip. Verify
# its prefix before repairing pip so no installation can target system Python.
if ! .venv-vision/bin/python -c \
  'import pathlib,sys; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(".venv-vision").resolve() and sys.prefix != sys.base_prefix'; then
  printf '%s\n' 'Existing .venv-vision is not a usable project virtual environment.' \
    'Move that environment aside after inspection, then rerun; no files were deleted.' >&2
  exit 2
fi
if ! .venv-vision/bin/python -m pip --version > /dev/null 2>&1; then
  require_venv_support
  printf '%s\n' 'Repairing missing pip inside the existing project virtual environment.'
  .venv-vision/bin/python -m ensurepip --upgrade
  .venv-vision/bin/python -m pip --version > /dev/null
fi

# Modern Torch may compile runtime kernels on the first real model forward.
# Catch missing Python.h/compiler before downloading the CUDA/model packages.
.venv-vision/bin/python tools/vision_gpu_probe.py --build-only \
  --out output/vision_critic/build-prerequisites.json

# Retrying after an interrupted installation must repair this project's venv.
# Executable existence alone is not evidence that the CUDA stack was installed.
if ! .venv-vision/bin/python - <<'PY'
import torch
import torchvision
assert tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6)
assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.ones(1, device="cuda").item() == 1
PY
then
  # Official matched CUDA 12.6 wheels; avoids silently selecting newer CUDA 13.
  # This modifies the project venv, never the vendor Python installation.
  # https://pytorch.org/get-started/previous-versions/
  timeout --signal=TERM --kill-after=30s "$INSTALL_TIMEOUT" \
    .venv-vision/bin/python -m pip install \
    torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126
  printf '%s\n' 'torch2.14.1 torchvision0.29.1 from official cu126 index' \
    > .venv-vision/chameleon-torch-origin.txt
fi

timeout --signal=TERM --kill-after=30s "$INSTALL_TIMEOUT" \
  .venv-vision/bin/python -m pip install -r requirements-vision.txt
.venv-vision/bin/python -m pip check
.venv-vision/bin/python -m pip freeze --all \
  > output/vision_critic/environment.freeze.txt
nvidia-smi > output/vision_critic/nvidia-smi.txt
.venv-vision/bin/python tools/vision_gpu_probe.py \
  --min-vram-gib 16 --out output/vision_critic/gpu-probe.json

# A sourceable, project-local convenience file. No shell profile is changed.
cat > vision-env.sh <<'ENV'
CHAMELEON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CHAMELEON_ROOT
export HF_HOME="$CHAMELEON_ROOT/local_models/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
source "$CHAMELEON_ROOT/.venv-vision/bin/activate"
ENV
printf '%s\n' \
  'Environment prepared. Model forward/backward/save/reload still needs validation.' \
  'Run the smoke through the local session controller and a bounded remote timeout.' \
  'A remote process timeout does not stop Brev billing.'
