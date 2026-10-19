#!/usr/bin/env bash
# Closed-network GPU venv for psdl2 via the pip mirror (Git Bash / sh on Windows; no PowerShell).
# Creates a NEW venv, installs pinned packages with pip, then checks CUDA torch and one psdl2 forward.
# Same torch build as DL1's scripts/setup_gpu_venv.sh (GTX 1660 Super 6 GB: CUDA 11.8 build, fp32).
# Existing venvs are never touched.
#
#   bash scripts/setup_gpu_venv.sh [VENV_DIR]
#
# Environment overrides:
#   PYTHON            python to build the venv with (default: python)
#   TORCH_SPEC        torch version (default 2.7.1+cu118, same as DL1). psdl2 needs torch>=2.7.1.
#   TORCH_INDEX_URL   index/mirror that hosts the +cu118 wheels (PyPI's Windows torch is CPU-only).
#                     Default: use the pip config as-is. If that mirror lacks +cu118, set this.
#
# Mirror address: pip reads PIP_INDEX_URL / PIP_TRUSTED_HOST / PIP_CERT from the environment, e.g.
#   export PIP_INDEX_URL=http://<mirror>/simple PIP_TRUSTED_HOST=<mirror-host>
set -euo pipefail

PYTHON="${PYTHON:-python}"
VENV="${1:-.venv-gpu}"
TORCH_SPEC="${TORCH_SPEC:-2.7.1+cu118}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Git Bash gives /c/... paths that a Windows python.exe cannot resolve; hand it a C:/... path.
if command -v cygpath >/dev/null 2>&1; then ROOT="$(cygpath -m "$ROOT")"; fi

if [ -e "$VENV" ]; then
  echo "Venv already exists and was preserved: $VENV (choose a new path)" >&2
  exit 1
fi

"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "CPython 3.12 required, got %s" % sys.version' >&2

"$PYTHON" -m venv "$VENV"
if [ -x "$VENV/Scripts/python.exe" ]; then VPY="$VENV/Scripts/python.exe"; else VPY="$VENV/bin/python"; fi
pip() { "$VPY" -I -m pip --disable-pip-version-check "$@"; }

index_args=()
if [ -n "${TORCH_INDEX_URL:-}" ]; then index_args=(--extra-index-url "$TORCH_INDEX_URL"); fi

pip install "numpy==2.5.3" "Pillow==12.3.0" "pytest==9.1.1" "hatchling==1.27.0" "editables"
pip install "${index_args[@]}" "torch==$TORCH_SPEC"
# Application from this checkout, editable: a later git pull updates the installed code. Deps are pinned above.
pip install --no-deps --no-build-isolation -e "$ROOT"
pip check

"$VPY" -I - <<'PY'
import sys
import torch, numpy, PIL, psdl2
from psdl2.model import Encoder, ModelConfig
print("python", sys.version.split()[0], "| torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| numpy", numpy.__version__, "| Pillow", PIL.__version__)
if not torch.version.cuda:
    sys.exit("CPU-only torch was installed. Set TORCH_INDEX_URL to a mirror hosting the +cu118 wheels and use a new venv.")
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("CUDA build installed but no usable GPU; check the NVIDIA driver (nvidia-smi). Use --device cpu meanwhile.")
props = torch.cuda.get_device_properties(0)
print("device:", props.name, "| memory %.1f GiB" % (props.total_memory / 2**30))
model = Encoder(ModelConfig(roles=("a",), layers=("M1",), width=8, feature_dim=16, projection_dim=8)).cuda()
out = model.cnn(torch.zeros(2, 5, 64, 64, device="cuda"))
print("psdl2 encoder CNN forward on cuda", tuple(out.shape))
PY
echo "Done. Run with --device cuda, e.g.  \"$VPY\" -I -m psdl2.train ... --device cuda"
