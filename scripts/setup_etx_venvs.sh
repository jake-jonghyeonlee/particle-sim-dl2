#!/usr/bin/env bash
# Local reproduction of the ETX runtime (Python 3.9.14, numpy 1.23.5, Pillow 10.1.0, torch 2.1.1) with uv.
# Creates two venvs OUTSIDE the repository (default ~/.cache/psdl2-etx/):
#   venv39-notorch   python 3.9 + numpy==1.23.5 + pillow==10.1.0           (needed by tests/test_etx_runtime.py)
#   venv39-torch     the same + torch==2.1.1 (CPU build; only `import torch` is checked)
# Uses PyPI once. Existing venvs are kept; delete one by hand to rebuild it.
#
#   bash scripts/setup_etx_venvs.sh
#
# Environment overrides:
#   PSDL2_ETX_HOME   target folder (default ~/.cache/psdl2-etx)
#   PSDL2_ETX_PYTHON uv python request (default 3.9.14; falls back to 3.9 if that exact patch is unavailable)
set -euo pipefail

HOME_DIR="${PSDL2_ETX_HOME:-$HOME/.cache/psdl2-etx}"
WANT="${PSDL2_ETX_PYTHON:-3.9.14}"
command -v uv >/dev/null 2>&1 || { echo "uv is required (https://docs.astral.sh/uv/)" >&2; exit 1; }
mkdir -p "$HOME_DIR"

if uv python find "$WANT" >/dev/null 2>&1 || uv python install "$WANT" >/dev/null 2>&1; then
  REQUEST="$WANT"
else
  echo "WARNING: CPython $WANT is unavailable; falling back to the closest 3.9.x" >&2
  uv python install 3.9 >/dev/null
  REQUEST="3.9"
fi

make_venv() {
  local name="$1"; shift
  local venv="$HOME_DIR/$name"
  if [ -e "$venv" ]; then
    echo "kept existing venv: $venv"
  else
    uv venv --python "$REQUEST" "$venv" >/dev/null
    uv pip install --python "$venv/bin/python" "$@"
  fi
}

make_venv venv39-notorch numpy==1.23.5 pillow==10.1.0
make_venv venv39-torch numpy==1.23.5 pillow==10.1.0 torch==2.1.1

report() {
  "$HOME_DIR/$1/bin/python" - "$1" <<'PY'
import sys
import numpy
import PIL
name = sys.argv[1]
line = f"{name}: python {sys.version.split()[0]} | numpy {numpy.__version__} | Pillow {PIL.__version__}"
try:
    import torch
    line += f" | torch {torch.__version__}"
except ImportError:
    line += " | torch (not installed)"
print(line)
PY
}

report venv39-notorch
report venv39-torch
if [ "$("$HOME_DIR/venv39-notorch/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')" != "3.9.14" ]; then
  echo "NOTE: the interpreter is not exactly 3.9.14 (the ETX patch release); record the printed version." >&2
fi
echo "Run the ETX tests with: PSDL2_ETX_PY=$HOME_DIR/venv39-notorch/bin/python"
