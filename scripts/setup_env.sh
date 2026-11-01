#!/usr/bin/env bash
# Create a Python environment with a CUDA build of PyTorch and every dependency needed
# to run the grids (DAC, Mimi, SpeechTokenizer) and to reproduce the reported results.
#
#   bash scripts/setup_env.sh                 # .venv, CUDA 12.8 wheels
#   TORCH_INDEX=cu130 bash scripts/setup_env.sh   # e.g. Blackwell / CUDA 13 drivers
#   VENV=/path/to/env bash scripts/setup_env.sh
#
# Uses `uv` when available (much faster), otherwise python -m venv + pip.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-.venv}"
TORCH_INDEX="${TORCH_INDEX:-cu128}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if command -v uv >/dev/null 2>&1; then
  [[ -d "$VENV" ]] || uv venv --python "$PYTHON_VERSION" "$VENV"
  PIP=(uv pip install --python "$VENV/bin/python")
else
  [[ -d "$VENV" ]] || python3 -m venv "$VENV"
  "$VENV/bin/python" -m pip install -U pip
  PIP=("$VENV/bin/python" -m pip install)
fi

echo "==> PyTorch (${TORCH_INDEX})"
"${PIP[@]}" torch torchaudio --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"

echo "==> Scientific stack"
"${PIP[@]}" -r requirements.txt

echo "==> Codec packages (installed without dependencies so they cannot downgrade torch / transformers)"
"${PIP[@]}" --no-deps descript-audio-codec descript-audiotools moshi speechtokenizer
"${PIP[@]}" -e . --no-deps

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "==> ffmpeg not on PATH: linking the imageio-ffmpeg static binary into $VENV/bin (needed for Opus)"
  "${PIP[@]}" imageio-ffmpeg
  ln -sf "$("$VENV/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" "$VENV/bin/ffmpeg"
fi

"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
echo "Done. Activate with: source $VENV/bin/activate"
