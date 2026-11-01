#!/usr/bin/env bash
# Fetch the official tokenizer checkpoints and the three CTC recognizers used in the paper.
#   DAC 16 kHz          github.com/descriptinc/descript-audio-codec (release 0.0.5)
#   Mimi                huggingface.co/kyutai/moshiko-pytorch-bf16
#   SpeechTokenizer     huggingface.co/fnlp/SpeechTokenizer (speechtokenizer_hubert_avg)
#   ASR (HF cache)      wav2vec2-base-960h (ranker + PGD surrogate),
#                       hubert-large-ls960-ft (held-out scorer), wav2vec2-large-960h-lv60-self (2nd held-out)
# Set HF_TOKEN for higher Hugging Face rate limits (optional).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"

mkdir -p checkpoints/dac checkpoints/mimi checkpoints/speechtokenizer

if [[ ! -s checkpoints/dac/weights_16khz.pth ]]; then
  echo "==> DAC 16 kHz"
  curl -L --fail --retry 5 -o checkpoints/dac/weights_16khz.pth.partial \
    "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth"
  mv checkpoints/dac/weights_16khz.pth.partial checkpoints/dac/weights_16khz.pth
fi

"$PY" - <<'PY'
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

def fetch(repo, filename, dest):
    dest = Path(dest)
    if dest.is_file() and dest.stat().st_size > 0:
        print("cached", dest); return
    shutil.copy(hf_hub_download(repo, filename), dest)
    print("saved", dest)

print("==> Mimi")
fetch("kyutai/moshiko-pytorch-bf16", "tokenizer-e351c8d8-checkpoint125.safetensors",
      "checkpoints/mimi/tokenizer-e351c8d8-checkpoint125.safetensors")
print("==> SpeechTokenizer")
for name in ("config.json", "SpeechTokenizer.pt"):
    fetch("fnlp/SpeechTokenizer", f"speechtokenizer_hubert_avg/{name}", f"checkpoints/speechtokenizer/{name}")
print("==> CTC recognizers")
for repo in ("facebook/wav2vec2-base-960h", "facebook/hubert-large-ls960-ft", "facebook/wav2vec2-large-960h-lv60-self"):
    snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"])
    print("ok", repo)
PY
echo "All checkpoints ready."
