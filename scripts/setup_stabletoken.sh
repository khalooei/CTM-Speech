#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"


mkdir -p third_party checkpoints
if [[ ! -d third_party/StableToken/.git ]]; then
  git clone https://github.com/Tencent/StableToken.git third_party/StableToken
fi

python -m pip install -r requirements-stabletoken.txt || true
if [[ -f third_party/StableToken/requirements.txt ]]; then
  python -m pip install -r third_party/StableToken/requirements.txt || true
fi

# HuggingFace checkpoint if available
python - <<'PY'
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
    out = Path("checkpoints/StableToken")
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="tencent/StableToken", local_dir=str(out))
    print("StableToken checkpoint downloaded")
except Exception as e:
    print("StableToken checkpoint download skipped:", e)
PY

echo "StableToken setup done. Prefer a separate conda env if dependency conflicts arise."
