#!/usr/bin/env bash
# Run a protocol grid (ctm.runner_v2), optionally sharded over several GPUs.
#
#   bash scripts/run_grid.sh configs/main_hier.yaml             # all visible GPUs
#   GPUS="0 1" bash scripts/run_grid.sh configs/main_hier.yaml  # pick GPUs
#   bash scripts/run_grid.sh configs/hier_smoke.yaml --max-utterances 1
#
# Results are appended to <output_dir>/utterance_results.jsonl (resume-safe: re-running skips
# finished (utterance, codec, condition) keys). Logs go to <output_dir>/shard_<i>.log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
CONFIG="${1:?usage: run_grid.sh CONFIG [extra runner args]}"; shift || true
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${GPUS:-}" ]]; then
  n=$("$PY" -c 'import torch; print(torch.cuda.device_count())')
  GPUS="$(seq -s ' ' 0 $(( n > 0 ? n - 1 : 0 )))"
fi
read -r -a gpu_list <<< "$GPUS"
N=${#gpu_list[@]}
OUT=$("$PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["output_dir"])' "$CONFIG")
mkdir -p "$OUT"
echo "config=$CONFIG  output=$OUT  shards=$N  gpus=[$GPUS]"

pids=()
for i in "${!gpu_list[@]}"; do
  CUDA_VISIBLE_DEVICES="${gpu_list[$i]}" "$PY" -m ctm.runner_v2 --config "$CONFIG" \
    --shard-id "$i" --num-shards "$N" "$@" > "$OUT/shard_$i.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "finished (exit $status); rows: $(wc -l < "$OUT/utterance_results.jsonl" 2>/dev/null || echo 0)"
exit $status
