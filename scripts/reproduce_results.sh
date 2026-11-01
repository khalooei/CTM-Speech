#!/usr/bin/env bash
# Recompute every reported result from the released per-utterance results (results/*.jsonl.gz):
# per-row metrics, summaries and paired tests, result tables, figures, theory checks, the
# generalization table and the utterance-level bootstrap. CPU only, a few minutes.
#
#   bash scripts/reproduce_results.sh              # uses results/ (released)
#   RESULTS=runs bash scripts/reproduce_results.sh # after re-running the grid yourself
#
# RESULTS=runs expects runs/<name>/utterance_results.jsonl as written by ctm.runner_v2.
# Everything is written to outputs/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" MPLBACKEND=Agg
RESULTS="${RESULTS:-results}"
OUT=outputs

src() {  # path of the per-utterance results of one run
  if [[ "$RESULTS" == results ]]; then echo "results/$1.jsonl.gz"; else echo "$RESULTS/$1/utterance_results.jsonl"; fi
}
MAIN="$(src main_hier)"
mkdir -p "$OUT/figures" "$OUT/tables"

echo "==> Per-row analysis"
"$PY" -m ctm.analysis_v2 --inputs "$MAIN" --out-dir "$OUT/analysis/main_hier"
for r in other_hier greedy_hier fleurs_hier; do
  if [[ -f "$(src $r)" ]]; then "$PY" -m ctm.analysis_v2 --inputs "$(src $r)" --out-dir "$OUT/analysis/$r"; fi
done

echo "==> Tables, figures, theory checks"
"$PY" scripts/make_tables_v2.py --inputs "$MAIN" --out-dir "$OUT/tables"
"$PY" -m ctm.plots_v2 --inputs "$MAIN" --out-dir "$OUT/figures"
"$PY" scripts/theory_validation.py --inputs "$MAIN" --out-dir "$OUT/figures" > "$OUT/figures/theory_summary.log" 2>&1
"$PY" scripts/make_generalization_table.py --out "$OUT/tables/tab_generalization.tex"
"$PY" scripts/cluster_bootstrap.py --per-row "$OUT/analysis/main_hier/per_row.csv" --out "$OUT/tables/tab_cluster.tex"

echo "Done: $OUT/ (tables/headline.json holds the headline numbers per tokenizer)"
