"""Compare a fresh run against the released per-utterance results on their shared keys.

    python scripts/compare_runs.py runs/hier_smoke/utterance_results.jsonl results/main_hier.jsonl.gz

For every (utterance, codec, condition) present in both files it counts exact matches of the
discrete quantities (changed cells, transcripts, gate) and reports absolute differences of the
numeric ones (input SNR, TCR, losses, AURC). On the same hardware and library versions a re-run is
reproducible; across GPUs, CTC-PGD inputs (gradient arithmetic), Opus inputs (ffmpeg/libopus build)
and tokenizer encoders differ at the floating-point level, so compare aggregates, not single rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ctm.analysis_v2 import curve_metrics  # noqa: E402
from ctm.jsonl_store import load_jsonl_last_wins  # noqa: E402

EXACT = ["n_changed", "n_frames_changed", "hyp_clean", "hyp_corrupt", "gate_passed"]
CLOSE = ["tcr", "input_snr_db", "wer_corrupt", "l_clean_eval", "l_corrupt_eval", "gap_eval", "gap_rank"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("new")
    p.add_argument("released")
    a = p.parse_args()
    new, ref = load_jsonl_last_wins(a.new), load_jsonl_last_wins(a.released)
    keys = sorted(k for k in new if k in ref and new[k].get("status") == "ok")
    if not keys:
        sys.exit("no shared (utterance, codec, condition) keys")
    print(f"{len(keys)} shared rows\n")
    exact_ok = {f: 0 for f in EXACT}
    diffs = {f: [] for f in CLOSE + ["oracle_aurc", "random_aurc"]}
    for k in keys:
        n, r = new[k], ref[k]
        for f in EXACT:
            exact_ok[f] += n.get(f) == r.get(f)
        for f in CLOSE:
            if n.get(f) is not None and r.get(f) is not None:
                diffs[f].append(abs(float(n[f]) - float(r[f])))
        for rk, name in (("intervention", "oracle_aurc"), ("random", "random_aurc")):
            mn, mr = curve_metrics(n, rk, "eval"), curve_metrics(r, rk, "eval")
            if mn and mr:
                diffs[name].append(abs(mn["aurc"] - mr["aurc"]))
    print("exact matches:")
    for f, c in exact_ok.items():
        print(f"  {f:<18} {c}/{len(keys)}")
    print("absolute differences (median / max):")
    for f, d in diffs.items():
        if d:
            print(f"  {f:<18} {np.median(d):.4g} / {np.max(d):.4g}  (n={len(d)})")


if __name__ == "__main__":
    main()
