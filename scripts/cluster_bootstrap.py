"""Row-level vs utterance-level (cluster) bootstrap for the main paired AURC contrasts.

Rows of the main grid share utterances (16 conditions per utterance), so a row bootstrap can
understate uncertainty. This script resamples utterances with replacement instead and writes a
LaTeX table:

    python scripts/cluster_bootstrap.py --per-row outputs/analysis/main_hier/per_row.csv \
        --out outputs/tables/tab_cluster.tex
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CODECS = (("dac", "DAC"), ("mimi", "Mimi"), ("speechtokenizer", "SpeechTok."))
CONTRASTS = (
    ("intervention", "random", "Oracle $-$ random"),
    ("intervention", "rvq_early", "Oracle $-$ RVQ-early"),
    ("displacement", "random", "Displacement $-$ random"),
)


def ci(stats: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--per-row", default="outputs/analysis/main_hier/per_row.csv")
    p.add_argument("--out", default="outputs/tables/tab_cluster.tex")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    df = pd.read_csv(a.per_row)
    ok = df[(df["ctm_status"] == "ok") & (df["eval_resolved"] == True)]  # noqa: E712
    rng = np.random.default_rng(a.seed)
    lines = [r"\begin{tabular}{llrcc}", r"\toprule",
             r"Codec & Contrast (held-out AURC) & Mean & Row CI & Utterance CI \\", r"\midrule"]
    for ci_idx, (codec, label) in enumerate(CODECS):
        g = ok[ok["codec"] == codec]
        for j, (x, y, name) in enumerate(CONTRASTS):
            d = (pd.to_numeric(g[f"{x}|eval|aurc"], errors="coerce") - pd.to_numeric(g[f"{y}|eval|aurc"], errors="coerce"))
            sub = pd.DataFrame({"u": g["utterance_id"].to_numpy(), "d": d.to_numpy()}).dropna()
            v = sub["d"].to_numpy()
            rows = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.n_boot)])
            groups = [grp["d"].to_numpy() for _, grp in sub.groupby("u")]
            sums = np.array([x.sum() for x in groups]); cnts = np.array([x.size for x in groups])
            utts = []
            for _ in range(a.n_boot):
                idx = rng.integers(0, len(groups), len(groups))
                utts.append(sums[idx].sum() / cnts[idx].sum())
            (rl, rh), (ul, uh) = ci(rows), ci(np.asarray(utts))
            lab = f"{label} ($n={len(v)}$, {len(groups)} utt.)" if j == 0 else ""
            lines.append(f"{lab} & {name} & ${v.mean():+.3f}$ & $[{rl:+.3f},{rh:+.3f}]$ & $[{ul:+.3f},{uh:+.3f}]$ \\\\")
        if ci_idx < len(CODECS) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
