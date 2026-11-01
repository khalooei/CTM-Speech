"""Generalization-check table: oracle vs random AURC/F-CTM90 on
the runs outside the main grid (second corpus, real noise, extra codec families)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ctm.analysis_v2 import load_rows, per_row_table

RUNS = [
    ("FLEURS (2nd corpus)", "results/fleurs_hier.jsonl.gz"),
    ("Real noise (MUSAN/DEMAND)", "results/realnoise_hier.jsonl.gz"),
    ("EnCodec", "results/extra_encodec_hier.jsonl.gz"),
    ("Qwen3-TTS tokenizer", "results/extra_qwen3tts_hier.jsonl.gz"),
    ("WavTokenizer", "results/extra_wavtokenizer_hier.jsonl.gz"),
    ("StableToken", "results/stabletoken_hier.jsonl.gz"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    root = Path(a.root)
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule",
             "Check & $n_{\\text{gated}}$ & AURC$_{\\text{orc}}$ & AURC$_{\\text{rnd}}$ & $\\FCTM{90}^{\\text{orc}}$ & $\\FCTM{90}^{\\text{rnd}}$ \\\\", "\\midrule"]
    for label, rel in RUNS:
        path = root / rel
        if not path.exists():
            continue
        rows = load_rows([str(path)])
        df = per_row_table(rows)
        g = df[(df["ctm_status"] == "ok") & (df.get("eval_resolved", True) == True)]
        if g.empty:
            lines.append(f"{label} & 0 & -- & -- & -- & -- \\\\")
            continue
        orc = np.asarray(g.get("intervention|eval|aurc"), float)
        rnd = np.asarray(g.get("random|eval|aurc"), float)
        c90o = np.asarray(g.get("intervention|eval|ctm90"), float)
        c90r = np.asarray(g.get("random|eval|ctm90"), float)
        lines.append(
            f"{label} & {len(g)} & {np.nanmean(orc):.2f} & {np.nanmean(rnd):.2f} & "
            f"{np.nanmedian(c90o):.2f} & {np.nanmedian(c90r):.2f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(a.out).write_text("\n".join(lines))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
