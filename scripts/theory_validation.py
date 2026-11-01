"""Theory diagnostics on real rows + Monte-Carlo selection-bias figure.

Outputs (out-dir):
  fig_theory_selection.pdf/png : F-CTM90 of pure noise under the old (48-subset, in-sample), in-sample and
                                 held-out protocols vs gap/sigma
  theory_summary.json          : redundancy index per codec, inflation-vs-prediction correlation, Gini checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ctm.analysis_v2 import load_rows, per_row_table, scorer_resolution, boot_ci
from ctm.hier import recovery_from_losses
from ctm.theory import gini, simulate_selection_bias


def selection_figure(out: Path):
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.55), sharey=True)
    gos = np.array([0.5, 1, 2, 4, 8, 16, 32, 128, 512])
    for ax, n_true, title in ((axes[0], 0, "no critical cells (null)"), (axes[1], 150, "10% of cells critical")):
        med = {k: [] for k in ("subset48_insample", "insample", "heldout")}
        for g in gos:
            r = simulate_selection_bias(1500, g, n_true=n_true, n_rep=100)
            for k in med:
                med[k].append(np.nanmedian(r[k]))
        ax.plot(gos, med["subset48_insample"], "s-", ms=2.5, color="#7A7F7C", label="48-cell subset, in-sample (old)")
        ax.plot(gos, med["insample"], "o-", ms=2.5, color="#B3412E", label="all of $\\mathcal{D}$, in-sample")
        ax.plot(gos, med["heldout"], "^-", ms=2.5, color="#2F6B8E", label="all of $\\mathcal{D}$, held-out")
        if n_true:
            ax.axhline(0.1 * 0.9, color="#999", lw=0.6, ls=":")
        ax.set_xscale("log"); ax.set_xlabel("gap / $\\sigma$"); ax.set_title(title, fontsize=7.5)
    axes[0].set_ylabel("median F-CTM$_{90}$"); axes[0].set_ylim(-0.02, 1.02)
    # A legend placed inside axes[1] sits on top of the held-out curve as it rises
    # toward 1.0 (the panel has no corner that stays clear of the data at every gap
    # value). Use one shared legend above both panels, with room reserved for it,
    # instead of guessing at a clear spot inside the plot.
    h, l = axes[1].get_legend_handles_labels()
    fig.tight_layout(pad=0.3, rect=(0.0, 0.0, 1.0, 0.92))
    fig.legend(h, l, frameon=False, fontsize=5.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.93), columnspacing=1.0, handletextpad=0.4)
    fig.savefig(out / "fig_theory_selection.pdf", bbox_inches="tight"); fig.savefig(out / "fig_theory_selection.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def real_data_checks(rows, out: Path) -> dict:
    df = per_row_table(rows)
    summary = {}
    for codec in ("dac", "mimi", "speechtokenizer"):
        g = df[(df["codec"] == codec) & (df["ctm_status"] == "ok") & (df["eval_resolved"] == True)]
        if g.empty:
            continue
        d = {}
        # redundancy index R = 2 AURC_random - 1 (0 under additivity), averaged over both random seeds
        ar = pd.concat([g["random|eval|aurc"], g.get("random2|eval|aurc", pd.Series(dtype=float))]).to_numpy()
        m, lo, hi = boot_ci(ar)
        d["redundancy_index"] = {"mean": 2 * m - 1, "lo": 2 * lo - 1, "hi": 2 * hi - 1, "n": int(np.isfinite(ar).sum())}
        # concentration proxy: 2 AURC_oracle - 1 (= Gini of weights under additivity)
        m2, lo2, hi2 = boot_ci(g["intervention|eval|aurc"].to_numpy())
        d["oracle_gini_equiv"] = {"mean": 2 * m2 - 1, "lo": 2 * lo2 - 1, "hi": 2 * hi2 - 1}
        # in-sample inflation vs prediction sigma*sqrt(2 ln |F|)/gap (sigma from negative frame influences)
        infl, pred, gini_ref = [], [], []
        for r in rows:
            if r["codec"] != codec or r.get("ctm_status") != "ok" or not scorer_resolution(r, "eval")[1] or not scorer_resolution(r, "rank")[1]:
                continue
            fi = np.asarray(r["frame_influence_rank"], float)
            neg = fi[fi < 0]
            if neg.size < 5:
                continue
            sigma = 1.4826 * np.median(np.abs(neg))  # robust sd of the null half
            c = r["curves"]["intervention"]
            k = r.get("n_cell_decodes", 0)
            counts = np.asarray(c["counts"])
            j = int(np.searchsorted(counts, max(k, 1), side="right") - 1)
            if j <= 0:
                continue
            rin = recovery_from_losses(c["loss_rank"], r["l_corrupt_rank"], r["l_clean_rank"])[j]
            rout = recovery_from_losses(c["loss_eval"], r["l_corrupt_eval"], r["l_clean_eval"])[j]
            infl.append(rin - rout)
            pred.append(sigma * np.sqrt(2 * np.log(max(fi.size, 2))) / r["gap_rank"])
            w = np.asarray([x[2] for x in r["refined_cells"]], float)
            gini_ref.append(gini(np.clip(w, 0, None)))
        if len(infl) >= 5:
            rho, p = spearmanr(pred, infl)
            d["inflation"] = {"n": len(infl), "median_inflation": float(np.median(infl)), "median_prediction": float(np.median(pred)), "spearman_rho": float(rho), "p": float(p)}
        if gini_ref:
            d["gini_refined_cells_median"] = float(np.median(gini_ref))
        summary[codec] = d
    (out / "theory_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    selection_figure(out)
    rows = load_rows(a.inputs)
    print(json.dumps(real_data_checks(rows, out), indent=1))


if __name__ == "__main__":
    main()
