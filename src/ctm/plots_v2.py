"""Paper figures for the hier-v2 protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ctm.analysis_v2 import load_rows, per_row_table
from ctm.hier import recovery_from_losses

PALETTE = {
    "intervention": "#B3412E",
    "frame_block": "#D98B5F",
    "random": "#7A7F7C",
    "random2": "#A9AEAB",
    "rvq_early": "#2F6B8E",
    "displacement": "#3F8F6B",
    "margin": "#9C6ADE",
    "gradient": "#C9A227",
    "energy": "#5A5A9E",
    "greedy_frame": "#1B1D1C",
}
LABELS = {
    "intervention": "oracle (frame→cell)",
    "frame_block": "oracle (frame only)",
    "random": "random",
    "random2": "random (seed 2)",
    "rvq_early": "RVQ-early",
    "displacement": "displacement",
    "margin": "VQ margin",
    "gradient": "gradient×Δ",
    "energy": "perturbation energy",
    "greedy_frame": "greedy (frame)",
}
CODEC_LABEL = {"dac": "DAC", "mimi": "Mimi", "speechtokenizer": "SpeechTokenizer"}
KIND_LABEL = {"gaussian": "Gaussian", "pgd_ctc": "CTC-PGD", "esc50": "ESC-50", "esc10": "ESC-10", "opus": "Opus",
              "pink": "pink", "brown": "brown", "bit_crush": "bit-crush"}
GRID = np.linspace(0, 1, 101)


def _style():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 6.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.dpi": 300, "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    })


def interp_curve(row: dict, ranking: str, scorer: str) -> Optional[np.ndarray]:
    from ctm.analysis_v2 import scorer_resolution

    c = row.get("curves", {}).get(ranking)
    if not c or f"loss_{scorer}" not in c or not scorer_resolution(row, scorer)[1]:
        return None
    lc, lk = row[f"l_corrupt_{scorer}"], row[f"l_clean_{scorer}"]
    if lc - lk <= 0:
        return None
    frac = np.asarray(c["counts"], float) / row["n_changed"]
    rec = recovery_from_losses(c[f"loss_{scorer}"], lc, lk)
    # step interpolation (recovery of largest prefix <= f)
    idx = np.searchsorted(frac, GRID, side="right") - 1
    return np.clip(rec[np.clip(idx, 0, len(rec) - 1)], -0.5, 1.2)


def mean_curves(rows: Sequence[dict], rankings: Sequence[str], scorer: str, n_boot: int = 500) -> Dict[str, Dict[str, np.ndarray]]:
    out = {}
    rng = np.random.default_rng(0)
    for rk in rankings:
        mats = [c for c in (interp_curve(r, rk, scorer) for r in rows) if c is not None]
        if not mats:
            continue
        M = np.stack(mats)
        med = np.median(M, axis=0)
        boots = np.stack([np.median(M[rng.integers(0, len(M), len(M))], axis=0) for _ in range(n_boot)])
        out[rk] = {"median": med, "lo": np.quantile(boots, 0.025, axis=0), "hi": np.quantile(boots, 0.975, axis=0), "n": len(M), "mean": M.mean(axis=0)}
    return out


def fig_recovery_by_codec(rows: List[dict], out: Path, scorer: str = "eval", rankings=("intervention", "rvq_early", "displacement", "energy", "random"), conditions: Optional[Sequence[str]] = None, title_suffix: str = ""):
    codecs = ["dac", "mimi", "speechtokenizer"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.6), sharey=True)
    for ax, codec in zip(axes, codecs):
        sub = [r for r in rows if r["codec"] == codec and r.get("ctm_status") == "ok" and (conditions is None or r["condition"] in conditions)]
        mc = mean_curves(sub, rankings, scorer)
        for rk in rankings:
            if rk not in mc:
                continue
            ax.plot(GRID, mc[rk]["median"], color=PALETTE[rk], lw=1.4 if rk == "intervention" else 1.0, label=LABELS[rk], ls="--" if rk == "random" else "-")
            if rk in ("intervention", "random"):
                ax.fill_between(GRID, mc[rk]["lo"], mc[rk]["hi"], color=PALETTE[rk], alpha=0.18, lw=0)
        ax.axhline(0.9, color="#999", lw=0.6, ls=":")
        ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.05)
        ax.set_title(f"{CODEC_LABEL[codec]} (n={mc.get('intervention', {}).get('n', 0)})")
        ax.set_xlabel("fraction of changed cells restored")
    axes[0].set_ylabel("recovery (median, 95% CI)")
    axes[-1].legend(loc="lower right", frameon=False, ncol=1)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_insample_vs_heldout(rows: List[dict], out: Path):
    """The winner's curse: intervention curve scored by the ranker vs by held-out evaluators.
    Stacked one codec per row (rather than three-across) so each panel is legible at
    single-column width -- three-across left every curve too compressed to read."""
    codecs = ["dac", "mimi", "speechtokenizer"]
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 6.4), sharex=True)
    for ax, codec in zip(axes, codecs):
        sub = [r for r in rows if r["codec"] == codec and r.get("ctm_status") == "ok"]
        for scorer, lab, col, ls in (("rank", "in-sample (wav2vec2-base = ranker)", "#B3412E", ":"), ("eval", "held-out (HuBERT-large)", "#B3412E", "-"), ("eval2", "held-out (wav2vec2-large)", "#8C2B2B", "--")):
            mc = mean_curves(sub, ["intervention", "random"], scorer)
            if "intervention" in mc:
                ax.plot(GRID, mc["intervention"]["median"], color=col, ls=ls, lw=1.2, label=f"oracle, {lab}")
            if "random" in mc and scorer == "eval":
                ax.plot(GRID, mc["random"]["median"], color="#7A7F7C", ls="--", lw=1.0, label="random, held-out")
        ax.set_xscale("symlog", linthresh=0.05)
        ax.set_xlim(0, 1); ax.set_ylim(-0.1, 1.2); ax.set_title(CODEC_LABEL[codec], fontsize=8.5)
        ax.axhline(1.0, color="#999", lw=0.5, ls=":")
        ax.set_ylabel("median recovery")
    axes[-1].set_xlabel("fraction restored (symlog)")
    # An in-axes legend has no corner that stays clear of the curves on any panel
    # (SpeechTokenizer's random/held-out lines pass through "lower right"). Shared
    # legend below the column instead, with room reserved for it.
    h, l = axes[-1].get_legend_handles_labels()
    fig.tight_layout(pad=0.4, rect=(0.0, 0.09, 1.0, 1.0))
    fig.legend(h, l, frameon=False, ncol=1, loc="lower center", bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_stage_restoration(rows: List[dict], out: Path, scorer: str = "eval"):
    """Per-stage: flip share vs recovery when restoring that stage alone vs cumulative (rvq_early)."""
    codecs = ["dac", "mimi", "speechtokenizer"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.55))
    for ax, codec in zip(axes, codecs):
        from ctm.analysis_v2 import scorer_resolution
        sub = [r for r in rows if r["codec"] == codec and r.get("ctm_status") == "ok" and r.get("stage_restore") and scorer_resolution(r, scorer)[1]]
        if not sub:
            continue
        Q = sub[0]["n_codebooks"]
        alone, cum, flips = [], [], []
        for r in sub:
            st = r["stage_restore"]
            gap = r[f"gap_{scorer}"]
            if gap <= 0 or f"loss_{scorer}" not in st:
                continue
            alone.append((r[f"l_corrupt_{scorer}"] - np.asarray(st[f"loss_{scorer}"])) / gap)
            fl = np.asarray(st["flip_counts"], float); flips.append(fl / max(fl.sum(), 1))
            c = r["curves"].get("rvq_early")
            if c:
                counts = np.asarray(c["counts"]); rec = recovery_from_losses(c[f"loss_{scorer}"], r[f"l_corrupt_{scorer}"], r[f"l_clean_{scorer}"])
                cum.append([rec[np.searchsorted(counts, k, side="right") - 1] for k in r["stage_counts_cum"]])
        alone, flips = np.median(np.stack(alone), 0), np.mean(np.stack(flips), 0)
        x = np.arange(Q)
        ax.bar(x - 0.2, flips, width=0.4, color="#A9AEAB", label="flip share")
        ax.bar(x + 0.2, np.clip(alone, -0.2, 1.2), width=0.4, color="#2F6B8E", label="recovery, stage alone")
        if cum:
            ax.plot(x, np.median(np.stack(cum), 0), color="#B3412E", marker="o", ms=2.5, lw=1.0, label="recovery, stages ≤ q")
        ax.axhline(0, color="#444", lw=0.5)
        ax.set_xticks(x); ax.set_xlabel("RVQ stage q"); ax.set_title(f"{CODEC_LABEL[codec]} (n={len(sub)})"); ax.set_ylim(-0.25, 1.1)
    axes[0].set_ylabel("share / median recovery")
    h, l = axes[0].get_legend_handles_labels()
    # Reserve a dedicated bottom strip for the shared legend before placing it, so it
    # never lands on top of the per-axes "RVQ stage q" labels (tight_layout does not
    # know about fig.legend artists added after it runs).
    fig.tight_layout(pad=0.4, rect=(0.0, 0.11, 1.0, 1.0))
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_severity(df: pd.DataFrame, out: Path, scorer: str = "eval"):
    """F-CTM90 and AURC (oracle vs random) against condition severity for each codec."""
    codecs = ["dac", "mimi", "speechtokenizer"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0), sharey=True)
    for ax, codec in zip(axes, codecs):
        sub = df[(df["codec"] == codec) & (df["ctm_status"] == "ok")]
        for kind, marker, col in (("gaussian", "o", "#2F6B8E"), ("pgd_ctc", "s", "#B3412E"), ("esc50", "^", "#3F8F6B"), ("opus", "D", "#9C6ADE")):
            g = sub[sub["condition_kind"] == kind]
            if g.empty:
                continue
            agg = g.groupby("condition").agg(x=("delta_wer", "median"), y=(f"intervention|{scorer}|aurc", "median"), yr=(f"random|{scorer}|aurc", "median"), n=("utterance_id", "count"))
            agg = agg[agg["n"] >= 5]
            ax.scatter(agg["x"], agg["y"], marker=marker, color=col, s=14, label=f"{kind} oracle")
            ax.scatter(agg["x"], agg["yr"], marker=marker, facecolors="none", edgecolors=col, s=14)
        ax.set_xlabel("median ΔWER (held-out)"); ax.set_title(CODEC_LABEL[codec]); ax.set_ylim(0.3, 1.0)
    axes[0].set_ylabel("median AURC (filled: oracle, open: random)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def fig_tcr_vs_damage(df: pd.DataFrame, out: Path):
    """Instability (TCR) vs functional damage rate per (codec, condition)."""
    fig, ax = plt.subplots(1, 1, figsize=(3.0, 2.1))
    marks = {"gaussian": "o", "pgd_ctc": "s", "esc50": "^", "esc10": "v", "opus": "D", "pink": "P", "brown": "X", "bit_crush": "*"}
    cols = {"dac": "#B3412E", "mimi": "#2F6B8E", "speechtokenizer": "#3F8F6B"}
    for (codec, cond), g in df.groupby(["codec", "condition"]):
        kind = g["condition_kind"].iloc[0]
        ax.scatter(np.nanmean(pd.to_numeric(g["tcr"], errors="coerce")), (g["gate_passed"] == True).mean(), marker=marks.get(kind, "o"), color=cols[codec], s=18, alpha=0.85, lw=0)
    for codec, c in cols.items():
        ax.scatter([], [], color=c, label=CODEC_LABEL[codec], s=18)
    for kind, m in marks.items():
        ax.scatter([], [], marker=m, color="#555", label=KIND_LABEL.get(kind, kind), s=14)
    ax.set_xlabel("mean token change rate (TCR)"); ax.set_ylabel("damage rate (transcript worse)")
    ax.set_xlim(0.4, 1.0); ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False, ncol=2, fontsize=5.5, loc="upper left")
    fig.tight_layout(pad=0.3); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args(argv)
    _style()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.inputs)
    df = per_row_table(rows)
    fig_recovery_by_codec(rows, out / "fig_recovery_all.pdf")
    fig_recovery_by_codec(rows, out / "fig_recovery_all.png")
    fig_recovery_by_codec(rows, out / "fig_recovery_pgd.pdf", conditions=[c for c in df["condition"].unique() if c.startswith("pgd")])
    fig_recovery_by_codec(rows, out / "fig_recovery_natural.pdf", conditions=[c for c in df["condition"].unique() if not c.startswith("pgd")])
    fig_insample_vs_heldout(rows, out / "fig_insample_vs_heldout.pdf")
    fig_insample_vs_heldout(rows, out / "fig_insample_vs_heldout.png")
    fig_stage_restoration(rows, out / "fig_stage.pdf")
    fig_stage_restoration(rows, out / "fig_stage.png")
    fig_severity(df, out / "fig_severity.pdf")
    fig_tcr_vs_damage(df, out / "fig_tcr_damage.pdf")
    fig_tcr_vs_damage(df, out / "fig_tcr_damage.png")
    print("figures ->", out)


if __name__ == "__main__":
    main()
