"""LaTeX tables and headline numbers from analysis_v2 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ctm.analysis_v2 import boot_ci, holm, load_rows, matched_pairs, matched_tests, paired_boot, per_row_table

CODEC = {"dac": "DAC", "mimi": "Mimi", "speechtokenizer": "SpeechTok."}
RANK_LABEL = {"random": "Random", "energy": "Perturbation energy", "rvq_early": "RVQ-early", "margin": "VQ margin", "displacement": "Codeword displacement", "gradient": "Gradient$\\times\\Delta$", "frame_block": "Oracle, frame only", "intervention": "Oracle, frame$\\to$cell", "greedy_frame": "Greedy (frame)"}
KIND_LABEL = {"gaussian": "Gaussian", "pink": "Pink", "brown": "Brown", "bit_crush": "Bit-crush", "esc50": "ESC-50", "esc10": "ESC-10", "opus": "Opus", "pgd_ctc": "CTC-PGD"}


def fmt(m, lo, hi, nd=2, bold=False):
    if not np.isfinite(m):
        return "--"
    s = f"{m:.{nd}f} [{lo:.{nd}f},\\,{hi:.{nd}f}]"
    return f"\\textbf{{{s}}}" if bold else s


def table_ranking_ablation(df: pd.DataFrame, scorer: str, out: Path, rankings=("random", "energy", "rvq_early", "margin", "displacement", "gradient", "frame_block", "intervention"), metric="aurc"):
    """Rows: rankings; columns: codecs; cell: mean AURC [95% CI] on identical rows; plus paired diff vs random.
    Bold the best-performing *oracle-free* ranking per codec column (the practical proxy a reader would pick),
    since the two Oracle rows are the known ceiling, not a competing method."""
    lines = ["\\begin{tabular}{l" + "c" * 3 + "}", "\\toprule", "Ranking & " + " & ".join(CODEC[c] for c in ("dac", "mimi", "speechtokenizer")) + " \\\\", "\\midrule"]
    ns = {}
    means = {rk: {} for rk in rankings}
    for rk in rankings:
        for codec in ("dac", "mimi", "speechtokenizer"):
            g = df[(df["codec"] == codec) & (df["ctm_status"] == "ok") & (df.get(f"{scorer}_resolved", True) == True)]
            col = f"{rk}|{scorer}|{metric}"
            if col not in g:
                means[rk][codec] = (np.nan, np.nan, np.nan); continue
            v = pd.to_numeric(g[col], errors="coerce").to_numpy()
            m, lo, hi = boot_ci(v)
            ns[codec] = int(np.isfinite(v).sum())
            means[rk][codec] = (m, lo, hi)
    proxy_rankings = [rk for rk in rankings if rk not in ("frame_block", "intervention")]
    best_proxy = {codec: max(proxy_rankings, key=lambda rk: (means[rk][codec][0] if np.isfinite(means[rk][codec][0]) else -np.inf)) for codec in ("dac", "mimi", "speechtokenizer")}
    for rk in rankings:
        cells = []
        for codec in ("dac", "mimi", "speechtokenizer"):
            m, lo, hi = means[rk][codec]
            is_best = rk in proxy_rankings and best_proxy[codec] == rk
            cells.append(fmt(m, lo, hi, bold=is_best))
        lines.append(f"{RANK_LABEL[rk]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\midrule", "$n$ (gated rows) & " + " & ".join(str(ns.get(c, 0)) for c in ("dac", "mimi", "speechtokenizer")) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))


def table_main(df: pd.DataFrame, scorer: str, out: Path):
    """Per codec x corruption kind: damage rate, TCR, dWER, oracle vs random AURC, F-CTM90 median (IQR), reach90@25%."""
    lines = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
             "Codec & Corruption & $n$ & dmg & TCR & $\\Delta$WER & AURC$_{\\text{oracle}}$ & AURC$_{\\text{rand}}$ & F-CTM$_{90}$ \\\\", "\\midrule"]
    for codec in ("dac", "mimi", "speechtokenizer"):
        first = True
        for kind in ("gaussian", "pink", "brown", "bit_crush", "esc50", "esc10", "opus", "pgd_ctc"):
            g = df[(df["codec"] == codec) & (df["condition_kind"] == kind)]
            if g.empty:
                continue
            ok = g[(g["ctm_status"] == "ok") & (g.get(f"{scorer}_resolved", True) == True)]
            n = len(ok)
            dmg = float((g["gate_passed"] == True).mean())
            tcr = float(np.nanmean(pd.to_numeric(g["tcr"], errors="coerce")))
            dwer = float(np.nanmean(pd.to_numeric(g["delta_wer"], errors="coerce")))
            ao = pd.to_numeric(ok.get(f"intervention|{scorer}|aurc"), errors="coerce").to_numpy() if n else np.array([])
            ar = pd.to_numeric(ok.get(f"random|{scorer}|aurc"), errors="coerce").to_numpy() if n else np.array([])
            c90 = pd.to_numeric(ok.get(f"intervention|{scorer}|ctm90"), errors="coerce").to_numpy() if n else np.array([])
            def ms(v, bold=False):
                v = v[np.isfinite(v)]
                if v.size == 0:
                    return "--"
                s = f"{np.mean(v):.2f}"
                return f"\\textbf{{{s}}}" if bold else s
            def med(v):
                v = v[np.isfinite(v)]
                return "--" if v.size == 0 else f"{np.median(v):.2f} ({np.quantile(v,0.25):.2f}--{np.quantile(v,0.75):.2f})"
            ao_f, ar_f = ao[np.isfinite(ao)], ar[np.isfinite(ar)]
            ao_m = np.mean(ao_f) if ao_f.size else np.nan
            ar_m = np.mean(ar_f) if ar_f.size else np.nan
            orc_better = np.isfinite(ao_m) and (not np.isfinite(ar_m) or ao_m >= ar_m)
            lines.append(f"{CODEC[codec] if first else ''} & {KIND_LABEL[kind]} & {n} & {dmg:.2f} & {tcr:.2f} & {dwer:.3f} & {ms(ao, orc_better)} & {ms(ar, not orc_better)} & {med(c90)} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    out.write_text("\n".join(lines))


FAMILY = {"gaussian": "Gaussian", "pink": "Coloured", "brown": "Coloured", "bit_crush": "Codec/quant.", "opus": "Codec/quant.", "esc50": "Real noise", "esc10": "Real noise", "pgd_ctc": "CTC-PGD"}


def table_main_compact(df: pd.DataFrame, scorer: str, out: Path):
    """Per codec x corruption family (5 families): damage rate, TCR, dWER, oracle vs random AURC, F-CTM90 median (IQR).
    Bolds the larger of AURC_orc/AURC_rnd per row: it is the direct row-level readout of "oracle beats random"
    (concentration), so the reader's eye should land on whichever one actually carries that evidence."""
    lines = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
             "Codec & Family & $n_{\\text{gated}}$ & dmg & TCR & $\\Delta$WER & AURC$_{\\text{orc}}$ & AURC$_{\\text{rnd}}$ & $\\FCTM{90}$ \\\\", "\\midrule"]
    d = df.copy(); d["family"] = d["condition_kind"].map(FAMILY)
    for codec in ("dac", "mimi", "speechtokenizer"):
        first = True
        for fam in ("Gaussian", "Coloured", "Codec/quant.", "Real noise", "CTC-PGD"):
            g = d[(d["codec"] == codec) & (d["family"] == fam)]
            if g.empty:
                continue
            ok = g[(g["ctm_status"] == "ok") & (g.get(f"{scorer}_resolved", True) == True)]
            dmg = float((g["gate_passed"] == True).mean()); tcr = float(np.nanmean(pd.to_numeric(g["tcr"], errors="coerce"))); dwer = float(np.nanmean(pd.to_numeric(g["delta_wer"], errors="coerce")))
            def mval(col):
                v = pd.to_numeric(ok.get(col), errors="coerce").to_numpy() if len(ok) else np.array([]); v = v[np.isfinite(v)]
                return np.nan if v.size == 0 else float(np.mean(v))
            def ms(col, bold=False):
                m = mval(col)
                s = "--" if not np.isfinite(m) else f"{m:.2f}"
                return f"\\textbf{{{s}}}" if bold and s != "--" else s
            def med(col):
                v = pd.to_numeric(ok.get(col), errors="coerce").to_numpy() if len(ok) else np.array([]); v = v[np.isfinite(v)]
                return "--" if v.size == 0 else f"{np.median(v):.2f} ({np.quantile(v,0.25):.2f}--{np.quantile(v,0.75):.2f})"
            orc_col, rnd_col = f"intervention|{scorer}|aurc", f"random|{scorer}|aurc"
            orc_m, rnd_m = mval(orc_col), mval(rnd_col)
            orc_better = np.isfinite(orc_m) and (not np.isfinite(rnd_m) or orc_m >= rnd_m)
            lines.append(f"{CODEC[codec] if first else ''} & {fam} & {len(ok)} & {dmg:.2f} & {tcr:.2f} & {dwer:.3f} & {ms(orc_col, orc_better)} & {ms(rnd_col, not orc_better)} & {med(f'intervention|{scorer}|ctm90')} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"; lines.append("\\end{tabular}")
    out.write_text("\n".join(lines))


def table_matched(df: pd.DataFrame, scorer: str, out: Path, caliper=0.5):
    outcomes = [f"intervention|{scorer}|aurc", f"intervention|{scorer}|ctm90", f"intervention|{scorer}|rec_at_5"]
    labels = {outcomes[0]: "AURC", outcomes[1]: "F-CTM$_{90}$", outcomes[2]: "Rec@5\\%"}
    lines = ["\\begin{tabular}{llrrrrr}", "\\toprule", "Codec & Outcome & pairs & PGD & Gaussian & $\\Delta$ [95\\% CI] & $p_{\\text{Holm}}$ \\\\", "\\midrule"]
    allp = []
    recs = []
    for codec in ("dac", "mimi", "speechtokenizer"):
        pairs = matched_pairs(df, codec, "pgd_ctc", "gaussian", outcomes, caliper=caliper)
        t = matched_tests(pairs, outcomes)
        for _, r in t.iterrows():
            recs.append((codec, r))
            allp.append(r.get("p_perm", 1.0))
    adj = holm([p if np.isfinite(p) else 1.0 for p in allp])
    for (codec, r), ph in zip(recs, adj):
        # Bold the significant (Holm-corrected) rows: this is the comparison the reader should land on.
        sig = np.isfinite(ph) and ph < 0.05
        diff_s = f"{r.get('mean_diff', np.nan):+.2f} [{r.get('ci_low', np.nan):+.2f},\\,{r.get('ci_high', np.nan):+.2f}]"
        ph_s = f"{ph:.3f}"
        if sig:
            diff_s, ph_s = f"\\textbf{{{diff_s}}}", f"\\textbf{{{ph_s}}}"
        lines.append(f"{CODEC[codec]} & {labels[r['outcome']]} & {r['n']} & {r.get('mean_a', np.nan):.2f} & {r.get('mean_b', np.nan):.2f} & {diff_s} & {ph_s} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out.write_text("\n".join(lines))
    return recs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--scorer", default="eval")
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(a.inputs)
    df = per_row_table(rows)
    for sc in (a.scorer, "eval2", "rank"):
        if any(f"|{sc}|" in c for c in df.columns):
            table_ranking_ablation(df, sc, out / f"tab_ranking_{sc}.tex")
            table_main(df, sc, out / f"tab_main_{sc}.tex")
            table_main_compact(df, sc, out / f"tab_main_compact_{sc}.tex")
            table_matched(df, sc, out / f"tab_matched_{sc}.tex")
    # headline numbers as json for the text
    head = {}
    for codec in ("dac", "mimi", "speechtokenizer"):
        g = df[(df["codec"] == codec) & (df["ctm_status"] == "ok")]
        head[codec] = {"n": int(len(g)), "damage_rate": float((df[df["codec"] == codec]["gate_passed"] == True).mean()),
                       "n_resolved_eval": int((g.get("eval_resolved", pd.Series(dtype=bool)) == True).sum()),
                       "n_rank_gap_positive": int((pd.to_numeric(g["gap_rank"], errors="coerce") > 0).sum())}
        for sc in ("eval", "eval2", "rank"):
            for rk in ("intervention", "random", "rvq_early", "displacement", "energy", "gradient", "margin", "frame_block"):
                for m in ("aurc", "ctm90", "ctm50", "rec_at_5", "rec_at_25"):
                    col = f"{rk}|{sc}|{m}"
                    if col in g:
                        v = pd.to_numeric(g[col], errors="coerce").to_numpy()
                        mean, lo, hi = boot_ci(v)
                        head[codec][f"{rk}.{sc}.{m}"] = {"mean": mean, "lo": lo, "hi": hi, "median": float(np.nanmedian(v)) if np.isfinite(v).any() else None}
            for rk in ("random", "rvq_early", "displacement", "energy", "gradient", "margin"):
                if f"intervention|{sc}|aurc" in g and f"{rk}|{sc}|aurc" in g:
                    head[codec][f"paired.{sc}.aurc.intervention_vs_{rk}"] = paired_boot(pd.to_numeric(g[f"intervention|{sc}|aurc"], errors="coerce").to_numpy(), pd.to_numeric(g[f"{rk}|{sc}|aurc"], errors="coerce").to_numpy())
    (out / "headline.json").write_text(json.dumps(head, indent=1))
    print("tables ->", out)


if __name__ == "__main__":
    main()
