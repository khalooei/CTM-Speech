"""Analysis for hier-v2 rows: per-row curve metrics, bootstrap aggregation, paired tests,
matched-severity analysis with caliper + Holm, cluster-robust regression."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ctm.hier import aurc, ctm_alpha, recovery_at, recovery_from_losses
from ctm.jsonl_store import load_jsonl_last_wins

ALPHAS = (0.5, 0.8, 0.9)
BUDGETS = (0.05, 0.10, 0.25)
SCORERS = ("eval", "rank", "eval2")


def load_rows(paths: Sequence[str | Path]) -> List[dict]:
    rows: Dict[str, dict] = {}
    for p in paths:
        rows.update(load_jsonl_last_wins(p))
    return [r for r in rows.values() if r.get("status") == "ok"]


def curve_metrics(row: dict, ranking: str, scorer: str) -> Optional[dict]:
    c = row.get("curves", {}).get(ranking)
    if not c or f"loss_{scorer}" not in c:
        return None
    lc, lk = row.get(f"l_corrupt_{scorer}"), row.get(f"l_clean_{scorer}")
    if lc is None or lk is None:
        return None
    gap = lc - lk
    n = int(row["n_changed"])
    frac = np.asarray(c["counts"], dtype=float) / n
    rec = recovery_from_losses(c[f"loss_{scorer}"], lc, lk)
    if not np.isfinite(rec).all():
        return None
    out = {
        "gap": float(gap),
        "aurc": aurc(frac, rec),
        "end_jitter": float(rec[-1] - 1.0),  # numeric-noise diagnostic; 0 by construction
        "max_rec": float(np.max(rec)),
        "min_rec": float(np.min(rec)),
        "monotone_viol": float(np.max(np.maximum.accumulate(rec) - rec)),
    }
    for a in ALPHAS:
        out[f"ctm{int(a * 100)}"] = ctm_alpha(frac, rec, a)
    for b in BUDGETS:
        out[f"rec_at_{int(b * 100)}"] = recovery_at(frac, rec, b)
        out[f"reach90_at_{int(b * 100)}"] = float(out[f"ctm90"] <= b + 1e-12)
    return out


RESOLUTION_FACTOR = 10.0  # a scorer's gap must exceed 10x the measured numeric noise of its endpoints


def scorer_resolution(row: dict, scorer: str) -> Tuple[float, bool]:
    """Numeric noise floor of a scorer on this row: largest endpoint discrepancy between the dedicated
    endpoint batch and the k=0 / k=|D| curve points (identical codes, different decode batches)."""
    lc, lk = row.get(f"l_corrupt_{scorer}"), row.get(f"l_clean_{scorer}")
    if lc is None or lk is None:
        return float("nan"), False
    noise = 0.0
    for c in row.get("curves", {}).values():
        if f"loss_{scorer}" not in c:
            continue
        noise = max(noise, abs(c[f"loss_{scorer}"][0] - lc), abs(c[f"loss_{scorer}"][-1] - lk))
    gap = lc - lk
    return float(noise), bool(gap > 0 and gap > RESOLUTION_FACTOR * noise)


def per_row_table(rows: Iterable[dict], scorers: Sequence[str] = SCORERS) -> pd.DataFrame:
    recs = []
    for r in rows:
        base = {
            "utterance_id": r["utterance_id"],
            "codec": r["codec"],
            "condition": r["condition"],
            "condition_kind": r.get("condition_kind"),
            "duration_s": r.get("duration_s"),
            "tcr": r.get("tcr"),
            "ued": r.get("ued"),
            "delta_wer": r.get("delta_wer"),
            "delta_cer": r.get("delta_cer"),
            "wer_clean": r.get("wer_clean"),
            "wer_corrupt": r.get("wer_corrupt"),
            "input_snr_db": r.get("input_snr_db"),
            "recon_snr_db": r.get("recon_snr_db"),
            "stoi": r.get("stoi"),
            "n_changed": r.get("n_changed"),
            "n_frames_changed": r.get("n_frames_changed"),
            "T": r.get("T_clean"),
            "Q": r.get("n_codebooks"),
            "gate_passed": r.get("gate_passed"),
            "ctm_status": r.get("ctm_status"),
            "gap_eval": r.get("gap_eval"),
            "gap_rank": r.get("gap_rank"),
            "stage_tcr0": (r.get("stage_tcr") or [None])[0],
            "wall_seconds": r.get("wall_seconds"),
        }
        if r.get("ctm_status") != "ok":
            recs.append(base)
            continue
        resolved = {}
        for sc in scorers:
            noise, ok = scorer_resolution(r, sc)
            base[f"{sc}_noise"], base[f"{sc}_resolved"] = noise, ok
            resolved[sc] = ok
        for ranking in r.get("curves", {}):
            for sc in scorers:
                if not resolved.get(sc):
                    continue  # unresolved scorer: metrics left NaN so every summary excludes the row
                m = curve_metrics(r, ranking, sc)
                if m is None:
                    continue
                for k, v in m.items():
                    base[f"{ranking}|{sc}|{k}"] = v
        # stage restoration shares (held-out scorer)
        st = r.get("stage_restore")
        if st and "loss_eval" in st and r.get("gap_eval") and resolved.get("eval"):
            infl = (r["l_corrupt_eval"] - np.asarray(st["loss_eval"])) / r["gap_eval"]
            flips = np.asarray(st["flip_counts"], dtype=float)
            base["stage_rec"] = json.dumps([float(x) for x in infl])
            base["stage_flip_share"] = json.dumps([float(x) for x in flips / max(flips.sum(), 1)])
        # additivity: sum of refined single-cell influences (rank scorer) vs joint recovery of the same prefix
        rc = r.get("refined_cells")
        cur = r.get("curves", {}).get("intervention")
        if rc and cur and r.get("gap_rank") and resolved.get("rank"):
            infl = sorted([x[2] for x in rc], reverse=True)
            k = len(infl)
            counts = np.asarray(cur["counts"])
            j = int(np.searchsorted(counts, k, side="right") - 1)
            if j > 0:
                kk = int(counts[j])
                joint = (r["l_corrupt_rank"] - cur["loss_rank"][j]) / r["gap_rank"]
                additive = float(np.sum(infl[:kk])) / r["gap_rank"]
                base["additivity_k"] = kk
                base["additivity_joint"] = float(joint)
                base["additivity_sum"] = additive
        recs.append(base)
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------


def boot_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, stat=np.mean) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    b = stat(x[idx], axis=1)
    return float(stat(x)), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def paired_boot(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size == 0:
        return {"n": 0}
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    bm = d[idx].mean(axis=1)
    # sign-flip permutation p-value
    signs = rng.choice([-1.0, 1.0], size=(n_boot, d.size))
    perm = np.abs((d[None, :] * signs).mean(axis=1))
    p = float((np.sum(perm >= abs(d.mean()) - 1e-15) + 1) / (n_boot + 1))
    return {"n": int(d.size), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
            "ci_low": float(np.quantile(bm, 0.025)), "ci_high": float(np.quantile(bm, 0.975)), "p_perm": p,
            "median_diff": float(np.median(d)), "frac_a_better": float(np.mean(d > 0))}


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def summarize(df: pd.DataFrame, group: Sequence[str], rankings: Sequence[str], scorer: str = "eval", metrics=("aurc", "ctm90", "ctm80", "ctm50", "rec_at_5", "rec_at_25", "reach90_at_25")) -> pd.DataFrame:
    recs = []
    for keys, g in df.groupby(list(group)):
        rec = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        rec["n_rows"] = int(len(g))
        rec["n_gated"] = int((g["ctm_status"] == "ok").sum())
        rec["damage_rate"] = float((g["gate_passed"] == True).mean())
        if f"{scorer}_resolved" in g:
            rec["n_resolved"] = int((g[f"{scorer}_resolved"] == True).sum())
        rec["mean_tcr"] = float(np.nanmean(pd.to_numeric(g["tcr"], errors="coerce")))
        rec["mean_delta_wer"] = float(np.nanmean(pd.to_numeric(g["delta_wer"], errors="coerce")))
        rec["mean_input_snr"] = float(np.nanmean(pd.to_numeric(g["input_snr_db"], errors="coerce")))
        for rk in rankings:
            for m in metrics:
                col = f"{rk}|{scorer}|{m}"
                if col not in g:
                    continue
                v = pd.to_numeric(g[col], errors="coerce").to_numpy()
                mean, lo, hi = boot_ci(v)
                rec[f"{rk}.{m}.mean"], rec[f"{rk}.{m}.lo"], rec[f"{rk}.{m}.hi"] = mean, lo, hi
                rec[f"{rk}.{m}.median"] = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
                rec[f"{rk}.{m}.n"] = int(np.isfinite(v).sum())
        recs.append(rec)
    return pd.DataFrame(recs)


def paired_ranking_comparisons(df: pd.DataFrame, group: Sequence[str], ref: str = "intervention", others: Sequence[str] = ("random", "rvq_early", "displacement", "margin", "gradient", "energy", "frame_block"), scorer: str = "eval", metric: str = "aurc") -> pd.DataFrame:
    recs = []
    for keys, g in df.groupby(list(group)):
        base = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        a = pd.to_numeric(g.get(f"{ref}|{scorer}|{metric}"), errors="coerce").to_numpy()
        for o in others:
            col = f"{o}|{scorer}|{metric}"
            if col not in g:
                continue
            b = pd.to_numeric(g[col], errors="coerce").to_numpy()
            res = paired_boot(a, b)
            recs.append({**base, "ref": ref, "other": o, "metric": metric, "scorer": scorer, **res})
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------
# Matched-severity analysis (same utterance, caliper on codec-level z-scores, Holm)
# ----------------------------------------------------------------------------


def matched_pairs(df: pd.DataFrame, codec: str, a_prefix: str, b_prefix: str, outcome_cols: Sequence[str], features=("tcr", "delta_wer"), caliper: float = 0.5) -> pd.DataFrame:
    sub = df[(df["codec"] == codec) & (df["ctm_status"] == "ok")].copy()
    for f in features:
        v = pd.to_numeric(sub[f], errors="coerce")
        sd = np.nanstd(v)
        sub[f"z_{f}"] = (v - np.nanmean(v)) / (sd if sd > 1e-12 else 1.0)
    recs = []
    for utt, g in sub.groupby("utterance_id"):
        A = g[g["condition"].str.startswith(a_prefix)]
        B = g[g["condition"].str.startswith(b_prefix)]
        if A.empty or B.empty:
            continue
        best, bd = None, np.inf
        for _, ar in A.iterrows():
            for _, br in B.iterrows():
                d = np.sqrt(sum((ar[f"z_{f}"] - br[f"z_{f}"]) ** 2 for f in features))
                if np.isfinite(d) and d < bd:
                    bd, best = d, (ar, br)
        if best is None or bd > caliper:
            recs.append({"utterance_id": utt, "matched": False, "distance": bd})
            continue
        ar, br = best
        rec = {"utterance_id": utt, "matched": True, "distance": bd, "a_condition": ar["condition"], "b_condition": br["condition"]}
        for f in features:
            rec[f"a_{f}"], rec[f"b_{f}"] = ar[f], br[f]
        for oc in outcome_cols:
            rec[f"a_{oc}"], rec[f"b_{oc}"] = ar.get(oc), br.get(oc)
        recs.append(rec)
    return pd.DataFrame(recs)


def matched_tests(pairs: pd.DataFrame, outcome_cols: Sequence[str]) -> pd.DataFrame:
    m = pairs[pairs["matched"] == True] if "matched" in pairs else pairs
    recs = []
    for oc in outcome_cols:
        if f"a_{oc}" not in m:
            continue
        res = paired_boot(pd.to_numeric(m[f"a_{oc}"], errors="coerce").to_numpy(), pd.to_numeric(m[f"b_{oc}"], errors="coerce").to_numpy())
        recs.append({"outcome": oc, "n_candidates": int(len(pairs)), "n_matched": int(len(m)), **res})
    out = pd.DataFrame(recs)
    if not out.empty and "p_perm" in out:
        out["p_holm"] = holm(out["p_perm"].fillna(1.0).tolist())
    return out


# ----------------------------------------------------------------------------
# Cluster-robust regression (sensitivity)
# ----------------------------------------------------------------------------


def cluster_ols(df: pd.DataFrame, outcome: str, formula_rhs: str = "C(condition_kind) + C(codec) + tcr + delta_wer + C(condition_kind):C(codec)") -> str:
    import statsmodels.formula.api as smf

    d = df[df["ctm_status"] == "ok"].copy()
    d["y"] = pd.to_numeric(d[outcome], errors="coerce")
    d["tcr"] = pd.to_numeric(d["tcr"], errors="coerce")
    d["delta_wer"] = pd.to_numeric(d["delta_wer"], errors="coerce")
    d = d.dropna(subset=["y", "tcr", "delta_wer"])
    model = smf.ols(f"y ~ {formula_rhs}", data=d)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["utterance_id"])[0]})
    lines = [f"OLS with utterance-clustered SE; outcome={outcome}; n={int(res.nobs)}; clusters={d['utterance_id'].nunique()}", str(res.summary())]
    # also a reduced model without interaction for readable coefficients
    res2 = smf.ols("y ~ C(condition_kind) + C(codec) + tcr + delta_wer", data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["utterance_id"])[0]})
    lines.append("\nReduced model (no interaction):\n" + str(res2.summary().tables[1]))
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.inputs)
    df = per_row_table(rows)
    df.to_csv(out / "per_row.csv", index=False)
    rankings = sorted({c.split("|")[0] for c in df.columns if "|" in c})
    for sc in ("eval", "rank", "eval2"):
        if not any(f"|{sc}|" in c for c in df.columns):
            continue
        summarize(df, ["codec", "condition"], rankings, scorer=sc).to_csv(out / f"summary_by_condition_{sc}.csv", index=False)
        summarize(df, ["codec", "condition_kind"], rankings, scorer=sc).to_csv(out / f"summary_by_kind_{sc}.csv", index=False)
        summarize(df, ["codec"], rankings, scorer=sc).to_csv(out / f"summary_by_codec_{sc}.csv", index=False)
        for metric in ("aurc", "ctm90", "rec_at_5"):
            paired_ranking_comparisons(df, ["codec"], scorer=sc, metric=metric).to_csv(out / f"paired_{metric}_{sc}.csv", index=False)
    print(f"rows={len(df)} gated={(df['ctm_status']=='ok').sum()} -> {out}")


if __name__ == "__main__":
    main()
