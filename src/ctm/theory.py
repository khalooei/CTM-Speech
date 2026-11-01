"""Theoretical diagnostics for recovery curves (see paper Sec. 'A theoretical view').

Additive model: restoring S changes the loss by sum_{i in S} w_i with per-cell weights w_i (sum = gap).
Under additivity:
  * a uniformly random ordering has E[Rec(S_k)] = k/|D|  ->  AURC_random = 1/2 exactly;
  * the oracle (descending w) curve is the Lorenz-complement curve -> AURC_oracle = (1 + G)/2, G = Gini(w).
Deviations of AURC_random from 1/2 measure interaction: R = 2*AURC_random - 1 > 0 means redundancy
(sub-additive / diminishing returns), < 0 synergy.

Selection bias (winner's curse): if single-cell influences are measured as w_i + e_i with e_i ~ N(0, s^2)
and the same noise re-enters the curve, the in-sample recovery of the top-k under the null (w = 0) is
about E[sum of top-k of n N(0,s^2)] / gap  ~=  k * s * phi(Phi^{-1}(1-k/n)) / (k/n) / gap  for k << n,
which exceeds 1 whenever gap <~ s * sqrt(2 ln n). Held-out scoring removes the term because the scorer's
noise is independent of the selection.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def gini(w: np.ndarray) -> float:
    w = np.sort(np.clip(np.asarray(w, float), 0, None))
    n = w.size
    if n == 0 or w.sum() <= 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(2.0 * np.sum(i * w) / (n * w.sum()) - (n + 1.0) / n)


def redundancy_index(aurc_random: float) -> float:
    return 2.0 * aurc_random - 1.0


def null_insample_recovery(n: int, gap_over_sigma: float, ks: np.ndarray) -> np.ndarray:
    """Expected in-sample recovery of the top-k of n pure-noise single-cell influences (w=0),
    when the curve is scored by re-using those same noisy measurements additively."""
    ks = np.asarray(ks, float)
    out = np.zeros_like(ks)
    for j, k in enumerate(ks):
        if k <= 0:
            continue
        q = min(k / n, 1 - 1e-9)
        # mean of the top fraction q of a standard normal sample
        mean_top = norm.pdf(norm.ppf(1 - q)) / q
        out[j] = k * mean_top / gap_over_sigma
    return out


def simulate_selection_bias(n: int, gap_over_sigma: float, n_true: int = 0, n_rep: int = 200, seed: int = 0, alpha: float = 0.9):
    """Monte-Carlo: n cells, n_true of them carry equal true weight summing to `gap` (rest 0);
    single-cell measurement noise N(0, sigma^2) with sigma = 1, gap = gap_over_sigma.
    Returns in-sample and held-out F-CTM_alpha (as fractions of n) and the k=48-subset variant."""
    rng = np.random.default_rng(seed)
    gap = float(gap_over_sigma)
    w = np.zeros(n)
    if n_true > 0:
        w[:n_true] = gap / n_true
    res = {"insample": [], "heldout": [], "subset48_insample": []}
    for _ in range(n_rep):
        meas = w + rng.normal(0, 1, n)  # ranking measurements
        order = np.argsort(-meas)
        ks = np.arange(1, n + 1)
        # in-sample: curve re-uses the same measurements (additive, same noise)
        cum_in = np.cumsum(meas[order]) / gap
        # held-out: fresh independent noise on the prefix evaluation (one draw per prefix)
        cum_out = (np.cumsum(w[order]) + rng.normal(0, 1, n)) / gap
        f_in = ks[np.argmax(cum_in >= alpha)] / n if (cum_in >= alpha).any() else 1.0
        f_out = ks[np.argmax(cum_out >= alpha)] / n if (cum_out >= alpha).any() else 1.0
        # old protocol: random 48-cell subset, curve over the subset, no Recovery(D)=1 guarantee
        sub = rng.choice(n, size=min(48, n), replace=False)
        ms = meas[sub]
        o = np.argsort(-ms)
        cum_sub = np.cumsum(ms[o]) / gap
        kk = np.arange(1, len(sub) + 1)
        f_sub = kk[np.argmax(cum_sub >= alpha)] / len(sub) if (cum_sub >= alpha).any() else np.nan
        res["insample"].append(f_in); res["heldout"].append(f_out); res["subset48_insample"].append(f_sub)
    return {k: np.asarray(v) for k, v in res.items()}
