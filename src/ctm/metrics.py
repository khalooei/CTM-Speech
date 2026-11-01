"""Stability and audio quality metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ctm.types import ensure_bqt


def token_change_rate(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    """Fraction of cells that differ. Requires equal shapes."""
    a = ensure_bqt(clean)
    b = ensure_bqt(corrupt)
    if a.shape != b.shape:
        raise ValueError(f"TCR requires equal shapes, got {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a != b).float().mean().item())


def unit_edit_distance(clean: torch.Tensor, corrupt: torch.Tensor) -> float:
    """
    Mean per-codebook normalized Levenshtein distance (UED).
    Works for equal or unequal lengths. Codes are [B,Q,T].
    """
    a = ensure_bqt(clean)[0]
    b = ensure_bqt(corrupt)[0]
    q = a.shape[0]
    dists = []
    for qi in range(q):
        dists.append(_normalized_levenshtein(a[qi].tolist(), b[qi].tolist()))
    return float(np.mean(dists)) if dists else 0.0


def _normalized_levenshtein(s: Sequence[int], t: Sequence[int]) -> float:
    n, m = len(s), len(t)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    # DP with two rows
    prev = list(range(m + 1))
    for i, cs in enumerate(s, start=1):
        cur = [i] + [0] * m
        for j, ct in enumerate(t, start=1):
            cost = 0 if cs == ct else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / max(n, m)


def codebook_occupancy(codes: torch.Tensor, codebook_size: Optional[int] = None) -> Dict[str, float]:
    """Unique/active ratio over flattened codes."""
    c = ensure_bqt(codes).reshape(-1)
    vals = c.unique()
    n_unique = int(vals.numel())
    size = int(codebook_size) if codebook_size is not None else int(c.max().item()) + 1
    return {
        "n_unique": float(n_unique),
        "occupancy": float(n_unique / max(size, 1)),
        "codebook_size": float(size),
    }


def js_divergence_code_hist(
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    codebook_size: Optional[int] = None,
) -> float:
    """Jensen–Shannon divergence between code histograms (bits)."""
    a = ensure_bqt(clean).reshape(-1)
    b = ensure_bqt(corrupt).reshape(-1)
    if codebook_size is None:
        codebook_size = int(max(int(a.max().item()), int(b.max().item())) + 1)
    ha = torch.bincount(a.clamp(min=0), minlength=codebook_size).float()
    hb = torch.bincount(b.clamp(min=0), minlength=codebook_size).float()
    pa = ha / ha.sum().clamp_min(1.0)
    pb = hb / hb.sum().clamp_min(1.0)
    m = 0.5 * (pa + pb)
    jsd = 0.5 * (_kl(pa, m) + _kl(pb, m))
    return float(jsd.item())


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    mask = p > 0
    return torch.sum(p[mask] * (torch.log2(p[mask].clamp_min(1e-12)) - torch.log2(q[mask].clamp_min(1e-12))))


def snr_db(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    r = reference.float().reshape(-1)
    e = estimate.float().reshape(-1)
    n = min(r.numel(), e.numel())
    r, e = r[:n], e[:n]
    noise = e - r
    return float((10.0 * torch.log10(r.pow(2).mean().clamp_min(1e-12) / noise.pow(2).mean().clamp_min(1e-12))).item())


def si_sdr(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-8) -> float:
    r = reference.float().reshape(-1)
    e = estimate.float().reshape(-1)
    n = min(r.numel(), e.numel())
    r, e = r[:n], e[:n]
    r = r - r.mean()
    e = e - e.mean()
    dot = torch.dot(e, r)
    s_target = (dot / (torch.dot(r, r) + eps)) * r
    e_noise = e - s_target
    return float((10.0 * torch.log10((s_target.pow(2).sum() + eps) / (e_noise.pow(2).sum() + eps))).item())


def stoi_score(reference: torch.Tensor, estimate: torch.Tensor, sample_rate: int = 16000) -> Optional[float]:
    try:
        from pystoi import stoi
    except ImportError:
        return None
    r = reference.detach().cpu().float().reshape(-1).numpy()
    e = estimate.detach().cpu().float().reshape(-1).numpy()
    n = min(len(r), len(e))
    return float(stoi(r[:n], e[:n], sample_rate, extended=False))


def stability_bundle(
    clean_codes: torch.Tensor,
    corrupt_codes: torch.Tensor,
    codebook_size: Optional[int] = None,
) -> Dict[str, float]:
    a = ensure_bqt(clean_codes)
    b = ensure_bqt(corrupt_codes)
    out: Dict[str, float] = {"ued": unit_edit_distance(a, b)}
    if a.shape == b.shape:
        out["tcr"] = token_change_rate(a, b)
        out["jsd"] = js_divergence_code_hist(a, b, codebook_size=codebook_size)
    else:
        out["tcr"] = float("nan")
        out["jsd"] = float("nan")
    occ = codebook_occupancy(b, codebook_size=codebook_size)
    out.update({f"corrupt_{k}": v for k, v in occ.items()})
    return out
