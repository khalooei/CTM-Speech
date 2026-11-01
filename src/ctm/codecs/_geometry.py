"""Shared VQ geometry helpers."""

from __future__ import annotations

import torch


def _codebook_nn_spacing(codebook: torch.Tensor) -> float:
    """Mean distance from each codeword to its nearest other codeword."""
    d = torch.cdist(codebook[None], codebook[None])[0]
    d.fill_diagonal_(float("inf"))
    return float(d.min(dim=1).values.mean().item())


def _margins_from_dist(dist: torch.Tensor, spacing: float) -> torch.Tensor:
    """dist [N, K] Euclidean distances -> normalised runner-up margin [N]."""
    top2 = dist.topk(2, dim=-1, largest=False).values
    return (top2[:, 1] - top2[:, 0]) / max(spacing, 1e-8)
