"""Deterministic fake codec for unit tests (no external checkpoints)."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ctm.codecs.base import CodecBase, CodecInfo
from ctm.types import Cell, ensure_bqt


class FakeCodec(CodecBase):
    """
    Simple non-learned codec:
    - Frames waveform with hop H
    - Quantizes frame mean energy into codebook indices across Q codebooks
    - Decode reconstructs piecewise-constant waveform from codes
    """

    def __init__(
        self,
        n_codebooks: int = 4,
        codebook_size: int = 32,
        hop: int = 160,
        sample_rate: int = 16000,
        name: str = "fake",
        device: Optional[str] = None,
    ):
        self.device = "cpu"
        self.info = CodecInfo(
            name=name,
            sample_rate=sample_rate,
            n_codebooks=n_codebooks,
            codebook_size=codebook_size,
            hop_length=hop,
            supports_hybrid_decode=True,
            supports_margin=True,
            supports_displacement=True,
        )
        self.hop = hop
        # Fixed codebook centroids in [0, 1]
        g = torch.Generator().manual_seed(0)
        self.codebooks = torch.rand(n_codebooks, codebook_size, generator=g)

    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        w = self.prepare_waveform(waveform, sample_rate)  # [B, T]
        b, t = w.shape
        # Pad
        pad = (self.hop - (t % self.hop)) % self.hop
        if pad:
            w = F.pad(w, (0, pad))
        frames = w.reshape(b, -1, self.hop).mean(dim=-1)  # [B, Tf] in roughly [-1,1]
        # Map to [0,1]
        feats = ((frames + 1.0) * 0.5).clamp(0, 1)
        codes = []
        for q in range(self.info.n_codebooks):
            cb = self.codebooks[q].to(feats.device)  # [K]
            # distance of feat to each centroid, with q-dependent scale
            scaled = (feats * (0.7 + 0.1 * q)).clamp(0, 1)
            dist = (scaled.unsqueeze(-1) - cb.view(1, 1, -1)).abs()  # [B, Tf, K]
            idx = dist.argmin(dim=-1)
            codes.append(idx)
        return torch.stack(codes, dim=1).long()  # [B, Q, T]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        codes = ensure_bqt(codes)
        b, q, tf = codes.shape
        # Average dequantized values across codebooks
        vals = []
        for qi in range(q):
            cb = self.codebooks[qi]
            idx = codes[:, qi].clamp(0, self.info.codebook_size - 1)
            vals.append(cb[idx])  # [B, Tf]
        feat = torch.stack(vals, dim=0).mean(dim=0)  # [B, Tf] in [0,1]
        frames = feat * 2.0 - 1.0
        wav = frames.unsqueeze(-1).repeat(1, 1, self.hop).reshape(b, -1)
        return wav

    # Linear "latent": contribution of stage q is codebook value / Q, so decode is exactly
    # decode_latent(sum of contributions).
    def latent_contributions(self, codes: torch.Tensor) -> torch.Tensor:
        codes = ensure_bqt(codes)
        b, q, tf = codes.shape
        out = []
        for qi in range(q):
            cb = self.codebooks[qi]
            idx = codes[:, qi].clamp(0, self.info.codebook_size - 1)
            out.append((cb[idx] / float(q)).unsqueeze(1))  # [B, 1, Tf]
        return torch.stack(out, dim=0)  # [Q, B, 1, Tf]

    def decode_latent(self, z: torch.Tensor, keep_grad: bool = False) -> torch.Tensor:
        feat = z[:, 0]  # [B, Tf]
        frames = feat * 2.0 - 1.0
        b = frames.shape[0]
        return frames.unsqueeze(-1).repeat(1, 1, self.hop).reshape(b, -1)

    def stage_margins(self, waveform: torch.Tensor, sample_rate: int):
        w = self.prepare_waveform(waveform, sample_rate)
        b, t = w.shape
        pad = (self.hop - (t % self.hop)) % self.hop
        if pad:
            w = F.pad(w, (0, pad))
        frames = w.reshape(b, -1, self.hop).mean(dim=-1)
        feats = ((frames + 1.0) * 0.5).clamp(0, 1)
        margins, codes = [], []
        for q in range(self.info.n_codebooks):
            cb = self.codebooks[q]
            scaled = (feats * (0.7 + 0.1 * q)).clamp(0, 1)
            dist = (scaled.unsqueeze(-1) - cb.view(1, 1, -1)).abs()[0]  # [Tf, K]
            top2 = dist.topk(2, largest=False, dim=-1).values
            margins.append((top2[:, 1] - top2[:, 0]).clamp_min(0))
            codes.append(dist.argmin(dim=-1))
        return torch.stack(margins), torch.stack(codes)
