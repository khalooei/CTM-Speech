"""Codec base interface. Canonical codes are [B, Q, T] int tensors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from ctm.types import Cell, ensure_bqt


@dataclass
class CodecInfo:
    name: str
    sample_rate: int
    n_codebooks: int
    codebook_size: int
    hop_length: Optional[int] = None
    supports_hybrid_decode: bool = True
    supports_margin: bool = False
    supports_displacement: bool = False


class CodecBase(ABC):
    info: CodecInfo

    @abstractmethod
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Return codes [B, Q, T]. waveform is [T] or [B, T] or [B, 1, T]."""

    @abstractmethod
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode codes [B, Q, T] to waveform [B, T] at codec sample rate."""

    def encode_decode(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
        codes = self.encode(waveform, sample_rate)
        recon = self.decode(codes)
        return codes, recon

    # ------------------------------------------------------------------
    # Projected-latent geometry API (implemented by RVQ codecs).
    # decode(codes) == decode_latent(latent_contributions(codes).sum(0)) up to numerics.
    # ------------------------------------------------------------------
    def latent_contributions(self, codes: torch.Tensor) -> torch.Tensor:
        """Per-stage projected latent contributions [Q, B, D, T] (on the codec device)."""
        raise NotImplementedError(f"{self.info.name} does not implement latent_contributions")

    def decode_latent(self, z: torch.Tensor, keep_grad: bool = False) -> torch.Tensor:
        """Decode a summed latent [B, D, T] to waveform [B, T'] (device tensor if keep_grad)."""
        raise NotImplementedError(f"{self.info.name} does not implement decode_latent")

    def stage_margins(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Clean quantisation margins [Q, T] (runner-up minus nearest distance, normalised by the
        codebook's mean nearest-neighbour spacing) and the codes [Q, T] found on the way."""
        raise NotImplementedError(f"{self.info.name} does not implement stage_margins")

    def codeword_displacements(
        self,
        clean_codes: torch.Tensor,
        corrupt_codes: torch.Tensor,
    ) -> Dict[Cell, float]:
        """Geometry: L2 distance between clean/corrupt codeword vectors in projected latent space."""
        with torch.no_grad():
            zc = self.latent_contributions(clean_codes)
            zp = self.latent_contributions(corrupt_codes)
            d = (zc - zp).norm(dim=2)[:, 0].detach().cpu()  # [Q, T]
        a = ensure_bqt(clean_codes)[0]
        b = ensure_bqt(corrupt_codes)[0]
        qs, ts = torch.where(a != b)
        return {(int(q), int(t)): float(d[q, t]) for q, t in zip(qs.tolist(), ts.tolist())}

    def displacement_matrix(self, clean_codes: torch.Tensor, corrupt_codes: torch.Tensor) -> torch.Tensor:
        """[Q, T] projected-latent displacement (0 where codes agree)."""
        with torch.no_grad():
            zc = self.latent_contributions(clean_codes)
            zp = self.latent_contributions(corrupt_codes)
            return (zc - zp).norm(dim=2)[:, 0].detach().cpu()

    def quantization_margins(self, waveform: torch.Tensor, sample_rate: int) -> Dict[Cell, float]:
        """Clean VQ margin per cell (smaller = closer to a decision boundary)."""
        margins, _codes = self.stage_margins(waveform, sample_rate)
        m = margins.detach().cpu()
        return {(q, t): float(m[q, t]) for q in range(m.shape[0]) for t in range(m.shape[1])}

    def prepare_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        from ctm.resample import resample_waveform

        w = waveform.float()
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.ndim == 3:
            w = w.mean(dim=1)
        if sample_rate != self.info.sample_rate:
            w = resample_waveform(w, sample_rate, self.info.sample_rate)
        return w

    @staticmethod
    def as_bqt(codes: torch.Tensor) -> torch.Tensor:
        return ensure_bqt(codes)
