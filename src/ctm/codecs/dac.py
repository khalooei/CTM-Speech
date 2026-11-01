"""DAC (descript-audio-codec) adapter — lazy import."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ctm.codecs.base import CodecBase, CodecInfo
from ctm.types import Cell, ensure_bqt


class DACCodec(CodecBase):
    def __init__(
        self,
        model_type: str = "44khz",
        bitrate: str = "8kbps",
        device: Optional[str] = None,
        n_quantizers: Optional[int] = None,
        load_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_quantizers = n_quantizers
        self._model = None
        self._model_type = model_type
        self._bitrate = bitrate
        self._load_path = load_path
        # Placeholders until load
        sr = {"44khz": 44100, "24khz": 24000, "16khz": 16000}.get(model_type, 44100)
        self.info = CodecInfo(
            name="dac",
            sample_rate=sr,
            n_codebooks=n_quantizers or 9,
            codebook_size=1024,
            supports_hybrid_decode=True,
            supports_margin=True,
            supports_displacement=True,
        )

    def _resolve_load_path(self) -> Optional[str]:
        if self._load_path:
            return self._load_path
        from pathlib import Path

        # Prefer locally downloaded weights (proxy-friendly).
        candidates = [
            Path("checkpoints/dac") / f"weights_{self._model_type}.pth",
            Path.home() / ".cache" / "descript" / "dac" / "weights" / f"weights_{self._model_type}.pth",
        ]
        for c in candidates:
            if c.exists() and c.stat().st_size > 0:
                return str(c.resolve())
        return None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from dac.utils import load_model
        except ImportError as e:
            raise ImportError(
                "descript-audio-codec is required for DACCodec. "
                "pip install descript-audio-codec"
            ) from e
        load_path = self._resolve_load_path()
        if load_path is not None:
            model = load_model(
                model_type=self._model_type,
                model_bitrate=self._bitrate,
                load_path=load_path,
            )
        else:
            model = load_model(model_type=self._model_type, model_bitrate=self._bitrate)
        model.to(self.device)
        model.eval()
        self._model = model
        self.info.sample_rate = int(model.sample_rate)
        self.info.n_codebooks = int(getattr(model, "n_codebooks", self.info.n_codebooks))
        self.info.codebook_size = int(getattr(model.quantizer.quantizers[0], "codebook_size", 1024))

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)
        # DAC expects [B, 1, T]
        if w.ndim == 2:
            w = w.unsqueeze(1)
        kwargs = {}
        if self.n_quantizers is not None:
            kwargs["n_quantizers"] = self.n_quantizers
        _, codes, _, _, _ = self._model.encode(w, **kwargs)
        # codes: [B, Q, T]
        return ensure_bqt(codes.cpu())

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes).to(self.device)
        # From codes to z via quantizer
        z = self._model.quantizer.from_codes(codes)[0]
        wav = self._model.decode(z)
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        return wav.cpu()

    # ---------------- projected-latent geometry ----------------
    def _quantizers(self):
        qs = list(self._model.quantizer.quantizers)
        if self.n_quantizers is not None:
            qs = qs[: int(self.n_quantizers)]
        return qs

    @torch.no_grad()
    def latent_contributions(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        import torch.nn.functional as F

        codes = ensure_bqt(codes).to(self.device)
        qs = self._quantizers()
        out = []
        for i in range(codes.shape[1]):
            vq = qs[i]
            e = F.embedding(codes[:, i], vq.codebook.weight).transpose(1, 2)  # [B, Dcb, T]
            out.append(vq.out_proj(e))  # [B, D, T]
        return torch.stack(out, dim=0)

    def decode_latent(self, z: torch.Tensor, keep_grad: bool = False) -> torch.Tensor:
        self._lazy_load()
        ctx = torch.enable_grad() if keep_grad else torch.no_grad()
        with ctx:
            wav = self._model.decode(z.to(self.device))
            if wav.ndim == 3:
                wav = wav.squeeze(1)
        return wav if keep_grad else wav.detach().cpu()

    @torch.no_grad()
    def stage_margins(self, waveform: torch.Tensor, sample_rate: int):
        """Replicates DAC's RVQ encode (normalised-space nearest codeword) and records margins."""
        from ctm.codecs._geometry import _codebook_nn_spacing, _margins_from_dist
        import torch.nn.functional as F

        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)
        if w.ndim == 2:
            w = w.unsqueeze(1)
        # DAC.encode (as used by encode()) applies no padding; mirror it exactly.
        z_e = self._model.encoder(w)
        residual = z_e
        margins, codes = [], []
        for vq in self._quantizers():
            z_in = vq.in_proj(residual)  # [B, Dcb, T]
            b, d, t = z_in.shape
            enc = F.normalize(z_in.permute(0, 2, 1).reshape(b * t, d), dim=-1)
            cbn = F.normalize(vq.codebook.weight, dim=-1)
            dist = torch.cdist(enc[None], cbn[None])[0]
            idx = dist.argmin(dim=-1)
            margins.append(_margins_from_dist(dist, _codebook_nn_spacing(cbn)).view(b, t)[0])
            codes.append(idx.view(b, t)[0])
            z_q = vq.out_proj(vq.decode_code(idx.view(b, t)))
            residual = residual - z_q
        return torch.stack(margins).cpu(), torch.stack(codes).cpu()
