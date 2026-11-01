"""EnCodec (facebook/encodec_24khz via transformers) adapter with projected-latent geometry."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ctm.codecs.base import CodecBase, CodecInfo
from ctm.types import ensure_bqt


class EncodecCodec(CodecBase):
    def __init__(self, model_name: str = "facebook/encodec_24khz", bandwidth: float = 6.0, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.bandwidth = float(bandwidth)
        self._model = None
        self.info = CodecInfo(name="encodec", sample_rate=24000, n_codebooks=8, codebook_size=1024, hop_length=320,
                              supports_hybrid_decode=True, supports_margin=True, supports_displacement=True)

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        from transformers import EncodecModel

        m = EncodecModel.from_pretrained(self.model_name).eval().to(self.device)
        self._model = m
        self.info.sample_rate = int(m.config.sampling_rate)
        self.info.codebook_size = int(m.config.codebook_size)
        # number of quantizers at this bandwidth
        self.info.n_codebooks = int(self.bandwidth * 1000 / (m.config.frame_rate * 10)) if hasattr(m.config, "frame_rate") else 8

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)
        if w.ndim == 2:
            w = w.unsqueeze(1)
        enc = self._model.encode(w, bandwidth=self.bandwidth)
        codes = enc.audio_codes  # [chunks, B, Q, T] (single chunk: non-chunked config)
        if codes.ndim == 4:
            codes = codes[0]
        self.info.n_codebooks = int(codes.shape[1])
        return ensure_bqt(codes.cpu())

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes).to(self.device)
        z = self._model.quantizer.decode(codes.transpose(0, 1))  # quantizer expects [Q, B, T]
        wav = self._model.decoder(z)
        return wav.squeeze(1).cpu()

    # ---- projected-latent geometry ----
    @torch.no_grad()
    def latent_contributions(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes).to(self.device)
        out = []
        for i in range(codes.shape[1]):
            layer = self._model.quantizer.layers[i]
            out.append(layer.decode(codes[:, i]))  # [B, D, T]
        return torch.stack(out, dim=0)

    def decode_latent(self, z: torch.Tensor, keep_grad: bool = False) -> torch.Tensor:
        self._lazy_load()
        ctx = torch.enable_grad() if keep_grad else torch.no_grad()
        with ctx, torch.backends.cudnn.flags(enabled=not keep_grad):
            wav = self._model.decoder(z.to(self.device)).squeeze(1)
        return wav if keep_grad else wav.detach().cpu()

    @torch.no_grad()
    def stage_margins(self, waveform: torch.Tensor, sample_rate: int):
        from ctm.codecs._geometry import _codebook_nn_spacing, _margins_from_dist

        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)
        if w.ndim == 2:
            w = w.unsqueeze(1)
        e = self._model.encoder(w)  # [B, D, T]
        residual = e
        margins, codes = [], []
        for layer in self._model.quantizer.layers[: self.info.n_codebooks]:
            cb = layer.codebook.embed  # [K, D]
            x = residual.permute(0, 2, 1)
            b, t, d = x.shape
            flat = x.reshape(b * t, d)
            dist = torch.cdist(flat[None], cb[None])[0]
            idx = dist.argmin(dim=-1)
            margins.append(_margins_from_dist(dist, _codebook_nn_spacing(cb)).view(b, t)[0])
            codes.append(idx.view(b, t)[0])
            residual = residual - layer.decode(idx.view(b, t))
        return torch.stack(margins).cpu(), torch.stack(codes).cpu()
