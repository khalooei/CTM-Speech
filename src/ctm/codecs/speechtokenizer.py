"""SpeechTokenizer adapter — lazy import; canonicalizes QBT -> BQT."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch

from ctm.codecs.base import CodecBase, CodecInfo
from ctm.types import canonicalize_qbt_to_bqt, ensure_bqt


class SpeechTokenizerCodec(CodecBase):
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        ckpt_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config_path = Path(config_path) if config_path else None
        self.ckpt_path = Path(ckpt_path) if ckpt_path else None
        self._model = None
        self.info = CodecInfo(
            name="speechtokenizer",
            sample_rate=16000,
            n_codebooks=8,
            codebook_size=1024,
            supports_hybrid_decode=True,
            supports_margin=True,
            supports_displacement=True,
        )

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from speechtokenizer import SpeechTokenizer
        except ImportError as e:
            raise ImportError(
                "speechtokenizer package required. "
                "Run: bash scripts/download_models.sh  (clones ZhangXInFD/SpeechTokenizer)"
            ) from e
        if self.config_path is None or self.ckpt_path is None:
            raise FileNotFoundError(
                "SpeechTokenizerCodec requires config_path and ckpt_path. "
                "Run scripts/download_models.sh first."
            )
        # Official API: load_from_checkpoint; older forks may use load(...)
        if hasattr(SpeechTokenizer, "load_from_checkpoint"):
            model = SpeechTokenizer.load_from_checkpoint(
                str(self.config_path), str(self.ckpt_path)
            )
        elif hasattr(SpeechTokenizer, "load"):
            model = SpeechTokenizer.load(str(self.config_path), str(self.ckpt_path))
        else:
            raise AttributeError("SpeechTokenizer has neither load_from_checkpoint nor load")
        model = model.to(self.device)
        model.eval()
        self._model = model
        self.info.sample_rate = int(getattr(model, "sample_rate", 16000))
        self.info.n_codebooks = int(getattr(model, "n_q", 8))
        self.info.codebook_size = int(getattr(model, "codebook_size", 1024))

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)
        # SpeechTokenizer expects [B, T] or [B, 1, T] depending on version
        codes = self._model.encode(w.unsqueeze(1) if w.ndim == 2 else w)
        # Official API often returns [n_q, B, T]
        if codes.ndim == 3 and codes.shape[0] <= 32 and codes.shape[1] <= 8:
            codes = canonicalize_qbt_to_bqt(codes)
        return ensure_bqt(codes.cpu())

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes).to(self.device)
        # Model may expect [Q, B, T]
        qbt = codes.permute(1, 0, 2).contiguous()
        wav = self._model.decode(qbt)
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        return wav.cpu()

    # ---------------- projected-latent geometry ----------------
    @torch.no_grad()
    def latent_contributions(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes).to(self.device)
        layers = self._model.quantizer.vq.layers
        out = [layers[i].decode(codes[:, i]) for i in range(codes.shape[1])]  # each [B, D, T]
        return torch.stack(out, dim=0)

    def decode_latent(self, z: torch.Tensor, keep_grad: bool = False) -> torch.Tensor:
        self._lazy_load()
        ctx = torch.enable_grad() if keep_grad else torch.no_grad()
        # The decoder contains an LSTM; cuDNN RNN backward is unavailable in eval mode.
        with ctx, torch.backends.cudnn.flags(enabled=not keep_grad):
            wav = self._model.decoder(z.to(self.device))
            if wav.ndim == 3:
                wav = wav.squeeze(1)
        return wav if keep_grad else wav.detach().cpu()

    @torch.no_grad()
    def stage_margins(self, waveform: torch.Tensor, sample_rate: int):
        from ctm.codecs._geometry import _codebook_nn_spacing, _margins_from_dist

        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)
        x = w.unsqueeze(1) if w.ndim == 2 else w
        e = self._model.encoder(x)  # [B, D, T]
        residual = e
        margins, codes = [], []
        for layer in self._model.quantizer.vq.layers[: self.info.n_codebooks]:
            xi = layer.project_in(residual.permute(0, 2, 1))  # [B, T, Dcb]
            b, t, d = xi.shape
            flat = xi.reshape(b * t, d)
            cb = layer._codebook.embed
            dist = torch.cdist(flat[None], cb[None])[0]
            idx = dist.argmin(dim=-1)
            margins.append(_margins_from_dist(dist, _codebook_nn_spacing(cb)).view(b, t)[0])
            codes.append(idx.view(b, t)[0])
            residual = residual - layer.decode(idx.view(b, t))
        return torch.stack(margins).cpu(), torch.stack(codes).cpu()
