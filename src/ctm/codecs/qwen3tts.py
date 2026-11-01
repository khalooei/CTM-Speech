"""Qwen3-TTS-Tokenizer-12Hz adapter (qwen-tts package; 16-layer multi-codebook at 12.5 Hz, causal ConvNet decoder)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from ctm.codecs.base import CodecBase, CodecInfo
from ctm.types import ensure_bqt


class Qwen3TTSCodec(CodecBase):
    def __init__(self, model_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._tok = None
        self.info = CodecInfo(name="qwen3tts", sample_rate=24000, n_codebooks=16, codebook_size=2048,
                              supports_hybrid_decode=True, supports_margin=False, supports_displacement=False)

    def _lazy_load(self) -> None:
        if self._tok is not None:
            return
        from qwen_tts import Qwen3TTSTokenizer

        self._tok = Qwen3TTSTokenizer.from_pretrained(self.model_name, device_map=self.device)
        self.info.sample_rate = int(self._tok.get_output_sample_rate())
        self._input_sr = int(self._tok.get_input_sample_rate())
        cfg = getattr(self._tok.model, "config", None)
        for name in ("codebook_size", "vocab_size"):
            v = getattr(cfg, name, None) if cfg is not None else None
            if isinstance(v, int) and v > 0:
                self.info.codebook_size = int(v)
                break

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._lazy_load()
        from ctm.resample import resample_waveform

        w = waveform.float().reshape(-1)
        if sample_rate != self._input_sr:
            w = resample_waveform(w, sample_rate, self._input_sr)
        enc = self._tok.encode(w.cpu().numpy().astype(np.float32), sr=self._input_sr)
        codes = enc.audio_codes[0]  # (T, Q)
        codes = torch.as_tensor(codes).long().T.unsqueeze(0)  # [1, Q, T]
        self.info.n_codebooks = int(codes.shape[1])
        return ensure_bqt(codes.cpu())

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes)
        batch = [{"audio_codes": codes[b].T.contiguous().to(self.device)} for b in range(codes.shape[0])]
        wavs, sr = self._tok.decode(batch)
        self.info.sample_rate = int(sr)
        ws = [torch.as_tensor(np.asarray(w, dtype=np.float32)).reshape(-1) for w in wavs]
        n = min(x.numel() for x in ws)
        return torch.stack([x[:n] for x in ws], dim=0)
