"""WavTokenizer (single-codebook, 75 Hz, 24 kHz) adapter via the official repository."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

from ctm.codecs.base import CodecBase, CodecInfo
from ctm.types import ensure_bqt


class WavTokenizerCodec(CodecBase):
    def __init__(self, repo_dir: str = "third_party/WavTokenizer", config_path: Optional[str] = None, ckpt_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_dir = Path(repo_dir)
        self.config_path = config_path or str(self.repo_dir / "configs" / "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml")
        self.ckpt_path = ckpt_path
        self._model = None
        self.info = CodecInfo(name="wavtokenizer", sample_rate=24000, n_codebooks=1, codebook_size=4096, hop_length=320,
                              supports_hybrid_decode=True, supports_margin=False, supports_displacement=False)

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        repo = str(self.repo_dir.resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from decoder.pretrained import WavTokenizer  # type: ignore

        ckpt = self.ckpt_path
        if ckpt is None:
            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download("novateur/WavTokenizer-large-speech-75token", "wavtokenizer_large_speech_320_v2.ckpt")
        m = WavTokenizer.from_pretrained0802(self.config_path, ckpt)
        self._model = m.eval().to(self.device)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)  # [1, T]
        bw = torch.tensor([0], device=self.device)
        _, codes = self._model.encode_infer(w, bandwidth_id=bw)  # [K=1, B, L]
        codes = codes.permute(1, 0, 2)  # [B, K, L]
        return ensure_bqt(codes.cpu())

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes).to(self.device).permute(1, 0, 2)  # [K, B, L]
        feats = self._model.codes_to_features(codes)  # [B, C, L]
        bw = torch.tensor([0], device=self.device)
        wav = self._model.decode(feats, bandwidth_id=bw)  # [B, T]
        return wav.reshape(codes.shape[1], -1).cpu()
