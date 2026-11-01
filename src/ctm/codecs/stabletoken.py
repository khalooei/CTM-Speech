"""StableToken adapter — official WhisperLFQEncoder + flow/HiFT decoder."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import torch
import torchaudio
from transformers import WhisperFeatureExtractor

from ctm.codecs.base import CodecBase, CodecInfo
from ctm.types import ensure_bqt


class StableTokenCodec(CodecBase):
    """
    Official Tencent StableToken path:
      tokenizer/ → WhisperLFQEncoder
      decoder/   → AudioDecoder (flow + HiFT)

    Codes are stored as [B, Q=1, T] integer tokens for CTM hybrid restore.
    Decode returns 16 kHz mono (resampled from decoder SR if needed).
    """

    def __init__(
        self,
        repo_dir: Optional[Union[str, Path]] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.repo_dir = Path(repo_dir) if repo_dir else Path("third_party/StableToken")
        self.checkpoint_dir = (
            Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints/StableToken")
        )
        self._tokenizer = None
        self._feature_extractor = None
        self._decoder = None
        self._target_sr = 16000
        self.decode_seed = 1234
        self.info = CodecInfo(
            name="stabletoken",
            sample_rate=16000,
            n_codebooks=1,
            codebook_size=8192,
            supports_hybrid_decode=True,
            supports_margin=False,
            supports_displacement=False,
        )

    def _ensure_paths(self) -> None:
        repo = self.repo_dir.resolve()
        if not repo.exists():
            raise FileNotFoundError(
                f"StableToken repo not found at {repo}. Run scripts/setup_stabletoken.sh"
            )
        # Official imports expect repo root + src/ + Matcha-TTS on sys.path
        for p in (
            str(repo),
            str(repo / "src"),
            str(repo / "third_party" / "Matcha-TTS"),
        ):
            if p not in sys.path:
                sys.path.insert(0, p)

    def _lazy_load(self) -> None:
        if self._tokenizer is not None:
            return
        self._ensure_paths()
        from src.model.modeling_whisper import WhisperLFQEncoder  # type: ignore
        from src.utils.flow_inference import AudioDecoder  # type: ignore

        tok_dir = self.checkpoint_dir / "tokenizer"
        dec_dir = self.checkpoint_dir / "decoder"
        if not tok_dir.exists() or not dec_dir.exists():
            raise FileNotFoundError(
                f"Expected {tok_dir} and {dec_dir}. Run scripts/setup_stabletoken.sh"
            )
        self._tokenizer = WhisperLFQEncoder.from_pretrained(str(tok_dir)).eval().to(self.device)
        self._feature_extractor = WhisperFeatureExtractor.from_pretrained(str(tok_dir))
        # transformers/from_pretrained leaves VotingLFQ.bit_weights corrupted (persistent=False
        # buffer). Re-init to [1,2,4,...,2^{d-1}] or indices explode and decode asserts.
        q = getattr(self._tokenizer, "quantizer", None)
        if q is not None and hasattr(q, "codebook_dim"):
            bw = 2 ** torch.arange(
                int(q.codebook_dim), device=self.device, dtype=torch.long
            )
            q.register_buffer("bit_weights", bw, persistent=False)
        self._decoder = AudioDecoder(
            config_path=str(dec_dir / "config.yaml"),
            flow_ckpt_path=str(dec_dir / "flow.pt"),
            hift_ckpt_path=str(dec_dir / "hift.pt"),
            device=self.device,
        )
        # codebook size from config when available
        cfg = getattr(self._tokenizer, "config", None)
        if cfg is not None:
            self.info.codebook_size = int(getattr(cfg, "quantize_vocab_size", None) or 8192)

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Official `extract_speech_token(..., require_only_quantized_token=True)` is broken under
        current transformers (returns garbage int64). Use full forward + `.quantized_token_ids`.
        """
        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate)  # [1, T]
        # mono numpy for WhisperFeatureExtractor
        wav_np = w.detach().cpu().float().reshape(-1).numpy()

        pooling_kernel_size = self._tokenizer.config.pooling_kernel_size or 1
        stride = (
            self._tokenizer.conv1.stride[0]
            * self._tokenizer.conv2.stride[0]
            * pooling_kernel_size
            * self._feature_extractor.hop_length
        )
        # Split into <=30s segments like the official util
        sr = self._target_sr
        tokens: list[int] = []
        t0 = 0
        while t0 * sr < wav_np.shape[0]:
            segment = wav_np[t0 * sr : (t0 + 30) * sr]
            features = self._feature_extractor(
                [segment],
                sampling_rate=sr,
                return_attention_mask=True,
                return_tensors="pt",
                padding="longest",
                pad_to_multiple_of=stride,
            )
            features = features.to(self.device)
            out = self._tokenizer(**features, require_only_quantized_token=False)
            q = out.quantized_token_ids  # [1, T']
            if q is None:
                raise RuntimeError("StableToken forward returned no quantized_token_ids")
            attn = features.attention_mask
            # Match official downsampling of attention mask to token rate
            attn = attn[:, :: self._tokenizer.conv1.stride[0] * self._tokenizer.conv2.stride[0]]
            attn = attn[:, :: self._tokenizer.config.pooling_kernel_size]
            if attn.shape != q.shape:
                # Fall back to length min if mask alignment drifts
                m = min(attn.shape[-1], q.shape[-1])
                attn = attn[:, :m]
                q = q[:, :m]
            ids = q[0][attn[0].bool()].detach().cpu().long()
            # Safety: valid embedding indices for flow decoder (vocab 8192)
            vocab = int(getattr(self._tokenizer.config, "quantize_vocab_size", 8192) or 8192)
            ids = ids.clamp(0, vocab - 1)
            tokens.extend(ids.tolist())
            t0 += 30

        if not tokens:
            codes = torch.zeros(1, 1, 1, dtype=torch.long)
        else:
            codes = torch.tensor(tokens, dtype=torch.long).view(1, 1, -1)
        return ensure_bqt(codes.cpu())

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode hybrid token grid [B,1,T] → waveform [B, samples] at 16 kHz."""
        self._lazy_load()
        from src.utils.utils import speech_token_to_wav  # type: ignore

        codes = ensure_bqt(codes)
        vocab = int(getattr(self._tokenizer.config, "quantize_vocab_size", 8192) or 8192)
        wavs = []
        for b in range(codes.shape[0]):
            token_ids = codes[b, 0].detach().cpu().long().clamp(0, vocab - 1).tolist()
            # The flow-matching decoder samples its initial noise: fix the seed per decode so that
            # identical codes decode to identical waveforms (required for counterfactual restoration)
            # and batch composition cannot change the result.
            torch.manual_seed(self.decode_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.decode_seed)
            tts_speech, sr = speech_token_to_wav(self._decoder, token_ids)
            if tts_speech.ndim == 1:
                tts_speech = tts_speech.unsqueeze(0)
            if int(sr) != self._target_sr:
                tts_speech = torchaudio.functional.resample(
                    tts_speech.float(), int(sr), self._target_sr
                )
            wavs.append(tts_speech.squeeze(0).cpu().float())
        maxlen = max(w.numel() for w in wavs)
        out = torch.zeros(len(wavs), maxlen)
        for i, w in enumerate(wavs):
            out[i, : w.numel()] = w
        return out
