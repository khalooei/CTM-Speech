"""Tokenizer-independent waveform attacks (CTC-PGD and spectral-band PGD)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PGDConfig:
    eps: float = 0.02
    step_size: float = 0.002
    steps: int = 20
    band: Optional[str] = None  # None | low | mid | high
    targeted: bool = False


def _band_mask(n_fft: int, sample_rate: int, band: str) -> torch.Tensor:
    """Binary mask over rFFT bins for spectral-band PGD."""
    freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    if band == "low":
        return (freqs < 1000).float()
    if band == "mid":
        return ((freqs >= 1000) & (freqs < 4000)).float()
    if band == "high":
        return (freqs >= 4000).float()
    raise ValueError(f"unknown band: {band}")


class CTCPGDAttack:
    """
    Projected gradient descent on the waveform against a surrogate CTC ASR.
    The resulting waveform is reused across all tokenizers (fairness invariant).
    """

    def __init__(self, surrogate_asr, sample_rate: int = 16000):
        self.surrogate = surrogate_asr
        self.sample_rate = sample_rate

    def _ensure_surrogate_loaded(self) -> None:
        if hasattr(self.surrogate, "_lazy_load"):
            self.surrogate._lazy_load()

    def attack(
        self,
        waveform: torch.Tensor,
        text: str,
        config: PGDConfig,
        seed: int = 0,
    ) -> torch.Tensor:
        if hasattr(self.surrogate, "name") and self.surrogate.name.startswith("fake"):
            return self._fake_attack(waveform, config, seed)

        self._ensure_surrogate_loaded()
        device = getattr(self.surrogate, "device", "cpu")
        x0 = waveform.float().reshape(-1).to(device)
        if x0.ndim != 1:
            x0 = x0.reshape(-1)
        delta = torch.zeros_like(x0, requires_grad=True)
        torch.manual_seed(seed)

        for _ in range(config.steps):
            x_adv = (x0 + delta).clamp(-1.0, 1.0)
            loss = self._ctc_loss_tensor(x_adv, text)
            if config.targeted:
                loss = -loss
            loss.backward()
            with torch.no_grad():
                g = delta.grad
                if config.band is not None:
                    g = self._filter_grad_band(g, config.band)
                delta += config.step_size * g.sign()
                delta.clamp_(-config.eps, config.eps)
                delta.grad = None
        return (x0 + delta.detach()).clamp(-1.0, 1.0).cpu()

    def _fake_attack(self, waveform: torch.Tensor, config: PGDConfig, seed: int) -> torch.Tensor:
        g = torch.Generator()
        g.manual_seed(seed)
        x = waveform.float().reshape(-1)
        noise = torch.randn(x.shape, generator=g) * config.eps
        return (x + noise).clamp(-1.0, 1.0)

    def _ctc_loss_tensor(self, waveform: torch.Tensor, text: str) -> torch.Tensor:
        """Differentiable CTC loss through HuggingFace surrogate when available."""
        asr = self.surrogate
        from ctm.asr import normalize_text
        from ctm.resample import resample_waveform

        w = waveform
        if getattr(asr, "target_sr", self.sample_rate) != self.sample_rate:
            w = resample_waveform(w, self.sample_rate, asr.target_sr)
        processor = asr._processor
        model = asr._model
        # Processor expects numpy for feature extraction; keep differentiable path via model input_values approx
        # Use raw waveform scaled as wav2vec2 expects (approx).
        input_values = w.unsqueeze(0)
        # Normalize like wav2vec2
        input_values = (input_values - input_values.mean()) / (input_values.std() + 1e-7)
        logits = model(input_values).logits
        from ctm.asr import encode_ctc_labels

        labels = encode_ctc_labels(processor, text)
        if not labels:
            return waveform.pow(2).mean()
        labels_t = torch.tensor([labels], dtype=torch.long, device=waveform.device)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
        input_lengths = torch.full((1,), log_probs.size(0), dtype=torch.long, device=waveform.device)
        target_lengths = torch.tensor([len(labels)], dtype=torch.long, device=waveform.device)
        blank = model.config.pad_token_id if hasattr(model.config, "pad_token_id") else 0
        return F.ctc_loss(log_probs, labels_t, input_lengths, target_lengths, blank=blank, zero_infinity=True)

    def _filter_grad_band(self, grad: torch.Tensor, band: str) -> torch.Tensor:
        n_fft = 512
        flat = grad.reshape(-1)
        window = torch.hann_window(n_fft, device=flat.device)
        spec = torch.stft(
            flat.unsqueeze(0),
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            return_complex=True,
        )
        mask = _band_mask(n_fft, self.sample_rate, band).to(flat.device)
        # spec: [1, F, T] ; mask: [F]
        spec = spec * mask.view(1, -1, 1)
        out = torch.istft(
            spec,
            n_fft=n_fft,
            hop_length=n_fft // 4,
            window=window,
            length=flat.numel(),
        )
        return out.reshape_as(grad)
