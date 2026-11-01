"""Waveform corruptions aligned with StableToken-style intrinsic protocols."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from ctm.types import ConditionSpec


def condition_seed(base_seed: int, utterance_id: str, condition_id: str) -> int:
    """Deterministic 31-bit seed from (base_seed, utterance_id, condition_id)."""
    payload = f"{base_seed}|{utterance_id}|{condition_id}".encode("utf-8")
    h = hashlib.sha256(payload).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _to_1d(wav: torch.Tensor) -> torch.Tensor:
    w = wav.float()
    if w.ndim == 2:
        w = w.mean(dim=0)
    return w


def mix_snr(clean: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Mix noise into clean at target SNR (dB)."""
    x = _to_1d(clean)
    n = _to_1d(noise)
    if n.numel() < x.numel():
        reps = int(np.ceil(x.numel() / max(n.numel(), 1)))
        n = n.repeat(reps)[: x.numel()]
    else:
        n = n[: x.numel()]
    x_pow = torch.mean(x**2).clamp_min(1e-12)
    n_pow = torch.mean(n**2).clamp_min(1e-12)
    scale = torch.sqrt(x_pow / (n_pow * (10 ** (snr_db / 10.0))))
    y = x + scale * n
    return y.clamp(-1.0, 1.0)


def gaussian_noise(wav: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    x = _to_1d(wav)
    g = _rng(seed)
    noise = torch.from_numpy(g.standard_normal(x.numel()).astype(np.float32))
    return mix_snr(x, noise, snr_db)


def _colored_noise(n: int, beta: float, seed: int) -> torch.Tensor:
    """Generate 1/f^beta noise via frequency-domain shaping."""
    g = _rng(seed)
    white = g.standard_normal(n).astype(np.float32)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1] if len(freqs) > 1 else 1.0
    shaped = spec / (freqs ** (beta / 2.0))
    out = np.fft.irfft(shaped, n=n).astype(np.float32)
    out = out / (np.std(out) + 1e-8)
    return torch.from_numpy(out)


def pink_noise(wav: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    x = _to_1d(wav)
    return mix_snr(x, _colored_noise(x.numel(), beta=1.0, seed=seed), snr_db)


def brown_noise(wav: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    x = _to_1d(wav)
    return mix_snr(x, _colored_noise(x.numel(), beta=2.0, seed=seed), snr_db)


def bit_crush(wav: torch.Tensor, bits: int = 10) -> torch.Tensor:
    x = _to_1d(wav).clamp(-1.0, 1.0)
    levels = float(2**bits - 1)
    return torch.round((x + 1.0) * 0.5 * levels) / levels * 2.0 - 1.0


def add_real_noise(
    wav: torch.Tensor,
    noise_paths: Sequence[Union[str, Path]],
    snr_db: float,
    seed: int,
    sample_rate: int = 16000,
    max_tries: int = 8,
) -> torch.Tensor:
    """Mix a real noise clip at the target SNR.

    The clip is resampled to `sample_rate` (ESC-50 is 44.1 kHz) and a random offset is drawn; offsets whose
    segment is nearly silent (which would need a scale that clips the mix) are re-drawn deterministically.
    """
    if not noise_paths:
        raise ValueError("add_real_noise requires at least one noise file")
    g = _rng(seed)
    x = _to_1d(wav)
    last = None
    for _ in range(max_tries):
        path = Path(noise_paths[int(g.integers(0, len(noise_paths)))])
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), always_2d=False)
            noise = torch.as_tensor(np.asarray(data, dtype=np.float32))
            if noise.ndim == 2:
                noise = noise.mean(dim=-1)
        except Exception as e:
            raise RuntimeError(f"failed to load noise file: {path}") from e
        if int(sr) != int(sample_rate):
            from ctm.resample import resample_waveform

            noise = resample_waveform(noise.reshape(-1), int(sr), int(sample_rate))
        n = noise.float().reshape(-1)
        if n.numel() == 0:
            continue
        if n.numel() > x.numel():
            start = int(g.integers(0, n.numel() - x.numel() + 1))
            n = n[start : start + x.numel()]
        # reject near-silent segments (< -50 dBFS RMS) which would require a clipping scale
        if float(n.pow(2).mean().sqrt()) < 10 ** (-50 / 20):
            last = n
            continue
        y = mix_snr(x, n, snr_db)
        if float((y.abs() >= 1.0).float().mean()) < 1e-3:
            return y
        last = n
    return mix_snr(x, last if last is not None else torch.zeros_like(x) + 1e-4, snr_db)


def _align_lag(ref: torch.Tensor, est: torch.Tensor, max_lag: int) -> int:
    """Integer lag (est is delayed by `lag` samples relative to ref), via FFT cross-correlation."""
    n = ref.numel() + est.numel()
    fa = torch.fft.rfft(ref.double(), n=n)
    fb = torch.fft.rfft(est.double(), n=n)
    xc = torch.fft.irfft(fb * fa.conj(), n=n)
    cand = torch.cat([xc[: max_lag + 1], xc[-max_lag:]])
    k = int(torch.argmax(cand).item())
    return k if k <= max_lag else k - (2 * max_lag + 1)


def opus_degrade(
    wav: torch.Tensor,
    sample_rate: int,
    bitrate: int = 16000,
    seed: int = 0,
) -> torch.Tensor:
    """Real Opus encode/decode round trip via ffmpeg (native encoder), lag-aligned to the input.

    Raises if ffmpeg is unavailable: a silent lowpass stand-in would misreport the condition.
    """
    import shutil
    import subprocess
    import tempfile

    import soundfile as sf

    x = _to_1d(wav)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found; the opus condition requires a real Opus round trip")
    with tempfile.TemporaryDirectory() as td:
        src = f"{td}/in.wav"
        enc = f"{td}/enc.opus"
        dec = f"{td}/dec.wav"
        sf.write(src, x.numpy(), sample_rate, subtype="FLOAT")
        cmd_enc = [
            ffmpeg, "-y", "-loglevel", "error", "-i", src,
            "-c:a", "libopus", "-b:a", str(int(bitrate)), "-vbr", "off", "-application", "voip", enc,
        ]
        r = subprocess.run(cmd_enc, capture_output=True)
        if r.returncode != 0:
            # Fall back to the experimental native encoder if libopus is not compiled in.
            cmd_enc = [
                ffmpeg, "-y", "-loglevel", "error", "-i", src,
                "-c:a", "opus", "-strict", "-2", "-b:a", str(int(bitrate)), enc,
            ]
            r = subprocess.run(cmd_enc, capture_output=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg opus encode failed: {r.stderr.decode(errors='ignore')[:300]}")
        cmd_dec = [ffmpeg, "-y", "-loglevel", "error", "-i", enc, "-ar", str(int(sample_rate)), "-ac", "1", "-f", "wav", "-c:a", "pcm_f32le", dec]
        r = subprocess.run(cmd_dec, capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg opus decode failed: {r.stderr.decode(errors='ignore')[:300]}")
        data, sr = sf.read(dec, always_2d=False, dtype="float32")
    y = torch.from_numpy(np.asarray(data, dtype=np.float32)).reshape(-1)
    # Align (Opus adds pre-skip/lookahead) and match length exactly.
    lag = _align_lag(x, y, max_lag=int(0.05 * sample_rate))
    if lag > 0:
        y = y[lag:]
    elif lag < 0:
        y = torch.cat([torch.zeros(-lag), y])
    if y.numel() < x.numel():
        y = torch.cat([y, torch.zeros(x.numel() - y.numel())])
    return y[: x.numel()].clamp(-1.0, 1.0)


def apply_corruption(
    wav: torch.Tensor,
    condition: ConditionSpec,
    *,
    base_seed: int,
    utterance_id: str,
    sample_rate: int = 16000,
    esc50_files: Optional[Sequence[Union[str, Path]]] = None,
    esc10_files: Optional[Sequence[Union[str, Path]]] = None,
    noise_banks: Optional[dict] = None,
) -> torch.Tensor:
    """Apply a named corruption. Condition ID must encode severity.

    `noise_banks` maps a kind name (e.g. "musan", "demand") to a list of wav paths."""
    seed = condition_seed(base_seed, utterance_id, condition.condition_id)
    kind = condition.kind.lower()
    p = condition.params

    if kind in {"gaussian", "white", "awgn"}:
        return gaussian_noise(wav, float(p["snr_db"]), seed)
    if kind == "pink":
        return pink_noise(wav, float(p["snr_db"]), seed)
    if kind == "brown":
        return brown_noise(wav, float(p["snr_db"]), seed)
    if kind in {"bitcrush", "bit_crush"}:
        return bit_crush(wav, int(p.get("bits", 10)))
    if kind in {"esc50", "real_noise", "esc-50"}:
        files = esc50_files or []
        return add_real_noise(wav, files, float(p["snr_db"]), seed, sample_rate=sample_rate)
    if kind in {"esc10", "ood_noise", "esc-10"}:
        files = esc10_files or esc50_files or []
        return add_real_noise(wav, files, float(p["snr_db"]), seed, sample_rate=sample_rate)
    if noise_banks and kind in noise_banks:
        return add_real_noise(wav, list(noise_banks[kind]), float(p["snr_db"]), seed, sample_rate=sample_rate)
    if kind == "opus":
        return opus_degrade(wav, sample_rate, bitrate=int(p.get("bitrate", 16000)), seed=seed)
    if kind in {"clean", "none"}:
        return _to_1d(wav).clone()
    raise ValueError(f"unknown corruption kind: {condition.kind}")


# StableToken intrinsic defaults (Table / Appendix protocol)
STABLETOKEN_CONDITIONS = [
    ConditionSpec("gaussian_snr25", "gaussian", {"snr_db": 25}),
    ConditionSpec("pink_snr22", "pink", {"snr_db": 22}),
    ConditionSpec("brown_snr16", "brown", {"snr_db": 16}),
    ConditionSpec("bitcrush_10bit", "bit_crush", {"bits": 10}),
    ConditionSpec("esc50_snr16", "esc50", {"snr_db": 16}),
    ConditionSpec("esc10_snr16", "esc10", {"snr_db": 16}),
]


def parse_condition(raw: dict) -> ConditionSpec:
    return ConditionSpec(
        condition_id=str(raw["condition_id"]),
        kind=str(raw["kind"]),
        params=dict(raw.get("params", {})),
    )
