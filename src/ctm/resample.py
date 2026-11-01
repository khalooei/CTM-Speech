"""Resampling helpers that do not require a working torchaudio CUDA extension."""

from __future__ import annotations

import numpy as np
import torch


def resample_waveform(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample [..., T] float waveform from orig_sr to target_sr."""
    if int(orig_sr) == int(target_sr):
        return wav
    # Prefer torchaudio when importable (matched CUDA build).
    try:
        import torchaudio

        squeezed = False
        w = wav
        if w.ndim == 1:
            w = w.unsqueeze(0)
            squeezed = True
        out = torchaudio.functional.resample(w, int(orig_sr), int(target_sr))
        return out.squeeze(0) if squeezed else out
    except Exception:
        pass
    # SciPy / polyphase fallback
    try:
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(int(orig_sr), int(target_sr))
        up, down = int(target_sr) // g, int(orig_sr) // g
        arr = wav.detach().cpu().float().numpy()
        if arr.ndim == 1:
            y = resample_poly(arr, up, down).astype(np.float32)
            return torch.from_numpy(y)
        ys = [resample_poly(arr[i], up, down).astype(np.float32) for i in range(arr.shape[0])]
        return torch.from_numpy(np.stack(ys, axis=0))
    except Exception:
        pass
    # Last resort: linear interpolation
    arr = wav.detach().cpu().float().numpy()
    n = arr.shape[-1]
    new_n = int(round(n * float(target_sr) / float(orig_sr)))
    x_old = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new = np.linspace(0.0, 1.0, new_n, endpoint=False)
    if arr.ndim == 1:
        y = np.interp(x_new, x_old, arr).astype(np.float32)
        return torch.from_numpy(y)
    ys = [np.interp(x_new, x_old, arr[i]).astype(np.float32) for i in range(arr.shape[0])]
    return torch.from_numpy(np.stack(ys, axis=0))
