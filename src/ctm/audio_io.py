"""Audio loading, resampling, and dataset manifests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np
import torch

from ctm.resample import resample_waveform
from ctm.types import Utterance


def load_waveform(path: str | Path, target_sr: Optional[int] = None) -> tuple[torch.Tensor, int]:
    """Load mono float32 waveform. Prefer soundfile (avoids broken torchaudio CUDA builds)."""
    path = Path(path)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=False)
        wav = torch.as_tensor(np.asarray(data, dtype=np.float32))
        if wav.ndim == 2:
            wav = wav.mean(dim=-1)
    except Exception as primary:
        # Optional torchaudio fallback only if the extension loads.
        try:
            import torchaudio

            wav, sr = torchaudio.load(str(path))
            if wav.ndim == 2 and wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0).float()
        except Exception as e:
            raise RuntimeError(f"failed to load audio: {path}") from e
    if target_sr is not None and int(sr) != int(target_sr):
        wav = resample_waveform(wav.reshape(-1), int(sr), int(target_sr))
        sr = target_sr
    return wav.float().reshape(-1), int(sr)


def write_manifest(path: str | Path, rows: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_manifest(path: str | Path) -> List[dict]:
    rows: List[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_utterances_from_manifest(
    manifest_path: str | Path,
    target_sr: int = 16000,
    max_utterances: Optional[int] = None,
) -> Iterator[Utterance]:
    rows = read_manifest(manifest_path)
    if max_utterances is not None:
        rows = rows[: max_utterances]
    for row in rows:
        wav, sr = load_waveform(row["path"], target_sr=target_sr)
        yield Utterance(
            utterance_id=str(row["utterance_id"]),
            waveform=wav,
            sample_rate=sr,
            text=row.get("text", ""),
            speaker_id=row.get("speaker_id"),
            path=row.get("path"),
        )


def build_librispeech_manifest(
    root: str | Path,
    split: str = "test-clean",
    out_path: Optional[str | Path] = None,
) -> List[dict]:
    """Scan a LibriSpeech-style tree and build a JSONL manifest."""
    root = Path(root)
    split_dir = root / split if (root / split).exists() else root
    rows: List[dict] = []
    for flac in sorted(split_dir.rglob("*.flac")):
        utt_id = flac.stem
        speaker = flac.parent.parent.name if flac.parent.parent != split_dir else None
        trans_path = flac.with_suffix(".trans.txt")
        # LibriSpeech stores transcripts as SPEAKER-CHAPTER.trans.txt in chapter dir
        chapter_trans = list(flac.parent.glob("*.trans.txt"))
        text = ""
        for tp in chapter_trans:
            with tp.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if parts and parts[0] == utt_id and len(parts) == 2:
                        text = parts[1]
                        break
            if text:
                break
        rows.append(
            {
                "utterance_id": utt_id,
                "path": str(flac.resolve()),
                "text": text,
                "speaker_id": speaker,
                "split": split,
            }
        )
    if out_path is not None:
        write_manifest(out_path, rows)
    return rows


def list_esc50_files(esc_root: str | Path) -> List[Path]:
    root = Path(esc_root)
    audio = root / "audio"
    if not audio.exists():
        audio = root
    return sorted(audio.glob("*.wav"))


def esc10_subset(esc_files: Sequence[Path], meta_csv: Optional[Path] = None) -> List[Path]:
    """Return ESC-10 subset if meta CSV is available; else first 400 files as fallback marker."""
    if meta_csv is None:
        # Try default location next to audio/
        candidates = [
            Path(esc_files[0]).parent.parent / "meta" / "esc50.csv" if esc_files else None,
        ]
        for c in candidates:
            if c is not None and c.exists():
                meta_csv = c
                break
    if meta_csv is None or not meta_csv.exists():
        return list(esc_files)[:400]
    esc10: List[Path] = []
    audio_dir = Path(esc_files[0]).parent if esc_files else meta_csv.parent.parent / "audio"
    with meta_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("esc10", "")).lower() in {"true", "1", "yes"}:
                p = audio_dir / row["filename"]
                if p.exists():
                    esc10.append(p)
    return esc10


def peak_normalize(wav: torch.Tensor, peak: float = 0.99) -> torch.Tensor:
    w = wav.float()
    m = w.abs().max().clamp_min(1e-8)
    return (w / m) * peak


def to_numpy_mono(wav: torch.Tensor) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        w = wav.detach().cpu().float()
        if w.ndim == 2:
            w = w.mean(dim=0)
        return w.numpy()
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr
