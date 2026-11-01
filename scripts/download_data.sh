#!/usr/bin/env bash
# Download the evaluation data (no torchaudio/CUDA needed) and write JSONL manifests.
#   minimal: LibriSpeech test-clean + ESC-50 (main grid)
#   full:    + LibriSpeech test-other + FLEURS en_us test (generalization checks)
# Usage: bash scripts/download_data.sh [minimal|full]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-minimal}"
export CTM_DATA_MODE="$MODE"

mkdir -p data/manifests data/librispeech data/esc50 data/fleurs

"${PYTHON:-python}" - <<'PY'
"""Download LibriSpeech + ESC-50 (+ optional FLEURS) with urllib only."""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(".").resolve()
MODE = os.environ.get("CTM_DATA_MODE", "minimal")

# Official OpenSLR mirrors
LIBRISPEECH_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}


def _urlopen(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "ctm-download/0.1"})
    return urllib.request.urlopen(req, timeout=600)


def download_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest}")
        return dest
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    with _urlopen(url) as resp, tmp.open("wb") as f:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dest)
    return dest


def build_librispeech_manifest(libri_root: Path, split: str, out_path: Path) -> int:
    split_dir = libri_root / split
    if not split_dir.exists():
        # accept either data/librispeech/LibriSpeech/<split> or data/librispeech/<split>
        alt = ROOT / "data" / "librispeech" / "LibriSpeech" / split
        split_dir = alt if alt.exists() else split_dir
    rows = []
    for flac in sorted(split_dir.rglob("*.flac")):
        utt_id = flac.stem
        speaker = flac.parent.parent.name
        text = ""
        for tp in flac.parent.glob("*.trans.txt"):
            with tp.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2 and parts[0] == utt_id:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def download_librispeech(split: str) -> None:
    out = ROOT / "data" / "librispeech"
    out.mkdir(parents=True, exist_ok=True)
    extract_root = out  # tar contains LibriSpeech/<split>/...
    marker = out / "LibriSpeech" / split
    man = ROOT / "data" / "manifests" / f"librispeech_{split.replace('-', '_')}.jsonl"
    if marker.exists() and any(marker.rglob("*.flac")):
        n = build_librispeech_manifest(out / "LibriSpeech", split, man)
        print(f"LibriSpeech {split} already present ({n} utts) -> {man}")
        return
    url = LIBRISPEECH_URLS[split]
    archive = out / f"{split}.tar.gz"
    print(f"Downloading LibriSpeech {split} ...")
    download_file(url, archive)
    print(f"  extracting {archive.name} ...")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=extract_root)
    n = build_librispeech_manifest(out / "LibriSpeech", split, man)
    print(f"  wrote {man} ({n} rows)")


def download_esc50() -> None:
    dest = ROOT / "data" / "esc50"
    dest.mkdir(parents=True, exist_ok=True)
    audio = dest / "audio"
    if audio.exists() and any(audio.glob("*.wav")):
        print("ESC-50 already present")
        return
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    archive = dest / "esc50-master.zip"
    print("Downloading ESC-50 ...")
    download_file(url, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest / "_tmp")
    nested = next((dest / "_tmp").glob("ESC-50-*"))
    for item in nested.iterdir():
        target = dest / item.name
        if target.exists():
            continue
        item.rename(target)
    print("ESC-50 ready")


def download_fleurs() -> None:
    """Download FLEURS en_us test without requiring torchcodec audio decoding."""
    try:
        from datasets import Audio, load_dataset
        import soundfile as sf
        import numpy as np
    except ImportError as e:
        print(f"FLEURS skipped (missing dependency): {e}")
        return
    print("Downloading FLEURS en_us test ...")
    # Keep audio as file path bytes to avoid torchcodec decode path.
    ds = load_dataset("google/fleurs", "en_us", split="test")
    try:
        ds = ds.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    audio_dir = ROOT / "data" / "fleurs" / "en_us" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    man = ROOT / "data" / "manifests" / "fleurs_en_us.jsonl"
    rows = []
    for i, ex in enumerate(ds):
        utt_id = f"fleurs_en_us_{i:05d}"
        path = audio_dir / f"{utt_id}.wav"
        audio = ex.get("audio")
        try:
            if isinstance(audio, dict) and audio.get("bytes"):
                path.write_bytes(audio["bytes"])
                # ensure wav; if not, try soundfile roundtrip later
            elif isinstance(audio, dict) and audio.get("path"):
                import shutil
                src = Path(audio["path"])
                if src.exists():
                    shutil.copy(src, path)
                else:
                    continue
            elif isinstance(audio, dict) and "array" in audio:
                arr = np.asarray(audio["array"], dtype="float32")
                sr = int(audio.get("sampling_rate", 16000))
                sf.write(str(path), arr, sr)
            else:
                continue
            # Normalize to wav mono via soundfile if needed
            try:
                data, sr = sf.read(str(path), always_2d=False)
                sf.write(str(path), np.asarray(data, dtype="float32"), int(sr))
            except Exception:
                pass
            rows.append(
                {
                    "utterance_id": utt_id,
                    "path": str(path.resolve()),
                    "text": ex.get("transcription") or ex.get("raw_transcription") or "",
                    "speaker_id": str(ex.get("id", "")),
                    "split": "fleurs_en_us_test",
                }
            )
        except Exception as e:
            print(f"skip fleurs {i}: {e}")
            continue
    with man.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {man} ({len(rows)} rows)")


download_librispeech("test-clean")
download_esc50()
if MODE == "full":
    download_librispeech("test-other")
    download_fleurs()
print("Done. DEMAND is not auto-downloaded; place a licensed copy under data/demand if needed.")
PY
