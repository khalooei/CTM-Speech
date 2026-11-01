"""Codec registry / factory."""

from __future__ import annotations

from typing import Any, Dict, List

from ctm.codecs.base import CodecBase


def list_codecs() -> List[str]:
    return ["fake", "dac", "mimi", "speechtokenizer", "stabletoken", "encodec", "wavtokenizer", "qwen3tts"]


def build_codec(name: str, **kwargs: Any) -> CodecBase:
    key = name.lower().strip()
    if key in {"fake", "fake_codec"}:
        from ctm.codecs.fake import FakeCodec

        return FakeCodec(**kwargs)
    if key == "dac":
        from ctm.codecs.dac import DACCodec

        return DACCodec(**kwargs)
    if key == "mimi":
        from ctm.codecs.mimi import MimiCodec

        return MimiCodec(**kwargs)
    if key in {"speechtokenizer", "speech_tokenizer", "st"}:
        from ctm.codecs.speechtokenizer import SpeechTokenizerCodec

        return SpeechTokenizerCodec(**kwargs)
    if key == "encodec":
        from ctm.codecs.encodec import EncodecCodec

        return EncodecCodec(**kwargs)
    if key == "wavtokenizer":
        from ctm.codecs.wavtokenizer import WavTokenizerCodec

        return WavTokenizerCodec(**kwargs)
    if key in {"qwen3tts", "qwen3_tts_tokenizer", "qwen"}:
        from ctm.codecs.qwen3tts import Qwen3TTSCodec

        return Qwen3TTSCodec(**kwargs)
    if key in {"stabletoken", "stable_token"}:
        from ctm.codecs.stabletoken import StableTokenCodec

        return StableTokenCodec(**kwargs)
    raise ValueError(f"unknown codec: {name}. choose from {list_codecs()}")
