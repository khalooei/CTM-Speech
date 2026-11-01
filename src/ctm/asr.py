"""ASR evaluators: continuous CTC loss for attribution; WER/CER for reporting."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

try:
    from jiwer import cer as jiwer_cer
    from jiwer import wer as jiwer_wer
except ImportError:  # pragma: no cover
    jiwer_wer = None
    jiwer_cer = None


def encode_ctc_labels(processor, text: str) -> list[int]:
    """Tokenize transcript for CTC (transformers v4/v5 compatible)."""
    norm = normalize_text(text).upper()
    tok = getattr(processor, "tokenizer", processor)
    if hasattr(tok, "encode"):
        ids = tok.encode(norm)
        # Some tokenizers wrap with bos/eos; wav2vec2 char tokenizer usually does not.
        return list(ids)
    if hasattr(processor, "as_target_processor"):
        with processor.as_target_processor():
            out = processor(norm)
            return list(out["input_ids"] if isinstance(out, dict) else out.input_ids)
    raise RuntimeError("Cannot encode CTC labels with this processor")


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if jiwer_wer is not None:
        return float(jiwer_wer(ref, hyp))
    # Fallback: token-level edit distance
    return _token_er(ref.split(), hyp.split())


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    if jiwer_cer is not None:
        return float(jiwer_cer(ref, hyp))
    return _token_er(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))


def _token_er(ref: Sequence[str], hyp: Sequence[str]) -> float:
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    prev = list(range(m + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * m
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / n


@dataclass
class ASROutput:
    loss: float
    hypothesis: str
    wer: Optional[float] = None
    cer: Optional[float] = None


class ASREvaluator:
    """Protocol-like base for continuous CTC loss + transcription."""

    name: str = "base"

    def ctc_loss(self, waveform: torch.Tensor, sample_rate: int, text: str) -> float:
        raise NotImplementedError

    def transcribe(self, waveform: torch.Tensor, sample_rate: int) -> str:
        raise NotImplementedError

    def evaluate(self, waveform: torch.Tensor, sample_rate: int, text: str) -> ASROutput:
        loss = self.ctc_loss(waveform, sample_rate, text)
        hyp = self.transcribe(waveform, sample_rate)
        return ASROutput(
            loss=float(loss),
            hypothesis=hyp,
            wer=compute_wer(text, hyp) if text else None,
            cer=compute_cer(text, hyp) if text else None,
        )


class FakeASREvaluator(ASREvaluator):
    """
    Deterministic fake evaluator for unit tests.
    Loss grows with L2 distance from a stored clean reference waveform,
    plus an optional additive bias keyed by waveform sum hash.
    """

    def __init__(self, name: str = "fake_asr"):
        self.name = name
        self._clean_refs: dict[str, torch.Tensor] = {}

    def set_clean_reference(self, key: str, waveform: torch.Tensor) -> None:
        w = waveform.float().reshape(-1).detach().cpu()
        self._clean_refs[key] = w

    def ctc_loss(self, waveform: torch.Tensor, sample_rate: int, text: str) -> float:
        w = waveform.float().reshape(-1).detach().cpu()
        key = text or "default"
        ref = self._clean_refs.get(key)
        if ref is None:
            # Loss from energy / roughness
            return float(w.pow(2).mean().sqrt().item())
        n = min(w.numel(), ref.numel())
        err = (w[:n] - ref[:n]).pow(2).mean().sqrt()
        return float(err.item())

    def ctc_loss_batch(self, waveforms: torch.Tensor, sample_rate: int, text: str):
        import numpy as np

        w = waveforms.float()
        if w.ndim == 1:
            w = w.unsqueeze(0)
        return np.asarray([self.ctc_loss(w[i], sample_rate, text) for i in range(w.shape[0])], dtype=float)

    def transcribe_batch(self, waveforms: torch.Tensor, sample_rate: int):
        w = waveforms.float()
        if w.ndim == 1:
            w = w.unsqueeze(0)
        return [self.transcribe(w[i], sample_rate) for i in range(w.shape[0])]

    def ctc_loss_differentiable(self, waveform: torch.Tensor, sample_rate: int, text: str) -> torch.Tensor:
        w = waveform.float().reshape(-1)
        ref = self._clean_refs.get(text or "default")
        if ref is None:
            return w.pow(2).mean().sqrt()
        n = min(w.numel(), ref.numel())
        return (w[:n] - ref[:n].to(w.device)).pow(2).mean().sqrt()

    def transcribe(self, waveform: torch.Tensor, sample_rate: int) -> str:
        # Produce a stable pseudo-transcript from energy buckets
        w = waveform.float().reshape(-1)
        energy = float(w.pow(2).mean().sqrt().item())
        if energy < 0.05:
            return "silence"
        if energy < 0.2:
            return "hello world"
        return "the quick brown fox"


class HuggingFaceCTCASR(ASREvaluator):
    """wav2vec2 / HuBERT-style CTC ASR via transformers (lazy import)."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: Optional[str] = None,
        target_sr: int = 16000,
    ):
        self.name = model_name
        self.model_name = model_name
        self.target_sr = target_sr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        import os

        # Ensure .env token aliases map to HF_TOKEN before hub I/O.
        try:
            from ctm.env_bootstrap import load_project_env

            load_project_env(quiet=True)
        except Exception:
            pass

        offline = os.environ.get("HF_HUB_OFFLINE", "0") in {"1", "true", "True"} or os.environ.get(
            "TRANSFORMERS_OFFLINE", "0"
        ) in {"1", "true", "True"}
        kwargs = {"local_files_only": True} if offline else {}
        # Prefer authenticated hub calls when a token is available (higher rate limits).
        token = (
            os.environ.get("HF_TOKEN")
            or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        )
        if token and not offline:
            kwargs["token"] = token
        name_l = self.model_name.lower()
        if "hubert" in name_l:
            from transformers import HubertForCTC, Wav2Vec2Processor

            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name, **kwargs)
            self._model = HubertForCTC.from_pretrained(self.model_name, **kwargs)
        else:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name, **kwargs)
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name, **kwargs)
        self._model.to(self.device)
        self._model.eval()

    def _prepare(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        from ctm.resample import resample_waveform

        w = waveform.float()
        if w.ndim == 2:
            w = w.mean(dim=0)
        if sample_rate != self.target_sr:
            w = resample_waveform(w, sample_rate, self.target_sr)
        return w

    # ------------------------------------------------------------------
    # Exact batched scoring for EQUAL-LENGTH waveforms (all hybrid decodes of one
    # utterance share a length, so no padding / attention-mask approximation is needed).
    # ------------------------------------------------------------------
    max_batch_seconds: float = 240.0  # audio seconds per forward pass (memory guard)

    def _normalize_batch(self, waveforms: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Replicates Wav2Vec2FeatureExtractor zero-mean/unit-var normalisation on-device."""
        from ctm.resample import resample_waveform

        w = waveforms.float()
        if w.ndim == 1:
            w = w.unsqueeze(0)
        if w.ndim == 3:
            w = w.mean(dim=1)
        if sample_rate != self.target_sr:
            w = resample_waveform(w, sample_rate, self.target_sr)
        w = w.to(self.device)
        fe = self._processor.feature_extractor
        if getattr(fe, "do_normalize", True):
            mean = w.mean(dim=-1, keepdim=True)
            var = w.var(dim=-1, keepdim=True, unbiased=False)
            w = (w - mean) / torch.sqrt(var + 1e-7)
        return w

    def _label_tensor(self, text: str) -> Optional[torch.Tensor]:
        labels = encode_ctc_labels(self._processor, text)
        if not labels:
            return None
        return torch.tensor(labels, dtype=torch.long, device=self.device)

    def _chunk_size(self, n_samples: int) -> int:
        dur = max(n_samples / float(self.target_sr), 1e-3)
        return max(1, int(self.max_batch_seconds / dur))

    @torch.no_grad()
    def logits_batch(self, waveforms: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._lazy_load()
        x = self._normalize_batch(waveforms, sample_rate)
        outs = []
        step = self._chunk_size(x.shape[-1])
        for i in range(0, x.shape[0], step):
            outs.append(self._model(x[i : i + step]).logits)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def ctc_loss_batch(self, waveforms: torch.Tensor, sample_rate: int, text: str):
        """Per-token-normalised CTC NLL for each waveform in [B, T] (equal lengths)."""
        import numpy as np

        self._lazy_load()
        labels = self._label_tensor(text)
        logits = self.logits_batch(waveforms, sample_rate)
        if labels is None:
            return np.full((logits.shape[0],), float("nan"))
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)  # [T, B, V]
        b = log_probs.shape[1]
        input_lengths = torch.full((b,), log_probs.size(0), dtype=torch.long, device=self.device)
        target_lengths = torch.full((b,), int(labels.numel()), dtype=torch.long, device=self.device)
        loss = F.ctc_loss(
            log_probs,
            labels.unsqueeze(0).repeat(b, 1),
            input_lengths,
            target_lengths,
            blank=self._blank_id(),
            reduction="none",
            zero_infinity=True,
        )
        return (loss / target_lengths.float()).detach().cpu().numpy().astype(float)

    @torch.no_grad()
    def transcribe_batch(self, waveforms: torch.Tensor, sample_rate: int) -> List[str]:
        logits = self.logits_batch(waveforms, sample_rate)
        ids = torch.argmax(logits, dim=-1)
        return list(self._processor.batch_decode(ids))

    def _blank_id(self) -> int:
        cfg = self._model.config
        return int(cfg.pad_token_id) if getattr(cfg, "pad_token_id", None) is not None else 0

    def ctc_loss_differentiable(self, waveform: torch.Tensor, sample_rate: int, text: str) -> torch.Tensor:
        """Differentiable per-token CTC NLL of a single on-device waveform tensor (grad flows to input)."""
        self._lazy_load()
        labels = self._label_tensor(text)
        x = self._normalize_batch(waveform.reshape(1, -1), sample_rate)
        logits = self._model(x).logits
        if labels is None:
            return logits.float().pow(2).mean()
        log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
        input_lengths = torch.full((1,), log_probs.size(0), dtype=torch.long, device=self.device)
        target_lengths = torch.tensor([int(labels.numel())], dtype=torch.long, device=self.device)
        return F.ctc_loss(
            log_probs, labels.unsqueeze(0), input_lengths, target_lengths, blank=self._blank_id(), zero_infinity=True
        )

    @torch.no_grad()
    def embed(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Mean-pooled last hidden state (L2-normalized)."""
        self._lazy_load()
        w = self._prepare(waveform, sample_rate)
        inputs = self._processor(w.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        # Prefer encoder submodule when present (Wav2Vec2ForCTC / HubertForCTC).
        if hasattr(self._model, "wav2vec2"):
            out = self._model.wav2vec2(input_values)
        elif hasattr(self._model, "hubert"):
            out = self._model.hubert(input_values)
        else:
            out = self._model(input_values, output_hidden_states=True)
            hid = out.hidden_states[-1]
            vec = hid.mean(dim=1).squeeze(0)
            return F.normalize(vec, dim=-1)
        hid = out.last_hidden_state  # [B,T,H]
        vec = hid.mean(dim=1).squeeze(0)
        return F.normalize(vec, dim=-1)

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor, sample_rate: int) -> str:
        self._lazy_load()
        w = self._prepare(waveform, sample_rate)
        inputs = self._processor(w.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        logits = self._model(input_values).logits
        ids = torch.argmax(logits, dim=-1)
        return self._processor.batch_decode(ids)[0]

    def ctc_loss(self, waveform: torch.Tensor, sample_rate: int, text: str) -> float:
        self._lazy_load()
        w = self._prepare(waveform, sample_rate)
        inputs = self._processor(w.cpu().numpy(), sampling_rate=self.target_sr, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        with torch.set_grad_enabled(True):
            logits = self._model(input_values).logits  # [B, T, V]
        labels = encode_ctc_labels(self._processor, text)
        if not labels:
            return float("nan")
        labels_t = torch.tensor([labels], dtype=torch.long, device=self.device)
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, V]
        input_lengths = torch.full((1,), log_probs.size(0), dtype=torch.long, device=self.device)
        target_lengths = torch.tensor([len(labels)], dtype=torch.long, device=self.device)
        loss = F.ctc_loss(
            log_probs,
            labels_t,
            input_lengths,
            target_lengths,
            blank=self._model.config.pad_token_id if hasattr(self._model.config, "pad_token_id") else 0,
            zero_infinity=True,
        )
        return float(loss.detach().item())


class EmbeddingCosineLoss:
    """
    Functional loss for E15-lite: 1 - cosine(emb(wav), emb(reference)).
    Reference should be the clean codec reconstruction (set once per utterance).
    """

    def __init__(self, encoder: HuggingFaceCTCASR):
        self.encoder = encoder
        self._ref: Optional[torch.Tensor] = None

    def set_reference(self, waveform: torch.Tensor, sample_rate: int) -> None:
        self._ref = self.encoder.embed(waveform, sample_rate).detach()

    def loss(self, waveform: torch.Tensor, sample_rate: int) -> float:
        if self._ref is None:
            raise RuntimeError("EmbeddingCosineLoss.set_reference() required before loss()")
        cur = self.encoder.embed(waveform, sample_rate)
        cos = float(torch.dot(self._ref, cur).clamp(-1, 1).item())
        return float(1.0 - cos)


def build_asr(name: str, device: Optional[str] = None) -> ASREvaluator:
    key = name.lower()
    if key in {"fake", "fake_asr"}:
        return FakeASREvaluator()
    return HuggingFaceCTCASR(model_name=name, device=device)
