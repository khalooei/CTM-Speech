"""Mimi (Kyutai moshi) codec adapter — lazy import."""

from __future__ import annotations

from typing import Optional

import torch

from ctm.codecs.base import CodecBase, CodecInfo
from ctm.types import ensure_bqt


class MimiCodec(CodecBase):
    def __init__(
        self,
        hf_repo: str = "kyutai/mimi",
        device: Optional[str] = None,
        num_codebooks: int = 8,
        load_path: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_repo = hf_repo
        self.num_codebooks = num_codebooks
        self.load_path = load_path
        self._model = None
        self.info = CodecInfo(
            name="mimi",
            sample_rate=24000,
            n_codebooks=num_codebooks,
            codebook_size=2048,
            hop_length=1920,  # 24kHz / 12.5 Hz
            supports_hybrid_decode=True,
            supports_margin=True,
            supports_displacement=True,
        )

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        # moshi uses torch.compile lazily; unsupported on Python 3.14+.
        import os
        from pathlib import Path

        os.environ.setdefault("NO_TORCH_COMPILE", "1")
        try:
            from moshi.models import loaders
        except ImportError as e:
            raise ImportError("moshi is required for MimiCodec. pip install moshi") from e

        # Prefer an explicit local checkpoint (offline-safe).
        local_candidates = []
        if self.load_path:
            local_candidates.append(Path(self.load_path))
        local_candidates.append(Path("checkpoints/mimi/tokenizer-e351c8d8-checkpoint125.safetensors"))
        checkpoint = None
        for cand in local_candidates:
            if cand.is_file():
                checkpoint = cand
                break

        if checkpoint is None:
            # moshi packages tokenizer weights with LM repos (e.g. moshiko), not kyutai/mimi.
            default_repo = getattr(loaders, "DEFAULT_REPO", "kyutai/moshiko-pytorch-bf16")
            repo = self.hf_repo
            if repo in {"kyutai/mimi", "mimi"}:
                repo = default_repo
            if hasattr(loaders, "hf_get"):
                checkpoint = loaders.hf_get(loaders.MIMI_NAME, hf_repo=repo)
            elif hasattr(loaders, "hf_get_path"):
                checkpoint = loaders.hf_get_path(repo, loaders.MIMI_NAME)
            else:
                from huggingface_hub import hf_hub_download

                checkpoint = hf_hub_download(repo, loaders.MIMI_NAME)

        model = loaders.get_mimi(str(checkpoint), device=self.device, num_codebooks=self.num_codebooks)
        if hasattr(model, "set_num_codebooks"):
            model.set_num_codebooks(self.num_codebooks)
        model.eval()
        self._model = model
        self.info.sample_rate = int(getattr(model, "sample_rate", 24000))
        self.info.n_codebooks = self.num_codebooks

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)
        if w.ndim == 2:
            w = w.unsqueeze(1)  # [B, 1, T]
        codes = self._model.encode(w)  # typically [B, Q, T]
        return ensure_bqt(codes.cpu())

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes).to(self.device)
        wav = self._model.decode(codes)
        if wav.ndim == 3:
            wav = wav.squeeze(1)
        return wav.cpu()

    # ---------------- projected-latent geometry ----------------
    def _stage_modules(self):
        """List of (rvq, layer) per active codebook in canonical stage order."""
        q = self._model.quantizer
        n_sem = int(getattr(q, "n_q_semantic", 1))
        out = []
        for i in range(self.num_codebooks):
            if i < n_sem:
                out.append((q.rvq_first, q.rvq_first.vq.layers[i]))
            else:
                out.append((q.rvq_rest, q.rvq_rest.vq.layers[i - n_sem]))
        return out

    @torch.no_grad()
    def latent_contributions(self, codes: torch.Tensor) -> torch.Tensor:
        self._lazy_load()
        codes = ensure_bqt(codes).to(self.device)
        out = []
        for i, (rvq, layer) in enumerate(self._stage_modules()[: codes.shape[1]]):
            out.append(rvq.output_proj(layer.decode(codes[:, i])))  # [B, D, T]
        return torch.stack(out, dim=0)

    def decode_latent(self, z: torch.Tensor, keep_grad: bool = False) -> torch.Tensor:
        self._lazy_load()
        m = self._model
        ctx = torch.enable_grad() if keep_grad else torch.no_grad()
        with ctx:
            emb = m._to_encoder_framerate(z.to(self.device))
            if m.decoder_transformer is not None:
                (emb,) = m.decoder_transformer(emb)
            wav = m.decoder(emb)
            if wav.ndim == 3:
                wav = wav.squeeze(1)
        return wav if keep_grad else wav.detach().cpu()

    @torch.no_grad()
    def stage_margins(self, waveform: torch.Tensor, sample_rate: int):
        from ctm.codecs._geometry import _codebook_nn_spacing, _margins_from_dist

        self._lazy_load()
        w = self.prepare_waveform(waveform, sample_rate).to(self.device)
        if w.ndim == 2:
            w = w.unsqueeze(1)
        emb = self._model._encode_to_unquantized_latent(w)  # [B, D, T]
        q = self._model.quantizer
        n_sem = int(getattr(q, "n_q_semantic", 1))
        margins, codes = [], []
        for rvq, n_layers in ((q.rvq_first, n_sem), (q.rvq_rest, self.num_codebooks - n_sem)):
            if n_layers <= 0:
                continue
            residual = rvq.input_proj(emb)
            for layer in rvq.vq.layers[:n_layers]:
                x = residual.permute(0, 2, 1)  # [B, T, D]
                x = layer.project_in(x)
                b, t, d = x.shape
                flat = x.reshape(b * t, d)
                cb = layer._codebook.embedding
                dist = torch.cdist(flat[None], cb[None])[0]
                idx = dist.argmin(dim=-1)
                margins.append(_margins_from_dist(dist, _codebook_nn_spacing(cb)).view(b, t)[0])
                codes.append(idx.view(b, t)[0])
                residual = residual - layer.decode(idx.view(b, t))
        return torch.stack(margins).cpu(), torch.stack(codes).cpu()
