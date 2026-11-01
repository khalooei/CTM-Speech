"""Shared types, status enums, and small tensor helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Canonical discrete codes: [B, Q, T]
Codes = torch.Tensor
Cell = Tuple[int, int]  # (q, t) within a single batch item
CellList = List[Cell]


class CTMStatus(str, Enum):
    OK = "ok"
    SKIPPED_LENGTH_MISMATCH = "skipped_length_mismatch"
    UNDEFINED_NO_POSITIVE_FUNCTIONAL_DEGRADATION = (
        "undefined_no_positive_functional_degradation"
    )
    SKIPPED_EMPTY_CHANGED_SET = "skipped_empty_changed_set"
    ERROR = "error"


class RowStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"


def ensure_bqt(codes: torch.Tensor, name: str = "codes") -> torch.Tensor:
    """Ensure codes are integer [B, Q, T]. Accepts [Q, T] or [B, Q, T]."""
    if not isinstance(codes, torch.Tensor):
        codes = torch.as_tensor(codes)
    if codes.ndim == 2:
        codes = codes.unsqueeze(0)
    if codes.ndim != 3:
        raise ValueError(f"{name} must be [B,Q,T] or [Q,T], got shape {tuple(codes.shape)}")
    return codes.long()


def canonicalize_qbt_to_bqt(codes: torch.Tensor) -> torch.Tensor:
    """Convert SpeechTokenizer-style [Q, B, T] to canonical [B, Q, T]."""
    if codes.ndim != 3:
        raise ValueError(f"expected 3D codes, got {tuple(codes.shape)}")
    # Heuristic: if dim0 looks like n_q (<=32) and dim1 looks like batch, treat as QBT.
    if codes.shape[0] <= 32 and codes.shape[1] <= 8:
        return codes.permute(1, 0, 2).contiguous().long()
    return codes.long()


def changed_cells(clean: torch.Tensor, corrupt: torch.Tensor) -> CellList:
    """Return list of (q, t) where clean[0] != corrupt[0]. Requires equal shapes."""
    clean = ensure_bqt(clean)
    corrupt = ensure_bqt(corrupt)
    if clean.shape != corrupt.shape:
        raise ValueError(
            f"shape mismatch for changed_cells: {tuple(clean.shape)} vs {tuple(corrupt.shape)}"
        )
    diff = clean[0] != corrupt[0]
    qs, ts = torch.where(diff)
    return [(int(q), int(t)) for q, t in zip(qs.tolist(), ts.tolist())]


def restore_cells(
    corrupt: torch.Tensor,
    clean: torch.Tensor,
    cells: Sequence[Cell],
) -> torch.Tensor:
    """Return a clone of corrupt with selected cells restored from clean."""
    out = ensure_bqt(corrupt).clone()
    clean = ensure_bqt(clean)
    for q, t in cells:
        out[:, q, t] = clean[:, q, t]
    return out


@dataclass
class Utterance:
    utterance_id: str
    waveform: torch.Tensor  # [T] or [1, T] float32
    sample_rate: int
    text: str = ""
    speaker_id: Optional[str] = None
    path: Optional[str] = None

    def mono(self) -> torch.Tensor:
        w = self.waveform
        if w.ndim == 2:
            w = w.mean(dim=0)
        return w.float()


@dataclass
class ConditionSpec:
    condition_id: str
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"condition_id": self.condition_id, "kind": self.kind, "params": dict(self.params)}


@dataclass
class CTMResult:
    ctm_status: str
    ctm50: Optional[float] = None
    ctm80: Optional[float] = None
    ctm90: Optional[float] = None
    recovery_fractions: List[float] = field(default_factory=list)
    recovery_values: List[float] = field(default_factory=list)
    ranking_name: str = "intervention"
    n_changed: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def result_key(utterance_id: str, codec: str, condition: str) -> str:
    return f"{utterance_id}||{codec}||{condition}"


ArrayLike = Union[np.ndarray, torch.Tensor, Sequence[float]]
