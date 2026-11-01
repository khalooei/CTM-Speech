"""Hierarchical (frame-then-cell) counterfactual restoration with held-out scoring.

Protocol (avoids the capped-subset and in-sample-selection problems analysed in Sec. 3 of the paper):

1. The changed set D = {(q, t): Z[q,t] != Z'[q,t]} is never subsampled. Every ranking is a
   permutation of all of D, so restoring the full prefix returns Z and Recovery(D) = 1 by
   construction.
2. Oracle influence is measured hierarchically: first per changed *frame* (restore all changed
   cells at time t), then per *cell* inside the top frames. Cost is O(|F| + Q * M) decodes.
3. Ranking is scored with one model (the "rank" scorer, e.g. the wav2vec2-base surrogate) and
   every recovery curve is scored with an independent evaluator (the "eval" scorer, e.g.
   HuBERT-large). The in-sample curve (rank scorer) is stored as well so the selection bias can
   be quantified.
4. Recovery curves use a log-spaced prefix grid so resolution is fine where curves are steep.

All hybrid decodes of one utterance have identical length, so batched scoring is exact.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ctm.types import ensure_bqt

FIB = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711]
FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]


def curve_grid(n: int, extra: Sequence[int] = ()) -> List[int]:
    """Prefix sizes 0..n: Fibonacci counts + fixed fractions of n + endpoints (+ extra)."""
    if n <= 0:
        return [0]
    pts = {0, n}
    pts.update(k for k in FIB if k < n)
    pts.update(int(round(f * n)) for f in FRACTIONS)
    pts.update(int(k) for k in extra)
    return sorted(p for p in pts if 0 <= p <= n)


@dataclass
class ChangedSet:
    """Changed cells of one (clean, corrupt) code pair, with frame grouping."""

    q: np.ndarray  # [n] stage index per changed cell
    t: np.ndarray  # [n] frame index per changed cell
    n_q: int
    n_t: int

    @property
    def n(self) -> int:
        return int(self.q.shape[0])

    @property
    def frames(self) -> np.ndarray:
        return np.unique(self.t)

    def cells_in_frame(self, frame: int) -> np.ndarray:
        return np.nonzero(self.t == frame)[0]

    @classmethod
    def from_codes(cls, clean: torch.Tensor, corrupt: torch.Tensor) -> "ChangedSet":
        a = ensure_bqt(clean)[0]
        b = ensure_bqt(corrupt)[0]
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
        qs, ts = torch.where(a != b)
        return cls(q=qs.cpu().numpy().astype(np.int64), t=ts.cpu().numpy().astype(np.int64), n_q=int(a.shape[0]), n_t=int(a.shape[1]))


class HybridOracle:
    """Batched decode + multi-scorer loss for arbitrary restored subsets of D."""

    def __init__(
        self,
        clean: torch.Tensor,
        corrupt: torch.Tensor,
        changed: ChangedSet,
        decode_fn: Callable[[torch.Tensor], torch.Tensor],
        scorers: Dict[str, Callable[[torch.Tensor], np.ndarray]],
        decode_batch_size: int = 32,
    ):
        self.clean = ensure_bqt(clean)
        self.corrupt = ensure_bqt(corrupt)
        self.changed = changed
        self.decode_fn = decode_fn
        self.scorers = scorers
        self.batch_size = max(1, int(decode_batch_size))
        self.n_decodes = 0
        self.decode_seconds = 0.0
        self.score_seconds = 0.0

    def hybrid_from_indices(self, idx: np.ndarray) -> torch.Tensor:
        """Corrupt codes with changed cells `idx` (indices into D) restored to clean."""
        out = self.corrupt.clone()
        if idx.size:
            q = torch.as_tensor(self.changed.q[idx])
            t = torch.as_tensor(self.changed.t[idx])
            out[0, q, t] = self.clean[0, q, t]
        return out

    def evaluate(self, subsets: Sequence[np.ndarray], scorer_names: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
        """Losses under the requested scorers (default: all) for each restored subset (arrays of D-indices)."""
        names = list(self.scorers) if scorer_names is None else [n for n in scorer_names if n in self.scorers]
        results = {name: np.empty(len(subsets), dtype=float) for name in names}
        for start in range(0, len(subsets), self.batch_size):
            chunk = subsets[start : start + self.batch_size]
            hybrids = torch.cat([self.hybrid_from_indices(np.asarray(s, dtype=np.int64)) for s in chunk], dim=0)
            t0 = time.time()
            wavs = self.decode_fn(hybrids)  # [b, T']
            if wavs.ndim == 3:
                wavs = wavs.squeeze(1)
            self.n_decodes += int(wavs.shape[0])
            self.decode_seconds += time.time() - t0
            t0 = time.time()
            for name in names:
                results[name][start : start + len(chunk)] = np.asarray(self.scorers[name](wavs), dtype=float)
            self.score_seconds += time.time() - t0
        return results


@dataclass
class Ranking:
    name: str
    order: np.ndarray  # permutation of range(n) (indices into D)
    scores: Optional[np.ndarray] = None  # per-cell score aligned with D (not with order), if any
    meta: Dict = field(default_factory=dict)

    def check(self, n: int) -> None:
        if self.order.shape[0] != n or np.unique(self.order).shape[0] != n:
            raise ValueError(f"ranking {self.name} is not a permutation of D (n={n})")


# ----------------------------------------------------------------------------
# Rankings
# ----------------------------------------------------------------------------


def _order_frames_then_cells(changed: ChangedSet, frame_order: Sequence[int], cell_key: Optional[np.ndarray] = None) -> np.ndarray:
    """Concatenate cells frame by frame (frames in given order); within a frame by cell_key asc (default: stage)."""
    key = changed.q.astype(float) if cell_key is None else cell_key
    parts = []
    for f in frame_order:
        idx = changed.cells_in_frame(int(f))
        parts.append(idx[np.argsort(key[idx], kind="stable")])
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def random_ranking(changed: ChangedSet, seed: int, name: str = "random") -> Ranking:
    rng = np.random.default_rng(seed)
    return Ranking(name=name, order=rng.permutation(changed.n))


def rvq_early_ranking(changed: ChangedSet) -> Ranking:
    order = np.lexsort((changed.t, changed.q))  # by stage, then time
    return Ranking(name="rvq_early", order=order)


def cell_score_ranking(changed: ChangedSet, scores: np.ndarray, name: str, descending: bool = True) -> Ranking:
    s = np.asarray(scores, dtype=float)
    s = np.where(np.isfinite(s), s, -np.inf if descending else np.inf)
    order = np.argsort(-s if descending else s, kind="stable")
    return Ranking(name=name, order=order, scores=np.asarray(scores, dtype=float))


def frame_score_ranking(changed: ChangedSet, frame_scores: Dict[int, float], name: str) -> Ranking:
    frames = changed.frames
    vals = np.array([frame_scores.get(int(f), -np.inf) for f in frames])
    frame_order = frames[np.argsort(-vals, kind="stable")]
    return Ranking(name=name, order=_order_frames_then_cells(changed, frame_order), meta={"frame_scores": {int(f): float(frame_scores.get(int(f), np.nan)) for f in frames}})


def perturbation_energy_frame_scores(x_clean: torch.Tensor, x_corrupt: torch.Tensor, n_frames: int) -> Dict[int, float]:
    """Signal-domain heuristic: energy of (corrupt - clean) inside each codec frame."""
    d = (x_corrupt.reshape(-1).float() - x_clean.reshape(-1).float()) ** 2
    n = d.numel()
    edges = np.linspace(0, n, n_frames + 1).astype(int)
    out = {}
    for f in range(n_frames):
        seg = d[edges[f] : max(edges[f + 1], edges[f] + 1)]
        out[f] = float(seg.mean().item()) if seg.numel() else 0.0
    return out


@dataclass
class HierarchicalResult:
    ranking: Ranking
    frame_block: Ranking
    frame_influence: Dict[int, float]
    refined_cells: np.ndarray  # D-indices with single-cell influence
    refined_influence: np.ndarray
    n_frame_decodes: int
    n_cell_decodes: int


def hierarchical_oracle_ranking(
    oracle: HybridOracle,
    l_corrupt_rank: float,
    rank_scorer: str = "rank",
    refine_top_frames: int = 16,
    refine_max_cells: int = 256,
) -> HierarchicalResult:
    """Frame-level oracle influence, then cell-level refinement inside the top frames."""
    ch = oracle.changed
    frames = ch.frames
    frame_sets = [ch.cells_in_frame(int(f)) for f in frames]
    res = oracle.evaluate(frame_sets, scorer_names=[rank_scorer])[rank_scorer]
    frame_infl = {int(f): float(l_corrupt_rank - l) for f, l in zip(frames, res)}
    vals = np.array([frame_infl[int(f)] for f in frames])
    frame_order = frames[np.argsort(-vals, kind="stable")]

    # refinement inside the top frames (bounded by a cell budget)
    top = []
    n_cells = 0
    for f in frame_order[: max(0, int(refine_top_frames))]:
        idx = ch.cells_in_frame(int(f))
        if n_cells + idx.size > int(refine_max_cells) and top:
            break
        top.append(idx)
        n_cells += idx.size
    refined = np.concatenate(top) if top else np.zeros(0, dtype=np.int64)
    if refined.size:
        cell_res = oracle.evaluate([np.array([i]) for i in refined], scorer_names=[rank_scorer])[rank_scorer]
        cell_infl = l_corrupt_rank - cell_res
    else:
        cell_infl = np.zeros(0)

    # final order: refined cells by influence, then remaining frames by frame influence (stage order inside)
    refined_sorted = refined[np.argsort(-cell_infl, kind="stable")]
    refined_frames = set(int(ch.t[i]) for i in refined)
    rest_frames = [int(f) for f in frame_order if int(f) not in refined_frames]
    rest = _order_frames_then_cells(ch, rest_frames)
    order = np.concatenate([refined_sorted, rest]).astype(np.int64)
    scores = np.full(ch.n, np.nan)
    scores[refined] = cell_infl
    ranking = Ranking(name="intervention", order=order, scores=scores, meta={"n_refined": int(refined.size)})
    block = Ranking(name="frame_block", order=_order_frames_then_cells(ch, frame_order))
    return HierarchicalResult(
        ranking=ranking,
        frame_block=block,
        frame_influence=frame_infl,
        refined_cells=refined,
        refined_influence=np.asarray(cell_infl, dtype=float),
        n_frame_decodes=int(len(frames)),
        n_cell_decodes=int(refined.size),
    )


def greedy_frame_ranking(
    oracle: HybridOracle,
    l_corrupt_rank: float,
    rank_scorer: str = "rank",
    max_frames: Optional[int] = None,
) -> Tuple[Ranking, List[float]]:
    """Fully greedy forward selection over frame blocks (O(|F|^2) decodes). Ablation only."""
    ch = oracle.changed
    frames = [int(f) for f in ch.frames]
    if max_frames is not None and len(frames) > int(max_frames):
        raise ValueError(f"greedy skipped: {len(frames)} frames > max_frames={max_frames}")
    chosen: List[int] = []
    chosen_idx = np.zeros(0, dtype=np.int64)
    current = float(l_corrupt_rank)
    gains: List[float] = []
    remaining = list(frames)
    while remaining:
        cand = [np.concatenate([chosen_idx, ch.cells_in_frame(f)]) for f in remaining]
        losses = oracle.evaluate(cand, scorer_names=[rank_scorer])[rank_scorer]
        j = int(np.argmin(losses))
        gains.append(float(current - losses[j]))
        current = float(losses[j])
        f = remaining.pop(j)
        chosen.append(f)
        chosen_idx = np.concatenate([chosen_idx, ch.cells_in_frame(f)])
    return Ranking(name="greedy_frame", order=_order_frames_then_cells(ch, chosen), meta={"frame_order": chosen}), gains


# ----------------------------------------------------------------------------
# Curves
# ----------------------------------------------------------------------------


def evaluate_curve(oracle: HybridOracle, ranking: Ranking, extra_counts: Sequence[int] = ()) -> Dict[str, object]:
    n = oracle.changed.n
    ranking.check(n)
    counts = curve_grid(n, extra=extra_counts)
    subsets = [ranking.order[:k] for k in counts]
    losses = oracle.evaluate(subsets)
    return {"counts": counts, **{f"loss_{k}": v.tolist() for k, v in losses.items()}}


def recovery_from_losses(losses: Sequence[float], l_corrupt: float, l_clean: float) -> np.ndarray:
    gap = float(l_corrupt - l_clean)
    if abs(gap) < 1e-12:
        return np.full(len(losses), np.nan)
    return (float(l_corrupt) - np.asarray(losses, dtype=float)) / gap


def ctm_alpha(fractions: np.ndarray, recovery: np.ndarray, alpha: float) -> float:
    """First prefix fraction whose recovery reaches alpha (1.0 at the latest since Recovery(D)=1)."""
    hit = np.nonzero(recovery >= alpha - 1e-9)[0]
    return float(fractions[hit[0]]) if hit.size else 1.0


def aurc(fractions: np.ndarray, recovery: np.ndarray, clip: Tuple[float, float] = (0.0, 1.0)) -> float:
    """Area under the (clipped) recovery curve over the fraction axis in [0, 1]."""
    r = np.clip(np.asarray(recovery, dtype=float), clip[0], clip[1])
    return float(np.trapezoid(r, np.asarray(fractions, dtype=float)))


def recovery_at(fractions: np.ndarray, recovery: np.ndarray, budget: float) -> float:
    """Recovery of the largest prefix with fraction <= budget (step interpolation)."""
    ok = np.nonzero(fractions <= budget + 1e-12)[0]
    return float(recovery[ok[-1]]) if ok.size else float("nan")


def stage_restoration(oracle: HybridOracle) -> Dict[str, object]:
    """Restore all changed cells of each stage alone: per-stage oracle influence (Q decodes)."""
    ch = oracle.changed
    subsets = [np.nonzero(ch.q == q)[0] for q in range(ch.n_q)]
    losses = oracle.evaluate(subsets)
    return {"flip_counts": [int(s.size) for s in subsets], **{f"loss_{k}": v.tolist() for k, v in losses.items()}}


def gradient_saliency_scores(
    codec,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    changed: ChangedSet,
    loss_fn_differentiable: Callable[[torch.Tensor], torch.Tensor],
) -> np.ndarray:
    """First-order (gradient x delta) estimate of the loss decrease from restoring each changed cell."""
    with torch.no_grad():
        zc = codec.latent_contributions(clean)  # [Q, B, D, T]
        zp = codec.latent_contributions(corrupt)
    z = zp.sum(0).detach().clone().requires_grad_(True)
    wav = codec.decode_latent(z, keep_grad=True)
    loss = loss_fn_differentiable(wav[0])
    (g,) = torch.autograd.grad(loss, z)
    with torch.no_grad():
        delta = (zc - zp)[:, 0]  # [Q, D, T]
        sal = -(g[0].unsqueeze(0) * delta).sum(1)  # [Q, T]: predicted loss decrease
        out = sal[torch.as_tensor(changed.q), torch.as_tensor(changed.t)].detach().cpu().numpy()
    del z, wav, loss, g
    return out.astype(float)
