"""Invariants of the hierarchical held-out protocol (fake codec / fake ASR)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from ctm.asr import FakeASREvaluator
from ctm.audio_io import write_manifest
from ctm.codecs.fake import FakeCodec
from ctm.corruptions import gaussian_noise
from ctm.hier import (
    ChangedSet,
    HybridOracle,
    aurc,
    cell_score_ranking,
    ctm_alpha,
    curve_grid,
    evaluate_curve,
    gradient_saliency_scores,
    greedy_frame_ranking,
    hierarchical_oracle_ranking,
    random_ranking,
    recovery_from_losses,
    rvq_early_ranking,
    stage_restoration,
)
from ctm.runner_v2 import HierRunner


@pytest.fixture
def setup():
    codec = FakeCodec(n_codebooks=4, codebook_size=16, hop=80)
    t = torch.linspace(0, 1.0, 16000)
    wav = 0.3 * torch.sin(2 * np.pi * 220 * t) * (1 + 0.5 * torch.sin(2 * np.pi * 3 * t))
    noisy = gaussian_noise(wav, snr_db=5.0, seed=3)
    z, zp = codec.encode(wav, 16000), codec.encode(noisy, 16000)
    assert not torch.equal(z, zp)
    ref = codec.decode(z)[0]

    def loss_batch(w):
        n = min(w.shape[-1], ref.numel())
        return ((w[:, :n] - ref[:n]) ** 2).mean(dim=-1).sqrt().numpy()

    changed = ChangedSet.from_codes(z, zp)
    oracle = HybridOracle(z, zp, changed, codec.decode, {"rank": loss_batch, "eval": loss_batch}, decode_batch_size=7)
    ends = oracle.evaluate([np.zeros(0, dtype=np.int64), np.arange(changed.n)])
    return codec, wav, noisy, z, zp, changed, oracle, float(ends["eval"][0]), float(ends["eval"][1])


def test_curve_grid_is_sorted_unique_with_endpoints():
    for n in (1, 2, 5, 48, 3148):
        g = curve_grid(n)
        assert g[0] == 0 and g[-1] == n and g == sorted(set(g))
        assert len(g) <= 40


def test_every_ranking_is_a_permutation_of_D_and_recovery_ends_at_one(setup):
    codec, wav, noisy, z, zp, changed, oracle, l_corrupt, l_clean = setup
    assert l_corrupt > l_clean
    hier = hierarchical_oracle_ranking(oracle, l_corrupt, "rank", refine_top_frames=3, refine_max_cells=8)
    rankings = [hier.ranking, hier.frame_block, random_ranking(changed, 0), rvq_early_ranking(changed)]
    for rk in rankings:
        rk.check(changed.n)  # permutation of all of D (never a subset)
        assert rk.order.shape[0] == changed.n
        curve = evaluate_curve(oracle, rk)
        rec = recovery_from_losses(curve["loss_eval"], l_corrupt, l_clean)
        assert curve["counts"][-1] == changed.n
        assert abs(rec[0]) < 1e-6 and abs(rec[-1] - 1.0) < 1e-6  # Recovery(D) = 1 by construction
        frac = np.asarray(curve["counts"]) / changed.n
        assert 0.0 <= aurc(frac, rec) <= 1.0
        assert 0.0 < ctm_alpha(frac, rec, 0.9) <= 1.0


def test_hierarchical_influence_covers_all_changed_frames(setup):
    codec, wav, noisy, z, zp, changed, oracle, l_corrupt, l_clean = setup
    hier = hierarchical_oracle_ranking(oracle, l_corrupt, "rank", refine_top_frames=2, refine_max_cells=100)
    assert set(hier.frame_influence) == set(int(f) for f in changed.frames)
    assert hier.n_frame_decodes == changed.frames.size
    assert hier.refined_cells.size == hier.refined_influence.size <= changed.n


def test_greedy_frame_ranking_is_permutation(setup):
    codec, wav, noisy, z, zp, changed, oracle, l_corrupt, l_clean = setup
    g, gains = greedy_frame_ranking(oracle, l_corrupt, "rank")
    g.check(changed.n)
    assert len(gains) == changed.frames.size


def test_stage_restoration_counts(setup):
    codec, wav, noisy, z, zp, changed, oracle, l_corrupt, l_clean = setup
    st = stage_restoration(oracle)
    assert sum(st["flip_counts"]) == changed.n and len(st["loss_eval"]) == changed.n_q


def test_latent_decomposition_and_gradient_saliency(setup):
    codec, wav, noisy, z, zp, changed, oracle, l_corrupt, l_clean = setup
    w_ref = codec.decode(z)
    w_lat = codec.decode_latent(codec.latent_contributions(z).sum(0))
    assert torch.allclose(w_ref, w_lat, atol=1e-6)
    ref = w_ref[0]
    sal = gradient_saliency_scores(codec, z, zp, changed, lambda w: ((w - ref) ** 2).mean())
    assert sal.shape == (changed.n,) and np.isfinite(sal).all()
    rk = cell_score_ranking(changed, sal, "gradient")
    rk.check(changed.n)
    margins, codes = codec.stage_margins(wav, 16000)
    assert torch.equal(codes, z[0])


def test_runner_v2_smoke_and_gate(tmp_path):
    t = torch.linspace(0, 1.0, 16000)
    wav = 0.3 * torch.sin(2 * np.pi * 220 * t) * (1 + 0.5 * torch.sin(2 * np.pi * 3 * t))
    import soundfile as sf

    (tmp_path / "audio").mkdir()
    sf.write(str(tmp_path / "audio" / "u0.wav"), wav.numpy(), 16000)
    man = tmp_path / "man.jsonl"
    write_manifest(man, [{"utterance_id": "u0", "path": str(tmp_path / "audio" / "u0.wav"), "text": "hello world"}])
    cfg = {
        "seed": 0,
        "output_dir": str(tmp_path / "runs"),
        "gate": "loss",
        "refine_top_frames": 2,
        "random_seeds": 2,
        "data": {"manifest": str(man), "sample_rate": 16000},
        "asr": {"surrogate": "fake", "evaluator": "fake"},
        "codecs": [{"name": "fake", "n_codebooks": 4, "codebook_size": 16, "hop": 80}],
        "rankings": ["intervention", "frame_block", "random", "rvq_early", "displacement", "margin", "gradient", "energy"],
        "conditions": [{"condition_id": "gaussian_snr5", "kind": "gaussian", "params": {"snr_db": 5}}],
    }
    out = HierRunner(cfg).run()
    row = json.loads(out.read_text().strip().splitlines()[-1])
    assert row["status"] == "ok", row.get("error")
    if row["ctm_status"] == "ok":
        for name, c in row["curves"].items():
            assert c["counts"][-1] == row["n_changed"], name
            rec = recovery_from_losses(c["loss_eval"], row["l_corrupt_eval"], row["l_clean_eval"])
            assert abs(rec[-1] - 1.0) < 1e-6, name
        assert set(row["curves"]) >= {"intervention", "frame_block", "random", "random2", "rvq_early", "displacement", "margin", "gradient", "energy"}
        assert "n_frame_decodes" in row and row["n_frame_decodes"] == row["n_frames_changed"]
    else:
        assert row["ctm_status"] in {"skipped_gate", "skipped_empty_changed_set"}
