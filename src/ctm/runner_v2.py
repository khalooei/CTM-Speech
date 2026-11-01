"""Tiered, resumable runner for the hierarchical held-out CTM protocol.

Tier A (every row): endpoints, stability metrics, transcript damage under the evaluator.
Tier B (rows passing the functional-damage gate): rankings + held-out recovery curves.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml
from tqdm import tqdm

from ctm.env_bootstrap import load_project_env

load_project_env(quiet=True)

from ctm.asr import FakeASREvaluator, build_asr, compute_cer, compute_wer  # noqa: E402
from ctm.attacks import CTCPGDAttack, PGDConfig  # noqa: E402
from ctm.audio_io import esc10_subset, iter_utterances_from_manifest, list_esc50_files  # noqa: E402
from ctm.codecs import build_codec  # noqa: E402
from ctm.corruptions import apply_corruption, condition_seed, parse_condition  # noqa: E402
from ctm.hier import (  # noqa: E402
    ChangedSet,
    HybridOracle,
    Ranking,
    cell_score_ranking,
    evaluate_curve,
    frame_score_ranking,
    gradient_saliency_scores,
    greedy_frame_ranking,
    hierarchical_oracle_ranking,
    perturbation_energy_frame_scores,
    random_ranking,
    rvq_early_ranking,
    stage_restoration,
)
from ctm.jsonl_store import append_jsonl, completed_keys, make_key  # noqa: E402
from ctm.metrics import si_sdr, snr_db, stability_bundle, stoi_score  # noqa: E402
from ctm.types import ConditionSpec, RowStatus  # noqa: E402

PROTOCOL_VERSION = "hier-v2"


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HierRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_seed = int(config.get("seed", 0))
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(config.get("output_dir", "runs/default"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = self.out_dir / "utterance_results.jsonl"

        data = config["data"]
        self.manifests = data["manifest"] if isinstance(data["manifest"], list) else [data["manifest"]]
        self.max_utterances = data.get("max_utterances")
        self.target_sr = int(data.get("sample_rate", 16000))
        esc_root = data.get("esc50_root")
        self.esc50_files = list_esc50_files(esc_root) if esc_root else []
        self.esc10_files = esc10_subset(self.esc50_files) if self.esc50_files else []
        # Extra real-noise banks: {"musan": "data/musan/noise", "demand": "data/demand"} -> recursive wav lists
        self.noise_banks = {}
        for kind, root in (data.get("noise_banks") or {}).items():
            files = sorted(Path(root).rglob("*.wav"))
            if not files:
                raise FileNotFoundError(f"noise bank '{kind}' has no wav files under {root}")
            self.noise_banks[kind] = files

        asr_cfg = config.get("asr", {})
        # rank scorer == attack surrogate (wav2vec2-base); eval scorer is architecturally independent.
        self.surrogate = build_asr(asr_cfg.get("surrogate", "fake"), device=self.device)
        self.evaluator = build_asr(asr_cfg.get("evaluator", "fake"), device=self.device)
        self.evaluator2 = build_asr(asr_cfg["evaluator2"], device=self.device) if asr_cfg.get("evaluator2") else None
        for a in (self.surrogate, self.evaluator, self.evaluator2):
            if a is not None and hasattr(a, "max_batch_seconds"):
                a.max_batch_seconds = float(config.get("score_max_batch_seconds", 240.0))

        self.codecs = {}
        for spec in config.get("codecs", [{"name": "fake"}]):
            name = spec["name"]
            codec_id = str(spec.get("id", name))
            kwargs = {k: v for k, v in spec.items() if k not in {"name", "id"}}
            kwargs.setdefault("device", self.device)
            self.codecs[codec_id] = build_codec(name, **kwargs)

        self.conditions = [parse_condition(c) for c in config.get("conditions", [])]
        self.rankings = list(config.get("rankings", ["intervention", "frame_block", "random", "rvq_early", "displacement", "margin", "gradient", "energy"]))
        self.random_seeds = int(config.get("random_seeds", 2))
        self.refine_top_frames = int(config.get("refine_top_frames", 16))
        self.refine_max_cells = int(config.get("refine_max_cells", 256))
        self.gate = str(config.get("gate", "transcript"))
        self.do_greedy = bool(config.get("do_greedy", False))
        self.greedy_max_frames = int(config.get("greedy_max_frames", 160))
        self.decode_batch_seconds = float(config.get("decode_batch_seconds", 320.0))
        self.decode_batch_max = int(config.get("decode_batch_max", 48))
        self.tier_b_max_per_condition = config.get("tier_b_max_per_condition")
        self.resume = bool(config.get("resume", True))
        self.shard_id = int(config.get("shard_id", 0))
        self.num_shards = max(1, int(config.get("num_shards", 1)))

    # ------------------------------------------------------------------
    def run(self) -> Path:
        done = completed_keys(self.results_path) if self.resume else set()
        utterances = []
        for man in self.manifests:
            utterances.extend(iter_utterances_from_manifest(man, target_sr=self.target_sr, max_utterances=self.max_utterances))
        if self.num_shards > 1:
            utterances = [u for i, u in enumerate(utterances) if i % self.num_shards == self.shard_id]
        desc = "utterances" if self.num_shards == 1 else f"utt[shard {self.shard_id}/{self.num_shards}]"
        for utt in tqdm(utterances, desc=desc):
            clean = utt.mono()
            cache: Dict[str, torch.Tensor] = {}
            for cond in self.conditions:
                pending = [c for c in self.codecs if make_key(utt.utterance_id, c, cond.condition_id) not in done]
                if not pending:
                    continue
                if cond.condition_id not in cache:
                    cache[cond.condition_id] = self._corrupt(clean, utt.utterance_id, cond, utt.text)
                x_corrupt = cache[cond.condition_id]
                for codec_name in pending:
                    key = make_key(utt.utterance_id, codec_name, cond.condition_id)
                    t0 = time.time()
                    try:
                        row = self._run_one(utt, clean, x_corrupt, cond, codec_name, self.codecs[codec_name])
                        row["status"] = RowStatus.OK.value
                    except Exception as e:  # explicit status, never silent zeros
                        row = {
                            "utterance_id": utt.utterance_id,
                            "codec": codec_name,
                            "condition": cond.condition_id,
                            "status": RowStatus.ERROR.value,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(limit=6),
                        }
                    row["protocol"] = PROTOCOL_VERSION
                    row["shard_id"] = self.shard_id
                    row["wall_seconds"] = round(time.time() - t0, 2)
                    append_jsonl(self.results_path, row)
                    done.add(key)
        return self.results_path

    # ------------------------------------------------------------------
    def _corrupt(self, clean, utterance_id, cond: ConditionSpec, text: str) -> torch.Tensor:
        kind = cond.kind.lower()
        if kind in {"pgd", "pgd_ctc", "ctc_pgd"}:
            seed = condition_seed(self.base_seed, utterance_id, cond.condition_id)
            cfg = PGDConfig(
                eps=float(cond.params.get("eps", 0.02)),
                step_size=float(cond.params.get("step_size", 0.002)),
                steps=int(cond.params.get("steps", 20)),
                band=cond.params.get("band"),
            )
            return CTCPGDAttack(self.surrogate, sample_rate=self.target_sr).attack(clean, text or "the", cfg, seed=seed)
        return apply_corruption(clean, cond, base_seed=self.base_seed, utterance_id=utterance_id, sample_rate=self.target_sr, esc50_files=self.esc50_files, esc10_files=self.esc10_files, noise_banks=self.noise_banks)

    def _batch_size_for(self, n_samples: int, codec_sr: int) -> int:
        dur = max(n_samples / float(codec_sr), 0.5)
        return int(max(2, min(self.decode_batch_max, math.floor(self.decode_batch_seconds / dur))))

    def _scorers(self, codec_sr: int, text: str) -> Dict[str, Callable[[torch.Tensor], np.ndarray]]:
        s = {
            "rank": lambda w: self.surrogate.ctc_loss_batch(w, codec_sr, text),
            "eval": lambda w: self.evaluator.ctc_loss_batch(w, codec_sr, text),
        }
        if self.evaluator2 is not None:
            s["eval2"] = lambda w: self.evaluator2.ctc_loss_batch(w, codec_sr, text)
        return s

    # ------------------------------------------------------------------
    def _run_one(self, utt, clean, x_corrupt, cond, codec_name, codec) -> Dict[str, Any]:
        text = utt.text or "a"
        z = codec.encode(clean, self.target_sr)
        zp = codec.encode(x_corrupt, self.target_sr)
        codec_sr = int(codec.info.sample_rate)

        if isinstance(self.evaluator, FakeASREvaluator):
            self.evaluator.set_clean_reference(text, codec.decode(z).reshape(-1))
        if isinstance(self.surrogate, FakeASREvaluator):
            self.surrogate.set_clean_reference(text, codec.decode(z).reshape(-1))

        row: Dict[str, Any] = {
            "utterance_id": utt.utterance_id,
            "codec": codec_name,
            "condition": cond.condition_id,
            "condition_kind": cond.kind,
            "condition_params": cond.params,
            "text": utt.text,
            "duration_s": round(clean.numel() / self.target_sr, 3),
            "input_snr_db": snr_db(clean, x_corrupt),
            "n_codebooks": int(z.shape[1]),
            "T_clean": int(z.shape[-1]),
            "T_corrupt": int(zp.shape[-1]),
        }
        stab = stability_bundle(z, zp, codebook_size=codec.info.codebook_size)
        row.update({f"stability_{k}": v for k, v in stab.items()})
        row["tcr"], row["ued"], row["jsd"] = stab.get("tcr"), stab.get("ued"), stab.get("jsd")

        if z.shape != zp.shape:
            # Endpoints only; CTM requires aligned sequences.
            rc, rr = codec.decode(z).reshape(-1), codec.decode(zp).reshape(-1)
            self._endpoint_metrics(row, rc, rr, codec_sr, text)
            row["ctm_status"] = "skipped_length_mismatch"
            row["n_changed"] = None
            return row

        changed = ChangedSet.from_codes(z, zp)
        row["n_changed"] = int(changed.n)
        row["n_frames_changed"] = int(changed.frames.size)
        row["stage_tcr"] = [float((z[0, q] != zp[0, q]).float().mean()) for q in range(z.shape[1])]

        # Endpoints decoded in one batch (same numerics as the curves).
        pair = codec.decode(torch.cat([zp, z], dim=0))
        if pair.ndim == 3:
            pair = pair.squeeze(1)
        rr, rc = pair[0], pair[1]
        self._endpoint_metrics(row, rc, rr, codec_sr, text)
        scorers = self._scorers(codec_sr, text)
        ends = {k: fn(pair) for k, fn in scorers.items()}  # [corrupt, clean]
        for k, v in ends.items():
            row[f"l_corrupt_{k}"] = float(v[0])
            row[f"l_clean_{k}"] = float(v[1])
            row[f"gap_{k}"] = float(v[0] - v[1])

        # ---- gate ----
        gate_ok, reason = self._gate(row)
        row["gate_passed"] = bool(gate_ok)
        row["gate_reason"] = reason
        if changed.n == 0:
            row["ctm_status"] = "skipped_empty_changed_set"
            return row
        if not gate_ok:
            row["ctm_status"] = "skipped_gate"
            return row
        if self.tier_b_max_per_condition is not None and self._tier_b_count(codec_name, cond.condition_id) >= int(self.tier_b_max_per_condition):
            row["ctm_status"] = "skipped_budget"
            return row

        # ---- Tier B ----
        bs = self._batch_size_for(pair.shape[-1], codec_sr)
        oracle = HybridOracle(z, zp, changed, codec.decode, scorers, decode_batch_size=bs)
        row["decode_batch_size"] = bs
        rankings: List[Ranking] = []
        hier = None
        if "intervention" in self.rankings or "frame_block" in self.rankings:
            hier = hierarchical_oracle_ranking(oracle, row["l_corrupt_rank"], "rank", self.refine_top_frames, self.refine_max_cells)
            if "intervention" in self.rankings:
                rankings.append(hier.ranking)
            if "frame_block" in self.rankings:
                rankings.append(hier.frame_block)
            fr = changed.frames
            row["changed_frames"] = [int(f) for f in fr]
            row["frame_influence_rank"] = [hier.frame_influence[int(f)] for f in fr]
            row["refined_cells"] = [[int(changed.q[i]), int(changed.t[i]), float(v)] for i, v in zip(hier.refined_cells, hier.refined_influence)]
            row["n_frame_decodes"], row["n_cell_decodes"] = hier.n_frame_decodes, hier.n_cell_decodes
        seed = condition_seed(self.base_seed, utt.utterance_id, cond.condition_id)
        if "random" in self.rankings:
            for s in range(self.random_seeds):
                rankings.append(random_ranking(changed, seed + s, name="random" if s == 0 else f"random{s + 1}"))
        if "rvq_early" in self.rankings:
            rankings.append(rvq_early_ranking(changed))
        if "displacement" in self.rankings:
            try:
                D = codec.displacement_matrix(z, zp)
                rankings.append(cell_score_ranking(changed, D[changed.q, changed.t].numpy(), "displacement", descending=True))
            except Exception as e:
                row["displacement_error"] = f"{type(e).__name__}: {e}"
        if "margin" in self.rankings:
            try:
                margins, mcodes = codec.stage_margins(clean, self.target_sr)
                row["margin_code_mismatch"] = float((mcodes != z[0]).float().mean())
                rankings.append(cell_score_ranking(changed, margins[changed.q, changed.t].numpy(), "margin", descending=False))
            except Exception as e:
                row["margin_error"] = f"{type(e).__name__}: {e}"
        if "gradient" in self.rankings:
            try:
                sal = gradient_saliency_scores(codec, z, zp, changed, lambda w: self.surrogate.ctc_loss_differentiable(w, codec_sr, text))
                rankings.append(cell_score_ranking(changed, sal, "gradient", descending=True))
            except Exception as e:
                row["gradient_error"] = f"{type(e).__name__}: {e}"
        if "energy" in self.rankings:
            rankings.append(frame_score_ranking(changed, perturbation_energy_frame_scores(clean, x_corrupt, changed.n_t), "energy"))
        if self.do_greedy or "greedy_frame" in self.rankings:
            try:
                g, gains = greedy_frame_ranking(oracle, row["l_corrupt_rank"], "rank", max_frames=self.greedy_max_frames)
                rankings.append(g)
                row["greedy_frame_gains"] = gains
            except Exception as e:
                row["greedy_error"] = f"{type(e).__name__}: {e}"

        # stage boundaries so the rvq_early curve is exact at each cumulative stage
        stage_counts = np.cumsum([int((changed.q == q).sum()) for q in range(changed.n_q)]).tolist()
        curves: Dict[str, Any] = {}
        heads: Dict[str, List[List[int]]] = {}
        for rk in rankings:
            extra = stage_counts if rk.name == "rvq_early" else ()
            curves[rk.name] = evaluate_curve(oracle, rk, extra_counts=extra)
            heads[rk.name] = [[int(changed.q[i]), int(changed.t[i])] for i in rk.order[:32]]
        row["curves"] = curves
        row["ranking_heads"] = heads
        row["stage_counts_cum"] = stage_counts
        row["stage_restore"] = stage_restoration(oracle)
        row["n_decodes"] = oracle.n_decodes
        row["decode_seconds"] = round(oracle.decode_seconds, 2)
        row["score_seconds"] = round(oracle.score_seconds, 2)
        row["ctm_status"] = "ok"
        return row

    def _tier_b_count(self, codec_name: str, condition: str) -> int:
        key = (codec_name, condition)
        if not hasattr(self, "_tier_b_counts"):
            self._tier_b_counts: Dict = {}
            from ctm.jsonl_store import load_jsonl_last_wins

            for r in load_jsonl_last_wins(self.results_path).values():
                if r.get("ctm_status") == "ok":
                    k = (r.get("codec"), r.get("condition"))
                    self._tier_b_counts[k] = self._tier_b_counts.get(k, 0) + 1
        n = self._tier_b_counts.get(key, 0)
        self._tier_b_counts[key] = n + 1
        return n

    def _gate(self, row: Dict[str, Any]):
        if self.gate == "none":
            return True, "no_gate"
        if row.get("gap_eval") is None or not (row["gap_eval"] > 0):
            return False, "no_positive_functional_degradation_eval"
        if self.gate == "loss":
            return True, "loss_gap_positive"
        wc, wk = row.get("wer_clean"), row.get("wer_corrupt")
        cc, ck = row.get("cer_clean"), row.get("cer_corrupt")
        if wc is not None and wk is not None and (wk > wc or (ck is not None and cc is not None and ck > cc)):
            return True, "transcript_damaged"
        return False, "transcript_unchanged"

    def _endpoint_metrics(self, row, rc: torch.Tensor, rr: torch.Tensor, codec_sr: int, text: str) -> None:
        n = min(rc.numel(), rr.numel())
        rc, rr = rc.reshape(-1)[:n], rr.reshape(-1)[:n]
        row["recon_snr_db"] = snr_db(rc, rr)
        row["si_sdr"] = si_sdr(rc, rr)
        row["stoi"] = stoi_score(rc, rr, sample_rate=codec_sr)
        hyps = self.evaluator.transcribe_batch(torch.stack([rc, rr]), codec_sr)
        row["hyp_clean"], row["hyp_corrupt"] = hyps[0], hyps[1]
        row["wer_clean"], row["wer_corrupt"] = compute_wer(text, hyps[0]), compute_wer(text, hyps[1])
        row["cer_clean"], row["cer_corrupt"] = compute_cer(text, hyps[0]), compute_cer(text, hyps[1])
        row["delta_wer"] = row["wer_corrupt"] - row["wer_clean"]
        row["delta_cer"] = row["cer_corrupt"] - row["cer_clean"]


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Hierarchical held-out CTM runner")
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--max-utterances", type=int, default=None)
    p.add_argument("--shard-id", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.max_utterances is not None:
        cfg.setdefault("data", {})["max_utterances"] = args.max_utterances
    if args.shard_id is not None:
        cfg["shard_id"] = args.shard_id
    if args.num_shards is not None:
        cfg["num_shards"] = args.num_shards
    if "CTM_SHARD_ID" in os.environ:
        cfg["shard_id"] = int(os.environ["CTM_SHARD_ID"])
    if "CTM_NUM_SHARDS" in os.environ:
        cfg["num_shards"] = int(os.environ["CTM_NUM_SHARDS"])
    runner = HierRunner(cfg)
    path = runner.run()
    print(f"Wrote results to {path}")


if __name__ == "__main__":
    main()
