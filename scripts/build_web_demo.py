"""Build the audio + data assets of the project page (docs/) on a GPU.

Case selection is fixed in advance and does not look at audio quality or wins:
for every codec x corruption family (CTC-PGD eps=0.01, Gaussian 15 dB, ESC-50 8 dB, Opus 8 kbit/s)
take the gated, held-out-resolved rows of the released main grid whose utterance lasts 3-7 s and
pick the one whose oracle AURC is closest to the median of that cell, i.e. a *typical* case.

For each case the exact paper pipeline is re-run (same seeded corruption, same codec, same
frame->cell oracle ranked by wav2vec2-base), then the corrupted codes are partially restored
along the oracle and the random ranking at fixed budgets. Every hybrid is decoded, transcribed
and scored by the held-out HuBERT-large recognizer, and saved as MP3 for the page.

    python scripts/build_web_demo.py --config configs/main_hier.yaml --out docs/assets
    python scripts/build_web_demo.py --summary-only --out docs/assets   # CPU: charts data only
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ctm.analysis_v2 import load_rows, per_row_table  # noqa: E402
from ctm.asr import compute_wer  # noqa: E402
from ctm.hier import aurc, recovery_from_losses  # noqa: E402

CODECS = ("dac", "mimi", "speechtokenizer")
FAMILIES = {  # label -> condition id
    "CTC-PGD (ε=0.01)": "pgd_ctc_eps0.01",
    "Gaussian 15 dB": "gaussian_snr15",
    "ESC-50 noise 8 dB": "esc50_snr8",
    "Opus 8 kbit/s": "opus_8k",
}
BUDGETS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
CURVE_RANKINGS = ("intervention", "random", "rvq_early", "displacement", "energy")


def select_cases(df) -> list[dict]:
    ok = df[(df["ctm_status"] == "ok") & (df["eval_resolved"] == True) & df["duration_s"].between(3.0, 7.0)]  # noqa: E712
    cases = []
    for codec in CODECS:
        for label, cond in FAMILIES.items():
            g = ok[(ok["codec"] == codec) & (ok["condition"] == cond)].copy()
            if g.empty:
                continue
            a = g["intervention|eval|aurc"].astype(float)
            g["_d"] = (a - a.median()).abs()
            r = g.sort_values(["_d", "utterance_id"]).iloc[0]
            cases.append({"codec": codec, "condition": cond, "family": label, "utterance_id": r["utterance_id"],
                          "cell_n": int(len(g)), "cell_median_aurc": float(a.median())})
    return cases


def curve_points(row: dict, ranking: str, scorer: str = "eval"):
    c = row["curves"].get(ranking)
    if not c or f"loss_{scorer}" not in c:
        return None
    n = max(int(row["n_changed"]), 1)
    frac = np.asarray(c["counts"], float) / n
    rec = recovery_from_losses(c[f"loss_{scorer}"], row[f"l_corrupt_{scorer}"], row[f"l_clean_{scorer}"])
    return frac, rec


def summary_data(rows: list[dict], df) -> dict:
    """Aggregates for the interactive charts (same data as Figs. 3-5 of the paper)."""
    grid = np.unique(np.concatenate([np.linspace(0, 0.1, 21), np.linspace(0.1, 1, 37)]))
    out = {"grid": grid.round(4).tolist(), "codecs": {}, "scatter": [], "stage": {}}
    for codec in CODECS:
        sub = [r for r in rows if r["codec"] == codec and r.get("ctm_status") == "ok"]
        res = set(df[(df["codec"] == codec) & (df["eval_resolved"] == True)]["utterance_id"] + "|" +  # noqa: E712
                  df[(df["codec"] == codec) & (df["eval_resolved"] == True)]["condition"])  # noqa: E712
        sub = [r for r in sub if f"{r['utterance_id']}|{r['condition']}" in res]
        curves = {}
        for rk in CURVE_RANKINGS:
            ys = []
            for r in sub:
                p = curve_points(r, rk)
                if p is not None:
                    ys.append(np.interp(grid, p[0], p[1]))
            if ys:
                Y = np.stack(ys)
                curves[rk] = {"median": np.median(Y, 0).round(4).tolist(),
                              "q25": np.quantile(Y, 0.25, 0).round(4).tolist(),
                              "q75": np.quantile(Y, 0.75, 0).round(4).tolist()}
        alone, flips = [], []
        for r in sub:
            st = r.get("stage_restore")
            if st and "loss_eval" in st and r["gap_eval"] > 0:
                alone.append((r["l_corrupt_eval"] - np.asarray(st["loss_eval"])) / r["gap_eval"])
                fl = np.asarray(st["flip_counts"], float)
                flips.append(fl / max(fl.sum(), 1))
        out["codecs"][codec] = {"n": len(sub), "curves": curves}
        if alone:
            out["stage"][codec] = {"alone": np.median(np.stack(alone), 0).round(3).tolist(),
                                   "flip_share": np.mean(np.stack(flips), 0).round(3).tolist()}
    for (codec, cond), g in df.groupby(["codec", "condition"]):
        out["scatter"].append({"codec": codec, "condition": cond,
                               "tcr": round(float(g["tcr"].mean()), 4),
                               "damage": round(float((g["gate_passed"] == True).mean()), 4)})  # noqa: E712
    return out


def to_mp3(wav: np.ndarray, sr: int, dest: Path) -> None:
    import soundfile as sf

    peak = float(np.max(np.abs(wav))) or 1.0
    wav = wav / max(peak, 1.0)  # never amplify; only prevent clipping
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        sf.write(tmp.name, wav.astype(np.float32), sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp.name, "-ac", "1",
                        "-codec:a", "libmp3lame", "-b:a", "64k", str(dest)], check=True)


def build_cases(cfg_path: str, cases: list[dict], rows_by_key: dict, out: Path) -> list[dict]:
    import torch

    from ctm.hier import ChangedSet, HybridOracle, hierarchical_oracle_ranking, random_ranking
    from ctm.audio_io import iter_utterances_from_manifest
    from ctm.corruptions import condition_seed
    from ctm.runner_v2 import HierRunner, load_config

    cfg = load_config(cfg_path)
    cfg["output_dir"] = str(Path(tempfile.mkdtemp(prefix="ctm_demo_")))
    runner = HierRunner(cfg)
    wanted = {c["utterance_id"] for c in cases}
    utts = {}
    for man in runner.manifests:
        for u in iter_utterances_from_manifest(man, target_sr=runner.target_sr, max_utterances=runner.max_utterances):
            if u.utterance_id in wanted:
                utts[u.utterance_id] = u
    conds = {c.condition_id: c for c in runner.conditions}
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for i, case in enumerate(cases):
        utt, cond = utts[case["utterance_id"]], conds[case["condition"]]
        codec = runner.codecs[case["codec"]]
        ref = rows_by_key[(case["utterance_id"], case["codec"], case["condition"])]
        print(f"[{i + 1}/{len(cases)}] {case['codec']} {case['condition']} {utt.utterance_id}", flush=True)
        clean = utt.mono()
        text = utt.text
        x_corrupt = runner._corrupt(clean, utt.utterance_id, cond, text).detach()  # PGD needs gradients
        with torch.no_grad():
            z = codec.encode(clean, runner.target_sr)
            zp = codec.encode(x_corrupt, runner.target_sr)
        sr = int(codec.info.sample_rate)
        changed = ChangedSet.from_codes(z, zp)
        scorers = runner._scorers(sr, text)
        oracle = HybridOracle(z, zp, changed, codec.decode, scorers, decode_batch_size=runner._batch_size_for(clean.numel(), runner.target_sr))
        with torch.no_grad():
            l_corrupt_rank = float(scorers["rank"](codec.decode(zp).reshape(1, -1))[0])
            hier = hierarchical_oracle_ranking(oracle, l_corrupt_rank, "rank", runner.refine_top_frames, runner.refine_max_cells)
        seed = condition_seed(runner.base_seed, utt.utterance_id, cond.condition_id)
        orders = {"oracle": hier.ranking.order, "random": random_ranking(changed, seed).order}
        n = changed.n
        cid = f"{i:02d}_{case['codec']}_{case['condition']}"
        files = {}
        # inputs (before the tokenizer) at 16 kHz
        to_mp3(clean.cpu().numpy(), runner.target_sr, audio_dir / f"{cid}_input_clean.mp3")
        to_mp3(x_corrupt.detach().cpu().numpy(), runner.target_sr, audio_dir / f"{cid}_input_corrupt.mp3")
        files["input_clean"] = f"audio/{cid}_input_clean.mp3"
        files["input_corrupt"] = f"audio/{cid}_input_corrupt.mp3"
        steps = {}
        for name, order in orders.items():
            ks = [int(round(b * n)) for b in BUDGETS]
            with torch.no_grad():
                hyb = torch.cat([oracle.hybrid_from_indices(order[:k]) for k in ks], 0)
                wavs = codec.decode(hyb)
                wavs = wavs.reshape(len(ks), -1)
                losses = runner.evaluator.ctc_loss_batch(wavs, sr, text)
                hyps = runner.evaluator.transcribe_batch(wavs, sr)
            l_corr, l_clean = float(losses[0]), float(losses[-1])
            rec = recovery_from_losses(losses, l_corr, l_clean)
            steps[name] = []
            for b, k, w, h, rv in zip(BUDGETS, ks, wavs, hyps, rec):
                f = f"audio/{cid}_{name}_{int(round(b * 100)):03d}.mp3"
                if name == "random" and b in (0.0, 1.0):  # identical to the oracle endpoints
                    f = f"audio/{cid}_oracle_{int(round(b * 100)):03d}.mp3"
                else:
                    to_mp3(w.detach().cpu().numpy(), sr, out / f)
                steps[name].append({"budget": b, "k": k, "recovery": round(float(rv), 4), "hyp": h,
                                    "wer": round(compute_wer(text, h), 4), "audio": f})
        # token grid: every changed cell with its rank under both orders
        pos = {nm: np.empty(n, dtype=np.int64) for nm in orders}
        for nm, order in orders.items():
            pos[nm][order] = np.arange(n)
        frame_infl = hier.frame_influence
        curves = {}
        for rk in CURVE_RANKINGS:
            p = curve_points(ref, rk)
            if p is not None:
                curves[rk] = {"x": p[0].round(5).tolist(), "y": p[1].round(4).tolist(), "aurc": round(float(aurc(p[0], p[1])), 4)}
        built.append({
            **case, "id": cid, "text": text, "duration_s": round(clean.numel() / runner.target_sr, 2),
            "Q": int(z.shape[1]), "T": int(z.shape[-1]), "n_changed": int(n), "tcr": round(n / (z.shape[1] * z.shape[-1]), 4),
            "frame_rate": round(z.shape[-1] / (clean.numel() / runner.target_sr), 2),
            "cells": {"q": changed.q.tolist(), "t": changed.t.tolist(),
                      "rank_oracle": pos["oracle"].tolist(), "rank_random": pos["random"].tolist()},
            "frame_influence": {str(k): round(float(v), 5) for k, v in frame_infl.items()},
            "files": files, "steps": steps, "curves": curves,
            "paper_row": {"wer_clean": ref.get("wer_clean"), "wer_corrupt": ref.get("wer_corrupt"),
                          "hyp_corrupt": ref.get("hyp_corrupt"), "gap_eval": ref.get("gap_eval")},
        })
        torch.cuda.empty_cache()
    return built


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/main_hier.yaml")
    p.add_argument("--results", default="results/main_hier.jsonl.gz")
    p.add_argument("--out", default="docs/assets")
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    rows = load_rows([a.results])
    df = per_row_table(rows)
    (out / "data" / "summary.json").write_text(json.dumps(summary_data(rows, df), separators=(",", ":")))
    print("wrote", out / "data" / "summary.json")
    if a.summary_only:
        return
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg is required to write MP3s")
    cases = select_cases(df)
    print(f"selected {len(cases)} cases")
    by_key = {(r["utterance_id"], r["codec"], r["condition"]): r for r in rows}
    built = build_cases(a.config, cases, by_key, out)
    meta = {"selection_rule": ("For each codec x corruption family: gated rows of the released main grid that are resolved "
                               "by the held-out scorer and last 3-7 s; the row whose oracle AURC is closest to the cell "
                               "median is shown (a typical case, chosen before listening)."),
            "budgets": BUDGETS, "scorer": "HuBERT-large (held-out)", "ranker": "wav2vec2-base", "cases": built}
    (out / "data" / "cases.json").write_text(json.dumps(meta, separators=(",", ":")))
    print("wrote", out / "data" / "cases.json")


if __name__ == "__main__":
    main()
