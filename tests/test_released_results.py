"""The released per-utterance results reproduce the reported headline numbers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ctm.analysis_v2 import load_rows, per_row_table

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "results" / "main_hier.jsonl.gz"

# Reported mean held-out (HuBERT-large) AURC on gated, resolved rows (ranking ablation table).
REPORTED_AURC = {
    ("dac", "intervention"): 0.85, ("dac", "random"): 0.70,
    ("mimi", "intervention"): 0.80, ("mimi", "random"): 0.48,
    ("speechtokenizer", "intervention"): 0.86, ("speechtokenizer", "random"): 0.70,
}


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    if not MAIN.exists():
        pytest.skip("results/main_hier.jsonl.gz not present")
    return per_row_table(load_rows([MAIN]))


def test_grid_is_complete(table):
    assert len(table) == 120 * 3 * 16  # utterances x codecs x conditions


@pytest.mark.parametrize("codec,ranking", sorted(REPORTED_AURC))
def test_headline_aurc(table, codec, ranking):
    g = table[(table["codec"] == codec) & (table["ctm_status"] == "ok")]
    mean = np.nanmean(pd.to_numeric(g[f"{ranking}|eval|aurc"], errors="coerce"))
    assert round(mean, 2) == pytest.approx(REPORTED_AURC[(codec, ranking)], abs=1e-9)


def test_full_restoration_recovers_the_gap(table):
    """Rec(D) = 1 by construction: every ranking is a permutation of the whole changed set."""
    rows = [r for r in load_rows([MAIN]) if r.get("ctm_status") == "ok"][:200]
    for r in rows:
        for curve in r["curves"].values():
            assert curve["counts"][-1] == r["n_changed"]
