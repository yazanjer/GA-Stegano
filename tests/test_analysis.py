"""Tests for the analysis half: estimators, classifier, statistics, baselines.

The steganalysis estimators are validated against *known* embedding rates.  An
estimator that silently returns nonsense would make the security section of the
paper meaningless, and nothing else in the pipeline would notice.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdt.baselines.adaptive import COST_FUNCTIONS, embed_with_cost
from amdt.baselines.classical import edge_adaptive_lsb, lsb_matching, lsb_replacement, pvd
from amdt.evaluation.metrics import evaluate_pair, mse, psnr, ssim
from amdt.evaluation.significance import (cliffs_delta, cohens_dz, compare_many,
                                          compare_paired, holm_bonferroni)
from amdt.steganalysis.ensemble import EnsembleClassifier
from amdt.steganalysis.features import spam686, srm_subset
from amdt.steganalysis.metrics import evaluate_scores, roc_curve, auc
from amdt.steganalysis.targeted import (chi_square_attack, payload_uniformity,
                                        rs_analysis, weighted_stego_estimate)


def _natural_image(side=192, seed=0):
    """A smooth, textured surrogate cover (no external file needed)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:side, 0:side].astype(np.float64)
    img = (110
           + 45 * np.sin(x / 17.0) * np.cos(y / 23.0)
           + 25 * np.sin((x + y) / 9.0)
           + rng.normal(0, 3.0, (side, side)))
    return np.clip(img, 0, 255).astype(np.uint8)


#: Candidate locations for the benchmark cover set, relative to the repo root.
_REAL_IMAGE_DIRS = (
    Path(__file__).resolve().parents[2] / "baseline code matlab" / "Images",
    Path(__file__).resolve().parents[1] / "data" / "images",
)


@pytest.fixture(scope="module")
def real_covers():
    """Several genuine photographic covers (see :func:`real_cover`)."""
    from amdt.data.dataset import load_dataset
    for d in _REAL_IMAGE_DIRS:
        if d.exists():
            imgs, _ = load_dataset(d, side=256, limit=5)
            return imgs
    pytest.skip("no natural cover images available; see _REAL_IMAGE_DIRS")


@pytest.fixture(scope="module")
def real_cover():
    """A genuine photographic cover.

    RS, WS and chi-square are *defined* for natural imagery: their models assume
    a smooth, non-uniform intensity histogram and correlated neighbours.  A
    synthetic sinusoid-plus-Gaussian surrogate violates both assumptions (its
    histogram is already pairs-of-values balanced), so validating the estimators
    against it would test the wrong thing.  Skip rather than weaken the test.
    """
    from amdt.data.dataset import load_dataset
    for d in _REAL_IMAGE_DIRS:
        if d.exists():
            imgs, _ = load_dataset(d, side=256, limit=1)
            return imgs[0]
    pytest.skip("no natural cover images available; see _REAL_IMAGE_DIRS")


def _lsb_embed(cover, rate, seed):
    rng = np.random.default_rng(seed)
    st = cover.copy().reshape(-1)
    k = int(rate * st.size)
    idx = rng.permutation(st.size)[:k]
    st[idx] = (st[idx] & 0xFE) | rng.integers(0, 2, k, dtype=np.uint8)
    return st.reshape(cover.shape)


# --------------------------------------------------------------------------- #
def test_metrics_agree_with_definitions():
    a = np.zeros((16, 16), dtype=np.uint8)
    b = a.copy()
    assert mse(a, b) == 0.0
    assert np.isinf(psnr(a, b))
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-9)
    b[0, 0] = 255
    assert mse(a, b) == pytest.approx(255 ** 2 / 256)


def test_quality_report_is_self_consistent():
    cover = _natural_image()
    stego = _lsb_embed(cover, 0.2, 1)
    q = evaluate_pair(cover, stego, payload_bits=int(0.2 * cover.size))
    assert q.n_changes > 0
    assert q.embedding_efficiency == pytest.approx(q.payload_bits / q.n_changes)
    assert q.psnr == pytest.approx(10 * np.log10(255 ** 2 / q.mse))


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rate", [0.0, 0.25, 0.5, 1.0])
def test_rs_and_ws_track_the_true_embedding_rate(real_covers, rate):
    """Both estimators must track the true rate *on average over covers*.

    Per-image accuracy is not the right assertion: RS and WS are both known to
    be biased on individual smooth images (WS in particular over-estimates when
    its local cover predictor is poor), which is exactly why the literature
    reports them averaged over a corpus.  The median over several covers is what
    the pipeline consumes, so it is what is tested here.
    """
    rs, ws = [], []
    for j, cover in enumerate(real_covers):
        stego = _lsb_embed(cover, rate, seed=int(rate * 100) + j + 1)
        rs.append(rs_analysis(stego)["estimated_rate"])
        ws.append(weighted_stego_estimate(stego)["estimated_rate"])
    assert np.all(np.isfinite(rs)), f"RS returned NaN at rate {rate}: {rs}"
    assert abs(float(np.median(rs)) - rate) < 0.12, f"RS median {np.median(rs)} vs {rate}"
    assert abs(float(np.median(ws)) - rate) < 0.15, f"WS median {np.median(ws)} vs {rate}"


def test_chi_square_separates_cover_from_fully_embedded(real_cover):
    assert chi_square_attack(real_cover)["p_embedded"] < 0.05
    assert chi_square_attack(_lsb_embed(real_cover, 1.0, 9))["p_embedded"] > 0.5


def test_payload_uniformity_flags_a_structured_payload():
    zeros = np.zeros(20000, dtype=np.uint8)
    unif = np.random.default_rng(0).integers(0, 2, 20000, dtype=np.uint8)
    assert payload_uniformity(zeros)["bias"] == pytest.approx(0.5)
    assert payload_uniformity(unif)["bias"] < 0.02


# --------------------------------------------------------------------------- #
def test_feature_extractors_are_deterministic_and_finite():
    img = _natural_image(128, 5)
    for fn in (spam686, srm_subset):
        a, b = fn(img), fn(img)
        assert np.array_equal(a, b)
        assert np.all(np.isfinite(a))
        assert a.ndim == 1 and a.size > 0


def test_features_differ_between_cover_and_stego():
    cover = _natural_image(128, 6)
    stego = _lsb_embed(cover, 1.0, 7)
    assert not np.allclose(spam686(cover), spam686(stego))


# --------------------------------------------------------------------------- #
def test_ensemble_detects_a_blatant_signal_and_not_a_null_one():
    rng = np.random.default_rng(0)
    n, d = 60, 40
    Xc = rng.normal(size=(n, d))
    Xs = Xc + 1.5                                  # trivially separable
    clf = EnsembleClassifier(d_sub=8, max_learners=40, random_state=0).fit(Xc, Xs)
    m = evaluate_scores(np.r_[np.zeros(n, int), np.ones(n, int)],
                        np.r_[clf.decision_function(Xc), clf.decision_function(Xs)])
    assert m.accuracy > 0.9 and m.auc > 0.95

    Xs_null = rng.normal(size=(n, d))              # same distribution -> chance
    clf2 = EnsembleClassifier(d_sub=8, max_learners=40, random_state=0).fit(Xc, Xs_null)
    assert 0.3 < clf2.oob_error_ <= 0.6


def test_roc_and_auc_are_sane():
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    perfect = np.r_[np.zeros(50), np.ones(50)]
    fpr, tpr, _ = roc_curve(y, perfect)
    assert auc(fpr, tpr) == pytest.approx(1.0)
    chance = np.zeros(100)
    fpr, tpr, _ = roc_curve(y, chance)
    assert 0.4 < auc(fpr, tpr) < 0.6


# --------------------------------------------------------------------------- #
def test_paired_tests_find_a_real_shift_and_ignore_a_null_one():
    rng = np.random.default_rng(0)
    base = rng.normal(50, 3, 40)
    better = base + 1.0
    c = compare_paired(better, base, "psnr")
    assert c.mean_diff == pytest.approx(1.0, abs=1e-9)
    assert c.preferred_p < 1e-6
    assert c.cohens_dz > 5

    same = base + rng.normal(0, 1e-3, 40)
    c2 = compare_paired(same, base, "psnr")
    assert c2.preferred_p > 0.001 or abs(c2.mean_diff) < 1e-3


def test_holm_correction_is_monotone_and_conservative():
    p = [0.001, 0.01, 0.04, 0.2]
    adj, rej = holm_bonferroni(p, 0.05)
    assert np.all(np.diff(adj) >= -1e-12)
    assert np.all(adj >= np.asarray(p) - 1e-12)
    assert rej.tolist() == [True, True, False, False]


def test_effect_below_noise_is_flagged():
    rng = np.random.default_rng(1)
    a = rng.normal(50, 5, 200)
    b = a - 0.05                     # real but tiny vs. an SD of 5
    c = compare_paired(a, b, "psnr")
    assert c.preferred_p < 0.05
    assert c.effect_below_noise


def test_cliffs_delta_bounds():
    assert cliffs_delta([3, 4, 5], [0, 1, 2]) == pytest.approx(1.0)
    assert cliffs_delta([0, 1, 2], [0, 1, 2]) == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
def test_classical_baselines_embed_what_they_claim():
    cover = _natural_image(128, 8)
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 2000, dtype=np.uint8)
    for fn in (lsb_replacement, lsb_matching, edge_adaptive_lsb, pvd):
        st, info = fn(cover, bits, rng)
        assert st.shape == cover.shape and st.dtype == np.uint8
        assert info["embedded_bits"] <= info["payload_bits"]
        assert info["n_changes"] == int(np.count_nonzero(st != cover))


def test_lsb_replacement_is_exactly_recoverable():
    cover = _natural_image(64, 9)
    bits = np.random.default_rng(0).integers(0, 2, 1000, dtype=np.uint8)
    st, _ = lsb_replacement(cover, bits, None)
    assert np.array_equal(st.reshape(-1)[:1000] & 1, bits)


@pytest.mark.parametrize("method", list(COST_FUNCTIONS))
def test_adaptive_simulator_hits_the_requested_payload(method):
    cover = _natural_image(128, 10)
    rng = np.random.default_rng(0)
    target = 3000
    st, info = embed_with_cost(cover, target, method, rng)
    assert abs(info["achieved_entropy_bits"] - target) / target < 0.05
    assert info["n_changes"] > 0
    assert st.dtype == np.uint8
    assert np.all(np.abs(st.astype(int) - cover.astype(int)) <= 1)
