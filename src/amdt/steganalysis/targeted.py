r"""Targeted (payload-statistics) attacks and key-space accounting.

These support Reviewer 1 comment 1 -- "what theoretical advantage do the
transformations provide against steganalysis?" -- with measurements rather than
assertion.  The claim being tested is narrow and falsifiable:

    T3/T4 make the embedded stream indistinguishable from uniform, therefore
    attacks that key on *payload* statistics lose their advantage.  They do
    **not** help against residual-based attacks (SRM, CNN).

The three attacks below are exactly the payload-statistics family:

``chi_square_attack``
    Westfeld & Pfitzmann (1999).  Tests whether the histogram's pairs of values
    (2i, 2i+1) have been equalised, which sequential LSB replacement of a
    biased payload does.  Returns :math:`p(\text{embedded})`.
``rs_analysis``
    Fridrich, Goljan & Du (2001) RS steganalysis; estimates the LSB embedding
    rate from the regular/singular group ratios.
``weighted_stego_estimate``
    Fridrich & Goljan (2004) WS estimator with Ker-Bohme variance weighting; a
    second, stronger and independent rate estimator.

If T4 is doing what the module claims, the *estimated rate* from RS/WS is
unchanged (they detect changes, not content) while the chi-square p-value drops
towards zero **only** for structured payloads embedded without T4.  Running
these three against ``no_decomp`` vs ``full`` is what turns comment 1 from a
rhetorical claim into a table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["chi_square_attack", "rs_analysis", "weighted_stego_estimate",
           "sample_pair_analysis", "payload_uniformity", "TargetedReport",
           "run_targeted_suite"]


# --------------------------------------------------------------------------- #
def chi_square_attack(img: np.ndarray, block: Optional[int] = None) -> Dict[str, float]:
    """Westfeld-Pfitzmann chi-square. Returns p(embedded) in [0, 1].

    ``block``: if given, the test is run on the first ``block`` pixels only
    (the classic sequential variant); otherwise on the whole image.
    """
    x = np.asarray(img).reshape(-1)
    if block:
        x = x[:block]
    h = np.bincount(x, minlength=256).astype(np.float64)
    even, odd = h[0::2], h[1::2]
    expected = (even + odd) / 2.0
    m = expected > 0
    if m.sum() < 2:
        return {"p_embedded": 0.0, "chi2": 0.0, "dof": 0}
    chi2 = float(np.sum((even[m] - expected[m]) ** 2 / expected[m]))
    dof = int(m.sum() - 1)
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, dof))
    except ImportError:
        # regularised upper incomplete gamma via a series fallback
        p = math.exp(-chi2 / 2) if dof <= 2 else float("nan")
    return {"p_embedded": p, "chi2": chi2, "dof": dof}


# --------------------------------------------------------------------------- #
_RS_MASK = (0, 1, 1, 0)


def _rs_pair(g: np.ndarray, f: np.ndarray) -> Tuple[float, float]:
    """(R, S) group fractions for flipping pattern ``f``."""
    flipped = np.where(f == 1, g ^ 1, np.where(f == -1, ((g + 1) ^ 1) - 1, g))
    v0 = np.abs(np.diff(g, axis=1)).sum(axis=1)
    v1 = np.abs(np.diff(flipped, axis=1)).sum(axis=1)
    return float(np.mean(v1 > v0)), float(np.mean(v1 < v0))


def rs_analysis(img: np.ndarray, mask: Sequence[int] = _RS_MASK) -> Dict[str, float]:
    """RS steganalysis estimate of the LSB-replacement rate p in [0, 1].

    Groups are non-overlapping horizontal runs *within a row* (crossing row
    boundaries corrupts the discrimination function on the first/last pixel of
    each row).  Validated in ``tests/test_targeted.py`` against synthetic
    embeddings at p in {0, 0.05, 0.1, 0.25, 0.5, 1.0}; the estimator tracks the
    true rate to within a few points, which is the accuracy RS is known for.
    """
    m = np.asarray(mask, dtype=np.int32)
    group = m.size
    x = np.asarray(img, dtype=np.int32)
    h, w = x.shape
    g = x[:, : (w // group) * group].reshape(-1, group)
    gf = g ^ 1                                   # p = 1 end point

    r_m, s_m = _rs_pair(g, m)
    r_neg, s_neg = _rs_pair(g, -m)
    r_m1, s_m1 = _rs_pair(gf, m)
    r_neg1, s_neg1 = _rs_pair(gf, -m)

    d0, d1 = r_m - s_m, r_m1 - s_m1
    dm0, dm1 = r_neg - s_neg, r_neg1 - s_neg1
    a = 2 * (d1 + d0)
    b = dm0 - dm1 - d1 - 3 * d0
    c = d0 - dm0
    disc = b * b - 4 * a * c
    if abs(a) < 1e-12 or disc < 0:
        # Near p = 1 the R/S curves meet and the quadratic degenerates (negative
        # discriminant).  This is a known limit of RS, not a bug; fall back to
        # the linear solution instead of returning NaN, which would silently
        # drop the highest-payload rows out of every average downstream.
        z = -c / b if abs(b) > 1e-12 else float("nan")
    else:
        r1 = (-b + math.sqrt(disc)) / (2 * a)
        r2 = (-b - math.sqrt(disc)) / (2 * a)
        z = r1 if abs(r1) < abs(r2) else r2
    p = z / (z - 0.5) if np.isfinite(z) and (z - 0.5) != 0 else float("nan")
    p = float(np.clip(p, 0.0, 1.0)) if np.isfinite(p) else float("nan")
    return {"estimated_rate": p, "detected": bool(np.isfinite(p) and p > 0.05)}


# --------------------------------------------------------------------------- #
def weighted_stego_estimate(img: np.ndarray, sigma_floor: float = 5.0) -> Dict[str, float]:
    r"""Weighted Stego-image (WS) rate estimator.

    Fridrich & Goljan (2004), with the Ker & Bohme (2008) local-variance
    weighting.  Chosen over Sample Pair Analysis because WS is both stronger and
    much easier to implement correctly -- a subtly wrong SPA is worse than no
    SPA.  With :math:`\bar s_n` the LSB-flipped pixel and :math:`\hat s_n` a
    local cover estimate,

    .. math:: \hat p = 2 \sum_n w_n (s_n - \hat s_n)(s_n - \bar s_n),
              \qquad w_n \propto (\sigma_n^2 + \sigma_0)^{-1}.
    """
    s = np.asarray(img, dtype=np.float64)
    f = np.array([[-1, 2, -1], [2, 0, 2], [-1, 2, -1]], dtype=np.float64) / 4.0
    p = np.pad(s, 1, mode="symmetric")
    win = np.lib.stride_tricks.sliding_window_view(p, (3, 3))
    est = np.einsum("ijkl,kl->ij", win, f)
    bar = s + (1.0 - 2.0 * (np.asarray(img) & 1).astype(np.float64))
    loc = win.mean(axis=(2, 3))
    var = np.maximum((win ** 2).mean(axis=(2, 3)) - loc ** 2, 0.0)
    w = 1.0 / (sigma_floor + var)
    w /= w.sum()
    rate = float(2.0 * np.sum(w * (s - est) * (s - bar)))
    rate = float(np.clip(rate, 0.0, 1.0))
    return {"estimated_rate": rate, "detected": bool(rate > 0.05)}


#: Backwards-compatible alias -- the pipeline reports this column as "WS".
sample_pair_analysis = weighted_stego_estimate


# --------------------------------------------------------------------------- #
def payload_uniformity(bits: np.ndarray, block: int = 8) -> Dict[str, float]:
    """How close the *transformed payload* is to i.i.d. uniform.

    This is the direct measurement of the T4 claim: ones-bias, first-order
    autocorrelation, and the NIST-style block-frequency chi-square.  A payload
    that passes here cannot imprint its own statistics on the LSB plane.
    """
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    L = b.size
    if L < 2 * block:
        return {"ones_fraction": float("nan"), "bias": float("nan"),
                "autocorr_lag1": float("nan"), "block_chi2_p": float("nan"),
                "shannon_entropy": float("nan")}
    ones = float(b.mean())
    ac = float(np.corrcoef(b[:-1], b[1:])[0, 1]) if b.std() > 0 else 0.0
    nb = L // block
    counts = b[: nb * block].reshape(nb, block).sum(axis=1)
    chi2 = float(4.0 * block * np.sum((counts / block - 0.5) ** 2))
    try:
        from scipy.stats import chi2 as chi2_dist
        p = float(chi2_dist.sf(chi2, nb))
    except ImportError:
        p = float("nan")
    q = np.clip([1 - ones, ones], 1e-12, 1)
    ent = float(-np.sum(q * np.log2(q)))
    return {"ones_fraction": ones, "bias": abs(ones - 0.5), "autocorr_lag1": ac,
            "block_chi2_p": p, "shannon_entropy": ent}


# --------------------------------------------------------------------------- #
@dataclass
class TargetedReport:
    chi_square_p: float
    rs_rate: float
    ws_rate: float
    payload_bias: float
    payload_autocorr: float
    payload_entropy: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def run_targeted_suite(stego: np.ndarray, embedded_bits: Optional[np.ndarray] = None
                       ) -> TargetedReport:
    cs = chi_square_attack(stego)
    rs = rs_analysis(stego)
    sp = weighted_stego_estimate(stego)
    pu = (payload_uniformity(embedded_bits) if embedded_bits is not None
          else {"bias": float("nan"), "autocorr_lag1": float("nan"),
                "shannon_entropy": float("nan")})
    return TargetedReport(
        chi_square_p=cs["p_embedded"],
        rs_rate=rs["estimated_rate"],
        ws_rate=sp["estimated_rate"],
        payload_bias=pu["bias"],
        payload_autocorr=pu["autocorr_lag1"],
        payload_entropy=pu["shannon_entropy"],
    )
