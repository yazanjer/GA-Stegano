"""Modern content-adaptive baselines: WOW, S-UNIWARD, HILL, MiPOD-lite.

Reviewer 1, comment 7: "The current benchmark methods are mainly classical
approaches. Please include comparisons with more recent adaptive and
learning-based steganography methods."

These four cost-based schemes are the standard modern comparison set in the
steganalysis literature (Holub & Fridrich 2012, 2013; Li et al. 2014;
Sedighi et al. 2016).  They are *distortion-minimising* rather than
capacity-maximising: a cost :math:`\\rho_{ij}` is assigned to changing each
pixel by :math:`\\pm 1`, and the payload is embedded so as to minimise the total
cost for a given payload.

Two honest caveats, both surfaced in the emitted ``info`` dict:

1. **Simulator, not STC.** Practical implementations use Syndrome-Trellis Codes
   to approach the rate-distortion bound.  Here the standard *payload-limited
   simulator* is used: change probabilities are derived from the costs by
   solving for the Lagrange multiplier :math:`\\lambda` such that the ternary
   entropy equals the payload, then changes are drawn from that distribution.
   This is the accepted way to benchmark cost functions (it is what the
   original papers report), and it is slightly *optimistic* relative to real
   STC coding -- so it favours the baselines, not the proposed method.
2. **No blind extractability.** The simulator does not produce a decodable
   stego object; it produces the correct *statistical* stego object for
   detectability comparison.  Capacity/imperceptibility numbers are therefore
   comparable, but these baselines are excluded from the round-trip extraction
   test, and the tables mark them accordingly.

A learning-based comparator (an ASDL-GAN / SteganoGAN-style generator) is
deliberately **not** faked here.  If a learned cost map is required, train one
and drop it in via :func:`register_cost_map`; the pipeline consumes any
``(cover) -> cost`` callable.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

__all__ = ["wow_cost", "suniward_cost", "hill_cost", "mipod_cost",
           "COST_FUNCTIONS", "register_cost_map", "simulate_embedding",
           "embed_with_cost"]

_WET = 1e10  # cost of a "wet" (unusable) pixel


# --------------------------------------------------------------------------- #
# convolution helpers (numpy only, mirror padding as in the reference code)
# --------------------------------------------------------------------------- #
def _conv2_mirror(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    kh, kw = k.shape
    p = np.pad(x, ((kh // 2, kh - 1 - kh // 2), (kw // 2, kw - 1 - kw // 2)), mode="symmetric")
    win = np.lib.stride_tricks.sliding_window_view(p, (kh, kw))
    return np.einsum("ijkl,kl->ij", win, k)


def _daubechies8_filters() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D wavelet filter bank (LH, HL, HH) from the db8 QMF pair."""
    try:
        import pywt
        w = pywt.Wavelet("db8")
        lo = np.asarray(w.dec_lo)[::-1]
        hi = np.asarray(w.dec_hi)[::-1]
    except Exception:  # pragma: no cover - pywt is a pinned dependency
        raise ImportError("PyWavelets is required for WOW / S-UNIWARD costs")
    return (np.outer(lo, hi), np.outer(hi, lo), np.outer(hi, hi))


# --------------------------------------------------------------------------- #
# cost functions
# --------------------------------------------------------------------------- #
def wow_cost(cover: np.ndarray, p: float = -1.0) -> np.ndarray:
    """WOW (Holub & Fridrich, WIFS 2012), reciprocal-Holder aggregation."""
    x = cover.astype(np.float64)
    xi = []
    for f in _daubechies8_filters():
        r = _conv2_mirror(x, f)
        xi.append(_conv2_mirror(np.abs(r), np.abs(f[::-1, ::-1])))
    xi = np.stack(xi)
    with np.errstate(divide="ignore"):
        rho = (np.sum(xi ** p, axis=0)) ** (-1.0 / p)
    rho = np.nan_to_num(rho, nan=_WET, posinf=_WET)
    return np.clip(rho, 0.0, _WET)


def suniward_cost(cover: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """S-UNIWARD (Holub, Fridrich & Denemark, EURASIP JIS 2014)."""
    x = cover.astype(np.float64)
    rho = np.zeros_like(x)
    for f in _daubechies8_filters():
        r = _conv2_mirror(x, f)
        w = 1.0 / (np.abs(r) + sigma)
        rho += _conv2_mirror(w, np.abs(f[::-1, ::-1]))
    return np.clip(np.nan_to_num(rho, nan=_WET, posinf=_WET), 0.0, _WET)


def hill_cost(cover: np.ndarray) -> np.ndarray:
    """HILL (Li, Wang, Huang & Li, ICIP 2014): high-pass then two low-passes."""
    x = cover.astype(np.float64)
    hpf = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64)
    l1 = np.ones((3, 3)) / 9.0
    l2 = np.ones((15, 15)) / 225.0
    r = np.abs(_conv2_mirror(x, hpf))
    with np.errstate(divide="ignore"):
        inv = 1.0 / (_conv2_mirror(r, l1) + 1e-10)
    rho = _conv2_mirror(inv, l2)
    return np.clip(np.nan_to_num(rho, nan=_WET, posinf=_WET), 0.0, _WET)


def mipod_cost(cover: np.ndarray, block: int = 9) -> np.ndarray:
    """MiPOD-lite: cost inversely proportional to local residual variance.

    A simplified stand-in for the full MiPOD Fisher-information model
    (Sedighi, Cogranne & Fridrich, TIFS 2016): the variance estimate uses a
    local window instead of the 2-D Wiener + PCA denoiser.  Labelled "-lite"
    everywhere it is reported so it is never mistaken for the published method.
    """
    x = cover.astype(np.float64)
    k = np.ones((block, block)) / (block * block)
    mu = _conv2_mirror(x, k)
    var = np.maximum(_conv2_mirror(x * x, k) - mu * mu, 1e-6)
    return np.clip(1.0 / var, 0.0, _WET)


COST_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "WOW": wow_cost,
    "S-UNIWARD": suniward_cost,
    "HILL": hill_cost,
    "MiPOD-lite": mipod_cost,
}


def register_cost_map(name: str, fn: Callable[[np.ndarray], np.ndarray]) -> None:
    """Plug in a learned or custom cost map (e.g. a trained generator)."""
    COST_FUNCTIONS[name] = fn


# --------------------------------------------------------------------------- #
# payload-limited sender simulator
# --------------------------------------------------------------------------- #
def _ternary_entropy(p: np.ndarray) -> float:
    q = np.clip(p, 1e-30, 1.0)
    r = np.clip(1.0 - 2.0 * p, 1e-30, 1.0)
    return float(-np.sum(2.0 * q * np.log2(q) + r * np.log2(r)))


def _lambda_for_payload(rho: np.ndarray, payload_bits: float,
                        tol: float = 1e-3, max_iter: int = 60) -> float:
    """Bisection on lambda so that H_3(p(lambda)) = payload_bits."""
    lo, hi = 1e-6, 1e6
    for _ in range(max_iter):
        mid = np.sqrt(lo * hi)
        p = _change_probs(rho, mid)
        h = _ternary_entropy(p)
        if abs(h - payload_bits) <= tol * max(1.0, payload_bits):
            return mid
        if h > payload_bits:
            lo = mid          # too much capacity -> raise lambda
        else:
            hi = mid
    return np.sqrt(lo * hi)


def _change_probs(rho: np.ndarray, lam: float) -> np.ndarray:
    e = np.exp(-lam * np.clip(rho, 0.0, 50.0 / max(lam, 1e-12)))
    return e / (1.0 + 2.0 * e)


def simulate_embedding(cover: np.ndarray, rho: np.ndarray, payload_bits: int,
                       rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    """Draw +-1 changes from the optimal payload-limited distribution."""
    lam = _lambda_for_payload(rho, float(payload_bits))
    p = _change_probs(rho, lam)
    u = rng.random(cover.shape)
    delta = np.zeros(cover.shape, dtype=np.int16)
    delta[u < p] = 1
    delta[(u >= p) & (u < 2 * p)] = -1
    st = cover.astype(np.int16) + delta
    # boundary handling: 0 cannot go down, 255 cannot go up -> flip the sign
    st = np.where(st < 0, 1, st)
    st = np.where(st > 255, 254, st)
    st = st.astype(np.uint8)
    return st, {
        "lambda": lam,
        "expected_changes": float(2.0 * p.sum()),
        "achieved_entropy_bits": _ternary_entropy(p),
    }


def embed_with_cost(cover: np.ndarray, payload_bits: int, method: str,
                    rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    """Full adaptive-baseline pipeline: cost map -> simulator -> stego."""
    if method not in COST_FUNCTIONS:
        raise KeyError(f"unknown adaptive method {method!r}; have {list(COST_FUNCTIONS)}")
    rho = COST_FUNCTIONS[method](cover)
    st, sim = simulate_embedding(cover, rho, payload_bits, rng)
    info = {
        "method": method,
        "payload_bits": int(payload_bits),
        "embedded_bits": int(payload_bits),
        "n_changes": int(np.count_nonzero(st != cover)),
        "capacity_limited": False,
        "blind_extractable": False,
        "simulator": "payload-limited sender (no STC); optimistic vs. real coding",
    }
    info.update(sim)
    return st, info
