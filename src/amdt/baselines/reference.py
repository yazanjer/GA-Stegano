"""Reference baselines for revision 2 (Reviewer 1 comments 4, 12; Reviewer 2 comments 1-3).

Three groups, all behind the registry signature ``fn(cover, payload, rng) -> (stego, info)``:

1. **Cost functions from a pinned reference implementation.**  WOW, S-UNIWARD,
   HILL and (full) MiPOD costs come from ``conseal`` (University of Innsbruck,
   pinned version recorded in the run provenance), not from our own ports.
   This replaces the v1 in-house costs and the simplified "MiPOD-lite".

2. **Two post-2024 spatial-domain methods, re-implemented under the identical
   protocol** (same BOSSBase covers, payloads, key, extraction check and
   steganalysis):

   * ``EvoHILL-STC`` -- the Evolved-HILL cost of Wang, Yi & Wu (IS&T
     Electronic Imaging 2026 / arXiv:2607.05868, Eqs. (7)-(9)): median-residual
     fusion ``R_F = sqrt(|R_H * R_M|)``, ``xi = R_F (*) L1`` (3x3 mean) and a
     variance-gated cost ``rho = 1/((xi + eps) * clip(V3x3(C), eps, 5))``,
     embedded with the STC of :mod:`amdt.stego.stc`.  The paper gives no
     low-pass spreading after the gating, so none is applied.
   * ``FM-PSO-LSB`` -- Aljughaiman & Alrawashdeh, "Optimization-driven
     steganographic system based on fused maps and Blowfish encryption",
     Scientific Reports (2026): a fused map ``F = a*E + (1-a)*N`` of the 9x9
     Shannon-entropy map and the normalised Laplacian noise map, priority
     ``F**g``, 1 bit/pixel LSB substitution in descending priority, PSO (18
     particles, 25 iterations) over ``a in [0,1], g in [0.5,5]`` minimising
     MSE.  Two documented adaptations make it blindly decodable and comparable:
     the maps are computed on the LSB-cleared image (``x & 0xFE``) so the
     receiver can recompute the ranking from the stego image, and the Blowfish
     layer is replaced by the same HMAC-SHA256 keystream used by AMDT (with a
     uniformly random payload, the encryption layer does not change any
     measured quantity).  ``(a, g)`` are shared side information, as in the
     original.  Because ``F**g`` is monotone in ``F >= 0``, ``g`` cannot change
     the ranking; it is kept for fidelity.

3. **Complete STC-coded versions** (``<cost>-STC``) of every cost function, so
   that "cost-function simulation" and "complete steganographic system" can be
   reported separately, as Reviewer 2 requested.  The payload-limited
   simulator versions (``<cost>-SIM``) are kept for continuity with v1.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from ..stego.stc import stc_embed_image, stc_extract_image
from ..stego.decomposition import keystream_bits

__all__ = ["cost_pair", "COSTS", "simulate", "stc_system", "stc_extract",
           "evohill_cost", "fm_pso_lsb", "fm_pso_extract", "BASELINE_SPEC"]

_WET = 1e10


def _conseal():
    import conseal
    return conseal


def _mirror_conv(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    from scipy.ndimage import correlate
    return correlate(x, k, mode="mirror")


def evohill_cost(cover: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Evolved HILL (Wang, Yi & Wu 2026), Eqs. (7)-(9)."""
    from scipy.ndimage import median_filter, uniform_filter
    c = cover.astype(np.float64)
    kb = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float64)
    r_h = _mirror_conv(c, kb)
    r_m = c - median_filter(c, size=3, mode="mirror")
    r_f = np.sqrt(np.abs(r_h * r_m))
    xi = uniform_filter(r_f, size=3, mode="mirror")
    mu = uniform_filter(c, size=3, mode="mirror")
    var = uniform_filter(c * c, size=3, mode="mirror") - mu * mu
    v = np.clip(var, eps, 5.0)
    rho = 1.0 / ((xi + eps) * v)
    return np.clip(np.nan_to_num(rho, nan=_WET, posinf=_WET), 0, _WET)


def _adjust(cover: np.ndarray, rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rp, rm = rho.copy(), rho.copy()
    rp[cover == 255] = _WET
    rm[cover == 0] = _WET
    return rp, rm


def cost_pair(name: str, cover: np.ndarray, rate_bpp: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """(rho_+1, rho_-1) for a cost-based scheme."""
    cl = _conseal()
    x = np.asarray(cover, dtype=np.uint8)
    if name == "HILL":
        r = cl.hill.compute_cost_adjusted(x)
    elif name == "S-UNIWARD":
        r = cl.suniward.compute_cost_adjusted(x)
    elif name == "WOW":
        r = cl.wow.compute_cost_adjusted(x)
    elif name == "MiPOD":
        (p, _), _ = cl.mipod.probability(x, rate_bpp)
        p = np.clip(p, 1e-12, 1 / 3 - 1e-12)
        rho = np.log(1.0 / p - 2.0)
        return _adjust(x, rho)
    elif name == "EvoHILL":
        return _adjust(x, evohill_cost(x))
    else:
        raise KeyError(name)
    if isinstance(r, tuple) and len(r) == 2:
        return np.asarray(r[0], float), np.asarray(r[1], float)
    r = np.asarray(r[0] if isinstance(r, tuple) else r, float)
    return _adjust(x, r)


COSTS = ("HILL", "S-UNIWARD", "WOW", "MiPOD", "EvoHILL")


def simulate(name: str, cover: np.ndarray, payload_bits: int,
             rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    """Payload-limited optimal simulator (reference: conseal) -- NOT decodable."""
    cl = _conseal()
    x = np.asarray(cover, dtype=np.uint8)
    alpha = payload_bits / x.size
    seed = int(rng.integers(0, 2**31 - 1))
    mod = {"HILL": cl.hill, "S-UNIWARD": cl.suniward, "WOW": cl.wow, "MiPOD": cl.mipod}.get(name)
    if mod is None:
        raise KeyError(f"no reference simulator for {name!r}")
    st = mod.simulate_single_channel(x, alpha, seed=seed)
    st = np.asarray(st).astype(np.uint8)
    return st, {"payload_bits": int(payload_bits), "embedded_bits": int(payload_bits),
                "n_changes": int(np.count_nonzero(st != x)), "blind_extractable": False,
                "capacity_limited": False, "coding": "simulator (payload-limited sender)"}


def stc_system(name: str, cover: np.ndarray, payload: np.ndarray, key: bytes,
               rng: np.random.Generator, h: int = 10) -> Tuple[np.ndarray, Dict[str, object]]:
    rate = np.asarray(payload).size / np.asarray(cover).size
    rp, rm = cost_pair(name, cover, rate)
    st, info = stc_embed_image(cover, rp, rm, payload, key + name.encode(), rng, h)
    info["coding"] = f"STC h={h}"
    return st, info


def stc_extract(name: str, stego: np.ndarray, key: bytes, h: int = 10) -> np.ndarray:
    return stc_extract_image(stego, key + name.encode(), h)


# --------------------------------------------------------------------------- #
# FM-PSO-LSB (Aljughaiman & Alrawashdeh, Sci. Rep. 2026)
# --------------------------------------------------------------------------- #
def _entropy9(x: np.ndarray) -> np.ndarray:
    from skimage.filters.rank import entropy
    return entropy(x.astype(np.uint8), np.ones((9, 9), dtype=bool)).astype(np.float64)


def _fm_maps(cover: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = (np.asarray(cover, dtype=np.uint8) & 0xFE)
    e = _entropy9(z)
    lap = np.abs(_mirror_conv(z.astype(np.float64),
                              np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)))
    e = e / e.max() if e.max() > 0 else e
    lap = lap / lap.max() if lap.max() > 0 else lap
    return e, lap


def _fm_order(e: np.ndarray, n: np.ndarray, a: float, g: float, top: int = -1) -> np.ndarray:
    """Pixel indices in descending priority, ties broken by index (stable).

    With ``top = L`` only the first ``L`` entries are produced, via an O(n)
    partition followed by a sort of the selected entries; the result is
    identical to ``np.argsort(-f, kind="stable")[:L]``.
    """
    f = (a * e + (1.0 - a) * n).reshape(-1) ** g
    if top < 0 or top >= f.size:
        return np.argsort(-f, kind="stable")
    thr = np.partition(f, f.size - top)[f.size - top]          # L-th largest value
    above = np.flatnonzero(f > thr)
    ties = np.flatnonzero(f == thr)[: top - above.size]
    sel = np.concatenate([above, ties])
    return sel[np.lexsort((sel, -f[sel]))]


def fm_pso_lsb(cover: np.ndarray, payload: np.ndarray, key: bytes, rng: np.random.Generator,
               particles: int = 18, iters: int = 25, w: float = 0.7, c1: float = 1.5,
               c2: float = 1.5) -> Tuple[np.ndarray, Dict[str, object]]:
    x = np.asarray(cover, dtype=np.uint8)
    flat = x.reshape(-1)
    bits = (np.asarray(payload, dtype=np.uint8).reshape(-1)
            ^ keystream_bits(key + b"FM-PSO", "enc", np.asarray(payload).size))
    L = bits.size
    e, lap = _fm_maps(x)
    lo, hi = np.array([0.0, 0.5]), np.array([1.0, 5.0])

    def emb(a, g):
        idx = _fm_order(e, lap, a, g, L)
        st = flat.copy()
        st[idx] = (st[idx] & 0xFE) | bits
        return st

    def fit(pos):
        st = emb(*pos)
        d = st.astype(np.float64) - flat
        return float(np.mean(d * d))

    pos = lo + rng.random((particles, 2)) * (hi - lo)
    vel = (rng.random((particles, 2)) - 0.5) * (hi - lo) * 0.2
    pf = np.array([fit(p) for p in pos])
    pbest, pbf = pos.copy(), pf.copy()
    gi = int(np.argmin(pbf)); gbest, gbf = pbest[gi].copy(), pbf[gi]
    for _ in range(iters):
        r1, r2 = rng.random((particles, 2)), rng.random((particles, 2))
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        f = np.array([fit(p) for p in pos])
        better = f < pbf
        pbest[better], pbf[better] = pos[better], f[better]
        gi = int(np.argmin(pbf))
        if pbf[gi] < gbf:
            gbest, gbf = pbest[gi].copy(), pbf[gi]
    st = emb(*gbest).reshape(x.shape)
    return st, {"payload_bits": int(L), "embedded_bits": int(L),
                "n_changes": int(np.count_nonzero(st != x)), "blind_extractable": True,
                "capacity_limited": False, "fm_alpha": float(gbest[0]), "fm_gamma": float(gbest[1]),
                "pso_evaluations": int(particles * (iters + 1))}


def fm_pso_extract(stego: np.ndarray, n_bits: int, a: float, g: float, key: bytes) -> np.ndarray:
    e, lap = _fm_maps(stego)
    idx = _fm_order(e, lap, a, g, n_bits)
    bits = np.asarray(stego, dtype=np.uint8).reshape(-1)[idx] & 1
    return (bits ^ keystream_bits(key + b"FM-PSO", "enc", n_bits)).astype(np.uint8)


#: Machine-readable baseline specification (Reviewer 2, comment 3).
BASELINE_SPEC: Dict[str, Dict[str, str]] = {
    "LSB": {"source": "this repo, baselines/classical.py", "rule": "LSB replacement, raster order",
            "params": "none", "boundary": "n/a", "decodable": "yes"},
    "LSB-M": {"source": "this repo", "rule": "LSB matching (+-1, random sign), key-seeded pixel permutation",
              "params": "none", "boundary": "0 -> +1, 255 -> -1", "decodable": "yes"},
    "EA-LSB": {"source": "this repo (after Luo et al. 2010)", "rule": "LSB replacement in descending Sobel magnitude of x & 0xFE",
               "params": "threshold adapts to payload", "boundary": "n/a", "decodable": "yes"},
    "PVD": {"source": "this repo (Wu & Tsai 2003)", "rule": "pixel-value differencing, range table {8,8,16,32,64,128}",
            "params": "non-overlapping horizontal pairs, raster", "boundary": "fall-off-boundary pairs skipped", "decodable": "yes"},
    "GA-FT": {"source": "this repo", "rule": "AMDT with direction/offsets fixed (raster), 1 segment",
              "params": "same GA as AMDT", "boundary": "n/a", "decodable": "yes"},
    "HILL": {"source": "conseal (pinned)", "rule": "KB high-pass, 3x3 and 15x15 averaging (Li et al. 2014)",
             "params": "wet cost 1e10 for infeasible +-1", "boundary": "mirror", "decodable": "STC: yes / SIM: no"},
    "S-UNIWARD": {"source": "conseal (pinned)", "rule": "db8 directional residuals (Holub et al. 2014)",
                  "params": "sigma = 1", "boundary": "mirror", "decodable": "STC: yes / SIM: no"},
    "WOW": {"source": "conseal (pinned)", "rule": "db8 residuals, Holder aggregation (Holub & Fridrich 2012)",
            "params": "p = -1", "boundary": "mirror", "decodable": "STC: yes / SIM: no"},
    "MiPOD": {"source": "conseal (pinned), full MiPOD", "rule": "Fisher-information model (Sedighi et al. 2016)",
              "params": "Wiener 2x2 + 9x9 variance, rho = ln(1/p - 2)", "boundary": "reference", "decodable": "STC: yes / SIM: no"},
    "EvoHILL": {"source": "this repo, from Wang, Yi & Wu 2026", "rule": "median-residual fusion + variance gating",
                "params": "eps = 1e-8, variance clipped to [eps, 5]", "boundary": "mirror", "decodable": "STC: yes"},
    "FM-PSO-LSB": {"source": "this repo, from Aljughaiman & Alrawashdeh 2026", "rule": "fused entropy/Laplacian priority, LSB substitution",
                   "params": "PSO 18 x 25, w = 0.7, c1 = c2 = 1.5; maps on x & 0xFE", "boundary": "n/a", "decodable": "yes"},
    "STC": {"source": "this repo, stego/stc.py", "rule": "binary STC on LSB plane, +-1 by cheaper sign",
            "params": "h = 10, key-seeded H_hat and pixel permutation, 32-bit length header", "boundary": "0 -> +1, 255 -> -1", "decodable": "yes"},
}
