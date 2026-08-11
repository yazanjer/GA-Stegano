"""Classical baselines: LSB, EA-LSB, GA-FT, PVD.

These reproduce the four comparison methods named in the manuscript.  Every
baseline exposes the same signature so the experiment driver treats them
uniformly::

    stego, info = method(cover, payload_bits, rng, **kwargs)

``info`` always reports ``payload_bits``, ``n_changes`` and ``embedded_bits``
(which can be *less* than requested for capacity-limited methods such as PVD --
silently truncating and then reporting the requested payload is a common way
these comparisons get inflated, so the driver asserts on the difference).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = ["lsb_replacement", "lsb_matching", "edge_adaptive_lsb", "pvd", "ga_fixed_traversal"]


def _info(cover: np.ndarray, stego: np.ndarray, requested: int, embedded: int,
          **extra) -> Dict[str, object]:
    d = {
        "payload_bits": int(requested),
        "embedded_bits": int(embedded),
        "n_changes": int(np.count_nonzero(cover != stego)),
        "capacity_limited": embedded < requested,
    }
    d.update(extra)
    return d


# --------------------------------------------------------------------------- #
def lsb_replacement(cover: np.ndarray, payload: np.ndarray,
                    rng: np.random.Generator | None = None,
                    random_order: bool = False) -> Tuple[np.ndarray, Dict[str, object]]:
    """Plain LSB substitution in raster order (the manuscript's ``LSB``).

    ``random_order=True`` gives the keyed-scatter variant (LSB-R with a
    pseudo-random path), which is the fairer modern form of the baseline.
    """
    st = cover.copy().reshape(-1)
    b = np.asarray(payload, dtype=np.uint8).reshape(-1)
    n = min(b.size, st.size)
    idx = (rng.permutation(st.size)[:n] if (random_order and rng is not None)
           else np.arange(n))
    st[idx] = (st[idx] & np.uint8(0xFE)) | b[:n]
    st = st.reshape(cover.shape)
    return st, _info(cover, st, b.size, n, method="LSB")


def lsb_matching(cover: np.ndarray, payload: np.ndarray,
                 rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, object]]:
    """LSB matching (+-1 embedding) -- strictly harder to detect than LSB-R."""
    st = cover.copy().reshape(-1).astype(np.int16)
    b = np.asarray(payload, dtype=np.uint8).reshape(-1)
    n = min(b.size, st.size)
    idx = rng.permutation(st.size)[:n]
    cur = (st[idx] & 1).astype(np.uint8)
    need = cur != b[:n]
    step = rng.choice(np.array([-1, 1], dtype=np.int16), size=int(need.sum()))
    vals = st[idx][need] + step
    vals = np.where(vals < 0, 1, vals)
    vals = np.where(vals > 255, 254, vals)
    tmp = st[idx]
    tmp[need] = vals
    st[idx] = tmp
    st = st.astype(np.uint8).reshape(cover.shape)
    return st, _info(cover, st, b.size, n, method="LSB-M")


# --------------------------------------------------------------------------- #
def _sobel_magnitude(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float64)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = kx.T
    p = np.pad(x, 1, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(p, (3, 3))
    gx = np.einsum("ijkl,kl->ij", win, kx)
    gy = np.einsum("ijkl,kl->ij", win, ky)
    return np.hypot(gx, gy)


def edge_adaptive_lsb(cover: np.ndarray, payload: np.ndarray,
                      rng: np.random.Generator,
                      percentile: float | None = None
                      ) -> Tuple[np.ndarray, Dict[str, object]]:
    """Edge-adaptive LSB (Luo-style): embed in the highest-gradient pixels first.

    The threshold adapts to the payload, exactly as in edge-adaptive schemes:
    only as many edge pixels as the payload needs are used, so short payloads
    stay entirely in textured regions where change is least visible and least
    detectable.
    """
    b = np.asarray(payload, dtype=np.uint8).reshape(-1)
    mag = _sobel_magnitude(cover).reshape(-1)
    order = np.argsort(-mag, kind="stable")
    n = min(b.size, order.size)
    idx = order[:n]
    st = cover.copy().reshape(-1)
    st[idx] = (st[idx] & np.uint8(0xFE)) | b[:n]
    st = st.reshape(cover.shape)
    thr = float(mag[order[n - 1]]) if n else float("nan")
    return st, _info(cover, st, b.size, n, method="EA-LSB", gradient_threshold=thr)


# --------------------------------------------------------------------------- #
_PVD_RANGES = ((0, 7, 3), (8, 15, 3), (16, 31, 4), (32, 63, 5),
               (64, 127, 6), (128, 255, 7))


def _pvd_capacity_bits(d: int) -> Tuple[int, int, int]:
    a = abs(int(d))
    for lo, hi, t in _PVD_RANGES:
        if lo <= a <= hi:
            return t, lo, hi
    return 7, 128, 255


def pvd(cover: np.ndarray, payload: np.ndarray,
        rng: np.random.Generator | None = None) -> Tuple[np.ndarray, Dict[str, object]]:
    """Wu & Tsai pixel-value differencing on horizontal non-overlapping pairs.

    Capacity is image-dependent, so ``embedded_bits`` may fall short of the
    request; the driver reports the shortfall rather than pretending the full
    payload fitted.  Falling-off-boundary pairs are skipped, which is what keeps
    the scheme reversible at the receiver.
    """
    b = np.asarray(payload, dtype=np.uint8).reshape(-1)
    st = cover.copy().astype(np.int32)
    h, w = st.shape
    pos = 0
    for i in range(h):
        for j in range(0, w - 1, 2):
            if pos >= b.size:
                break
            p1, p2 = int(st[i, j]), int(st[i, j + 1])
            d = p2 - p1
            t, lo, hi = _pvd_capacity_bits(d)
            take = min(t, b.size - pos)
            if take <= 0:
                break
            val = 0
            for k in range(take):
                val = (val << 1) | int(b[pos + k])
            val <<= (t - take)                     # pad low bits with zeros
            d_new = lo + val if d >= 0 else -(lo + val)
            m = d_new - d
            n1 = p1 - int(np.ceil(m / 2))
            n2 = p2 + int(np.floor(m / 2))
            if not (0 <= n1 <= 255 and 0 <= n2 <= 255):
                continue                            # falling-off-boundary: skip
            st[i, j], st[i, j + 1] = n1, n2
            pos += take
        if pos >= b.size:
            break
    st = st.astype(np.uint8)
    return st, _info(cover, st, b.size, pos, method="PVD")


# --------------------------------------------------------------------------- #
def ga_fixed_traversal(cover: np.ndarray, payload: np.ndarray,
                       rng: np.random.Generator,
                       key: bytes = b"\x00" * 32,
                       population: int = 25, generations: int = 100,
                       patience: int = 25) -> Tuple[np.ndarray, Dict[str, object]]:
    """GA with a *fixed* raster traversal -- the manuscript's ``GA-FT``.

    Implemented as the AMDT ablation ``fixed_path`` so the only difference from
    the proposed method is the frozen direction/offset genes.  This is the
    comparison that isolates the contribution of multi-directional traversal;
    re-implementing GA-FT as separate code would confound it with
    implementation differences.
    """
    from ..ga.optimizer import GAConfig
    from ..stego.amdt import run_amdt

    res = run_amdt(
        cover, payload, key,
        GAConfig(population=population, generations=generations,
                 patience=patience, n_segments=1),
        rng, variant="fixed_path",
    )
    return res.stego, _info(
        cover, res.stego, payload.size, payload.size, method="GA-FT",
        ga_evaluations=res.ga.evaluations, ga_wall_time_s=res.ga.wall_time_s,
    )
