"""Syndrome-trellis codes (STC) for complete, decodable cost-based embedding.

Reviewer 2 (comment 2) pointed out that the cost-based baselines were run only
under the payload-limited *simulator*, which produces statistically correct but
non-decodable stego images, while AMDT is a complete encoder/decoder.  This
module supplies the missing coding layer so that HILL, S-UNIWARD, WOW, MiPOD
and the 2026 Evolved-HILL cost can be run as *complete* steganographic systems
whose payload is actually extracted and checked.

Code.  Binary STC (Filler, Judas & Fridrich, IEEE TIFS 6(3), 2011) on the LSB
plane with constraint height ``h`` (default 10).  The parity-check matrix H is
the usual band of copies of a random ``h x w`` sub-matrix H_hat whose columns
have their first and last bit set; the column widths ``w_i`` are distributed as
``floor((i+1)n/m) - floor(i n/m)`` so that exactly the whole cover sequence of
length ``n`` is used for any rate ``m/n``.  Rows beyond the message length are
truncated in the last ``h - 1`` blocks, so the final trellis state is 0.

Embedding (sender): the Viterbi algorithm finds the LSB vector ``y`` with
``H y = m`` minimising ``sum_k rho_k [y_k != x_k]``; a required LSB flip at pixel
``k`` is realised as the cheaper of +1 / -1 (``rho_k = min(rho+_k, rho-_k)``),
i.e. the standard "binary STC with +-1 embedding" construction.

Extraction (receiver): ``m = H y`` from the stego LSBs.  The receiver needs
the key (which seeds H_hat and the pixel permutation) and nothing else: the
32-bit message length is carried by 32 key-selected header pixels that are
excluded from the STC cover sequence.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or not callable(a[0]) else a[0]

__all__ = ["make_hhat", "column_widths", "stc_embed_bits", "stc_extract_bits",
           "stc_embed_image", "stc_extract_image", "LEN_BITS"]

LEN_BITS = 32


def _seed_from(key: bytes, label: str) -> int:
    return int.from_bytes(hashlib.sha256(key + label.encode()).digest()[:8], "little")


def make_hhat(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cols = rng.integers(0, 1 << h, size=max(w, 1), dtype=np.int64)
    cols |= 1 | (1 << (h - 1))
    return cols


def column_widths(n: int, m: int) -> np.ndarray:
    if m <= 0 or n < m:
        raise ValueError(f"STC needs 0 < m <= n (got m={m}, n={n})")
    edges = (np.arange(m + 1, dtype=np.int64) * n) // m
    return np.diff(edges).astype(np.int64)


@njit(cache=True)
def _viterbi(x, rho, msg, hhat, widths, h):
    n = x.size
    m = msg.size
    S = 1 << h
    INF = 1e300
    cost = np.full(S, INF)
    cost[0] = 0.0
    newc = np.empty(S)
    path = np.zeros((n, S), dtype=np.uint8)
    k = 0
    for i in range(m):
        rows_left = m - i
        hb = h if rows_left > h else rows_left
        hmask = (1 << hb) - 1
        for j in range(widths[i]):
            col = hhat[j] & hmask
            c0 = rho[k] if x[k] == 1 else 0.0
            c1 = rho[k] if x[k] == 0 else 0.0
            for s in range(S):
                a = cost[s] + c0
                b = cost[s ^ col] + c1
                if b < a:
                    newc[s] = b
                    path[k, s] = 1
                else:
                    newc[s] = a
                    path[k, s] = 0
            tmp = cost
            cost = newc
            newc = tmp
            k += 1
        bit = msg[i]
        half = S >> 1
        for s in range(half):
            newc[s] = cost[(s << 1) | bit]
        for s in range(half, S):
            newc[s] = INF
        tmp = cost
        cost = newc
        newc = tmp
    total = cost[0]
    y = np.empty(n, dtype=np.uint8)
    s = 0
    k = n - 1
    for i in range(m - 1, -1, -1):
        s = ((s << 1) | msg[i]) & (S - 1)
        rows_left = m - i
        hb = h if rows_left > h else rows_left
        hmask = (1 << hb) - 1
        for j in range(widths[i] - 1, -1, -1):
            c = path[k, s]
            y[k] = c
            if c:
                s ^= hhat[j] & hmask
            k -= 1
    return y, total


@njit(cache=True)
def _syndrome(y, hhat, widths, h, m):
    out = np.zeros(m, dtype=np.uint8)
    state = 0
    k = 0
    for i in range(m):
        rows_left = m - i
        hb = h if rows_left > h else rows_left
        hmask = (1 << hb) - 1
        for j in range(widths[i]):
            if y[k]:
                state ^= hhat[j] & hmask
            k += 1
        out[i] = state & 1
        state >>= 1
    return out


def stc_embed_bits(x: np.ndarray, rho: np.ndarray, msg: np.ndarray, h: int = 10,
                   seed: int = 0) -> Tuple[np.ndarray, float]:
    """Return the minimum-cost LSB vector ``y`` with syndrome ``msg``."""
    x = np.ascontiguousarray(x, dtype=np.uint8)
    rho = np.ascontiguousarray(np.minimum(rho, 1e12), dtype=np.float64)
    msg = np.ascontiguousarray(msg, dtype=np.uint8)
    widths = column_widths(x.size, msg.size)
    hhat = make_hhat(h, int(widths.max()), seed)
    y, total = _viterbi(x, rho, msg, hhat, widths, h)
    return y, float(total)


def stc_extract_bits(y: np.ndarray, m: int, h: int = 10, seed: int = 0) -> np.ndarray:
    y = np.ascontiguousarray(y, dtype=np.uint8)
    widths = column_widths(y.size, int(m))
    hhat = make_hhat(h, int(widths.max()), seed)
    return _syndrome(y, hhat, widths, h, int(m))


def _layout(shape: Tuple[int, int], key: bytes) -> np.ndarray:
    n = int(shape[0]) * int(shape[1])
    return np.random.default_rng(_seed_from(key, "stc-perm")).permutation(n)


def stc_embed_image(cover: np.ndarray, rho_p1: np.ndarray, rho_m1: np.ndarray,
                    payload: np.ndarray, key: bytes, rng: np.random.Generator,
                    h: int = 10) -> Tuple[np.ndarray, Dict[str, object]]:
    """Complete STC sender: length header + STC over a key-permuted pixel order."""
    cover = np.asarray(cover, dtype=np.uint8)
    flat = cover.reshape(-1).astype(np.int16)
    rp = np.asarray(rho_p1, dtype=np.float64).reshape(-1)
    rm = np.asarray(rho_m1, dtype=np.float64).reshape(-1)
    perm = _layout(cover.shape, key)
    hdr, seq = perm[:LEN_BITS], perm[LEN_BITS:]
    m = int(np.asarray(payload).size)
    if m > seq.size:
        raise ValueError("payload exceeds STC capacity")
    st = flat.copy()

    def _flip(idx: np.ndarray) -> None:
        up = rp[idx] <= rm[idx]
        v = st[idx]
        up = np.where(v == 0, True, np.where(v == 255, False, up))
        st[idx] = v + np.where(up, 1, -1)

    # 32-bit length header by LSB matching on key-selected pixels
    lbits = np.array([(m >> (LEN_BITS - 1 - i)) & 1 for i in range(LEN_BITS)], dtype=np.uint8)
    need = (st[hdr] & 1).astype(np.uint8) != lbits
    _flip(hdr[need])
    # STC over the remaining pixels
    x = (flat[seq] & 1).astype(np.uint8)
    y, total = stc_embed_bits(x, np.minimum(rp[seq], rm[seq]),
                              np.asarray(payload, dtype=np.uint8).reshape(-1), h,
                              _seed_from(key, "stc-hhat"))
    _flip(seq[y != x])
    stego = st.astype(np.uint8).reshape(cover.shape)
    return stego, {"stc_h": h, "stc_distortion": total, "stc_cover_len": int(seq.size),
                   "n_changes": int(np.count_nonzero(stego != cover)),
                   "payload_bits": m, "embedded_bits": m, "blind_extractable": True,
                   "capacity_limited": False}


def stc_extract_image(stego: np.ndarray, key: bytes, h: int = 10) -> np.ndarray:
    flat = np.asarray(stego, dtype=np.uint8).reshape(-1)
    perm = _layout(stego.shape, key)
    hdr, seq = perm[:LEN_BITS], perm[LEN_BITS:]
    lb = flat[hdr] & 1
    m = 0
    for b in lb:
        m = (m << 1) | int(b)
    if m <= 0 or m > seq.size:
        raise ValueError("implausible STC length header")
    return stc_extract_bits(flat[seq] & 1, m, h, _seed_from(key, "stc-hhat"))
