
"""Fast, bit-identical replacement for sealwatch.srm.cooccurrence.cooccurrence4.

sealwatch 2025.9 counts 4-D co-occurrences with 625 nested boolean masks per
call, which makes the full 34,671-D SRM cost ~18 s per 512x512 image.  The
counts are integers, so replacing the loop by one ``np.bincount`` over the
flattened 4-tuple index yields exactly the same table (verified feature-for-
feature in tests/test_revision2.py) at a fraction of the cost.
"""
from __future__ import annotations
import numpy as np

def cooccurrence4_fast(D, type, T):
    B = 2 * T + 1
    if type == 'hor':
        L, C, E, R = D[:, :-3], D[:, 1:-2], D[:, 2:-1], D[:, 3:]
    elif type == 'ver':
        L, C, E, R = D[:-3], D[1:-2], D[2:-1], D[3:]
    elif type == 'diag':
        L, C, E, R = D[:-3, :-3], D[1:-2, 1:-2], D[2:-1, 2:-1], D[3:, 3:]
    elif type == 'mdiag':
        L, C, E, R = D[3:, :-3], D[2:-1, 1:-2], D[1:-2, 2:-1], D[:-3, 3:]
    elif type == 'square':
        L, C, E, R = D[1:, :-1], D[1:, 1:], D[:-1, 1:], D[:-1, :-1]
    else:
        raise NotImplementedError(f'type {type} not implemented')
    cooc = _count4(np.ascontiguousarray(L, dtype=np.int64), np.ascontiguousarray(C, dtype=np.int64),
                   np.ascontiguousarray(E, dtype=np.int64), np.ascontiguousarray(R, dtype=np.int64), T)
    return cooc / np.sum(cooc)


try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = None


def _count4_py(L, C, E, R, T):
    B = 2 * T + 1
    L = L + T; C = C + T; E = E + T; R = R + T
    ok = (L >= 0) & (L < B) & (C >= 0) & (C < B) & (E >= 0) & (E < B) & (R >= 0) & (R < B)
    idx = (((L * B + C) * B + E) * B + R)[ok]
    return np.bincount(idx.ravel(), minlength=B ** 4).reshape(B, B, B, B)


if njit is not None:
    @njit(cache=True)
    def _count4(L, C, E, R, T):
        B = 2 * T + 1
        out = np.zeros((B, B, B, B), dtype=np.int64)
        a = L.ravel(); b = C.ravel(); c = E.ravel(); d = R.ravel()
        for i in range(a.size):
            p = a[i] + T; q = b[i] + T; r = c[i] + T; s = d[i] + T
            if 0 <= p < B and 0 <= q < B and 0 <= r < B and 0 <= s < B:
                out[p, q, r, s] += 1
        return out
else:  # pragma: no cover
    _count4 = _count4_py

def install():
    import sys
    import sealwatch.srm  # noqa: F401  (populates sys.modules)
    co = sys.modules['sealwatch.srm.cooccurrence']
    orig = co.cooccurrence4
    co.cooccurrence4 = cooccurrence4_fast
    # all1st..all5x5 bind ``CoocN=cooccurrence4`` as a default argument at
    # definition time, so the defaults themselves must be rebound.
    for name in ("all1st", "all2nd", "all3rd", "all3x3", "all5x5"):
        fn = getattr(co, name)
        if fn.__defaults__:
            fn.__defaults__ = tuple(cooccurrence4_fast if d is orig or getattr(d, "__name__", "") == "cooccurrence4" else d
                                    for d in fn.__defaults__)
        if fn.__kwdefaults__ and "CoocN" in fn.__kwdefaults__:
            fn.__kwdefaults__["CoocN"] = cooccurrence4_fast
    return co

def srm_full(img):
    """Full SRM (Fridrich & Kodovsky 2012), 34,671-D, via sealwatch with the fast counter."""
    install()
    import sealwatch.srm as s
    f = s.extract(np.asarray(img))
    # sealwatch builds part of its submodel dict from a set, so the insertion order of
    # the last submodels depends on PYTHONHASHSEED.  Sort the names so that the feature
    # layout is identical in every process.
    return np.concatenate([np.asarray(f[k], dtype=np.float64).ravel() for k in sorted(f)])
