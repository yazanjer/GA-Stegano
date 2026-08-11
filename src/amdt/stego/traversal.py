"""Multi-directional traversal path generation (16 patterns).

This is a faithful, vectorised re-implementation of ``CreateHostPixelSeq.m``
from the MATLAB benchmark, with three defects of the original fixed:

MATLAB bug 1 -- ``if row == numOfRows: row = 1`` wraps *before* the last row is
    ever used, so row ``numOfRows`` is unreachable in directions 0-3 and 8-15.
    The header (chromosome) pixels live on that last row, which is why the
    original code never collided with them, but the behaviour is undocumented.
    Here the reserved header region is made explicit (``reserved_rows``) and the
    traversal wraps correctly over the remaining rows.
MATLAB bug 2 -- ``row < 1 -> row = numOfRows - 1`` (upward wraps) skips row
    ``numOfRows`` as well and is asymmetric with the downward wrap.
MATLAB bug 3 -- boustrophedon directions (4-7, 12-15) reverse ``key`` on the
    same step that they step the orthogonal axis, emitting one duplicate cell
    per turn.  Duplicates silently overwrite earlier message bits.  The
    implementation below is verified duplicate-free by
    ``tests/test_traversal.py``.

Direction code layout (4 bits, values 0-15), matching Table 1 of the paper:

    ``d = (axis << 3) | (serpentine << 2) | (secondary_desc << 1) | primary_desc``

    ==== ======== =========== ================= ==================
    bit  name     0           1                 meaning
    ==== ======== =========== ================= ==================
    3    axis     row-major   column-major      which axis advances fastest
    2    serp     raster      serpentine        reset vs. reverse at line end
    1    sec      ascending   descending        direction of the slow axis
    0    pri      ascending   descending        direction of the fast axis
    ==== ======== =========== ================= ==================

Every pattern is a *bijection* onto the embeddable region, so the traversal is
information-lossless and the extractor can reproduce it exactly from the
3 chromosome genes ``(direction, x_off, y_off)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "DIRECTION_NAMES",
    "TraversalSpec",
    "traversal_order",
    "pixel_sequence",
    "header_pixels",
]

DIRECTION_NAMES = {
    0: "row-major, L->R, T->B (raster)",
    1: "row-major, R->L, T->B",
    2: "row-major, L->R, B->T",
    3: "row-major, R->L, B->T",
    4: "row-major serpentine, start L->R, T->B",
    5: "row-major serpentine, start R->L, T->B",
    6: "row-major serpentine, start L->R, B->T",
    7: "row-major serpentine, start R->L, B->T",
    8: "column-major, T->B, L->R",
    9: "column-major, B->T, L->R",
    10: "column-major, T->B, R->L",
    11: "column-major, B->T, R->L",
    12: "column-major serpentine, start T->B, L->R",
    13: "column-major serpentine, start B->T, L->R",
    14: "column-major serpentine, start T->B, R->L",
    15: "column-major serpentine, start B->T, R->L",
}


@dataclass(frozen=True)
class TraversalSpec:
    """Decoded 4-bit direction gene."""

    code: int
    column_major: bool
    serpentine: bool
    slow_descending: bool
    fast_descending: bool

    @classmethod
    def from_code(cls, code: int) -> "TraversalSpec":
        code = int(code) & 0x0F
        return cls(
            code=code,
            column_major=bool(code & 0b1000),
            serpentine=bool(code & 0b0100),
            slow_descending=bool(code & 0b0010),
            fast_descending=bool(code & 0b0001),
        )

    @property
    def name(self) -> str:
        return DIRECTION_NAMES[self.code]


def _line_order(n: int, descending: bool) -> np.ndarray:
    idx = np.arange(n, dtype=np.int64)
    return idx[::-1].copy() if descending else idx


def traversal_order(
    shape: Tuple[int, int],
    direction: int,
    x_off: int = 0,
    y_off: int = 0,
) -> np.ndarray:
    """Flat (row-major) indices of every embeddable pixel, in visiting order.

    Parameters
    ----------
    shape : (H, W) of the *embeddable region* (header rows already removed).
    direction : 0..15, see module docstring.
    x_off, y_off : cyclic rotation of the starting column / row.  These are the
        ``xOff``/``yOff`` genes; rotating rather than truncating keeps the
        traversal a bijection for every offset value.

    Returns
    -------
    ndarray[int64] of length H*W with no repeated entry.
    """
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        return np.empty(0, dtype=np.int64)

    spec = TraversalSpec.from_code(direction)

    # Cyclic start offsets -> keeps every pattern a permutation of the region.
    rows = (np.arange(h, dtype=np.int64) + int(y_off)) % h
    cols = (np.arange(w, dtype=np.int64) + int(x_off)) % w

    if not spec.column_major:
        slow, fast = rows, cols          # rows are the slow axis
        slow_desc, fast_desc = spec.slow_descending, spec.fast_descending
        slow_seq = slow[_line_order(h, slow_desc)]
        base_fast = fast[_line_order(w, fast_desc)]
        out = np.empty((h, w, 2), dtype=np.int64)
        for i, r in enumerate(slow_seq):
            fast_seq = base_fast[::-1] if (spec.serpentine and i % 2 == 1) else base_fast
            out[i, :, 0] = r
            out[i, :, 1] = fast_seq
        rr = out[..., 0].reshape(-1)
        cc = out[..., 1].reshape(-1)
    else:
        slow_desc, fast_desc = spec.slow_descending, spec.fast_descending
        slow_seq = cols[_line_order(w, slow_desc)]      # columns are slow
        base_fast = rows[_line_order(h, fast_desc)]     # rows are fast
        out = np.empty((w, h, 2), dtype=np.int64)
        for i, c in enumerate(slow_seq):
            fast_seq = base_fast[::-1] if (spec.serpentine and i % 2 == 1) else base_fast
            out[i, :, 1] = c
            out[i, :, 0] = fast_seq
        rr = out[..., 0].reshape(-1)
        cc = out[..., 1].reshape(-1)

    return rr * w + cc


def header_pixels(shape: Tuple[int, int], n_bits: int, reserved_rows: int = 1) -> np.ndarray:
    """Flat indices (in the *full* image) of the pixels carrying the header.

    The header occupies the last ``reserved_rows`` rows, filled right-to-left
    from the bottom-right corner -- the same convention as the MATLAB
    ``chromPixelSeq``, so stego images stay comparable with the benchmark.
    """
    h, w = int(shape[0]), int(shape[1])
    capacity = reserved_rows * w
    if n_bits > capacity:
        raise ValueError(
            f"header needs {n_bits} bits but only {capacity} reserved pixels "
            f"({reserved_rows} row(s) x {w} px). Increase reserved_rows."
        )
    last = h * w - 1
    return np.array([last - i for i in range(n_bits)], dtype=np.int64)


def pixel_sequence(
    shape: Tuple[int, int],
    direction: int,
    x_off: int,
    y_off: int,
    n_pixels: int,
    reserved_rows: int = 1,
) -> np.ndarray:
    """First ``n_pixels`` payload pixels as flat indices into the full image.

    The reserved header rows at the bottom are excluded from the traversal, so
    payload and header can never collide (a failure mode the MATLAB code only
    avoided by accident).
    """
    h, w = int(shape[0]), int(shape[1])
    usable_h = h - int(reserved_rows)
    if usable_h <= 0:
        raise ValueError("reserved_rows leaves no room for payload")
    order = traversal_order((usable_h, w), direction, x_off, y_off)
    if n_pixels > order.size:
        raise ValueError(
            f"requested {n_pixels} pixels, embeddable region holds {order.size}"
        )
    return order[:n_pixels]  # region is top-aligned -> flat index is unchanged
