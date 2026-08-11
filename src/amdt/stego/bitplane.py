"""Adaptive bit-plane selection and mapping (Algorithm 5 of the manuscript).

The ``mask`` gene is an integer in 1..15 whose 4-bit binary expansion selects
which of the four least-significant bit-planes carry payload:

    mask = 0b1010  ->  planes {b3, b1}  (0-indexed from the LSB, b0)

``bp_dir`` fixes the order in which the selected planes are filled within a
pixel: 0 = ascending (LSB first), 1 = descending (4th LSB first).  Ascending
order concentrates changes in the lowest planes and is almost always cheaper in
MSE; descending is retained because it is a gene in the benchmark chromosome
and the ablation needs it.

Per-pixel capacity is ``popcount(mask)``.  Worst-case squared error per changed
pixel is bounded by ``(sum of selected plane weights)^2``, which
:func:`max_pixel_error` exposes so the GA's search space can be reasoned about
analytically rather than only empirically.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = ["mask_to_planes", "plane_capacity", "max_pixel_error", "embed_bits_into_pixels",
           "extract_bits_from_pixels"]


def mask_to_planes(mask: int, bp_dir: int = 0) -> np.ndarray:
    """Selected plane indices (0 = LSB) in fill order.

    ``mask`` bit 3 (MSB of the nibble) corresponds to plane 3 (the 4th LSB),
    matching ``dec2bin(mask,4)`` indexing in the MATLAB reference.
    """
    m = int(mask) & 0x0F
    planes = [p for p in range(4) if (m >> p) & 1]      # ascending: b0, b1, ...
    planes = np.asarray(planes, dtype=np.int64)
    return planes[::-1].copy() if int(bp_dir) else planes


def plane_capacity(mask: int) -> int:
    """Payload bits carried by one pixel."""
    return int(bin(int(mask) & 0x0F).count("1"))


def max_pixel_error(mask: int) -> int:
    """Upper bound on |stego - cover| for one pixel under this mask."""
    m = int(mask) & 0x0F
    return sum(1 << p for p in range(4) if (m >> p) & 1)


def embed_bits_into_pixels(
    values: np.ndarray,
    bits: np.ndarray,
    mask: int,
    bp_dir: int,
) -> Tuple[np.ndarray, int]:
    """Write ``bits`` into the masked planes of ``values`` (uint8, 1-D).

    Returns the modified copy and the number of bits actually consumed.  If the
    payload runs out mid-pixel the remaining planes of that pixel are left
    untouched -- unwritten planes are *never* zeroed, which the MATLAB code got
    right and many re-implementations get wrong (zeroing leaks the payload
    length through a suspiciously clean tail).
    """
    planes = mask_to_planes(mask, bp_dir)
    k = planes.size
    if k == 0:
        return values.copy(), 0

    v = values.astype(np.uint8, copy=True)
    n_pix = v.size
    n_bits = min(int(bits.size), n_pix * k)
    full_pix = n_bits // k
    tail = n_bits - full_pix * k

    if full_pix:
        b = bits[: full_pix * k].reshape(full_pix, k).astype(np.uint8)
        for j, p in enumerate(planes):
            clear = np.uint8(~(1 << int(p)) & 0xFF)
            v[:full_pix] = (v[:full_pix] & clear) | (b[:, j] << int(p))
    if tail:
        i = full_pix
        b = bits[full_pix * k: n_bits].astype(np.uint8)
        for j in range(tail):
            p = int(planes[j])
            v[i] = (v[i] & np.uint8(~(1 << p) & 0xFF)) | np.uint8(int(b[j]) << p)

    return v, n_bits


def extract_bits_from_pixels(
    values: np.ndarray,
    n_bits: int,
    mask: int,
    bp_dir: int,
) -> np.ndarray:
    """Inverse of :func:`embed_bits_into_pixels`."""
    planes = mask_to_planes(mask, bp_dir)
    k = planes.size
    if k == 0 or n_bits <= 0:
        return np.zeros(0, dtype=np.uint8)

    v = values.astype(np.uint8, copy=False)
    n_pix = int(np.ceil(n_bits / k))
    n_pix = min(n_pix, v.size)
    cols = [(v[:n_pix] >> int(p)) & np.uint8(1) for p in planes]
    out = np.stack(cols, axis=1).reshape(-1)
    return out[:n_bits].astype(np.uint8)
