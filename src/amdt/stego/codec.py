"""AMDT embedding / extraction codec: chromosome, header, embed, extract.

Chromosome (one per *segment*), 33 bits -- the single canonical layout used
by the code, the header and the manuscript (Table 2 / Eq. (11)):

    ===========  ====  ==========================================
    gene         bits  range / meaning
    ===========  ====  ==========================================
    direction      4   0-15   traversal pattern (Table 1)
    x_off          9   0-511  cyclic column start
    y_off          9   0-511  cyclic row start
    mask           4   1-15   bit-plane selection (0 is invalid)
    alpha          1   T1 complement flag
    beta           1   T2 reverse flag
    bp_dir         1   plane fill order
    sigma          1   T3 keyed block scrambling enable
    block_idx      2   T3 block size index -> {64,128,256,512}
    delta          1   T4 keyed diffusion enable
    ===========  ====  ==========================================

Search space.  With the mask value 0 excluded (zero capacity), the
per-segment genotype space is

    |Theta| = 16 * 512 * 512 * 15 * 2 (alpha) * 2 (beta) * 2 (bp_dir)
              * 2 (sigma) * 4 (block_idx) * 2 (delta) = 8.05e9.

The deployed method locks the security constraint sigma = delta = 1, so the
space actually searched is |Theta| / 4 = 2.01e9 per segment (block_idx stays
free).  :func:`amdt.ga.optimizer.search_space_size` reports both.

Header layout (revision 2), written into the reserved bottom rows by LSB
replacement, one bit per pixel, right-to-left from the bottom-right corner:

    nonce      64 bits  fresh per stego object, in clear
    ENC[ version   4 bits  (= 2)
         n_seg     8 bits  number of segments (1-255)
         length   24 bits  total payload length L in bits
         genes    33 bits x n_seg ]
    tag       16 bits   HMAC-SHA256(K, "AMDT-tag" || nonce || plaintext), truncated

ENC is XOR with KS_{K_h}("hdr", .) where K_h = HMAC(K, "AMDT-hdr" || nonce).
Total = 116 + 33*n_seg bits (248 bits for the default four segments).

The v1 header carried a fixed plaintext magic byte (0xA7) and the genes in
clear.  A fixed magic in a fixed position is itself a targeted-steganalysis
signature, and clear genes publish the traversal path; revision 2 therefore
encrypts everything except the nonce and replaces magic + CRC by a keyed tag.
Without K the header is indistinguishable from random LSBs (under the PRF
assumption on HMAC-SHA256); with K the receiver authenticates it (a wrong key
or a non-stego image is rejected with probability 1 - 2^-16).

Everything the extractor needs *except the 256-bit secret key* is in the
header, so extraction is blind given the key.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .bitplane import (
    embed_bits_into_pixels,
    extract_bits_from_pixels,
    mask_to_planes,
    plane_capacity,
)
import hashlib
import hmac as _hmac
import os

from .decomposition import (DecompositionParams, decompose, derive_subkey, keystream_bits,
                            recompose, segment_message)
from .traversal import header_pixels, traversal_order

__all__ = [
    "Chromosome",
    "GENE_BOUNDS",
    "CHROM_BITS",
    "HEADER_FIXED_BITS",
    "EmbedResult",
    "new_nonce",
    "build_header",
    "read_header",
    "embed",
    "extract",
    "header_bit_count",
    "reserved_rows_for",
    "max_payload_bits",
    "bytes_to_bits",
    "bits_to_bytes",
]

VERSION = 2
CHROM_BITS = 33
NONCE_BITS = 64
TAG_BITS = 16
HEADER_PLAIN_FIXED = 4 + 8 + 24                    # version, n_seg, length
HEADER_FIXED_BITS = NONCE_BITS + HEADER_PLAIN_FIXED + TAG_BITS   # = 116

#: (low, high) inclusive bounds for each gene, in chromosome order.
GENE_BOUNDS: Tuple[Tuple[int, int], ...] = (
    (0, 15),    # direction
    (0, 511),   # x_off
    (0, 511),   # y_off
    (1, 15),    # mask (0 would give zero capacity)
    (0, 1),     # alpha
    (0, 1),     # beta
    (0, 1),     # bp_dir
    (0, 1),     # sigma
    (0, 3),     # block_idx
    (0, 1),     # delta
)
_GENE_WIDTHS = (4, 9, 9, 4, 1, 1, 1, 1, 2, 1)
GENE_NAMES = (
    "direction", "x_off", "y_off", "mask",
    "alpha", "beta", "bp_dir", "sigma", "block_idx", "delta",
)


# --------------------------------------------------------------------------- #
# bit helpers
# --------------------------------------------------------------------------- #
def _int_to_bits(value: int, width: int) -> np.ndarray:
    return np.array([(int(value) >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def bytes_to_bits(data: bytes) -> np.ndarray:
    """MSB-first bit expansion (matches ``dec2bin(...,8)`` in MATLAB)."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    b = np.asarray(bits, dtype=np.uint8)
    pad = (-b.size) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(b).tobytes()


def _crc16(bits: np.ndarray) -> int:
    """CRC-16/CCITT-FALSE over a bit array."""
    crc = 0xFFFF
    for bit in bits:
        crc ^= (int(bit) & 1) << 15
        for _ in range(1):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


# --------------------------------------------------------------------------- #
# chromosome
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chromosome:
    direction: int = 0
    x_off: int = 0
    y_off: int = 0
    mask: int = 1
    alpha: int = 0
    beta: int = 0
    bp_dir: int = 0
    sigma: int = 0
    block_idx: int = 0
    delta: int = 0

    # -- conversions -------------------------------------------------------
    def to_vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in GENE_NAMES], dtype=np.int64)

    @classmethod
    def from_vector(cls, vec: Sequence[int]) -> "Chromosome":
        return cls(**{n: int(v) for n, v in zip(GENE_NAMES, vec)})

    def to_bits(self) -> np.ndarray:
        return np.concatenate(
            [_int_to_bits(getattr(self, n), w) for n, w in zip(GENE_NAMES, _GENE_WIDTHS)]
        )

    @classmethod
    def from_bits(cls, bits: np.ndarray) -> "Chromosome":
        vals, i = [], 0
        for w in _GENE_WIDTHS:
            vals.append(_bits_to_int(bits[i: i + w]))
            i += w
        return cls.from_vector(vals)

    # -- derived -----------------------------------------------------------
    @property
    def decomposition(self) -> DecompositionParams:
        return DecompositionParams(
            alpha=self.alpha, beta=self.beta, sigma=self.sigma,
            block_idx=self.block_idx, delta=self.delta,
        )

    @property
    def bits_per_pixel(self) -> int:
        return plane_capacity(self.mask)

    def clipped(self) -> "Chromosome":
        vals = {}
        for n, (lo, hi) in zip(GENE_NAMES, GENE_BOUNDS):
            vals[n] = int(np.clip(int(getattr(self, n)), lo, hi))
        return Chromosome(**vals)

    def as_dict(self) -> Dict[str, int]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# capacity accounting
# --------------------------------------------------------------------------- #
def header_bit_count(n_segments: int) -> int:
    return HEADER_FIXED_BITS + CHROM_BITS * int(n_segments)


def reserved_rows_for(width: int, n_segments: int) -> int:
    return int(np.ceil(header_bit_count(n_segments) / int(width)))


def max_payload_bits(shape: Tuple[int, int], chroms: Sequence[Chromosome]) -> int:
    """Total payload the given per-segment masks can hold in this cover.

    Accounts for the exact band partition used by :func:`_segment_pixels`, so
    this is a hard capacity, not an estimate.  Note the payload is split
    *equally by length* across segments, so the binding constraint is the
    minimum over segments of ``band_pixels * bits_per_pixel`` scaled by the
    number of segments.
    """
    h, w = shape
    n = max(1, len(chroms))
    rows = reserved_rows_for(w, n)
    usable_h = h - rows
    per_seg = []
    for s, c in enumerate(chroms):
        band_h = ((s + 1) * usable_h) // n - (s * usable_h) // n
        per_seg.append(band_h * w * c.bits_per_pixel)
    # segment s receives ~L/n bits, so L <= n * min_s(capacity_s)
    return int(n * min(per_seg)) if per_seg else 0


# --------------------------------------------------------------------------- #
# region partitioning for multi-segment embedding
# --------------------------------------------------------------------------- #
def _segment_pixels(
    shape: Tuple[int, int],
    chrom: Chromosome,
    n_pixels: int,
    reserved_rows: int,
    seg_idx: int,
    n_segments: int,
) -> np.ndarray:
    """Pixels for one segment, guaranteed disjoint from the other segments.

    The embeddable region is split into ``n_segments`` horizontal bands using a
    *canonical* (direction-independent) row partition.  Each segment then
    traverses **its own band** with its own ``direction``/``x_off``/``y_off``
    genes.  Disjointness therefore holds by construction, whatever the genes
    are: this replaces the manuscript's collision *check* (Algorithm 7) with a
    collision *impossibility*, which is faster and removes a discontinuity from
    the GA fitness landscape.  A band-local traversal still yields all 16
    patterns, so no expressiveness is lost.
    """
    h, w = shape
    usable_h = h - reserved_rows
    if usable_h < n_segments:
        raise ValueError(
            f"{n_segments} segments need >= {n_segments} embeddable rows, "
            f"cover provides {usable_h}"
        )
    r0 = (seg_idx * usable_h) // n_segments
    r1 = ((seg_idx + 1) * usable_h) // n_segments
    band_h = r1 - r0
    order = traversal_order((band_h, w), chrom.direction, chrom.x_off, chrom.y_off)
    if n_pixels > order.size:
        raise ValueError(
            f"segment {seg_idx}: needs {n_pixels} px, band holds {order.size}. "
            "Reduce payload, raise mask popcount, or use fewer segments."
        )
    return order[:n_pixels] + r0 * w


# --------------------------------------------------------------------------- #
# embed / extract
# --------------------------------------------------------------------------- #
@dataclass
class EmbedResult:
    stego: np.ndarray
    n_changes: int
    payload_bits: int
    header_bits: int
    reserved_rows: int
    chromosomes: List[Chromosome]
    nonce: bytes = b""

    @property
    def total_bits(self) -> int:
        return self.payload_bits + self.header_bits


def new_nonce(rng: "np.random.Generator | None" = None) -> bytes:
    """64-bit nonce.  Experiments draw it from the seeded RNG so runs are
    byte-reproducible; production use should pass ``rng=None`` (os.urandom)."""
    if rng is None:
        return os.urandom(8)
    return rng.integers(0, 256, size=8, dtype=np.uint8).tobytes()


def embed(
    cover: np.ndarray,
    payload: np.ndarray,
    chroms: Sequence[Chromosome],
    key: bytes,
    nonce: "bytes | None" = None,
) -> EmbedResult:
    """Embed ``payload`` bits into ``cover`` using one chromosome per segment.

    Parameters
    ----------
    cover : (H, W) uint8 grayscale image.
    payload : 1-D uint8 array of 0/1 bits.
    chroms : ``n_segments`` chromosomes; ``len(chroms)`` defines the split.
    key : 256-bit shared secret for T3/T4 and the header.
    nonce : 8-byte per-message nonce (fresh random if ``None``).
    """
    cover = np.asarray(cover, dtype=np.uint8)
    if cover.ndim != 2:
        raise ValueError("cover must be 2-D grayscale uint8")
    payload = np.asarray(payload, dtype=np.uint8).reshape(-1)

    chroms = [c.clipped() for c in chroms]
    n_seg = len(chroms)
    if not 1 <= n_seg <= 255:
        raise ValueError("n_segments must be in 1..255")
    L = int(payload.size)
    if L >= 1 << 24:
        raise ValueError("payload longer than the 24-bit length field allows")

    h, w = cover.shape
    rows = reserved_rows_for(w, n_seg)
    stego = cover.copy()

    nonce = bytes(nonce) if nonce is not None else new_nonce()
    if len(nonce) != NONCE_BITS // 8:
        raise ValueError("nonce must be 8 bytes")
    segments = segment_message(payload, n_seg)
    flat = stego.reshape(-1)

    for s, (chrom, seg) in enumerate(zip(chroms, segments)):
        t = decompose(seg, chrom.decomposition, derive_subkey(key, nonce, s))
        k = chrom.bits_per_pixel
        n_pix = int(np.ceil(t.size / k)) if t.size else 0
        if n_pix == 0:
            continue
        idx = _segment_pixels(cover.shape, chrom, n_pix, rows, s, n_seg)
        vals, used = embed_bits_into_pixels(flat[idx], t, chrom.mask, chrom.bp_dir)
        if used != t.size:  # pragma: no cover -- guarded by _segment_pixels
            raise RuntimeError("bit accounting mismatch in segment embedding")
        flat[idx] = vals

    # ---- header (written last so it cannot be overwritten by payload) -----
    hbits = build_header(chroms, L, key, nonce)
    hidx = header_pixels(cover.shape, hbits.size, rows)
    hv = flat[hidx]
    flat[hidx] = (hv & np.uint8(0xFE)) | hbits

    n_changes = int(np.count_nonzero(stego != cover))
    return EmbedResult(
        stego=stego,
        n_changes=n_changes,
        payload_bits=L,
        header_bits=int(hbits.size),
        reserved_rows=rows,
        chromosomes=list(chroms),
        nonce=nonce,
    )


def _hdr_key(key: bytes, nonce: bytes) -> bytes:
    return _hmac.new(key, b"AMDT-hdr" + bytes(nonce), hashlib.sha256).digest()


def _hdr_tag(key: bytes, nonce: bytes, plain: np.ndarray) -> np.ndarray:
    d = _hmac.new(key, b"AMDT-tag" + bytes(nonce) + np.packbits(plain).tobytes()
                  + len(plain).to_bytes(4, "big"), hashlib.sha256).digest()
    return np.unpackbits(np.frombuffer(d[:TAG_BITS // 8], dtype=np.uint8))


def build_header(chroms: Sequence[Chromosome], length: int, key: bytes, nonce: bytes,
                 gene_bits: "Sequence[np.ndarray] | None" = None, version: int = VERSION) -> np.ndarray:
    """nonce || ENC(version, n_seg, length, genes) || tag  (see module docstring)."""
    parts = [_int_to_bits(version, 4), _int_to_bits(len(chroms), 8), _int_to_bits(length, 24)]
    parts += list(gene_bits) if gene_bits is not None else [c.to_bits() for c in chroms]
    plain = np.concatenate(parts).astype(np.uint8)
    enc = plain ^ keystream_bits(_hdr_key(key, nonce), "hdr", plain.size)
    nonce_bits = np.unpackbits(np.frombuffer(bytes(nonce), dtype=np.uint8))
    return np.concatenate([nonce_bits, enc, _hdr_tag(key, nonce, plain)]).astype(np.uint8)


def read_header(flat: np.ndarray, shape: Tuple[int, int], key: bytes, chrom_bits: int = CHROM_BITS,
                expect_version: int = VERSION) -> Tuple[bytes, int, int, List[np.ndarray], int]:
    """Parse and authenticate a header.  Returns (nonce, n_seg, L, gene_bit_rows, reserved_rows)."""
    h, w = shape
    probe_n = NONCE_BITS + HEADER_PLAIN_FIXED
    probe = flat[header_pixels(shape, probe_n, int(np.ceil(probe_n / w)))] & 1
    nonce = np.packbits(probe[:NONCE_BITS]).tobytes()
    hk = _hdr_key(key, nonce)
    fixed = probe[NONCE_BITS:] ^ keystream_bits(hk, "hdr", HEADER_PLAIN_FIXED)
    version = _bits_to_int(fixed[:4])
    n_seg = _bits_to_int(fixed[4:12])
    if version != expect_version or not 1 <= n_seg <= 255:
        raise ValueError("no valid AMDT header for this key")
    total = HEADER_FIXED_BITS + chrom_bits * n_seg
    rows = int(np.ceil(total / w))
    hbits = flat[header_pixels(shape, total, rows)] & 1
    enc = hbits[NONCE_BITS: total - TAG_BITS]
    plain = enc ^ keystream_bits(hk, "hdr", enc.size)
    if not np.array_equal(_hdr_tag(key, nonce, plain), hbits[total - TAG_BITS:]):
        raise ValueError("header authentication failed -- wrong key or not an AMDT stego image")
    L = _bits_to_int(plain[12:36])
    genes = [plain[36 + i * chrom_bits: 36 + (i + 1) * chrom_bits] for i in range(n_seg)]
    return nonce, n_seg, L, genes, rows


def extract(stego: np.ndarray, key: bytes) -> Tuple[np.ndarray, List[Chromosome]]:
    """Blind extraction: recover the payload bits and the per-segment genes.

    Raises ``ValueError`` if the header does not authenticate under ``key``, so
    a cover image (or a wrong key) is rejected rather than returning noise.
    """
    stego = np.asarray(stego, dtype=np.uint8)
    flat = stego.reshape(-1)
    nonce, n_seg, L, gene_rows, rows = read_header(flat, stego.shape, key)
    chroms = [Chromosome.from_bits(g) for g in gene_rows]

    edges = [(s * L) // n_seg for s in range(n_seg + 1)]
    out = np.zeros(L, dtype=np.uint8)
    for s, chrom in enumerate(chroms):
        seg_len = edges[s + 1] - edges[s]
        if seg_len == 0:
            continue
        k = chrom.bits_per_pixel
        n_pix = int(np.ceil(seg_len / k))
        idx = _segment_pixels(stego.shape, chrom, n_pix, rows, s, n_seg)
        t = extract_bits_from_pixels(flat[idx], seg_len, chrom.mask, chrom.bp_dir)
        out[edges[s]: edges[s + 1]] = recompose(t, chrom.decomposition, derive_subkey(key, nonce, s))

    return out, chroms
