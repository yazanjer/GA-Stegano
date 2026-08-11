r"""Hierarchical message decomposition -- formal definitions and algorithms.

Addresses Reviewer 1 comments 1 and 2 directly.

Comment 2 objected that the manuscript's Table 2 names *scrambling* and
*message diffusion* with no mathematical formulation, algorithm or parameters.
This module supplies all three, for every layer, and every layer is an
invertible map verified by ``tests/test_decomposition.py``.

Notation
--------
Let the raw payload be :math:`m = (m_1,\dots,m_L) \in \{0,1\}^L`.
Let :math:`K \in \{0,1\}^{256}` be a secret key shared by sender and receiver
(it is *never* embedded in the cover; only the layer-enable flags and the
block-size index travel in the header).  Write
:math:`\mathrm{KS}_K(\texttt{label}, n)` for the first :math:`n` bits of
SHA-256 in counter mode keyed by :math:`K`:

.. math::
    \mathrm{KS}_K(\ell, n) = \mathrm{bits}\big(
        \mathrm{SHA256}(K \| \ell \| \langle 0\rangle_{32}) \,\|\,
        \mathrm{SHA256}(K \| \ell \| \langle 1\rangle_{32}) \,\|\, \cdots
    \big)_{1:n}.

The four layers
---------------

**T1 -- bit complementation** (parameter :math:`\alpha \in \{0,1\}`, 1 header bit)

.. math:: T_1(m)_i = m_i \oplus \alpha .

Involutive: :math:`T_1^{-1} = T_1`.

**T2 -- bit reversal** (parameter :math:`\beta \in \{0,1\}`, 1 header bit)

.. math:: T_2(m)_i = m_{\,L+1-i} \text{ if } \beta = 1, \text{ else } m_i .

Involutive.

**T3 -- keyed block scrambling** (parameters: enable bit :math:`\sigma`,
block-size index :math:`b \in \{0,1,2,3\} \mapsto B \in \{64,128,256,512\}`;
3 header bits.  The permutation itself is key-derived, not transmitted.)

Partition :math:`m` into :math:`n_B=\lfloor L/B\rfloor` **full** blocks
:math:`m^{(1)},\dots,m^{(n_B)}` plus a tail of :math:`r = L - n_B B` bits (no
padding is used, so the layer is exactly length-preserving; the tail is
rotated within itself by :math:`\mathrm{KS}_K(\text{"tail"},16) \bmod r`).
Draw

.. math::
    \pi \leftarrow \mathrm{FisherYates}\big(n_B;\ \mathrm{KS}_K(\text{"blk"},\cdot)\big)
    \in S_{n_B},
    \qquad
    \rho_j = \mathrm{KS}_K(\text{"rot"}\|j, 16) \bmod B .

Then

.. math::
    T_3(m)^{(j)} = \mathrm{ROTL}_{\rho_{\pi(j)}}\big(m^{(\pi(j))}\big),

i.e. blocks are permuted *and* each block is cyclically rotated by its own
key-derived amount.  Inverse: rotate right by :math:`\rho_{\pi(j)}` and apply
:math:`\pi^{-1}`.

**T4 -- keyed diffusion** (enable bit :math:`\delta`, 1 header bit)

A CBC-style chaining over GF(2) with a key-derived keystream
:math:`k = \mathrm{KS}_K(\text{"dif"}, L)` and IV :math:`s_0 = \mathrm{KS}_K(\text{"iv"},1)`:

.. math:: s_i = t_i \oplus s_{i-1} \oplus k_i, \qquad i = 1,\dots,L,

with inverse :math:`t_i = s_i \oplus s_{i-1} \oplus k_i`.  Computed in
:math:`O(L)` by a prefix-XOR (``np.bitwise_xor.accumulate``), so the layer costs
one pass over the payload.

**S -- segmentation** (parameter :math:`n_s`, 8 header bits)

:math:`m` is cut into :math:`n_s` contiguous segments; each segment is
decomposed and embedded under its *own* GA-optimised parameter vector
:math:`\theta^{(s)}`, giving :math:`n_s` independent traversal paths, bit-plane
masks and transformation flags inside one cover.

Why this is more than "simple bit tricks" (Reviewer 1, comment 1)
-----------------------------------------------------------------
The reviewer is right that :math:`T_1` and :math:`T_2` alone are trivial and
carry no security argument -- they are *distortion* controls, not security
controls: they let the GA pick whichever of the four bit-orderings happens to
agree best with the cover's LSB plane, which lowers the number of embedding
changes.  That is their honest role and the code labels it as such
(:func:`distortion_layers`).

The security argument rests on :math:`T_3\circ T_4`:

1. *Uniformity.* Modelling SHA-256 as a random oracle, :math:`k` is uniform on
   :math:`\{0,1\}^L` and independent of :math:`m`, hence :math:`s = T_4(t)` is
   uniform on :math:`\{0,1\}^L` for **any** payload distribution.  Concretely,
   the per-bit bias :math:`|\Pr[s_i=1]-\tfrac12|` is negligible even for
   pathological payloads (all-zero, ASCII text, a bitmap) -- the case where
   plain LSB embedding is most detectable, because natural payloads are far
   from uniform and imprint their own histogram on the LSB plane.
2. *Reduction.* With :math:`s` uniform and independent of the cover, the
   detector can no longer exploit payload structure; the detection problem
   reduces to distinguishing the *embedding-change* process alone.  Formally,
   for any detector :math:`D`,
   :math:`|\Pr[D(\text{stego})=1] - \Pr[D(\text{stego}_{\text{rand}})=1]|
   \le \mathrm{Adv}^{\mathrm{prf}}_{\mathrm{SHA256}}`, i.e. content-dependent
   attacks (chi-square / pairs-of-values / histogram attacks, which all test
   for payload bias) gain no advantage over attacking a random payload.
3. *Key-space.* :math:`T_3` contributes :math:`\log_2(n_B!) + n_B\log_2 B` bits
   of uncertainty to an attacker who has *recovered the bit-plane and path*
   but not :math:`K`; see :func:`keyspace_bits`.  Combined with the
   :math:`4+9+9+4=26` bits of path/plane entropy per segment, this is what the
   ablation in Table 2 quantifies empirically.
4. *Difference from prior preprocessing.* Existing schemes encrypt the payload
   *before* embedding with a fixed key and fixed positions (e.g. chaos-LSB,
   AES-then-LSB).  Here the decomposition parameters are themselves
   **jointly optimised with the embedding path by the GA**, per segment, so the
   transformation and the carrier selection are chosen together to minimise
   distortion under a security constraint, instead of being two independent
   stages.  :func:`describe` emits this as machine-readable provenance.

Nothing in this module claims security beyond points 1-3: :math:`T_3`/:math:`T_4`
defeat *payload-statistics* attacks, they do **not** by themselves defeat
residual-based detectors such as SRM or a CNN.  That is exactly what the
steganalysis experiments in ``src/amdt/steganalysis`` are for.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence

import numpy as np

__all__ = [
    "BLOCK_SIZES",
    "DecompositionParams",
    "keystream_bits",
    "keyed_permutation",
    "complement",
    "reverse",
    "block_scramble",
    "block_unscramble",
    "diffuse",
    "undiffuse",
    "decompose",
    "recompose",
    "segment_message",
    "keyspace_bits",
    "distortion_layers",
    "security_layers",
    "describe",
]

BLOCK_SIZES = (64, 128, 256, 512)
distortion_layers = ("T1_complement", "T2_reverse")
security_layers = ("T3_block_scramble", "T4_diffusion")


# --------------------------------------------------------------------------- #
# key-derived randomness
# --------------------------------------------------------------------------- #
def keystream_bits(key: bytes, label: str, n_bits: int) -> np.ndarray:
    """First ``n_bits`` of SHA-256-CTR keyed by ``key`` as a uint8 0/1 array."""
    if n_bits <= 0:
        return np.zeros(0, dtype=np.uint8)
    n_bytes = (n_bits + 7) // 8
    blocks: List[bytes] = []
    counter = 0
    label_b = label.encode("utf-8")
    while sum(len(b) for b in blocks) < n_bytes:
        blocks.append(
            hashlib.sha256(key + label_b + counter.to_bytes(4, "little")).digest()
        )
        counter += 1
    raw = b"".join(blocks)[:n_bytes]
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits[:n_bits].copy()


def keyed_permutation(key: bytes, label: str, n: int) -> np.ndarray:
    """Fisher-Yates permutation of ``range(n)`` driven by the keystream.

    Uses 64 key bits per swap and rejection-free modular reduction; the modulo
    bias is < 2^-40 for any n <= 2^24, which is far below anything an attacker
    could exploit and is documented here rather than hidden.
    """
    if n <= 1:
        return np.arange(max(n, 0), dtype=np.int64)
    raw = keystream_bits(key, label, 64 * n)
    words = np.packbits(raw.reshape(n, 64), axis=1).view(">u8").reshape(n)
    perm = np.arange(n, dtype=np.int64)
    for i in range(n - 1, 0, -1):
        j = int(words[i] % (i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    return perm


# --------------------------------------------------------------------------- #
# T1 / T2 -- distortion-shaping layers
# --------------------------------------------------------------------------- #
def complement(bits: np.ndarray, alpha: int) -> np.ndarray:
    """T1. Involutive."""
    return bits ^ np.uint8(1) if alpha else bits.copy()


def reverse(bits: np.ndarray, beta: int) -> np.ndarray:
    """T2. Involutive."""
    return bits[::-1].copy() if beta else bits.copy()


# --------------------------------------------------------------------------- #
# T3 -- keyed block scrambling
# --------------------------------------------------------------------------- #
def _rot_amounts(key: bytes, n_blocks: int, block: int) -> np.ndarray:
    if n_blocks <= 0:
        return np.zeros(0, dtype=np.int64)
    raw = keystream_bits(key, "rot", 16 * n_blocks)
    words = np.packbits(raw.reshape(n_blocks, 16), axis=1).view(">u2").reshape(n_blocks)
    return (words.astype(np.int64) % block).astype(np.int64)


def _tail_rotation(key: bytes, r: int) -> int:
    """Rotation applied to the final partial block (length ``r`` < B)."""
    if r <= 1:
        return 0
    w = int(np.packbits(keystream_bits(key, "tail", 16).reshape(1, 16), axis=1).view(">u2")[0, 0])
    return w % r


def block_scramble(bits: np.ndarray, key: bytes, block: int) -> np.ndarray:
    """T3 forward.  Strictly length-preserving -- **no padding**.

    ``L`` splits into ``n = floor(L/B)`` full blocks plus a tail of
    ``r = L - nB`` bits.  Full blocks are permuted among themselves and each is
    rotated; the tail is rotated within itself (a permutation of a length-``r``
    cyclic group).  Padding is deliberately avoided: padding then truncating
    would discard real payload bits and make the layer non-invertible, which is
    the classic bug in block-scrambling implementations.
    """
    L = int(bits.size)
    if L == 0:
        return bits.copy()
    n = L // block
    r = L - n * block
    out = np.empty(L, dtype=np.uint8)

    if n:
        buf = bits[: n * block].reshape(n, block)
        perm = keyed_permutation(key, "blk", n)
        rot = _rot_amounts(key, n, block)
        moved = buf[perm]
        idx = (np.arange(block)[None, :] + rot[perm][:, None]) % block
        out[: n * block] = np.take_along_axis(moved, idx, axis=1).reshape(-1)
    if r:
        t = _tail_rotation(key, r)
        out[n * block:] = np.roll(bits[n * block:], -t)
    return out


def block_unscramble(bits: np.ndarray, key: bytes, block: int, length: int) -> np.ndarray:
    """T3 inverse.  ``length`` must equal the forward input length."""
    L = int(length)
    if L == 0:
        return np.zeros(0, dtype=np.uint8)
    n = L // block
    r = L - n * block
    out = np.empty(L, dtype=np.uint8)

    if n:
        buf = bits[: n * block].reshape(n, block)
        perm = keyed_permutation(key, "blk", n)
        rot = _rot_amounts(key, n, block)
        idx = (np.arange(block)[None, :] - rot[perm][:, None]) % block
        unrot = np.take_along_axis(buf, idx, axis=1)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(n, dtype=np.int64)
        out[: n * block] = unrot[inv].reshape(-1)
    if r:
        t = _tail_rotation(key, r)
        out[n * block:] = np.roll(bits[n * block:], t)
    return out


# --------------------------------------------------------------------------- #
# T4 -- keyed diffusion
# --------------------------------------------------------------------------- #
def diffuse(bits: np.ndarray, key: bytes) -> np.ndarray:
    r"""T4 forward: :math:`s_i = t_i \oplus s_{i-1} \oplus k_i`.

    Implemented as a prefix XOR: with :math:`u_i = t_i \oplus k_i`,
    :math:`s_i = s_0 \oplus \bigoplus_{j\le i} u_j`.
    """
    L = bits.size
    if L == 0:
        return bits.copy()
    k = keystream_bits(key, "dif", L)
    iv = keystream_bits(key, "iv", 1)[0]
    u = bits ^ k
    return (np.bitwise_xor.accumulate(u) ^ iv).astype(np.uint8)


def undiffuse(bits: np.ndarray, key: bytes) -> np.ndarray:
    r"""T4 inverse: :math:`t_i = s_i \oplus s_{i-1} \oplus k_i`."""
    L = bits.size
    if L == 0:
        return bits.copy()
    k = keystream_bits(key, "dif", L)
    iv = keystream_bits(key, "iv", 1)[0]
    prev = np.empty(L, dtype=np.uint8)
    prev[0] = iv
    prev[1:] = bits[:-1]
    return (bits ^ prev ^ k).astype(np.uint8)


# --------------------------------------------------------------------------- #
# parameter object + full pipeline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DecompositionParams:
    """Header-transmitted decomposition parameters (7 bits total)."""

    alpha: int = 0          # 1 bit  -- T1 complement
    beta: int = 0           # 1 bit  -- T2 reverse
    sigma: int = 0          # 1 bit  -- T3 enable
    block_idx: int = 0      # 2 bits -- T3 block size index
    delta: int = 0          # 1 bit  -- T4 enable
    _reserved: int = 0      # 1 bit  -- future use, keeps the header byte-aligned

    @property
    def block(self) -> int:
        return BLOCK_SIZES[int(self.block_idx) & 0b11]

    def as_dict(self) -> Dict[str, int]:
        d = asdict(self)
        d.pop("_reserved", None)
        d["block_size"] = self.block
        return d


def decompose(bits: np.ndarray, params: DecompositionParams, key: bytes) -> np.ndarray:
    """Apply T1 -> T2 -> T3 -> T4 in order."""
    x = np.asarray(bits, dtype=np.uint8)
    x = complement(x, params.alpha)
    x = reverse(x, params.beta)
    if params.sigma:
        x = block_scramble(x, key, params.block)
    if params.delta:
        x = diffuse(x, key)
    return x


def recompose(bits: np.ndarray, params: DecompositionParams, key: bytes) -> np.ndarray:
    """Apply the inverses in reverse order: T4^-1 -> T3^-1 -> T2^-1 -> T1^-1."""
    x = np.asarray(bits, dtype=np.uint8)
    if params.delta:
        x = undiffuse(x, key)
    if params.sigma:
        x = block_unscramble(x, key, params.block, x.size)
    x = reverse(x, params.beta)
    x = complement(x, params.alpha)
    return x


def segment_message(bits: np.ndarray, n_segments: int) -> List[np.ndarray]:
    """Contiguous near-equal split (Algorithm 3 of the manuscript).

    Segment ``s`` spans indices ``[floor(sL/n), floor((s+1)L/n))`` so lengths
    differ by at most one bit and the boundaries are recoverable from
    ``(L, n)`` alone -- no per-segment length field is needed in the header.
    """
    n_segments = max(1, int(n_segments))
    L = int(np.asarray(bits).size)
    edges = [(s * L) // n_segments for s in range(n_segments + 1)]
    return [np.asarray(bits, dtype=np.uint8)[edges[s]: edges[s + 1]] for s in range(n_segments)]


# --------------------------------------------------------------------------- #
# security accounting
# --------------------------------------------------------------------------- #
def keyspace_bits(length: int, params: DecompositionParams, n_segments: int = 1) -> Dict[str, float]:
    r"""Attacker uncertainty contributed by each layer, in bits.

    Reported per segment and in total; used to build the key-space column of the
    security table requested in Reviewer 1 comment 1.
    """
    per_seg_path = 4 + 9 + 9 + 4          # direction, x_off, y_off, mask genes
    t12 = 2 * (params.alpha is not None)  # alpha, beta -- 2 bits, always present
    t3 = 0.0
    if params.sigma:
        n_b = max(1, math.ceil(length / params.block))
        t3 = math.lgamma(n_b + 1) / math.log(2) + n_b * math.log2(params.block)
    t4 = 256.0 if params.delta else 0.0   # bounded by the key length
    per_segment = per_seg_path + t12 + t3
    return {
        "path_and_plane_bits_per_segment": float(per_seg_path),
        "T1_T2_bits_per_segment": float(t12),
        "T3_bits_per_segment": float(t3),
        "T4_key_bits_shared": float(t4),
        "total_bits": float(per_segment * n_segments + t4),
    }


def describe() -> Dict[str, object]:
    """Machine-readable spec dumped into every run directory (comment 2)."""
    return {
        "layers": [
            {
                "id": "T1",
                "name": "bit complementation",
                "formula": "T1(m)_i = m_i XOR alpha",
                "parameters": {"alpha": "1 bit, in header, GA-optimised"},
                "role": "distortion shaping (not a security layer)",
                "invertible": True,
            },
            {
                "id": "T2",
                "name": "bit reversal",
                "formula": "T2(m)_i = m_{L+1-i} if beta=1 else m_i",
                "parameters": {"beta": "1 bit, in header, GA-optimised"},
                "role": "distortion shaping (not a security layer)",
                "invertible": True,
            },
            {
                "id": "T3",
                "name": "keyed block scrambling",
                "formula": (
                    "partition into ceil(L/B) blocks; permute blocks by "
                    "pi ~ FisherYates(KS_K('blk')); rotate block j left by "
                    "rho_{pi(j)} = KS_K('rot'||j) mod B"
                ),
                "parameters": {
                    "sigma": "1 bit enable, in header",
                    "block_idx": f"2 bits -> B in {BLOCK_SIZES}",
                    "pi, rho": "derived from secret key K, never transmitted",
                },
                "role": "security layer: destroys local payload structure",
                "invertible": True,
            },
            {
                "id": "T4",
                "name": "keyed diffusion",
                "formula": "s_i = t_i XOR s_{i-1} XOR k_i,  k = KS_K('dif', L)",
                "parameters": {
                    "delta": "1 bit enable, in header",
                    "k, IV": "derived from secret key K, never transmitted",
                },
                "role": (
                    "security layer: makes the embedded stream computationally "
                    "indistinguishable from uniform, so payload-statistics "
                    "attacks (chi-square, pairs-of-values) gain no advantage"
                ),
                "invertible": True,
            },
            {
                "id": "S",
                "name": "segmentation",
                "formula": "segment s = m[floor(sL/n) : floor((s+1)L/n)]",
                "parameters": {"n_segments": "8 bits, in global header"},
                "role": "each segment gets an independent GA-optimised theta",
                "invertible": True,
            },
        ],
        "header_bits": {"alpha": 1, "beta": 1, "sigma": 1, "block_idx": 2, "delta": 1, "reserved": 1},
        "key": "256-bit shared secret, never embedded",
        "distortion_layers": list(distortion_layers),
        "security_layers": list(security_layers),
    }
