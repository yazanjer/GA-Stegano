"""Correctness tests. Run with ``pytest -q`` from the project root.

These are not smoke tests: each one guards a property that, if broken, would
silently invalidate a table in the paper.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdt.stego.bitplane import (embed_bits_into_pixels, extract_bits_from_pixels,
                                 mask_to_planes, plane_capacity)
from amdt.stego.codec import (Chromosome, bits_to_bytes, bytes_to_bits, embed, extract,
                              max_payload_bits)
from amdt.stego.decomposition import (BLOCK_SIZES, DecompositionParams, decompose,
                                      keyed_permutation, keystream_bits, recompose,
                                      segment_message)
from amdt.stego.traversal import traversal_order

KEY = bytes.fromhex("2a" * 32)


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("direction", range(16))
@pytest.mark.parametrize("shape", [(7, 9), (16, 16), (5, 32)])
def test_traversal_is_a_bijection(direction, shape):
    """Every pattern must visit each pixel exactly once, or bits are overwritten."""
    order = traversal_order(shape, direction, x_off=3, y_off=4)
    assert order.size == shape[0] * shape[1]
    assert np.array_equal(np.sort(order), np.arange(order.size))


@pytest.mark.parametrize("direction", range(16))
def test_traversal_offsets_stay_bijective(direction):
    for x, y in ((0, 0), (511, 511), (13, 7)):
        o = traversal_order((12, 20), direction, x, y)
        assert len(set(o.tolist())) == o.size


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mask", range(1, 16))
@pytest.mark.parametrize("bp_dir", [0, 1])
def test_bitplane_roundtrip(mask, bp_dir):
    rng = np.random.default_rng(0)
    px = rng.integers(0, 256, 100, dtype=np.uint8)
    n = plane_capacity(mask) * 100
    bits = rng.integers(0, 2, n, dtype=np.uint8)
    out, used = embed_bits_into_pixels(px, bits, mask, bp_dir)
    assert used == n
    assert np.array_equal(extract_bits_from_pixels(out, n, mask, bp_dir), bits)


def test_bitplane_partial_tail_leaves_planes_untouched():
    """Unused planes must not be zeroed -- a clean tail leaks the payload length."""
    px = np.full(4, 0b11111111, dtype=np.uint8)
    out, used = embed_bits_into_pixels(px, np.zeros(3, dtype=np.uint8), mask=15, bp_dir=0)
    assert used == 3
    assert out[0] & 0b1000 == 0b1000          # 4th selected plane of pixel 0 untouched
    assert out[1] == 0b11111111               # pixel 1 never written


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("length", [0, 1, 7, 64, 100, 255, 1224, 4096])
def test_decomposition_is_invertible_for_every_flag_combination(length):
    rng = np.random.default_rng(length)
    bits = rng.integers(0, 2, length, dtype=np.uint8)
    for a, b, s, bi, d in itertools.product((0, 1), (0, 1), (0, 1), range(4), (0, 1)):
        p = DecompositionParams(a, b, s, bi, d)
        assert np.array_equal(recompose(decompose(bits, p, KEY), p, KEY), bits), p


def test_diffusion_makes_pathological_payloads_uniform():
    """The T4 security claim, as a test: an all-zero payload must come out unbiased."""
    zeros = np.zeros(200_000, dtype=np.uint8)
    p = DecompositionParams(delta=1)
    out = decompose(zeros, p, KEY)
    assert abs(out.mean() - 0.5) < 0.005
    assert abs(np.corrcoef(out[:-1], out[1:])[0, 1]) < 0.01


def test_keystream_and_permutation_are_key_dependent():
    a = keystream_bits(KEY, "dif", 4096)
    b = keystream_bits(bytes.fromhex("2b" * 32), "dif", 4096)
    assert not np.array_equal(a, b)
    pa = keyed_permutation(KEY, "blk", 64)
    pb = keyed_permutation(bytes.fromhex("2b" * 32), "blk", 64)
    assert not np.array_equal(pa, pb)
    assert np.array_equal(np.sort(pa), np.arange(64))


def test_segmentation_partitions_without_loss():
    bits = np.arange(1000, dtype=np.uint8) % 2
    for n in (1, 2, 3, 7, 13):
        segs = segment_message(bits, n)
        assert sum(s.size for s in segs) == bits.size
        assert np.array_equal(np.concatenate(segs), bits)


# --------------------------------------------------------------------------- #
def _cover(rng, side=64):
    return rng.integers(0, 256, (side, side), dtype=np.uint8)


@pytest.mark.parametrize("n_seg", [1, 2, 5, 13])
def test_codec_roundtrip_multi_segment(n_seg):
    rng = np.random.default_rng(1)
    cover = _cover(rng)
    bits = bytes_to_bits(b"Steganography is the practice of concealing a file." * 3)
    chroms = [Chromosome(direction=(3 * i + 7) % 16, x_off=11 * i, y_off=5 * i,
                         mask=[5, 15, 9, 3, 10][i % 5], alpha=i % 2, beta=(i // 2) % 2,
                         bp_dir=i % 2, sigma=1, block_idx=i % 4, delta=1)
              for i in range(n_seg)]
    res = embed(cover, bits, chroms, KEY)
    out, got = extract(res.stego, KEY)
    assert np.array_equal(out, bits)
    assert [c.as_dict() for c in got] == [c.as_dict() for c in chroms]


def test_codec_roundtrip_over_all_direction_and_mask_genes():
    rng = np.random.default_rng(2)
    cover = _cover(rng)
    bits = bytes_to_bits(b"payload" * 40)
    for d in range(16):
        for m in range(1, 16):
            c = Chromosome(direction=d, x_off=257, y_off=13, mask=m, alpha=1, beta=1,
                           bp_dir=d % 2, sigma=1, block_idx=1, delta=1)
            out, _ = extract(embed(cover, bits, [c], KEY).stego, KEY)
            assert np.array_equal(out, bits), (d, m)


def test_extraction_rejects_a_plain_cover():
    rng = np.random.default_rng(3)
    with pytest.raises(ValueError):
        extract(_cover(rng), KEY)


def test_wrong_key_does_not_recover_the_payload():
    rng = np.random.default_rng(4)
    cover = _cover(rng)
    bits = bytes_to_bits(b"top secret" * 20)
    c = Chromosome(mask=3, sigma=1, delta=1, block_idx=2)
    stego = embed(cover, bits, [c], KEY).stego
    wrong, _ = extract(stego, bytes.fromhex("ff" * 32))
    assert not np.array_equal(wrong, bits)
    # and it should look like noise, not a near-miss
    assert 0.4 < wrong.mean() < 0.6


def test_header_never_collides_with_payload():
    rng = np.random.default_rng(5)
    cover = _cover(rng)
    chroms = [Chromosome(mask=15) for _ in range(4)]
    cap = max_payload_bits(cover.shape, chroms)
    bits = rng.integers(0, 2, cap, dtype=np.uint8)
    res = embed(cover, bits, chroms, KEY)
    out, _ = extract(res.stego, KEY)
    assert np.array_equal(out, bits)


def test_capacity_limit_is_enforced_not_silently_truncated():
    rng = np.random.default_rng(6)
    cover = _cover(rng, side=32)
    chroms = [Chromosome(mask=1)]
    too_much = rng.integers(0, 2, cover.size * 2, dtype=np.uint8)
    with pytest.raises(ValueError):
        embed(cover, too_much, chroms, KEY)


def test_bytes_bits_roundtrip():
    data = bytes(range(256))
    assert bits_to_bytes(bytes_to_bits(data)) == data
