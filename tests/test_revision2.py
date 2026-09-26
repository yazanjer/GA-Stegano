"""Revision-2 tests: keying, header, chromosome accounting, STC, AMDT-D, new baselines, full SRM."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amdt.stego.codec import (CHROM_BITS, HEADER_FIXED_BITS, Chromosome, _GENE_WIDTHS, embed,  # noqa: E402
                              extract, header_bit_count)
from amdt.stego.decomposition import DecompositionParams, decompose, derive_subkey  # noqa: E402
from amdt.ga.optimizer import search_space_size  # noqa: E402

KEY = bytes(range(32))


def _texture(rng, n=128):
    base = rng.integers(0, 256, (n, n)).astype(np.float64)
    from scipy.ndimage import uniform_filter
    img = 0.5 * base + 0.5 * uniform_filter(base, 5)
    img[: n // 4] = 128.0                      # a flat strip, like sky
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# chromosome / search-space accounting (Reviewer 1, comment 5)
# --------------------------------------------------------------------------- #
def test_chromosome_is_33_bits_everywhere():
    assert CHROM_BITS == 33 == sum(_GENE_WIDTHS)
    assert len(Chromosome().to_bits()) == 33
    assert len(DecompositionParams.__dataclass_fields__) == 5          # 6 bits: a,b,s,block(2),d
    assert header_bit_count(4) == HEADER_FIXED_BITS + 4 * 33 == 248


def test_search_space_sizes_match_manuscript():
    full = search_space_size(1)
    cons = search_space_size(1, constrained=True)
    assert full == 16 * 512 * 512 * 15 * 2 ** 5 * 4
    assert f"{full:.2e}" == "8.05e+09"
    assert cons == full / 4 and f"{cons:.2e}" == "2.01e+09"


# --------------------------------------------------------------------------- #
# nonce, sub-keys, encrypted/authenticated header (Reviewer 2, comment 5)
# --------------------------------------------------------------------------- #
def test_keystream_not_reused_across_messages_or_segments():
    bits = np.zeros(4096, dtype=np.uint8)
    p = DecompositionParams(sigma=1, delta=1, block_idx=1)
    a = decompose(bits, p, derive_subkey(KEY, b"A" * 8, 0))
    b = decompose(bits, p, derive_subkey(KEY, b"B" * 8, 0))
    c = decompose(bits, p, derive_subkey(KEY, b"A" * 8, 1))
    assert not np.array_equal(a, b) and not np.array_equal(a, c)
    assert abs(np.mean(a ^ b) - 0.5) < 0.05                           # independent streams


def test_header_has_no_fixed_signature():
    rng = np.random.default_rng(0)
    cover = _texture(rng, 64)
    heads = []
    for i in range(64):
        st = embed(cover, rng.integers(0, 2, 300, dtype=np.uint8), [Chromosome(mask=1, sigma=1, delta=1)],
                   KEY, nonce=rng.integers(0, 256, 8, dtype=np.uint8).tobytes()).stego
        heads.append(st.reshape(-1)[-header_bit_count(1):] & 1)
    m = np.mean(heads, axis=0)
    assert m.min() > 0.1 and m.max() < 0.9                            # no constant bit position


def test_tampered_header_is_rejected():
    rng = np.random.default_rng(1)
    cover = _texture(rng, 64)
    pay = rng.integers(0, 2, 500, dtype=np.uint8)
    st = embed(cover, pay, [Chromosome(mask=3, sigma=1, delta=1)], KEY, nonce=b"n" * 8).stego
    rec, _ = extract(st, KEY)
    assert np.array_equal(rec, pay)
    bad = st.copy().reshape(-1)
    bad[-100] ^= 1
    with pytest.raises(ValueError):
        extract(bad.reshape(st.shape), KEY)


# --------------------------------------------------------------------------- #
# traversal vectorisation is exact
# --------------------------------------------------------------------------- #
def test_vectorised_traversal_equals_reference_loop():
    from amdt.stego.traversal import _traversal_order_loop, traversal_order
    for d in range(16):
        for sh in [(7, 9), (1, 5), (5, 1), (16, 16)]:
            for xo, yo in [(0, 0), (3, 4)]:
                assert np.array_equal(traversal_order(sh, d, xo, yo), _traversal_order_loop(sh, d, xo, yo))


# --------------------------------------------------------------------------- #
# STC
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,m,h", [(1000, 100, 7), (1000, 400, 10), (997, 331, 8), (64, 64, 6)])
def test_stc_roundtrip(n, m, h):
    from amdt.stego.stc import stc_embed_bits, stc_extract_bits
    rng = np.random.default_rng(n + m + h)
    x = rng.integers(0, 2, n, dtype=np.uint8)
    rho = rng.random(n) + 0.01
    msg = rng.integers(0, 2, m, dtype=np.uint8)
    y, cost = stc_embed_bits(x, rho, msg, h, seed=5)
    assert np.array_equal(stc_extract_bits(y, m, h, seed=5), msg)
    assert math.isclose(cost, float(rho[y != x].sum()), rel_tol=1e-9)


def test_stc_beats_naive_embedding_in_distortion():
    from amdt.stego.stc import stc_embed_bits
    rng = np.random.default_rng(3)
    n, m = 4000, 800
    x = rng.integers(0, 2, n, dtype=np.uint8)
    rho = rng.exponential(1.0, n)
    msg = rng.integers(0, 2, m, dtype=np.uint8)
    _, cost = stc_embed_bits(x, rho, msg, 10, 1)
    naive = rho[:m][x[:m] != msg].sum()
    assert cost < 0.5 * naive


@pytest.mark.parametrize("name", ["HILL", "S-UNIWARD", "WOW", "MiPOD", "EvoHILL"])
def test_stc_complete_system_roundtrip(name):
    pytest.importorskip("conseal")
    from amdt.baselines.reference import stc_extract, stc_system
    rng = np.random.default_rng(7)
    x = _texture(rng, 128)
    pay = rng.integers(0, 2, int(0.2 * x.size), dtype=np.uint8)
    st, info = stc_system(name, x, pay, KEY, rng)
    assert np.array_equal(stc_extract(name, st, KEY), pay)
    assert np.abs(st.astype(int) - x).max() <= 1


# --------------------------------------------------------------------------- #
# AMDT-D
# --------------------------------------------------------------------------- #
def test_amdt_d_roundtrip_and_gating_invariance():
    pytest.importorskip("conseal")
    from amdt.ga.optimizer import GAConfig
    from amdt.stego.amdt_d import extract_d, gating_cost, run_amdt_d
    rng = np.random.default_rng(11)
    x = _texture(rng, 128)
    pay = rng.integers(0, 2, int(0.1 * x.size), dtype=np.uint8)
    cfg = GAConfig(population=8, generations=5, patience=5, n_segments=2)
    st, info = run_amdt_d(x, pay, KEY, cfg, rng)
    assert info["extraction_ok"]
    assert np.array_equal(gating_cost(st), gating_cost(x))
    assert np.array_equal(st >> 2, x >> 2)                            # carry-safe
    rec, _ = extract_d(st, KEY)
    assert np.array_equal(rec, pay)


# --------------------------------------------------------------------------- #
# new baselines
# --------------------------------------------------------------------------- #
def test_fm_pso_lsb_roundtrip_and_fast_order():
    pytest.importorskip("skimage")
    from amdt.baselines.reference import _fm_maps, _fm_order, fm_pso_extract, fm_pso_lsb
    rng = np.random.default_rng(2)
    x = _texture(rng, 96)
    e, n = _fm_maps(x)
    for L in (1, 100, 3000):
        assert np.array_equal(_fm_order(e, n, 0.3, 2.0, L), np.argsort(-((0.3 * e + 0.7 * n).reshape(-1) ** 2.0),
                                                                   kind="stable")[:L])
    pay = rng.integers(0, 2, 900, dtype=np.uint8)
    st, info = fm_pso_lsb(x, pay, KEY, rng, particles=4, iters=2)
    assert np.array_equal(fm_pso_extract(st, pay.size, info["fm_alpha"], info["fm_gamma"], KEY), pay)


def test_evohill_cost_is_finite_positive():
    from amdt.baselines.reference import evohill_cost
    rho = evohill_cost(_texture(np.random.default_rng(0), 64))
    assert np.all(np.isfinite(rho)) and np.all(rho > 0)


def test_ea_lsb_is_blindly_decodable():
    from amdt.baselines.classical import _sobel_magnitude, edge_adaptive_lsb
    rng = np.random.default_rng(9)
    x = _texture(rng, 64)
    pay = rng.integers(0, 2, 800, dtype=np.uint8)
    st, _ = edge_adaptive_lsb(x, pay, rng)
    mag = _sobel_magnitude(st & np.uint8(0xFE)).reshape(-1)
    assert np.array_equal(st.reshape(-1)[np.argsort(-mag, kind="stable")[:800]] & 1, pay)


# --------------------------------------------------------------------------- #
# full SRM (Reviewer 2, comment 4)
# --------------------------------------------------------------------------- #
def test_full_srm_dimension_and_bit_identity():
    pytest.importorskip("sealwatch")
    import importlib
    from amdt.steganalysis.srm_full import srm_full
    x = _texture(np.random.default_rng(1), 96)
    fast = srm_full(x)
    assert fast.shape == (34671,)
    co = sys.modules["sealwatch.srm.cooccurrence"]
    importlib.reload(co)
    import sealwatch.srm.srm as ss
    importlib.reload(ss)
    f = ss.extract(x)
    ref = np.concatenate([np.asarray(f[k], dtype=np.float64).ravel() for k in sorted(f)])
    assert np.array_equal(fast, ref)


def test_srm_layout_does_not_depend_on_hash_seed():
    """sealwatch orders some submodels by set iteration; srm_full must not."""
    pytest.importorskip("sealwatch")
    import os, subprocess
    code = ("import sys, hashlib, numpy as np; sys.path.insert(0, 'src');"
            "from amdt.steganalysis.srm_full import srm_full;"
            "x = (np.random.default_rng(3).random((64, 64)) * 255).astype(np.uint8);"
            "print(hashlib.md5(srm_full(x).tobytes()).hexdigest())")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outs = {subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip() for h in (1, 2, 3)}
    assert len(outs) == 1
