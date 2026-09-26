"""AMDT-D: detectability-aware AMDT (revision 2, Reviewer 2 comment 7).

Reviewer 2 observed that an MSE fitness cannot express detectability, so more
GA search under MSE cannot close the security gap.  AMDT-D keeps everything
that makes AMDT a *blind, keyed, exactly decodable* scheme -- per-segment GA
search over traversal direction, offsets and T1/T2 flags, the keyed T3/T4
layers, the authenticated self-describing header -- and changes three things:

1. **Fitness.**  The GA minimises the additive detectability-aware distortion
   ``D = sum_{changed pixels} rho_HILL(cover)`` instead of the MSE.  (HILL cost
   from the pinned ``conseal`` implementation; any additive cost can be used.)

2. **Content-adaptive gating that the receiver can recompute.**  A gating map
   ``tau`` is the HILL cost of ``4*(x >> 2) + 2`` -- a function of bit-planes
   2..7 only.  In segment ``s`` the traversal visits only the
   ``ceil(r_g * L_s)`` pixels of lowest ``tau`` in its band (``L_s`` = bits of
   the segment, ``r_g`` = gate ratio chosen by the GA from
   ``GATE_RATIOS``; ``g = 0`` disables gating).  Because embedding never
   touches bit-planes 2..7 (point 3) and the header is written by LSB
   replacement, ``tau`` is bit-identical on cover and stego, so the receiver
   rebuilds the same visiting order from the stego image and the key.

3. **Carry-safe +-1 embedding.**  A required LSB flip is made by +1 or -1.  If
   the two low bits are 11 the change must be -1, if 00 it must be +1 (either
   other choice would carry into bit-plane 2 and break point 2); otherwise the
   sign is drawn at random.  Half of the changes are thus LSB-matching-like and
   the LSB-replacement asymmetry that RS/WS/SRM exploit is reduced; one bit per
   visited pixel (LSB plane only), so ``mask = 1`` is fixed and the mask /
   bp_dir genes are dropped.

Chromosome (per segment, 30 bits):

    direction 4 | x_off 9 | y_off 9 | alpha 1 | beta 1 | sigma 1 | block_idx 2
    | delta 1 | gate 2

Header: the revision-2 encrypted header with version 3 and 30-bit genes
(116 + 30 * n_seg bits).  Security constraint sigma = delta = 1 as for AMDT.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..evaluation.metrics import evaluate_pair, mse
from ..ga.optimizer import GAConfig, GAResult, optimise, random_search
from .codec import (HEADER_FIXED_BITS, _bits_to_int, _int_to_bits, build_header, new_nonce,
                    read_header)
from .decomposition import DecompositionParams, decompose, derive_subkey, recompose, segment_message
from .traversal import header_pixels, traversal_order

__all__ = ["DChromosome", "D_GENE_BOUNDS", "D_CHROM_BITS", "GATE_RATIOS", "D_ABLATIONS",
           "embed_d", "extract_d", "run_amdt_d", "gating_cost", "detect_cost"]

VERSION_D = 3
GATE_RATIOS = (0.0, 1.25, 1.6, 2.5)
D_GENE_NAMES = ("direction", "x_off", "y_off", "alpha", "beta", "sigma", "block_idx", "delta", "gate")
D_GENE_BOUNDS = ((0, 15), (0, 511), (0, 511), (0, 1), (0, 1), (0, 1), (0, 3), (0, 1), (0, 3))
D_GENE_WIDTHS = (4, 9, 9, 1, 1, 1, 2, 1, 2)
D_CHROM_BITS = sum(D_GENE_WIDTHS)   # 30
SECURITY_LOCKS = {"sigma": 1, "delta": 1}


@dataclass(frozen=True)
class DChromosome:
    direction: int = 0
    x_off: int = 0
    y_off: int = 0
    alpha: int = 0
    beta: int = 0
    sigma: int = 1
    block_idx: int = 0
    delta: int = 1
    gate: int = 0

    @classmethod
    def from_vector(cls, v) -> "DChromosome":
        return cls(**{n: int(x) for n, x in zip(D_GENE_NAMES, v)})

    def to_bits(self) -> np.ndarray:
        return np.concatenate([_int_to_bits(getattr(self, n), w) for n, w in zip(D_GENE_NAMES, D_GENE_WIDTHS)])

    @classmethod
    def from_bits(cls, bits) -> "DChromosome":
        vals, i = [], 0
        for w in D_GENE_WIDTHS:
            vals.append(_bits_to_int(bits[i:i + w])); i += w
        return cls.from_vector(vals)

    @property
    def decomposition(self) -> DecompositionParams:
        return DecompositionParams(alpha=self.alpha, beta=self.beta, sigma=self.sigma,
                                   block_idx=self.block_idx, delta=self.delta)

    def locked(self, locks: Dict[str, int]) -> "DChromosome":
        d = {n: getattr(self, n) for n in D_GENE_NAMES}
        d.update(locks)
        return DChromosome(**d)

    def as_dict(self) -> Dict[str, int]:
        return {n: int(getattr(self, n)) for n in D_GENE_NAMES}


def _hill(x: np.ndarray) -> np.ndarray:
    import conseal
    r = conseal.hill.compute_cost_adjusted(np.asarray(x, dtype=np.uint8))
    if isinstance(r, tuple):
        r = r[0] if len(r) == 1 else np.minimum(r[0], r[1])
    return np.asarray(r, dtype=np.float64)


def gating_cost(img: np.ndarray) -> np.ndarray:
    """HILL cost of the bit-plane-2..7 image; invariant under AMDT-D embedding."""
    y = ((np.asarray(img, dtype=np.uint8) >> 2) << 2) + 2
    return _hill(y.astype(np.uint8))


def detect_cost(cover: np.ndarray) -> np.ndarray:
    """Detectability-aware per-pixel cost used as the GA fitness (HILL)."""
    import conseal
    r = conseal.hill.compute_cost_adjusted(np.asarray(cover, dtype=np.uint8))
    if isinstance(r, tuple) and len(r) == 2:
        return np.minimum(np.asarray(r[0], float), np.asarray(r[1], float))
    return np.asarray(r[0] if isinstance(r, tuple) else r, float)


def _rows(w: int, n_seg: int) -> int:
    return int(np.ceil((HEADER_FIXED_BITS + D_CHROM_BITS * n_seg) / w))


@dataclass
class _Plan:
    """Per-(image, payload length, n_seg) precomputation shared by all GA candidates."""
    shape: Tuple[int, int]
    n_seg: int
    rows: int
    bands: List[Tuple[int, int]]
    seg_len: List[int]
    keep: List[List[np.ndarray]]      # keep[s][g] -> bool mask over band-local indices


def _plan(tau: np.ndarray, L: int, n_seg: int) -> _Plan:
    h, w = tau.shape
    rows = _rows(w, n_seg)
    usable = h - rows
    bands = [((s * usable) // n_seg, ((s + 1) * usable) // n_seg) for s in range(n_seg)]
    edges = [(s * L) // n_seg for s in range(n_seg + 1)]
    seg_len = [edges[s + 1] - edges[s] for s in range(n_seg)]
    keep = []
    for (r0, r1), n_need in zip(bands, seg_len):
        tb = tau[r0:r1].reshape(-1)
        order = np.lexsort((np.arange(tb.size), tb))       # stable: tau, then index
        masks = []
        for r in GATE_RATIOS:
            m = np.ones(tb.size, dtype=bool)
            if r > 0:
                k = min(tb.size, int(np.ceil(r * n_need)))
                m[:] = False
                m[order[:k]] = True
            masks.append(m)
        keep.append(masks)
    return _Plan((h, w), n_seg, rows, bands, seg_len, keep)


def _seg_indices(plan: _Plan, s: int, c: DChromosome) -> np.ndarray:
    r0, r1 = plan.bands[s]
    w = plan.shape[1]
    order = traversal_order((r1 - r0, w), c.direction, c.x_off, c.y_off)
    order = order[plan.keep[s][c.gate][order]]
    n = plan.seg_len[s]
    if n > order.size:
        raise ValueError("segment does not fit its gated band")
    return order[:n] + r0 * w


def embed_d(cover: np.ndarray, payload: np.ndarray, chroms: Sequence[DChromosome], key: bytes,
            nonce: bytes, rng: np.random.Generator, plan: Optional[_Plan] = None,
            tau: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, object]]:
    cover = np.asarray(cover, dtype=np.uint8)
    payload = np.asarray(payload, dtype=np.uint8).reshape(-1)
    n_seg = len(chroms)
    if plan is None:
        plan = _plan(gating_cost(cover) if tau is None else tau, payload.size, n_seg)
    flat = cover.reshape(-1).astype(np.int16).copy()
    for s, (c, seg) in enumerate(zip(chroms, segment_message(payload, n_seg))):
        if not seg.size:
            continue
        t = decompose(seg, c.decomposition, derive_subkey(key, nonce, s))
        idx = _seg_indices(plan, s, c)
        v = flat[idx]
        need = (v & 1).astype(np.uint8) != t
        vi = v[need]
        low = vi & 3
        sign = np.where(low == 3, -1, np.where(low == 0, 1, rng.choice(np.array([-1, 1], dtype=np.int16), size=vi.size)))
        v[need] = vi + sign
        flat[idx] = v
    hb = build_header(chroms, payload.size, key, nonce, [c.to_bits() for c in chroms], VERSION_D)
    hidx = header_pixels(cover.shape, hb.size, plan.rows)
    flat[hidx] = (flat[hidx] & 0xFE) | hb
    st = flat.astype(np.uint8).reshape(cover.shape)
    return st, {"n_changes": int(np.count_nonzero(st != cover)), "header_bits": int(hb.size)}


def extract_d(stego: np.ndarray, key: bytes) -> Tuple[np.ndarray, List[DChromosome]]:
    stego = np.asarray(stego, dtype=np.uint8)
    flat = stego.reshape(-1)
    nonce, n_seg, L, genes, rows = read_header(flat, stego.shape, key, D_CHROM_BITS, VERSION_D)
    chroms = [DChromosome.from_bits(g) for g in genes]
    plan = _plan(gating_cost(stego), L, n_seg)
    out = np.zeros(L, dtype=np.uint8)
    edges = [(s * L) // n_seg for s in range(n_seg + 1)]
    for s, c in enumerate(chroms):
        if edges[s + 1] == edges[s]:
            continue
        idx = _seg_indices(plan, s, c)
        t = (flat[idx] & 1).astype(np.uint8)
        out[edges[s]:edges[s + 1]] = recompose(t, c.decomposition, derive_subkey(key, nonce, s))
    return out, chroms


@dataclass(frozen=True)
class DSpec:
    name: str
    locks: Dict[str, int] = field(default_factory=dict)
    fitness: str = "cost"          # cost | mse
    n_segments: Optional[int] = None
    search: str = "ga"             # ga | random (matched budget) | none


D_ABLATIONS: Dict[str, DSpec] = {
    "full":       DSpec("full", dict(SECURITY_LOCKS)),
    "mse_fitness": DSpec("mse_fitness", dict(SECURITY_LOCKS), fitness="mse"),
    "no_gate":    DSpec("no_gate", {**SECURITY_LOCKS, "gate": 0}),
    "fixed_path": DSpec("fixed_path", {**SECURITY_LOCKS, "direction": 0, "x_off": 0, "y_off": 0}),
    "single_seg": DSpec("single_seg", dict(SECURITY_LOCKS), n_segments=1),
    "no_ga":      DSpec("no_ga", dict(SECURITY_LOCKS), search="random"),
    "no_search":  DSpec("no_search", dict(SECURITY_LOCKS), search="none"),
}


def run_amdt_d(cover: np.ndarray, payload: np.ndarray, key: bytes, ga_cfg: GAConfig,
               rng: np.random.Generator, variant: str = "full", verify: bool = True,
               budget: Optional[int] = None) -> Dict[str, object]:
    spec = D_ABLATIONS[variant]
    n_seg = spec.n_segments or ga_cfg.n_segments
    cfg = GAConfig(**{**ga_cfg.as_dict(), "n_segments": n_seg})
    cover = np.asarray(cover, dtype=np.uint8)
    payload = np.asarray(payload, dtype=np.uint8).reshape(-1)
    nonce = new_nonce(rng)
    tau = gating_cost(cover)
    plan = _plan(tau, payload.size, n_seg)
    rho = detect_cost(cover).reshape(-1)
    cflat = cover.reshape(-1)
    subkeys = [derive_subkey(key, nonce, s) for s in range(n_seg)]
    segs = segment_message(payload, n_seg)
    t_cache: Dict[Tuple[int, int, int, int, int, int], np.ndarray] = {}

    def seg_bits(s, c):
        k = (s, c.alpha, c.beta, c.sigma, c.block_idx, c.delta)
        if k not in t_cache:
            t_cache[k] = decompose(segs[s], c.decomposition, subkeys[s])
        return t_cache[k]

    def fitness(chroms: List[DChromosome]) -> float:
        chroms = [c.locked(spec.locks) for c in chroms]
        tot = 0.0
        for s, c in enumerate(chroms):
            try:
                idx = _seg_indices(plan, s, c)
            except ValueError:
                return -np.inf
            ch = idx[(cflat[idx] & 1) != seg_bits(s, c)]
            tot += float(rho[ch].sum()) if spec.fitness == "cost" else float(ch.size)
        return -tot

    t0 = __import__("time").perf_counter()
    if spec.search == "ga":
        ga = optimise(fitness, cfg, rng, bounds=D_GENE_BOUNDS, decode=DChromosome.from_vector)
    elif spec.search == "none":
        ga = random_search(fitness, 1, n_seg, rng, bounds=D_GENE_BOUNDS, decode=DChromosome.from_vector)
        tries = 1
        while not np.isfinite(ga.fitness) and tries < 1000:
            ga = random_search(fitness, 1, n_seg, rng, bounds=D_GENE_BOUNDS, decode=DChromosome.from_vector)
            tries += 1
    else:
        b = int(budget) if budget else cfg.population * (cfg.generations + 1)
        ga = random_search(fitness, b, n_seg, rng, bounds=D_GENE_BOUNDS, decode=DChromosome.from_vector)
    chroms = [c.locked(spec.locks) for c in ga.chromosomes]
    st, info = embed_d(cover, payload, chroms, key, nonce, rng, plan=plan)
    wall = __import__("time").perf_counter() - t0
    ok = True
    if verify:
        try:
            rec, _ = extract_d(st, key)
            ok = bool(np.array_equal(rec, payload))
        except ValueError:
            ok = False
    changed = (st != cover).reshape(-1)
    info.update({
        "variant": variant, "extraction_ok": ok, "payload_bits": int(payload.size),
        "embedded_bits": int(payload.size), "blind_extractable": True, "capacity_limited": False,
        "hill_distortion": float(rho[changed].sum()), "ga_fitness": ga.fitness,
        "ga_evaluations": ga.evaluations, "ga_generations": ga.generations_run,
        "ga_wall_time_s": ga.wall_time_s, "ga_converged_at": ga.converged_at,
        "embed_wall_s": wall, "chromosomes": [c.as_dict() for c in chroms],
        "gates": [GATE_RATIOS[c.gate] for c in chroms], "ga_history": ga.history,
    })
    return st, info
