"""Uniform method registry: every embedding scheme behind one signature.

    fn(cover, payload_bits_array, rng) -> (stego, info)

``info`` must contain ``payload_bits``, ``embedded_bits``, ``n_changes`` and
``blind_extractable``.  The driver checks ``embedded_bits`` against the request
and refuses to put a capacity-limited result in the same table cell as a
complete one without flagging it.
"""

from __future__ import annotations

from functools import partial
from typing import Callable, Dict, List, Tuple

import numpy as np

from ..baselines.adaptive import COST_FUNCTIONS, embed_with_cost
from ..baselines.classical import edge_adaptive_lsb, lsb_matching, lsb_replacement, pvd
from ..ga.optimizer import GAConfig
from ..stego.amdt import ABLATIONS, run_amdt

__all__ = ["MethodFn", "build_registry", "CLASSICAL_NAMES", "ADAPTIVE_NAMES"]

MethodFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], Tuple[np.ndarray, Dict]]

CLASSICAL_NAMES = ("LSB", "LSB-M", "EA-LSB", "GA-FT", "PVD")
ADAPTIVE_NAMES = tuple(COST_FUNCTIONS.keys())


def _wrap_classical(fn, name: str) -> MethodFn:
    def call(cover, payload, rng):
        st, info = fn(cover, payload, rng)
        info.setdefault("method", name)
        info["blind_extractable"] = True
        return st, info
    return call


def _wrap_adaptive(name: str) -> MethodFn:
    def call(cover, payload, rng):
        return embed_with_cost(cover, int(payload.size), name, rng)
    return call


def _wrap_amdt(variant: str, ga_cfg: GAConfig, key: bytes, verify: bool) -> MethodFn:
    def call(cover, payload, rng):
        res = run_amdt(cover, payload, key, ga_cfg, rng, variant=variant, verify=verify)
        info = {
            "method": f"AMDT-{variant}" if variant != "full" else "AMDT",
            "payload_bits": res.embed_result.payload_bits,
            "embedded_bits": res.embed_result.payload_bits,
            "n_changes": res.embed_result.n_changes,
            "capacity_limited": False,
            "blind_extractable": True,
            "extraction_ok": res.extraction_ok,
            "header_bits": res.embed_result.header_bits,
            "ga_evaluations": res.ga.evaluations if res.ga else None,
            "ga_generations": res.ga.generations_run if res.ga else None,
            "ga_wall_time_s": res.ga.wall_time_s if res.ga else None,
            "ga_converged_at": res.ga.converged_at if res.ga else None,
            "ga_history": res.ga.history if res.ga else None,
            "chromosomes": [c.as_dict() for c in res.chromosomes],
        }
        return res.stego, info
    return call


def build_registry(ga_cfg: GAConfig, key: bytes, verify: bool = True,
                   include: Tuple[str, ...] = ("proposed", "classical", "adaptive"),
                   ablations: Tuple[str, ...] = ()) -> Dict[str, MethodFn]:
    reg: Dict[str, MethodFn] = {}

    if "proposed" in include:
        reg["AMDT"] = _wrap_amdt("full", ga_cfg, key, verify)

    if "classical" in include:
        reg["LSB"] = _wrap_classical(lambda c, p, r: lsb_replacement(c, p, r), "LSB")
        reg["LSB-M"] = _wrap_classical(lsb_matching, "LSB-M")
        reg["EA-LSB"] = _wrap_classical(edge_adaptive_lsb, "EA-LSB")
        reg["PVD"] = _wrap_classical(pvd, "PVD")
        # GA-FT is the AMDT fixed_path ablation -- same code, one gene frozen,
        # so the comparison isolates traversal and nothing else.
        reg["GA-FT"] = _wrap_amdt("fixed_path", GAConfig(**{**ga_cfg.as_dict(),
                                                            "n_segments": 1}), key, verify)

    if "adaptive" in include:
        for n in ADAPTIVE_NAMES:
            reg[n] = _wrap_adaptive(n)

    for v in ablations:
        if v not in ABLATIONS:
            raise KeyError(f"unknown ablation variant {v!r}")
        reg[f"AMDT-{v}"] = _wrap_amdt(v, ga_cfg, key, verify)

    return reg
