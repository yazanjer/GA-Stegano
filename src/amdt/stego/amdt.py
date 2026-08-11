"""The AMDT method: GA-optimised multi-directional traversal embedding.

Wraps the codec and the GA into one callable and exposes the ablation switches
the reviewers asked to see isolated (Reviewer 1 comments 1, 2 and 4).

Ablation variants -- each removes exactly one component, so every row of the
ablation table attributes an effect to a single cause:

    ============= ================================================
    variant        what is disabled
    ============= ================================================
    ``full``       nothing (all genes free)
    ``no_T1T2``    alpha = beta = 0 (no complement / reverse)
    ``no_T3``      sigma = 0        (no keyed block scrambling)
    ``no_T4``      delta = 0        (no keyed diffusion)
    ``no_decomp``  T1-T4 all off    (raw payload bits embedded)
    ``fixed_path`` direction = 0    (raster only; = GA-FT baseline)
    ``fixed_plane`` mask = 1        (LSB plane only)
    ``no_ga``      random genes, no search (matched-budget control)
    ``single_seg`` n_segments = 1
    ============= ================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..evaluation.metrics import QualityReport, evaluate_pair, mse
from ..ga.optimizer import GAConfig, GAResult, genome_to_chroms, optimise, random_search
from .codec import Chromosome, EmbedResult, embed, extract, max_payload_bits, reserved_rows_for

__all__ = ["AblationSpec", "ABLATIONS", "AMDTResult", "run_amdt", "capacity_bits"]


@dataclass(frozen=True)
class AblationSpec:
    """Gene locks that define an ablation variant."""

    name: str
    locks: Dict[str, int] = field(default_factory=dict)
    force_segments: Optional[int] = None
    use_ga: bool = True

    def apply(self, chroms: Sequence[Chromosome]) -> List[Chromosome]:
        if not self.locks:
            return [c.clipped() for c in chroms]
        out = []
        for c in chroms:
            d = c.as_dict()
            d.update(self.locks)
            out.append(Chromosome(**d).clipped())
        return out


#: The security constraint of Eq. (2) in the manuscript, made operational.
#:
#: Minimising MSE alone always switches the security layers **off**, because T3
#: and T4 randomise the payload and a randomised payload agrees with the cover's
#: LSB plane slightly less often than a structured one.  The manuscript states
#: the problem as "minimise distortion *subject to* S(.) >= S_min"; the
#: constraint is enforced here by locking ``sigma = delta = 1`` in the deployed
#: method, and it is *unlocked* only in the ablation rows that exist to measure
#: what those layers buy.  The ``unconstrained`` variant shows what the GA does
#: when the constraint is dropped -- it is reported to make the effect explicit
#: rather than hidden in a default.
SECURITY_LOCKS = {"sigma": 1, "delta": 1}

ABLATIONS: Dict[str, AblationSpec] = {
    "full":          AblationSpec("full", dict(SECURITY_LOCKS)),
    "no_T1T2":       AblationSpec("no_T1T2", {**SECURITY_LOCKS, "alpha": 0, "beta": 0}),
    "no_T3":         AblationSpec("no_T3", {"sigma": 0, "delta": 1}),
    "no_T4":         AblationSpec("no_T4", {"sigma": 1, "delta": 0}),
    "no_decomp":     AblationSpec("no_decomp",
                                  {"alpha": 0, "beta": 0, "sigma": 0, "delta": 0}),
    "fixed_path":    AblationSpec("fixed_path",
                                  {**SECURITY_LOCKS, "direction": 0, "x_off": 0, "y_off": 0}),
    "fixed_plane":   AblationSpec("fixed_plane",
                                  {**SECURITY_LOCKS, "mask": 1, "bp_dir": 0}),
    "single_seg":    AblationSpec("single_seg", dict(SECURITY_LOCKS), force_segments=1),
    "no_ga":         AblationSpec("no_ga", dict(SECURITY_LOCKS), use_ga=False),
    "unconstrained": AblationSpec("unconstrained"),
}


def capacity_bits(shape: Tuple[int, int], n_segments: int, mask: int = 15) -> int:
    """Maximum payload for a given segment count and uniform bit-plane mask."""
    chroms = [Chromosome(mask=mask) for _ in range(n_segments)]
    return max_payload_bits(shape, chroms)


@dataclass
class AMDTResult:
    stego: np.ndarray
    chromosomes: List[Chromosome]
    quality: QualityReport
    ga: Optional[GAResult]
    embed_result: EmbedResult
    variant: str
    extraction_ok: bool

    def as_dict(self) -> Dict[str, object]:
        d = {"variant": self.variant, "extraction_ok": self.extraction_ok}
        d.update(self.quality.as_dict())
        if self.ga is not None:
            d.update({
                "ga_fitness": self.ga.fitness,
                "ga_evaluations": self.ga.evaluations,
                "ga_generations": self.ga.generations_run,
                "ga_wall_time_s": self.ga.wall_time_s,
                "ga_converged_at": self.ga.converged_at,
            })
        d["chromosomes"] = [c.as_dict() for c in self.chromosomes]
        return d


def run_amdt(
    cover: np.ndarray,
    payload: np.ndarray,
    key: bytes,
    ga_cfg: GAConfig,
    rng: np.random.Generator,
    variant: str = "full",
    verify: bool = True,
) -> AMDTResult:
    """Optimise embedding parameters for one (cover, payload) pair and embed.

    The fitness is ``-MSE`` restricted to *feasible* genomes; infeasible ones
    (payload larger than the chosen mask can carry) get ``-inf`` so they are
    never selected but the population is not artificially truncated.
    """
    spec = ABLATIONS[variant]
    cfg = GAConfig(**{**ga_cfg.as_dict(),
                      "n_segments": spec.force_segments or ga_cfg.n_segments})
    cover = np.asarray(cover, dtype=np.uint8)
    payload = np.asarray(payload, dtype=np.uint8).reshape(-1)

    def fitness(chroms: List[Chromosome]) -> float:
        chroms = spec.apply(chroms)
        try:
            res = embed(cover, payload, chroms, key)
        except ValueError:
            return -np.inf                      # infeasible capacity
        return -mse(cover, res.stego)

    ga_res: Optional[GAResult]
    if spec.use_ga:
        ga_res = optimise(fitness, cfg, rng)
    else:
        # matched-budget random control: same number of fitness evaluations
        budget = cfg.population * (cfg.generations + 1)
        ga_res = random_search(fitness, budget, cfg.n_segments, rng)

    chroms = spec.apply(ga_res.chromosomes)
    result = embed(cover, payload, chroms, key)
    quality = evaluate_pair(cover, result.stego, result.payload_bits, result.n_changes)

    ok = True
    if verify:
        try:
            recovered, _ = extract(result.stego, key)
            ok = bool(np.array_equal(recovered, payload))
        except ValueError:
            ok = False

    return AMDTResult(
        stego=result.stego,
        chromosomes=chroms,
        quality=quality,
        ga=ga_res,
        embed_result=result,
        variant=variant,
        extraction_ok=ok,
    )
