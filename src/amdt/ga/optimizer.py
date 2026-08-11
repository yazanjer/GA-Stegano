"""Genetic algorithm for joint embedding-parameter optimisation.

Operators follow the manuscript ("population 25, tournament size 3, two-point
crossover p=0.8, gene mutation p=0.1"), with three additions that the reviewers'
questions require:

* **Convergence logging** (Reviewer 1 comment 6): best/mean/std fitness, unique
  genotype count and cumulative evaluation count are recorded per generation and
  written to CSV, so the convergence plot is a data artifact rather than a
  screenshot of MATLAB's ``gaplotbestf``.
* **Determinism**: the whole run is driven by one ``np.random.Generator``
  derived from the root seed, so a given (seed, image, payload) triple always
  yields the same chromosome.
* **Elitism + early stopping**: the best individual always survives, and the run
  stops after ``patience`` generations without improvement.  Both are reported
  in the runtime table -- a fixed 1000-generation budget (the MATLAB default)
  wastes ~90 % of the wall-clock on this problem, which matters for comment 6.

Search space: :math:`|\\Theta| = 16 \\cdot 512 \\cdot 512 \\cdot 15 \\cdot 2^3
\\cdot 4 \\cdot 2 = 4.02\\times10^9` per segment, so exhaustive search is not an
option and the GA's value is measurable against random search (see
``run_experiments.py --study ga_vs_random``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..stego.codec import GENE_BOUNDS, GENE_NAMES, Chromosome

__all__ = ["GAConfig", "GAHistory", "GAResult", "optimise", "random_search", "search_space_size"]


@dataclass
class GAConfig:
    population: int = 25
    generations: int = 100
    tournament_size: int = 3
    crossover_prob: float = 0.8
    mutation_prob: float = 0.1
    elitism: int = 1
    patience: int = 25              # generations without improvement -> stop
    n_segments: int = 1

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


@dataclass
class GAHistory:
    generation: List[int] = field(default_factory=list)
    best: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)
    unique: List[int] = field(default_factory=list)
    evaluations: List[int] = field(default_factory=list)
    elapsed_s: List[float] = field(default_factory=list)

    def to_records(self) -> List[Dict[str, float]]:
        return [
            {
                "generation": g, "best_fitness": b, "mean_fitness": m,
                "std_fitness": s, "unique_genotypes": u, "cum_evaluations": e,
                "elapsed_s": t,
            }
            for g, b, m, s, u, e, t in zip(
                self.generation, self.best, self.mean, self.std,
                self.unique, self.evaluations, self.elapsed_s
            )
        ]


@dataclass
class GAResult:
    chromosomes: List[Chromosome]
    fitness: float
    history: GAHistory
    evaluations: int
    generations_run: int
    wall_time_s: float
    converged_at: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "fitness": self.fitness,
            "evaluations": self.evaluations,
            "generations_run": self.generations_run,
            "wall_time_s": self.wall_time_s,
            "converged_at": self.converged_at,
            "chromosomes": [c.as_dict() for c in self.chromosomes],
        }


def search_space_size(n_segments: int = 1) -> float:
    per = 1.0
    for lo, hi in GENE_BOUNDS:
        per *= (hi - lo + 1)
    return per ** n_segments


# --------------------------------------------------------------------------- #
# genotype helpers -- a genome is the concatenation of n_segments gene vectors
# --------------------------------------------------------------------------- #
_LO = np.array([lo for lo, _ in GENE_BOUNDS], dtype=np.int64)
_HI = np.array([hi for _, hi in GENE_BOUNDS], dtype=np.int64)
_NGENE = len(GENE_BOUNDS)


def _random_genome(rng: np.random.Generator, n_seg: int) -> np.ndarray:
    return np.concatenate([rng.integers(_LO, _HI + 1) for _ in range(n_seg)])


def _clip(genome: np.ndarray, n_seg: int) -> np.ndarray:
    g = genome.reshape(n_seg, _NGENE)
    return np.clip(g, _LO, _HI).reshape(-1)


def genome_to_chroms(genome: np.ndarray, n_seg: int) -> List[Chromosome]:
    g = np.asarray(genome, dtype=np.int64).reshape(n_seg, _NGENE)
    return [Chromosome.from_vector(row) for row in g]


def _tournament(rng: np.random.Generator, fit: np.ndarray, k: int) -> int:
    cand = rng.integers(0, fit.size, size=k)
    return int(cand[np.argmax(fit[cand])])


def _two_point_crossover(rng: np.random.Generator, a: np.ndarray, b: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray]:
    n = a.size
    if n < 2:
        return a.copy(), b.copy()
    i, j = sorted(rng.choice(np.arange(1, n), size=2, replace=False)) if n > 2 else (1, n)
    c1, c2 = a.copy(), b.copy()
    c1[i:j], c2[i:j] = b[i:j], a[i:j]
    return c1, c2


def _mutate(rng: np.random.Generator, genome: np.ndarray, p: float, n_seg: int) -> np.ndarray:
    g = genome.reshape(n_seg, _NGENE).copy()
    hit = rng.random(g.shape) < p
    if hit.any():
        fresh = rng.integers(np.broadcast_to(_LO, g.shape),
                             np.broadcast_to(_HI + 1, g.shape))
        g[hit] = fresh[hit]
    return g.reshape(-1)


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def optimise(
    fitness_fn: Callable[[List[Chromosome]], float],
    cfg: GAConfig,
    rng: np.random.Generator,
    seed_genomes: Optional[Sequence[np.ndarray]] = None,
) -> GAResult:
    """Maximise ``fitness_fn``.

    ``fitness_fn`` receives a list of ``n_segments`` chromosomes and returns a
    scalar to be **maximised** (the pipeline uses ``-MSE``; use ``PSNR`` for a
    literal match with the MATLAB objective -- they induce the same ordering).
    Infeasible genomes should return ``-inf``.
    """
    t0 = time.perf_counter()
    n_seg = cfg.n_segments
    pop = np.stack([_random_genome(rng, n_seg) for _ in range(cfg.population)])
    if seed_genomes:
        for i, g in enumerate(seed_genomes[: cfg.population]):
            pop[i] = _clip(np.asarray(g, dtype=np.int64), n_seg)

    cache: Dict[bytes, float] = {}
    evals = 0

    def evaluate(genome: np.ndarray) -> float:
        nonlocal evals
        key = genome.tobytes()
        if key in cache:
            return cache[key]
        val = float(fitness_fn(genome_to_chroms(genome, n_seg)))
        cache[key] = val
        evals += 1
        return val

    fit = np.array([evaluate(g) for g in pop])
    hist = GAHistory()
    best_idx = int(np.argmax(fit))
    best_genome, best_fit = pop[best_idx].copy(), float(fit[best_idx])
    converged_at, stale = 0, 0

    def log(gen: int) -> None:
        finite = fit[np.isfinite(fit)]
        hist.generation.append(gen)
        hist.best.append(best_fit)
        hist.mean.append(float(finite.mean()) if finite.size else float("nan"))
        hist.std.append(float(finite.std(ddof=0)) if finite.size else float("nan"))
        hist.unique.append(len({g.tobytes() for g in pop}))
        hist.evaluations.append(evals)
        hist.elapsed_s.append(time.perf_counter() - t0)

    log(0)
    gen = 0
    for gen in range(1, cfg.generations + 1):
        order = np.argsort(-fit)
        new = [pop[i].copy() for i in order[: cfg.elitism]]
        while len(new) < cfg.population:
            p1 = pop[_tournament(rng, fit, cfg.tournament_size)]
            p2 = pop[_tournament(rng, fit, cfg.tournament_size)]
            if rng.random() < cfg.crossover_prob:
                c1, c2 = _two_point_crossover(rng, p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            new.append(_clip(_mutate(rng, c1, cfg.mutation_prob, n_seg), n_seg))
            if len(new) < cfg.population:
                new.append(_clip(_mutate(rng, c2, cfg.mutation_prob, n_seg), n_seg))

        pop = np.stack(new)
        fit = np.array([evaluate(g) for g in pop])

        i = int(np.argmax(fit))
        if fit[i] > best_fit:
            best_fit, best_genome = float(fit[i]), pop[i].copy()
            converged_at, stale = gen, 0
        else:
            stale += 1
        log(gen)
        if cfg.patience and stale >= cfg.patience:
            break

    return GAResult(
        chromosomes=genome_to_chroms(best_genome, n_seg),
        fitness=best_fit,
        history=hist,
        evaluations=evals,
        generations_run=gen,
        wall_time_s=time.perf_counter() - t0,
        converged_at=converged_at,
    )


def random_search(
    fitness_fn: Callable[[List[Chromosome]], float],
    budget: int,
    n_segments: int,
    rng: np.random.Generator,
) -> GAResult:
    """Matched-budget random search -- the control the GA must beat.

    Reviewers routinely ask whether the GA is doing anything a random draw of
    the same size would not; this makes the answer a measured number.
    """
    t0 = time.perf_counter()
    hist = GAHistory()
    best_fit, best_genome = -np.inf, None
    for i in range(1, budget + 1):
        g = _random_genome(rng, n_segments)
        v = float(fitness_fn(genome_to_chroms(g, n_segments)))
        if v > best_fit:
            best_fit, best_genome = v, g
        if i % max(1, budget // 100) == 0 or i == budget:
            hist.generation.append(i)
            hist.best.append(best_fit)
            hist.mean.append(float("nan"))
            hist.std.append(float("nan"))
            hist.unique.append(i)
            hist.evaluations.append(i)
            hist.elapsed_s.append(time.perf_counter() - t0)
    return GAResult(
        chromosomes=genome_to_chroms(best_genome, n_segments),
        fitness=best_fit,
        history=hist,
        evaluations=budget,
        generations_run=budget,
        wall_time_s=time.perf_counter() - t0,
        converged_at=budget,
    )
