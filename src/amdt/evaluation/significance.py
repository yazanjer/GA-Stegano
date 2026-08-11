r"""Statistical validation of the reported improvements.

Reviewer 1, comment 4: "The reported improvements are presented only through
average metrics and boxplots.  Please include statistical significance testing,
confidence intervals, standard deviations, Wilcoxon or paired t-test
comparisons."

Design decisions that matter for the rebuttal:

**Pairing.**  Every comparison is *paired on the cover image* (and on the seed),
because the between-image variance of PSNR is an order of magnitude larger than
the between-method difference.  An unpaired test on this data has almost no
power and would understate a real effect.

**Which test.**  Both the paired t-test and the Wilcoxon signed-rank test are
reported for every pair, together with a Shapiro-Wilk normality check on the
differences.  When Shapiro rejects, the Wilcoxon p-value is the one quoted in
the manuscript; both are printed so the reader can see they agree.

**Multiplicity.**  Comparing the proposed method against k baselines is k
simultaneous tests.  Holm-Bonferroni correction is applied within each metric
family and the adjusted p-values are what appear in the table.  Uncorrected
p-values are kept in the CSV for transparency.

**Effect size.**  A p-value is not an effect.  Cohen's :math:`d_z` (paired) and
Cliff's :math:`\delta` (non-parametric) are reported alongside, plus a
bootstrap CI on the mean difference.  The pipeline explicitly flags any case
where the mean improvement is smaller than the seed-to-seed standard deviation,
because that is a difference no reviewer should accept regardless of p.

**Power.**  With n images x s seeds paired observations, the achieved power for
the observed effect is estimated by simulation and reported.  This pre-empts
"is n=29 enough?", which is the obvious follow-up question given the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["PairedComparison", "compare_paired", "holm_bonferroni",
           "bootstrap_mean_ci", "cohens_dz", "cliffs_delta", "descriptive",
           "compare_many", "power_estimate"]


# --------------------------------------------------------------------------- #
def descriptive(x: Sequence[float], alpha: float = 0.05) -> Dict[str, float]:
    """Mean, SD, SEM, median, IQR and a t-based CI for the mean."""
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = a.size
    if n == 0:
        return {k: float("nan") for k in
                ("n", "mean", "std", "sem", "ci_low", "ci_high", "median", "iqr")}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    try:
        from scipy import stats
        t = float(stats.t.ppf(1 - alpha / 2, df=max(n - 1, 1)))
    except ImportError:
        t = 1.96
    q1, q3 = np.percentile(a, [25, 75])
    return {"n": float(n), "mean": mean, "std": std, "sem": float(sem),
            "ci_low": mean - t * sem, "ci_high": mean + t * sem,
            "median": float(np.median(a)), "iqr": float(q3 - q1)}


def cohens_dz(diff: np.ndarray) -> float:
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    s = d.std(ddof=1)
    return float(d.mean() / s) if s > 0 else float("inf" if d.mean() else 0.0)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Non-parametric effect size in [-1, 1]; |d| >= .474 is 'large'."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    return float((gt - lt) / (x.size * y.size))


def bootstrap_mean_ci(diff: Sequence[float], n_boot: int = 10000, alpha: float = 0.05,
                      rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    rng = rng or np.random.default_rng(0)
    d = np.asarray(diff, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


# --------------------------------------------------------------------------- #
@dataclass
class PairedComparison:
    metric: str
    method_a: str
    method_b: str
    n_pairs: int
    mean_a: float
    std_a: float
    mean_b: float
    std_b: float
    mean_diff: float
    diff_ci_low: float
    diff_ci_high: float
    t_statistic: float
    t_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    shapiro_p_value: float
    normality_ok: bool
    cohens_dz: float
    cliffs_delta: float
    preferred_test: str
    preferred_p: float
    p_adjusted: float = float("nan")
    significant: bool = False
    effect_below_noise: bool = False

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


def compare_paired(a: Sequence[float], b: Sequence[float], metric: str,
                   name_a: str = "proposed", name_b: str = "baseline",
                   alpha: float = 0.05,
                   rng: Optional[np.random.Generator] = None) -> PairedComparison:
    """Paired t-test + Wilcoxon + effect sizes + bootstrap CI on the difference."""
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired comparison needs equal-length inputs, got {x.shape} vs {y.shape}")
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    d = x - y
    n = d.size

    t_stat = t_p = w_stat = w_p = sh_p = float("nan")
    if n >= 2:
        try:
            from scipy import stats
            t_stat, t_p = stats.ttest_rel(x, y)
            if np.any(d != 0):
                w = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
                w_stat, w_p = float(w.statistic), float(w.pvalue)
            sh_p = float(stats.shapiro(d).pvalue) if 3 <= n <= 5000 else float("nan")
        except ImportError:  # pragma: no cover
            pass

    normal = bool(np.isnan(sh_p) or sh_p > alpha)
    preferred = "paired t-test" if normal else "Wilcoxon signed-rank"
    preferred_p = t_p if normal else w_p

    da = descriptive(x)
    db = descriptive(y)
    _, lo, hi = bootstrap_mean_ci(d, rng=rng)

    return PairedComparison(
        metric=metric, method_a=name_a, method_b=name_b, n_pairs=int(n),
        mean_a=da["mean"], std_a=da["std"], mean_b=db["mean"], std_b=db["std"],
        mean_diff=float(d.mean()) if n else float("nan"),
        diff_ci_low=lo, diff_ci_high=hi,
        t_statistic=float(t_stat), t_p_value=float(t_p),
        wilcoxon_statistic=float(w_stat), wilcoxon_p_value=float(w_p),
        shapiro_p_value=float(sh_p), normality_ok=normal,
        cohens_dz=cohens_dz(d), cliffs_delta=cliffs_delta(x, y),
        preferred_test=preferred, preferred_p=float(preferred_p),
        effect_below_noise=bool(abs(d.mean()) < max(da["std"], db["std"]))
        if n else False,
    )


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Step-down Holm correction. Returns (adjusted p, reject flags)."""
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, i in enumerate(order):
        val = (n - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj, adj <= alpha


def compare_many(proposed: Sequence[float], baselines: Dict[str, Sequence[float]],
                 metric: str, name: str = "AMDT", alpha: float = 0.05,
                 rng: Optional[np.random.Generator] = None) -> List[PairedComparison]:
    """Proposed vs. every baseline, with Holm correction across the family."""
    comps = [compare_paired(proposed, vals, metric, name, b, alpha, rng)
             for b, vals in baselines.items()]
    adj, rej = holm_bonferroni([c.preferred_p for c in comps], alpha)
    for c, a, r in zip(comps, adj, rej):
        c.p_adjusted = float(a)
        c.significant = bool(r)
    return comps


def power_estimate(effect_dz: float, n: int, alpha: float = 0.05,
                   n_sim: int = 20000, rng: Optional[np.random.Generator] = None) -> float:
    """Simulated power of the paired t-test for the observed effect size."""
    if not np.isfinite(effect_dz) or n < 2:
        return float("nan")
    rng = rng or np.random.default_rng(0)
    try:
        from scipy import stats
        crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        samples = rng.standard_normal((n_sim, n)) + effect_dz
        t = samples.mean(axis=1) / (samples.std(axis=1, ddof=1) / np.sqrt(n))
        return float(np.mean(np.abs(t) > crit))
    except ImportError:  # pragma: no cover
        return float("nan")
