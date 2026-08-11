"""Detection metrics for steganalysis.

Reviewer 1, comment 3 asks for Accuracy, Precision, Recall, F1 and ROC/AUC.
Two further numbers are added because the steganalysis literature reports them
and their absence is conspicuous:

``P_E``
    The minimal total probability of error under equal priors,
    :math:`P_E = \\min_{\\tau} \\tfrac12 (P_{FA}(\\tau) + P_{MD}(\\tau))`.
    This is *the* standard figure of merit in the field; accuracy at a fixed
    threshold is not.
``MD5``
    Missed-detection rate at 5 % false alarms -- the operating point a real
    warden would use.

All metrics are computed from the *score* vector, so they are threshold-free
except where a threshold is stated.  Chance level is 0.5 accuracy / 0.5 P_E:
for a steganographic method, **lower detector accuracy is better**, and the
plots are labelled to prevent the direction being misread.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["DetectionMetrics", "roc_curve", "auc", "evaluate_scores",
           "bootstrap_ci"]


def roc_curve(y_true: np.ndarray, score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds), ties handled correctly."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=np.float64)
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.where(np.diff(s))[0]
    idx = np.r_[distinct, y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = 1 + idx - tps
    p, n = max(int(y.sum()), 1), max(int((1 - y).sum()), 1)
    return np.r_[0, fps / n], np.r_[0, tps / p], np.r_[np.inf, s[idx]]


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


@dataclass
class DetectionMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    p_e: float
    p_e_threshold: float
    md_at_fa5: float
    n_cover: int
    n_stego: int
    threshold: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_scores(y_true: Sequence[int], score: Sequence[float],
                    threshold: float = 0.0) -> DetectionMetrics:
    """``y_true``: 1 = stego, 0 = cover. ``score``: higher = more stego-like."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=np.float64)
    pred = (s > threshold).astype(int)

    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))

    acc = (tp + tn) / max(y.size, 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    fpr, tpr, thr = roc_curve(y, s)
    a = auc(fpr, tpr)
    pe_curve = 0.5 * (fpr + (1.0 - tpr))
    k = int(np.argmin(pe_curve))
    md5 = float(1.0 - np.interp(0.05, fpr, tpr))

    return DetectionMetrics(
        accuracy=float(acc), precision=float(prec), recall=float(rec), f1=float(f1),
        auc=float(a), p_e=float(pe_curve[k]), p_e_threshold=float(thr[k]),
        md_at_fa5=md5, n_cover=int((y == 0).sum()), n_stego=int((y == 1).sum()),
        threshold=float(threshold),
    )


def bootstrap_ci(y_true: Sequence[int], score: Sequence[float], metric: str = "accuracy",
                 n_boot: int = 2000, alpha: float = 0.05,
                 rng: Optional[np.random.Generator] = None) -> Tuple[float, float, float]:
    """Percentile bootstrap CI for any :class:`DetectionMetrics` field.

    Resampling is done **pairwise** (a cover and its stego are resampled
    together) when the arrays are interleaved cover/stego of equal size, which
    is how the pipeline builds them; this respects the dependence structure and
    avoids the too-narrow intervals that i.i.d. resampling would give.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype=np.float64)
    n = y.size
    paired = (n % 2 == 0) and np.array_equal(y[: n // 2], np.zeros(n // 2, dtype=int)) \
        and np.array_equal(y[n // 2:], np.ones(n // 2, dtype=int))

    vals = []
    for _ in range(n_boot):
        if paired:
            h = n // 2
            idx = rng.integers(0, h, size=h)
            bi = np.r_[idx, idx + h]
        else:
            bi = rng.integers(0, n, size=n)
        if len(np.unique(y[bi])) < 2:
            continue
        vals.append(getattr(evaluate_scores(y[bi], s[bi]), metric))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    v = np.asarray(vals)
    point = getattr(evaluate_scores(y, s), metric)
    return float(point), float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))
