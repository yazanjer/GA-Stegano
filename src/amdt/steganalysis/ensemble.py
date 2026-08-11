"""FLD ensemble classifier (Kodovsky, Fridrich & Holub, TIFS 2012).

The standard classifier for rich-model steganalysis, and the one the reviewer
named.  Implementation notes:

* Each base learner is a Fisher Linear Discriminant trained on a random
  ``d_sub``-dimensional feature subspace and a bootstrap sample of the training
  pairs.  Cover/stego pairs are bootstrapped **together**, never split, so a
  cover and its stego never straddle the bag/out-of-bag boundary.
* ``d_sub`` and the number of learners ``L`` are chosen automatically by the
  original out-of-bag search: grow ``L`` until the OOB error stops improving,
  then step ``d_sub`` in the direction that lowers OOB error.  The test set is
  never consulted -- OOB error is computed from training data only.
* Singular within-class scatter is handled by ridge regularisation on the
  diagonal, escalating until Cholesky succeeds (as in the reference MATLAB).

The output is a real-valued score (mean projection margin over learners), so
ROC/AUC is well defined rather than being computed from hard labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = ["FLDBase", "EnsembleClassifier"]


@dataclass
class FLDBase:
    subspace: np.ndarray          # feature indices
    w: np.ndarray                 # projection vector
    threshold: float

    def score(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.subspace] @ self.w - self.threshold


def _train_fld(Xc: np.ndarray, Xs: np.ndarray) -> Tuple[np.ndarray, float]:
    mu_c, mu_s = Xc.mean(0), Xs.mean(0)
    d = Xc.shape[1]
    Sw = np.cov(Xc, rowvar=False, bias=True) + np.cov(Xs, rowvar=False, bias=True)
    Sw = np.atleast_2d(Sw)
    diff = mu_s - mu_c
    reg = 1e-10 * np.trace(Sw) / max(d, 1)
    for _ in range(12):
        try:
            L = np.linalg.cholesky(Sw + reg * np.eye(d))
            w = np.linalg.solve(L.T, np.linalg.solve(L, diff))
            break
        except np.linalg.LinAlgError:
            reg = max(reg * 10.0, 1e-10)
    else:  # pragma: no cover
        w = diff
    thr = 0.5 * (w @ mu_c + w @ mu_s)
    return w, float(thr)


@dataclass
class EnsembleClassifier:
    """Kodovsky-Fridrich ensemble.

    Parameters
    ----------
    d_sub : subspace dimension; ``None`` triggers the OOB search.
    n_learners : ``None`` triggers automatic growth up to ``max_learners``.
    """

    d_sub: Optional[int] = None
    n_learners: Optional[int] = None
    max_learners: int = 200
    min_learners: int = 25
    oob_tolerance: float = 0.005
    random_state: int = 0
    learners: List[FLDBase] = field(default_factory=list)
    oob_error_: float = float("nan")
    d_sub_: int = 0

    # ------------------------------------------------------------------ #
    def _grow(self, Xc: np.ndarray, Xs: np.ndarray, d_sub: int,
              rng: np.random.Generator) -> Tuple[List[FLDBase], float]:
        n, d = Xc.shape
        learners: List[FLDBase] = []
        oob_sum = np.zeros((2, n))
        oob_cnt = np.zeros((2, n))
        prev = np.inf
        history: List[float] = []

        for i in range(self.max_learners):
            bag = rng.integers(0, n, size=n)
            oob = np.setdiff1d(np.arange(n), np.unique(bag), assume_unique=False)
            sub = rng.choice(d, size=min(d_sub, d), replace=False)
            w, thr = _train_fld(Xc[np.ix_(bag, sub)], Xs[np.ix_(bag, sub)])
            learners.append(FLDBase(sub, w, thr))

            if oob.size:
                oob_sum[0, oob] += Xc[np.ix_(oob, sub)] @ w - thr
                oob_sum[1, oob] += Xs[np.ix_(oob, sub)] @ w - thr
                oob_cnt[:, oob] += 1

            if (i + 1) >= self.min_learners and (i + 1) % 5 == 0:
                m = oob_cnt[0] > 0
                if not m.any():
                    continue
                fa = np.mean(oob_sum[0, m] > 0)
                md = np.mean(oob_sum[1, m] <= 0)
                err = 0.5 * (fa + md)
                history.append(err)
                if abs(prev - err) < self.oob_tolerance * max(err, 1e-9):
                    prev = err
                    break
                prev = err

        m = oob_cnt[0] > 0
        fa = np.mean(oob_sum[0, m] > 0) if m.any() else 0.5
        md = np.mean(oob_sum[1, m] <= 0) if m.any() else 0.5
        return learners, float(0.5 * (fa + md))

    # ------------------------------------------------------------------ #
    def fit(self, X_cover: np.ndarray, X_stego: np.ndarray) -> "EnsembleClassifier":
        """``X_cover[i]`` and ``X_stego[i]`` must come from the *same* cover."""
        Xc = np.asarray(X_cover, dtype=np.float64)
        Xs = np.asarray(X_stego, dtype=np.float64)
        if Xc.shape != Xs.shape:
            raise ValueError("cover/stego feature matrices must be paired and equal-sized")
        rng = np.random.default_rng(self.random_state)
        d = Xc.shape[1]

        if self.d_sub is not None:
            self.learners, self.oob_error_ = self._grow(Xc, Xs, int(self.d_sub), rng)
            self.d_sub_ = int(self.d_sub)
            return self

        # OOB search over d_sub (geometric ladder, as in the reference code)
        ladder = sorted({max(1, int(v)) for v in
                         (d * f for f in (0.02, 0.05, 0.1, 0.2, 0.4)) if v >= 1}
                        | {min(d, 200)})
        best = (np.inf, None, 0)
        for ds in ladder:
            ls, err = self._grow(Xc, Xs, min(ds, d), rng)
            if err < best[0]:
                best = (err, ls, min(ds, d))
        self.oob_error_, self.learners, self.d_sub_ = best[0], best[1], best[2]
        return self

    # ------------------------------------------------------------------ #
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        s = np.zeros(X.shape[0])
        for l in self.learners:
            s += l.score(X)
        return s / max(len(self.learners), 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0).astype(int)

    def as_dict(self) -> dict:
        return {
            "classifier": "FLD ensemble (Kodovsky-Fridrich)",
            "n_learners": len(self.learners),
            "d_sub": self.d_sub_,
            "oob_error": self.oob_error_,
        }
