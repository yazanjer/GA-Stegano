#!/usr/bin/env python
"""Kodovsky-Fridrich FLD ensemble on full-SRM features, GPU-accelerated.

    python revision/classify_v2.py --run /workspace/runs/stego

Protocol (unchanged from v1 except the feature set): 1,000 BOSSBase covers,
cover-wise 5-fold cross-validation with one fixed fold assignment shared by all
methods (a cover and its stego are always in the same fold); inside each
training fold the subspace dimension d_sub is chosen from a fixed ladder by the
out-of-bag (OOB) error and the number of base learners L grows until the OOB
error stabilises (Kodovsky, Fridrich & Holub, IEEE TIFS 2012); the test fold
is touched once.  Decisions are majority votes of the base learners; the vote
fraction is the score used for ROC/AUC and P_E.  Output: ``steganalysis_srm.csv``
(pooled test predictions of the 5 folds, 2,000 test decisions per set) and
``roc_srm.csv``.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: F401,E402
from amdt.steganalysis.metrics import evaluate_scores, roc_curve  # noqa: E402

DSUB = (250, 500, 750, 1000, 1500, 2000)


class GPUEnsemble:
    def __init__(self, dev, seed=0, l_max=300, step=10, tol=0.002):
        self.dev, self.g, self.l_max, self.step, self.tol = dev, torch.Generator(device="cpu").manual_seed(seed), l_max, step, tol

    def _learner(self, Xc, Xs, n, d, ds):
        b = torch.randint(0, n, (n,), generator=self.g)
        sub = torch.randperm(d, generator=self.g)[:ds]
        oob = torch.ones(n, dtype=torch.bool); oob[b] = False
        bd, sd = b.to(self.dev), sub.to(self.dev)
        c = Xc[bd][:, sd]; s = Xs[bd][:, sd]
        mc, ms = c.mean(0), s.mean(0)
        C = (c - mc).T @ (c - mc) / n + (s - ms).T @ (s - ms) / n
        reg = 1e-10 * torch.trace(C) / ds
        eye = torch.eye(ds, device=self.dev, dtype=C.dtype)
        for _ in range(12):
            L, info = torch.linalg.cholesky_ex(C + reg * eye)
            if int(info) == 0:
                break
            reg = max(float(reg) * 10, 1e-10)
        w = torch.cholesky_solve((ms - mc)[:, None], L)[:, 0]
        thr = 0.5 * (w @ mc + w @ ms)
        return sd, w, thr, oob.to(self.dev)

    def _grow(self, Xc, Xs, ds):
        n, d = Xc.shape
        learners, votes, cnt = [], torch.zeros(2, n, device=self.dev), torch.zeros(n, device=self.dev)
        prev, err = None, 0.5
        while len(learners) < self.l_max:
            for _ in range(self.step):
                sd, w, thr, oob = self._learner(Xc, Xs, n, d, ds)
                learners.append((sd, w, thr))
                votes[0] += oob * (Xc[:, sd] @ w > thr).float()
                votes[1] += oob * (Xs[:, sd] @ w > thr).float()
                cnt += oob.float()
            m = cnt > 0
            fa = ((votes[0][m] / cnt[m]) > 0.5).float().mean()
            md = ((votes[1][m] / cnt[m]) <= 0.5).float().mean()
            err = float(0.5 * (fa + md))
            if prev is not None and abs(prev - err) < self.tol:
                break
            prev = err
        return learners, err

    def fit(self, Xc, Xs):
        best = None
        for ds in DSUB:
            if ds > Xc.shape[1]:
                continue
            ls, err = self._grow(Xc, Xs, ds)
            if best is None or err < best[1]:
                best = (ls, err, ds)
        self.learners, self.oob_, self.dsub_ = best
        return self

    def score(self, X):
        v = torch.zeros(X.shape[0], device=self.dev)
        for sd, w, thr in self.learners:
            v += (X[:, sd] @ w > thr).float()
        return (v / len(self.learners)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--sets", default="")
    a = ap.parse_args()
    run = Path(a.run); fdir = run / "features"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xc_all = np.load(fdir / "cover.npy")
    n = Xc_all.shape[0]
    fold = np.random.default_rng(12345).permutation(n) % a.folds
    out_csv = run / "steganalysis_srm.csv"
    done = set()
    if out_csv.exists():
        import csv
        done = {r["set"] for r in csv.DictReader(open(out_csv))}
    rows_roc = []
    Xc_t = torch.from_numpy(Xc_all).to(dev)
    for f in sorted(fdir.glob("*.npy")):
        if f.stem == "cover" or f.stem in done or (a.sets and f.stem not in a.sets.split(",")):
            continue
        t0 = time.time()
        Xs_t = torch.from_numpy(np.load(f)).to(dev)
        y, sc, pe_fold, dsub, nl = [], [], [], [], []
        for k in range(a.folds):
            tr = torch.from_numpy(np.flatnonzero(fold != k)).to(dev)
            te = torch.from_numpy(np.flatnonzero(fold == k)).to(dev)
            ens = GPUEnsemble(dev, seed=k).fit(Xc_t[tr], Xs_t[tr])
            s = np.r_[ens.score(Xc_t[te]), ens.score(Xs_t[te])]
            yy = np.r_[np.zeros(len(te)), np.ones(len(te))]
            pe_fold.append(evaluate_scores(yy, s, 0.5).p_e); dsub.append(ens.dsub_); nl.append(len(ens.learners))
            y.append(yy); sc.append(s)
        y, sc = np.concatenate(y), np.concatenate(sc)
        m = evaluate_scores(y, sc, 0.5)
        # pairwise bootstrap CI on P_E (cover and its stego resampled together)
        rng = np.random.default_rng(0)
        h = y.size // 2
        yc, ys = sc[y == 0], sc[y == 1]
        boots = []
        for _ in range(2000):
            i = rng.integers(0, h, h)
            boots.append(evaluate_scores(np.r_[np.zeros(h), np.ones(h)], np.r_[yc[i], ys[i]], 0.5).p_e)
        method, _, rate = f.stem.rpartition("@")
        row = {"set": f.stem, "method": method.replace("_", ":"), "rate": float(rate), **m.as_dict(),
               "p_e_ci_low": float(np.quantile(boots, 0.025)), "p_e_ci_high": float(np.quantile(boots, 0.975)),
               "p_e_fold_mean": float(np.mean(pe_fold)), "p_e_fold_sd": float(np.std(pe_fold, ddof=1)),
               "d_sub": json.dumps(dsub), "n_learners": json.dumps(nl), "detector": "SRM(34671)+FLD-ensemble",
               "n_covers": n, "folds": a.folds}
        import csv
        new = not out_csv.exists()
        with open(out_csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row)); 
            if new: w.writeheader()
            w.writerow(row)
        fpr, tpr, _ = roc_curve(y, sc)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
        with open(run / "roc_srm.csv", "a") as fh:
            for i in keep:
                fh.write(f"{f.stem},{fpr[i]:.5f},{tpr[i]:.5f}\n")
        print(f"{f.stem}: P_E={m.p_e:.4f} [{row['p_e_ci_low']:.3f},{row['p_e_ci_high']:.3f}] acc={m.accuracy:.3f} "
              f"auc={m.auc:.3f} dsub={dsub} L={nl} {time.time()-t0:.0f}s", flush=True)
        del Xs_t


if __name__ == "__main__":
    main()
