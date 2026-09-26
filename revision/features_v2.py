#!/usr/bin/env python
"""Full 34,671-D SRM features for the covers and every stego set of a run.

    python revision/features_v2.py --run /workspace/runs/stego --workers 30 [--n 1000]

Writes ``features/<set>.npy`` (float32, rows in the cover order of
``features/covers.json``).  Uses sealwatch's SRM with the bit-identical fast
co-occurrence counter of ``amdt.steganalysis.srm_full``.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cover_sets, load  # noqa: E402
from amdt.steganalysis.srm_full import srm_full  # noqa: E402


def _one(path):
    from PIL import Image
    x = np.array(Image.open(path)) if str(path).endswith(".png") else load(Path(path))
    return srm_full(x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--sets", default="")
    a = ap.parse_args()
    run = Path(a.run); fdir = run / "features"; fdir.mkdir(exist_ok=True)
    covers = cover_sets()["stego"][: a.n]
    ids = [p.stem for p in covers]
    (fdir / "covers.json").write_text(json.dumps(ids))
    jobs = [("cover", [str(p) for p in covers])]
    for d in sorted((run / "stego").iterdir()):
        if a.sets and d.name not in a.sets.split(","):
            continue
        files = [d / f"{i}.png" for i in ids]
        if all(f.exists() for f in files):
            jobs.append((d.name, [str(f) for f in files]))
        else:
            print("skip incomplete set", d.name, flush=True)
    with mp.get_context("fork").Pool(a.workers) as pool:
        for name, files in jobs:
            out = fdir / f"{name}.npy"
            if out.exists():
                continue
            t0 = time.time()
            X = np.stack(pool.map(_one, files, chunksize=4))
            np.save(out, X)
            print(f"{name}: {X.shape} in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
