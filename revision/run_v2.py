#!/usr/bin/env python
"""Revision-2 canonical experiment driver.

One seeded run per (method, rate, cover, seed) -> one row of a raw per-image
CSV.  Every table and figure of the revised manuscript is computed from these
CSVs by ``revision/make_tables.py``; nothing is copied by hand.

    python revision/run_v2.py quality  --out runs/quality  --workers 30
    python revision/run_v2.py ablation --out runs/ablation --workers 30
    python revision/run_v2.py stego    --out runs/stego    --workers 30 --n-covers 1000
    python revision/run_v2.py srnet    --out runs/stego    --workers 30      # extra covers @0.4
    python revision/run_v2.py runtime  --out runs/runtime  --workers 1

Runs are resumable: finished (method, rate, cover, seed) keys are skipped.
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import multiprocessing as mp
import platform
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KEY, cover_sets, load, method_rng, payload_bits, sha256  # noqa: E402

from amdt.baselines import classical as C  # noqa: E402
from amdt.baselines import reference as REF  # noqa: E402
from amdt.evaluation.metrics import evaluate_pair  # noqa: E402
from amdt.ga.optimizer import GAConfig  # noqa: E402
from amdt.stego.amdt import run_amdt  # noqa: E402
from amdt.stego.amdt_d import detect_cost, extract_d, run_amdt_d  # noqa: E402
from amdt.stego.codec import extract  # noqa: E402

GA = GAConfig(population=25, generations=100, tournament_size=3, crossover_prob=0.8,
              mutation_prob=0.1, elitism=1, patience=25, n_segments=4)
RATES = (0.05, 0.1, 0.2, 0.4)
STC = [f"{c}-STC" for c in REF.COSTS]
SIM = [f"{c}-SIM" for c in ("HILL", "S-UNIWARD", "WOW", "MiPOD")]
MAIN = ["AMDT", "AMDT-D", "GA-FT", "LSB", "LSB-M", "EA-LSB", "PVD", "FM-PSO-LSB"] + STC + SIM
AMDT_ABL = ["AMDT:no_T1T2", "AMDT:no_T3", "AMDT:no_T4", "AMDT:no_decomp", "AMDT:fixed_path",
            "AMDT:fixed_plane", "AMDT:single_seg", "AMDT:no_ga", "AMDT:no_search", "AMDT:unconstrained"]
D_ABL = ["AMDT-D:mse_fitness", "AMDT-D:no_gate", "AMDT-D:fixed_path", "AMDT-D:single_seg",
         "AMDT-D:no_ga", "AMDT-D:no_search"]
COST_RANK = {"AMDT": 10, "AMDT:": 9, "FM-PSO-LSB": 6, "AMDT-D": 5, "GA-FT": 4}
FIELDS = ["method", "rate", "cover", "seed", "psnr", "ssim", "mse", "correlation", "payload_bits",
          "n_changes", "embedding_efficiency", "hill_distortion", "extraction_ok", "blind_extractable",
          "wall_s", "cpu_s", "ga_evaluations", "ga_generations", "ga_converged_at", "extra"]


@lru_cache(maxsize=4)
def _cover(path: str):
    x = load(Path(path))
    return x, detect_cost(x)


def _lsb_extract(st, n):
    return st.reshape(-1)[:n] & 1


def _ealsb_extract(st, n):
    mag = C._sobel_magnitude(st & np.uint8(0xFE)).reshape(-1)
    return st.reshape(-1)[np.argsort(-mag, kind="stable")[:n]] & 1


def embed_one(method: str, x: np.ndarray, pay: np.ndarray, rng_tag, verify=True):
    """Returns (stego, info, extraction_ok or None)."""
    cid, rate, seed = rng_tag
    base, _, var = method.partition(":")
    rng = method_rng(base, cid, rate, seed)
    ok = None
    if base == "AMDT":
        v = var or "full"
        budget = None
        if v == "no_ga" and os.environ.get("NO_GA_BUDGET"):
            budget = int(os.environ["NO_GA_BUDGET"])     # runtime study: time the random search alone
        elif v == "no_ga":
            ref = run_amdt(x, pay, KEY, GA, method_rng("AMDT", cid, rate, seed), "full", verify=False)
            budget = ref.ga.evaluations
        r = run_amdt(x, pay, KEY, GA, rng, v, verify=verify, budget=budget)
        info = {"n_changes": r.embed_result.n_changes, "blind_extractable": True,
                "ga_evaluations": r.ga.evaluations, "ga_generations": r.ga.generations_run,
                "ga_converged_at": r.ga.converged_at, "matched_budget": budget,
                "chromosomes": [c.as_dict() for c in r.chromosomes],
                "hist": [round(b, 6) for b in r.ga.history.best] if rate == 0.1 and not var else None}
        return r.stego, info, (r.extraction_ok if verify else None)
    if base == "AMDT-D":
        v = var or "full"
        budget = None
        if v == "no_ga":
            _, ri = run_amdt_d(x, pay, KEY, GA, method_rng("AMDT-D", cid, rate, seed), "full", verify=False)
            budget = ri["ga_evaluations"]
        st, info = run_amdt_d(x, pay, KEY, GA, rng, v, verify=verify, budget=budget)
        info["matched_budget"] = budget
        info["hist"] = [round(b, 3) for b in info.pop("ga_history").best] if rate == 0.1 and not var else None
        return st, info, (info["extraction_ok"] if verify else None)
    if base == "GA-FT":
        cfg = GAConfig(**{**GA.as_dict(), "n_segments": 1})
        r = run_amdt(x, pay, KEY, cfg, rng, "fixed_path", verify=verify)
        info = {"n_changes": r.embed_result.n_changes, "blind_extractable": True,
                "ga_evaluations": r.ga.evaluations, "ga_generations": r.ga.generations_run,
                "ga_converged_at": r.ga.converged_at,
                "hist": [round(b, 6) for b in r.ga.history.best] if rate == 0.1 else None}
        return r.stego, info, (r.extraction_ok if verify else None)
    if base == "LSB":
        st, info = C.lsb_replacement(x, pay, rng)
        ok = bool(np.array_equal(_lsb_extract(st, pay.size), pay)) if verify else None
    elif base == "LSB-M":
        st, info = C.lsb_matching(x, pay, rng)
        if verify:
            perm = method_rng(base, cid, rate, seed).permutation(x.size)[:pay.size]
            ok = bool(np.array_equal(st.reshape(-1)[perm] & 1, pay))
    elif base == "EA-LSB":
        st, info = C.edge_adaptive_lsb(x, pay, rng)
        ok = bool(np.array_equal(_ealsb_extract(st, pay.size), pay)) if verify else None
    elif base == "PVD":
        st, info = C.pvd(x, pay, rng)
    elif base == "FM-PSO-LSB":
        st, info = REF.fm_pso_lsb(x, pay, KEY, rng)
        if verify:
            ok = bool(np.array_equal(REF.fm_pso_extract(st, pay.size, info["fm_alpha"], info["fm_gamma"], KEY), pay))
    elif base.endswith("-STC"):
        name = base[:-4]
        st, info = REF.stc_system(name, x, pay, KEY + cid.encode(), rng)
        if verify:
            ok = bool(np.array_equal(REF.stc_extract(name, st, KEY + cid.encode()), pay))
    elif base.endswith("-SIM"):
        st, info = REF.simulate(base[:-4], x, pay.size, rng)
    else:
        raise KeyError(method)
    info.setdefault("blind_extractable", base not in {m.split(":")[0] for m in SIM})
    return st, info, ok


def run_task(t):
    method, rate, cpath, seed, save_dir = t
    cid = Path(cpath).stem
    x, rho = _cover(cpath)
    pay = payload_bits(cid, rate, seed)
    w0, c0 = time.perf_counter(), time.process_time()
    st, info, ok = embed_one(method, x, pay, (cid, rate, seed))
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    q = evaluate_pair(x, st, pay.size)
    changed = (st != x)
    if save_dir:
        from PIL import Image
        d = Path(save_dir) / f"{method.replace(':', '_')}@{rate}"
        d.mkdir(parents=True, exist_ok=True)
        Image.fromarray(st).save(d / f"{cid}.png")
    extra = {k: info[k] for k in ("chromosomes", "gates", "fm_alpha", "fm_gamma", "matched_budget",
                                  "hist", "stc_h", "coding") if k in info and info[k] is not None}
    return {"method": method, "rate": rate, "cover": cid, "seed": seed, "psnr": q.psnr, "ssim": q.ssim,
            "mse": q.mse, "correlation": q.correlation, "payload_bits": pay.size,
            "n_changes": int(changed.sum()), "embedding_efficiency": q.embedding_efficiency,
            "hill_distortion": float(rho[changed].sum()), "extraction_ok": ok,
            "blind_extractable": info.get("blind_extractable"), "wall_s": wall, "cpu_s": cpu,
            "ga_evaluations": info.get("ga_evaluations"), "ga_generations": info.get("ga_generations"),
            "ga_converged_at": info.get("ga_converged_at"), "extra": json.dumps(extra)}


def _rank(m):
    for k, v in COST_RANK.items():
        if m.startswith(k):
            return v
    return 1


def build_tasks(study, a):
    sets = cover_sets()
    T = []
    if study in ("quality", "runtime"):
        covers = sets["quality"][: a.n_covers or 100]
        seeds = a.seeds if study == "quality" else [0]
        if study == "runtime":
            covers = covers[: a.n_covers or 20]
        for m in a.methods or MAIN:
            for r in a.rates or RATES:
                for c in covers:
                    for s in seeds:
                        T.append((m, r, str(c), s, None))
    elif study == "ablation":
        for m in a.methods or (AMDT_ABL + D_ABL):
            for c in sets["quality"][: a.n_covers or 100]:
                for s in a.seeds:
                    T.append((m, 0.1, str(c), s, None))
    elif study == "stego":
        covers = sets["stego"][: a.n_covers or 1000]
        for m in a.methods or MAIN:
            for r in a.rates or (0.1, 0.4):
                for c in covers:
                    T.append((m, r, str(c), 0, str(Path(a.out) / "stego")))
        for m in (["AMDT:fixed_path", "AMDT:single_seg", "AMDT-D:mse_fitness", "AMDT-D:no_gate",
                   "AMDT-D:fixed_path", "AMDT-D:single_seg", "AMDT-D:no_search"] if not a.methods else []):
            for c in covers:
                T.append((m, 0.1, str(c), 0, str(Path(a.out) / "stego")))
    elif study == "srnet":
        covers = sets["stego"][1000:2000]
        for m in a.methods or ["AMDT", "AMDT-D", "LSB-M", "HILL-STC", "S-UNIWARD-STC", "EvoHILL-STC", "FM-PSO-LSB"]:
            for c in covers:
                T.append((m, 0.4, str(c), 0, str(Path(a.out) / "stego")))
    T.sort(key=lambda t: -_rank(t[0]))
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["quality", "ablation", "stego", "srnet", "runtime"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--n-covers", type=int, default=0)
    ap.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")], default=[0, 1, 2, 3, 4])
    ap.add_argument("--rates", type=lambda s: [float(v) for v in s.split(",")], default=None)
    ap.add_argument("--methods", type=lambda s: s.split(","), default=None)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{a.study}.csv"
    tasks = build_tasks(a.study, a)
    k, n = (int(v) for v in a.shard.split("/"))
    tasks = [t for i, t in enumerate(tasks) if i % n == k]
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((row["method"], float(row["rate"]), row["cover"], int(row["seed"])))
    todo = [t for t in tasks if (t[0], t[1], Path(t[2]).stem, t[3]) not in done]
    meta = {"study": a.study, "n_tasks": len(tasks), "todo": len(todo), "workers": a.workers,
            "platform": platform.platform(), "processor": platform.processor(), "python": sys.version,
            "cpu_model": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if "model name" in l), "?")
            if os.path.exists("/proc/cpuinfo") else "?", "ga": GA.as_dict(),
            "argv": sys.argv, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / f"{a.study}_meta.json").write_text(json.dumps(meta, indent=1))
    if a.study in ("quality", "stego"):
        sets = cover_sets()
        man = out / "cover_manifest.json"
        if not man.exists():
            man.write_text(json.dumps({k: [{"file": p.name, "sha256": sha256(p)} for p in v]
                                       for k, v in sets.items()}, indent=0))
    print(f"{a.study}: {len(todo)}/{len(tasks)} tasks to run", flush=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            wr.writeheader()
        t0 = time.time()
        with mp.get_context("fork").Pool(a.workers, maxtasksperchild=40) as pool:
            for i, row in enumerate(pool.imap_unordered(run_task, todo, chunksize=1), 1):
                wr.writerow(row); f.flush()
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"{i}/{len(todo)} {el/60:.1f} min, eta {el/i*(len(todo)-i)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
