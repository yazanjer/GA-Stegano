"""The studies. Each function answers one reviewer comment and writes artifacts.

    study_quality        -> comments 4, 7   quality/capacity vs. all baselines
    study_ablation       -> comments 1, 2   what each decomposition layer buys
    study_targeted       -> comment 1       payload-statistics attacks + key space
    study_steganalysis   -> comment 3       SRM-subset + FLD ensemble
    study_cnn            -> comment 3       Yedroudj-Net / SRNet (needs torch+GPU)
    study_runtime        -> comment 6       runtime, complexity, convergence
    study_stats          -> comment 4       significance testing over everything
    study_reproducibility-> comment 5       dataset/payload/hardware/seed manifest

Every study returns a ``pandas.DataFrame`` and writes it to
``<run_dir>/results/<name>.csv``; nothing is recomputed between studies, the
CSVs are the interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..data.dataset import (DatasetSpec, cover_wise_split, describe_payloads,
                            payload_bits_for_rate)
from ..evaluation import plots, tables
from ..evaluation.metrics import evaluate_pair
from ..evaluation.significance import compare_many, descriptive, power_estimate
from ..ga.optimizer import GAConfig
from ..stego.amdt import capacity_bits
from ..stego.decomposition import DecompositionParams, decompose, describe, keyspace_bits
from ..steganalysis.ensemble import EnsembleClassifier
from ..steganalysis.features import extract_batch, feature_dimension
from ..steganalysis.metrics import bootstrap_ci, evaluate_scores, roc_curve, auc
from ..steganalysis.targeted import payload_uniformity, run_targeted_suite
from ..utils.execution import resolve_n_jobs
from ..utils.profiling import StageProfiler, complexity_table, profile_model
from ..utils.provenance import capture, hardware_string
from ..utils.seeding import SeedPolicy, seeded_rng
from .registry import build_registry
from .store import StegoStore

log = logging.getLogger("amdt.studies")

__all__ = ["RunContext", "study_quality", "study_ablation", "study_targeted",
           "study_steganalysis", "study_cnn", "study_runtime", "study_stats",
           "study_reproducibility"]


# --------------------------------------------------------------------------- #
class RunContext:
    """Paths, images, config and the shared profiler for one invocation."""

    def __init__(self, run_dir: str | Path, images: Sequence[np.ndarray],
                 spec: DatasetSpec, cfg, key: bytes, tracker=None,
                 target=None) -> None:
        self.dir = Path(run_dir)
        self.results = self.dir / cfg.output.results
        self.figures = self.dir / cfg.output.figures
        self.tables = self.dir / cfg.output.tables
        for p in (self.results, self.figures, self.tables):
            p.mkdir(parents=True, exist_ok=True)
        self.images = list(images)
        self.spec = spec
        self.cfg = cfg
        self.key = key
        from ..utils.execution import ExecutionTarget
        self.target = target if target is not None else ExecutionTarget()
        self.profiler = StageProfiler(target=self.target)
        self.provenance = capture({"dataset": spec.as_dict(),
                                   "execution": self.target.as_dict()})
        # Never None: with tracking off this is a NullTracker writing
        # metrics.jsonl, so a run always leaves a record behind.
        from ..utils.tracking import NullTracker
        self.tracker = tracker if tracker is not None else NullTracker(self.dir)
        self.formats = tuple(cfg.output.formats)
        plots.apply_style()

    # -- helpers ---------------------------------------------------------
    @property
    def seeds(self) -> List[int]:
        return list(self.cfg.experiment.seeds)

    @property
    def rates(self) -> List[float]:
        return list(self.cfg.experiment.payload_rates_bpp)

    def ga_config(self, **over) -> GAConfig:
        d = {k: v for k, v in dict(self.cfg.ga).items()}
        d.update(over)
        return GAConfig(**d)

    def save_csv(self, df: pd.DataFrame, name: str) -> Path:
        p = self.results / f"{name}.csv"
        df.to_csv(p, index=False)
        return p

    def save_json(self, obj, name: str) -> Path:
        p = self.results / f"{name}.json"
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def save_table(self, tex: str, name: str) -> Path:
        return tables.write_table(tex, self.tables, name)

    def save_fig(self, fig, name: str) -> List[Path]:
        return plots.save(fig, self.figures, name, self.formats)

    @property
    def prov_line(self) -> str:
        return (f"git={self.provenance.get('git_commit')} "
                f"seeds={self.seeds} dataset={self.spec.name} "
                f"n_images={self.spec.n_images}")


def _payload(rng: np.random.Generator, n_bits: int) -> np.ndarray:
    return rng.integers(0, 2, n_bits, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# comment 4 + 7 : quality and capacity against every baseline
# --------------------------------------------------------------------------- #
def study_quality(ctx: RunContext, methods: Optional[Dict] = None,
                  store: Optional[StegoStore] = None) -> pd.DataFrame:
    """Embed every (image, rate, seed, method) and record quality metrics.

    ``store`` (optional) caches ``(method, rate, seed, image_idx) -> stego`` so
    the steganalysis studies can reuse the images.  It is a *cache*: on a large
    corpus the store may decline to keep anything, and the consumers re-embed.
    """
    reg = methods or build_registry(ctx.ga_config(), ctx.key,
                                    verify=bool(ctx.cfg.experiment.verify_extraction))
    rows: List[Dict] = []

    # Stream to the tracker as we go. An 8-hour run that only reports at the
    # end is a run you cannot tell apart from a hung one, and a problem visible
    # at hour 2 costs six fewer hours than the same problem found at hour 8.
    total_units = len(ctx.rates) * len(ctx.seeds) * len(ctx.images) * len(reg)
    unit = 0

    for rate in ctx.rates:
        n_bits = payload_bits_for_rate(ctx.images[0].shape, rate)
        for seed in ctx.seeds:
            for i, cover in enumerate(ctx.images):
                payload = _payload(seeded_rng(seed, f"payload:{rate}:{i}"), n_bits)
                for name, fn in reg.items():
                    rng = seeded_rng(seed, f"{name}:{rate}:{i}")
                    with ctx.profiler.stage(f"embed:{name}"):
                        stego, info = fn(cover, payload, rng)
                    q = evaluate_pair(cover, stego, info["embedded_bits"], info["n_changes"])
                    row = {
                        "method": name, "rate_bpp": rate, "seed": seed, "image": i,
                        "image_name": ctx.spec.records[i].name,
                        "requested_bits": int(n_bits),
                        "embedded_bits": int(info["embedded_bits"]),
                        "capacity_limited": bool(info.get("capacity_limited", False)),
                        "blind_extractable": bool(info.get("blind_extractable", True)),
                        "extraction_ok": info.get("extraction_ok", None),
                        **q.as_dict(),
                    }
                    for k in ("ga_evaluations", "ga_generations", "ga_wall_time_s",
                              "ga_converged_at"):
                        if info.get(k) is not None:
                            row[k] = info[k]
                    rows.append(row)

                    unit += 1
                    ctx.tracker.log({
                        f"quality/{name}/psnr": q.psnr,
                        f"quality/{name}/ssim": q.ssim,
                        f"quality/{name}/mse": q.mse,
                        f"quality/{name}/embedding_efficiency": q.embedding_efficiency,
                        "progress/fraction": unit / max(total_units, 1),
                        "progress/rate_bpp": rate,
                        "progress/seed": seed,
                        "progress/image": i,
                    }, step=unit)
                    if row.get("extraction_ok") is False:
                        # Surfaced immediately: every downstream number from a
                        # failed extraction is meaningless.
                        ctx.tracker.log({"alert/extraction_failed": 1}, step=unit)
                        log.error("EXTRACTION FAILED: %s rate=%s seed=%s image=%s",
                                  name, rate, seed, i)

                    if store is not None:
                        store.put((name, rate, seed, i), stego)
                        if info.get("ga_history") is not None:
                            store.put_meta((name, rate, seed, i), info["ga_history"])

    df = pd.DataFrame(rows)
    ctx.save_csv(df, "quality")
    ctx.tracker.summary({
        "quality/rows": len(df),
        "quality/extraction_failures": int((df.extraction_ok == False).sum()),  # noqa: E712
        "quality/done": True,
    })

    # ---- figures 2,3,4,6,8 + payload sweep --------------------------------
    ref_rate = ctx.rates[len(ctx.rates) // 2]
    sub = df[df.rate_bpp == ref_rate]
    for metric, ylabel, fname in (
        ("psnr", "PSNR (dB)", "fig02_psnr"),
        ("ssim", "SSIM", "fig03_ssim"),
        ("mse", "MSE", "fig04_mse"),
        ("embedding_efficiency", "bits per change", "fig06_efficiency"),
        ("correlation", "correlation", "fig08_correlation"),
    ):
        data = {m: g[metric].to_numpy() for m, g in sub.groupby("method")}
        fig = plots.metric_boxplot(data, ylabel,
                                   title=f"payload = {ref_rate} bpp", highlight="AMDT")
        ctx.save_fig(fig, fname)

    series = {}
    for m, g in df.groupby("method"):
        agg = g.groupby("rate_bpp")["psnr"].agg(["mean", "std"]).reset_index()
        series[m] = (agg.rate_bpp.tolist(), agg["mean"].tolist(), agg["std"].tolist())
    ctx.save_fig(plots.metric_vs_rate(series, ylabel="PSNR (dB)"), "fig07_psnr_vs_rate")

    # ---- table ------------------------------------------------------------
    per_method = {m: {k: g[k].to_numpy() for k in
                      ("psnr", "ssim", "mse", "embedding_efficiency", "correlation")}
                  for m, g in sub.groupby("method")}
    ctx.save_table(
        tables.results_table(
            per_method,
            metrics=("psnr", "ssim", "mse", "embedding_efficiency", "correlation"),
            caption=(f"Imperceptibility at {ref_rate} bpp over "
                     f"{ctx.spec.n_images} covers $\\times$ {len(ctx.seeds)} seeds "
                     r"(mean $\pm$ std)."),
            label="tab:quality", provenance=ctx.prov_line),
        "tab_quality")
    return df


# --------------------------------------------------------------------------- #
# comments 1 + 2 : ablation over decomposition layers
# --------------------------------------------------------------------------- #
def study_ablation(ctx: RunContext) -> pd.DataFrame:
    variants = tuple(ctx.cfg.ablation.variants)
    rate = float(ctx.cfg.ablation.rate_bpp)
    reg = build_registry(ctx.ga_config(), ctx.key,
                         verify=bool(ctx.cfg.experiment.verify_extraction),
                         include=(), ablations=variants)
    n_bits = payload_bits_for_rate(ctx.images[0].shape, rate)

    rows: List[Dict] = []
    # Only the images needed for Fig. 5 are retained; keeping all
    # variants x images x seeds would cost hundreds of MB for one figure.
    fig_stegos: Dict[str, np.ndarray] = {}
    for seed in ctx.seeds:
        for i, cover in enumerate(ctx.images):
            payload = _payload(seeded_rng(seed, f"payload:{rate}:{i}"), n_bits)
            for name, fn in reg.items():
                rng = seeded_rng(seed, f"{name}:{rate}:{i}")
                with ctx.profiler.stage(f"ablation:{name}"):
                    stego, info = fn(cover, payload, rng)
                q = evaluate_pair(cover, stego, info["embedded_bits"], info["n_changes"])
                tr = run_targeted_suite(stego, None)
                rows.append({"variant": name.replace("AMDT-", ""), "seed": seed,
                             "image": i, "rate_bpp": rate,
                             "extraction_ok": info.get("extraction_ok"),
                             **q.as_dict(), **tr.as_dict()})
                v = name.replace("AMDT-", "")
                ctx.tracker.log({f"ablation/{v}/psnr": q.psnr,
                                 f"ablation/{v}/ssim": q.ssim,
                                 f"ablation/{v}/mse": q.mse})
                if seed == ctx.seeds[0] and i == 0:
                    fig_stegos[name.replace("AMDT-", "")] = stego

    df = pd.DataFrame(rows)
    ctx.save_csv(df, "ablation")

    data = {v: g["psnr"].to_numpy() for v, g in df.groupby("variant")}
    ctx.save_fig(plots.metric_boxplot(data, "PSNR (dB)",
                                      title=f"ablation at {rate} bpp", highlight="full"),
                 "fig09_ablation_psnr")

    # Fig. 5: Laplacian histograms, cover vs. each variant, first image, first seed
    ctx.save_fig(plots.laplacian_histogram(ctx.images[0], fig_stegos), "fig05_laplacian")

    per = {v: {k: g[k].to_numpy() for k in ("psnr", "ssim", "mse", "embedding_efficiency")}
           for v, g in df.groupby("variant")}
    ctx.save_table(
        tables.results_table(
            per, metrics=("psnr", "ssim", "mse", "embedding_efficiency"),
            caption=(r"Ablation of the hierarchical message decomposition at "
                     f"{rate} bpp. Each row disables exactly one component."),
            label="tab:ablation", provenance=ctx.prov_line),
        "tab_ablation")
    return df


# --------------------------------------------------------------------------- #
# comment 1 : payload-statistics attacks + key-space accounting
# --------------------------------------------------------------------------- #
def study_targeted(ctx: RunContext) -> pd.DataFrame:
    """Does T3/T4 actually neutralise payload-statistics attacks?

    Three payload types are used, because the whole claim is about payload
    structure: uniform random (the easy case every paper reports), ASCII text,
    and an all-zero block (the adversarial worst case).
    """
    rate = float(ctx.cfg.ablation.rate_bpp)
    n_bits = payload_bits_for_rate(ctx.images[0].shape, rate)

    def make_payload(kind: str, rng: np.random.Generator) -> np.ndarray:
        if kind == "uniform":
            return rng.integers(0, 2, n_bits, dtype=np.uint8)
        if kind == "ascii_text":
            txt = (b"Steganography is the practice of concealing information. " * 10000)
            return np.unpackbits(np.frombuffer(txt, dtype=np.uint8))[:n_bits].copy()
        if kind == "all_zeros":
            return np.zeros(n_bits, dtype=np.uint8)
        raise KeyError(kind)

    variants = {
        "no_decomp": DecompositionParams(0, 0, 0, 0, 0),
        "T1T2_only": DecompositionParams(1, 1, 0, 0, 0),
        "T3_only": DecompositionParams(0, 0, 1, 1, 0),
        "T4_only": DecompositionParams(0, 0, 0, 0, 1),
        "full": DecompositionParams(1, 1, 1, 1, 1),
    }

    rows = []
    for kind in ("uniform", "ascii_text", "all_zeros"):
        for seed in ctx.seeds:
            raw = make_payload(kind, seeded_rng(seed, f"targeted:{kind}"))
            for vname, p in variants.items():
                t = decompose(raw, p, ctx.key)
                u = payload_uniformity(t)
                ks = keyspace_bits(raw.size, p, int(ctx.cfg.ga.n_segments))
                rows.append({"payload_type": kind, "variant": vname, "seed": seed,
                             "payload_bits": int(raw.size), **u,
                             "keyspace_total_bits": ks["total_bits"],
                             "keyspace_T3_bits": ks["T3_bits_per_segment"]})

    df = pd.DataFrame(rows)
    ctx.save_csv(df, "targeted_payload")

    # image-domain attacks on the ablation stego images
    reg = build_registry(ctx.ga_config(), ctx.key, verify=False, include=(),
                         ablations=("no_decomp", "full"))
    img_rows = []
    for kind in ("uniform", "ascii_text", "all_zeros"):
        for seed in ctx.seeds[:2]:
            raw = make_payload(kind, seeded_rng(seed, f"targeted:{kind}"))
            for i, cover in enumerate(ctx.images):
                for name, fn in reg.items():
                    stego, info = fn(cover, raw, seeded_rng(seed, f"{name}:{kind}:{i}"))
                    tr = run_targeted_suite(stego, None)
                    img_rows.append({"payload_type": kind,
                                     "variant": name.replace("AMDT-", ""),
                                     "seed": seed, "image": i, **tr.as_dict()})
            # cover control
            for i, cover in enumerate(ctx.images):
                tr = run_targeted_suite(cover, None)
                img_rows.append({"payload_type": kind, "variant": "cover",
                                 "seed": seed, "image": i, **tr.as_dict()})

    dfi = pd.DataFrame(img_rows).drop_duplicates(
        subset=["payload_type", "variant", "seed", "image"])
    ctx.save_csv(dfi, "targeted_image")

    ctx.save_json(describe(), "decomposition_spec")

    header = ["Payload", "Variant", "Bias", r"$\rho_1$", "Entropy",
              r"$\chi^2$ $p$", "RS rate", "WS rate"]
    rows_tex = []
    for (k, v), g in df.groupby(["payload_type", "variant"], sort=False):
        gi = dfi[(dfi.payload_type == k) & (dfi.variant == v)]
        rows_tex.append([
            k.replace("_", r"\_"), v.replace("_", r"\_"),
            f"{g['bias'].mean():.4f}", f"{g['autocorr_lag1'].mean():+.4f}",
            f"{g['shannon_entropy'].mean():.5f}",
            f"{gi['chi_square_p'].mean():.3f}" if len(gi) else "--",
            f"{gi['rs_rate'].mean():.3f}" if len(gi) else "--",
            f"{gi['ws_rate'].mean():.3f}" if len(gi) else "--",
        ])
    ctx.save_table(
        tables.latex_table(
            header, rows_tex,
            caption=(r"Payload-statistics analysis. Bias, lag-1 autocorrelation and "
                     r"entropy are measured on the \emph{transformed payload}; "
                     r"$\chi^2$, RS and WS are attacks on the stego image. T4 drives "
                     r"the payload to uniform for every payload type, which is the "
                     r"mechanism behind the security claim; RS/WS are unchanged, "
                     r"confirming that the layers do not affect change-based detection."),
            label="tab:targeted", column_spec="ll" + "r" * 6,
            provenance=ctx.prov_line),
        "tab_targeted")
    return df


# --------------------------------------------------------------------------- #
# comment 3 : rich-model steganalysis
# --------------------------------------------------------------------------- #
def _stego_set(ctx: RunContext, store: Optional[StegoStore], reg: Dict,
               name: str, rate: float, seed: int) -> List[np.ndarray]:
    """Stego images for one (method, rate); from cache when available.

    Re-embedding on a miss is exact, not approximate: the payload and the RNG
    are both derived from ``(seed, method, rate, image)``, so a recomputed stego
    is byte-identical to the cached one.  Peak memory is one method-rate set.
    """
    n_bits = payload_bits_for_rate(ctx.images[0].shape, rate)
    out = []
    for i, cover in enumerate(ctx.images):
        cached = store.get((name, rate, seed, i)) if store is not None else None
        if cached is not None:
            out.append(cached)
            continue
        payload = _payload(seeded_rng(seed, f"payload:{rate}:{i}"), n_bits)
        st, _ = reg[name](cover, payload, seeded_rng(seed, f"{name}:{rate}:{i}"))
        out.append(st)
    return out


def study_steganalysis(ctx: RunContext, store: Optional[StegoStore] = None,
                       methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """SRM-subset features + FLD ensemble, cover-wise k-fold cross-validation."""
    scfg = ctx.cfg.steganalysis
    feat = str(scfg.features)
    n_folds = int(scfg.cross_validation.n_folds)
    rate_list = ctx.rates
    seed0 = ctx.seeds[0]

    reg = build_registry(ctx.ga_config(), ctx.key, verify=False)
    names = list(methods) if methods else sorted(reg)

    # Explicit, machine-independent worker count: n_jobs=-1 would tie the
    # feature values to the host's core count via BLAS reduction order.
    n_jobs = resolve_n_jobs(scfg.n_jobs, ctx.target)

    with ctx.profiler.stage("features:cover"):
        Xc = extract_batch(ctx.images, feat, n_jobs=n_jobs, progress=True)

    rows, roc_store = [], {}
    for rate in rate_list:
        for name in names:
            stegos = _stego_set(ctx, store, reg, name, rate, seed0)
            with ctx.profiler.stage("features:stego"):
                Xs = extract_batch(stegos, feat, n_jobs=n_jobs)

            # Cover-wise k-fold: a cover and its stego are always in the same
            # fold, so the classifier can never see a test cover during training.
            folds = _kfold(len(ctx.images), n_folds, seeded_rng(seed0, "kfold"))
            scores, labels = [], []
            oob = []
            for tr_idx, te_idx in folds:
                clf = EnsembleClassifier(
                    d_sub=scfg.classifier.d_sub,
                    max_learners=int(scfg.classifier.max_learners),
                    random_state=int(scfg.classifier.random_state),
                )
                with ctx.profiler.stage("fld_ensemble"):
                    clf.fit(Xc[tr_idx], Xs[tr_idx])
                oob.append(clf.oob_error_)
                scores.append(np.r_[clf.decision_function(Xc[te_idx]),
                                    clf.decision_function(Xs[te_idx])])
                labels.append(np.r_[np.zeros(te_idx.size, int), np.ones(te_idx.size, int)])
            s = np.concatenate(scores)
            y = np.concatenate(labels)
            m = evaluate_scores(y, s)
            acc, lo, hi = bootstrap_ci(y, s, "accuracy", rng=seeded_rng(seed0, "boot"))
            rows.append({"method": name, "rate_bpp": rate, "detector":
                         f"{feat}+FLD-ensemble", "n_folds": n_folds,
                         "oob_error": float(np.mean(oob)),
                         "accuracy_ci_low": lo, "accuracy_ci_high": hi,
                         **m.as_dict()})
            if rate == rate_list[len(rate_list) // 2]:
                fpr, tpr, _ = roc_curve(y, s)
                roc_store[name] = (fpr, tpr, auc(fpr, tpr))

    df = pd.DataFrame(rows)
    ctx.save_csv(df, "steganalysis_classical")

    if roc_store:
        ctx.save_fig(plots.roc_panel(roc_store), "fig10_roc")
    series = {}
    for m, g in df.groupby("method"):
        a = g.groupby("rate_bpp")["p_e"].agg(["mean", "std"]).reset_index()
        series[m] = (a.rate_bpp.tolist(), a["mean"].tolist(),
                     a["std"].fillna(0).tolist())
    if series:
        ctx.save_fig(plots.detectability_curve(series), "fig11_detectability")

    ref = ctx.rates[len(ctx.rates) // 2]
    sub = df[df.rate_bpp == ref]
    ctx.save_table(
        tables.detection_table(
            {str(name): row.to_dict()
             for name, row in sub.set_index("method").iterrows()} if not sub.empty else {},
            detector=f"{feat} + FLD ensemble",
            caption=(f"Steganalysis at {ref} bpp with the {feat} rich model and the "
                     r"Kodovsky--Fridrich FLD ensemble, "
                     f"{n_folds}-fold cover-wise cross-validation. "
                     r"Lower accuracy / higher $P_E$ favours the steganographic method."),
            label="tab:detection", provenance=ctx.prov_line),
        "tab_detection")

    warn = ctx.spec.steganalysis_warning()
    if warn:
        ctx.save_json({"warning": warn, "n_images": ctx.spec.n_images},
                      "steganalysis_caveat")
    return df


def _kfold(n: int, k: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = rng.permutation(n)
    folds = np.array_split(idx, max(k, 2))
    out = []
    for i in range(len(folds)):
        te = np.sort(folds[i])
        tr = np.sort(np.concatenate([f for j, f in enumerate(folds) if j != i]))
        out.append((tr, te))
    return out


# --------------------------------------------------------------------------- #
# comment 3 : CNN steganalysis
# --------------------------------------------------------------------------- #
def study_cnn(ctx: RunContext, store: Optional[StegoStore] = None,
              methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Yedroudj-Net / SRNet. Raises if torch is unavailable -- by design."""
    from ..steganalysis.cnn import PairedStegoDataset, TrainConfig, evaluate_cnn, train_cnn
    import torch

    ccfg = ctx.cfg.steganalysis
    seed0 = ctx.seeds[0]
    reg = build_registry(ctx.ga_config(), ctx.key, verify=False)
    names = list(methods) if methods else sorted(reg)
    rows = []
    for rate in ctx.rates:
        for name in names:
            stegos = _stego_set(ctx, store, reg, name, rate, seed0)
            sp = cover_wise_split(len(ctx.images), seeded_rng(seed0, "cnnsplit"),
                                  train=float(ctx.cfg.dataset.split.train),
                                  val=float(ctx.cfg.dataset.split.val))
            tr = PairedStegoDataset([ctx.images[i] for i in sp["train"]],
                                    [stegos[i] for i in sp["train"]])
            va = PairedStegoDataset([ctx.images[i] for i in sp["val"]],
                                    [stegos[i] for i in sp["val"]])
            te = PairedStegoDataset([ctx.images[i] for i in sp["test"]],
                                    [stegos[i] for i in sp["test"]])
            cfg = TrainConfig(
                model=str(ccfg.model), epochs=int(ccfg.epochs),
                batch_pairs=int(ccfg.batch_pairs), lr=float(ccfg.lr),
                weight_decay=float(ccfg.weight_decay), optimizer=str(ccfg.optimizer),
                scheduler=str(ccfg.scheduler), seed=seed0,
                num_workers=int(ccfg.num_workers), amp=bool(ccfg.amp),
                checkpoint_dir=(str(ccfg.checkpoint_dir) if ccfg.checkpoint_dir else
                                str(ctx.dir / "checkpoints" / f"{name}_{rate}")),
                early_stop_patience=int(ccfg.early_stop_patience),
                curriculum_from=(str(ccfg.curriculum_from) if ccfg.curriculum_from else None),
            )
            # The stub applies patches published by the out-of-runtime watcher.
            # It is handed validation metrics only; `te` never enters its scope.
            stub = None
            if bool(getattr(ctx.cfg, "supervisor", {}).get("enabled", False)):
                from .supervisor import PatchStub
                stub = PatchStub(patch_dir=ctx.dir, run_dir=ctx.dir,
                                 max_rounds=int(ctx.cfg.supervisor.trip.max_rounds),
                                 branch=str(ctx.cfg.supervisor.patch_branch))

            with ctx.profiler.stage(f"cnn_train:{cfg.model}"):
                model, hist, prof = train_cnn(tr, va, cfg, tracker=ctx.tracker,
                                              supervisor_stub=stub)
            from torch.utils.data import DataLoader
            from ..steganalysis.cnn import _collate
            loader = DataLoader(te, batch_size=cfg.batch_pairs, collate_fn=_collate)
            dev = next(model.parameters()).device
            m = evaluate_cnn(model, loader, dev)

            # Cost profile: params, FLOPs, synchronized per-sample latency, VRAM.
            side = ctx.images[0].shape[0]
            cost = profile_model(model, (1, 1, side, side), device=str(dev))
            rows.append({"method": name, "rate_bpp": rate,
                         "detector": cfg.model, **m.as_dict(), **prof, **cost,
                         "agent_rounds_used": (stub.rounds_used if stub else 0)})
    df = pd.DataFrame(rows)
    ctx.save_csv(df, "steganalysis_cnn")
    if not df.empty:
        ref = ctx.rates[len(ctx.rates) // 2]
        sub = df[df.rate_bpp == ref]
        ctx.save_table(
            tables.detection_table(
                {str(name): row.to_dict()
                 for name, row in sub.set_index("method").iterrows()},
                detector=str(ccfg.model),
                caption=(f"CNN steganalysis ({ccfg.model}) at {ref} bpp, cover-wise "
                         r"split, selection on validation $P_E$, test set touched once."),
                label="tab:detection_cnn", provenance=ctx.prov_line),
            "tab_detection_cnn")
    return df


# --------------------------------------------------------------------------- #
# comment 6 : runtime, complexity, convergence
# --------------------------------------------------------------------------- #
def study_runtime(ctx: RunContext, quality_df: Optional[pd.DataFrame] = None,
                  store: Optional[StegoStore] = None) -> pd.DataFrame:
    rows = ctx.profiler.summary()
    df = pd.DataFrame(rows)
    ctx.save_csv(df, "runtime")
    ctx.save_csv(pd.DataFrame(complexity_table()), "complexity")
    if not df.empty:
        if ctx.profiler.spans_devices:
            log.warning("runtime rows span more than one device; the table gets a "
                        "hardware column and the means must not be compared across it")
        if not ctx.target.timing_valid:
            log.warning("this run's timings are not reportable (%s); the table "
                        "marks them and the manuscript should omit them",
                        ctx.target.timing_invalid_reason)
        ctx.save_fig(plots.runtime_plot(rows), "fig12_runtime")
        ctx.save_table(tables.runtime_table(rows, provenance=ctx.prov_line,
                                            hardware=ctx.target.label), "tab_runtime")

    if store is not None:
        hists: Dict[str, List[List[float]]] = {}
        for key, hist in store.meta_items():
            hists.setdefault(str(key[0]), []).append(list(hist.best))
        if hists:
            ctx.save_fig(plots.convergence_curve(hists), "fig13_convergence")
            conv = pd.DataFrame([
                {"method": m, "runs": len(rs),
                 "mean_generations": float(np.mean([len(r) for r in rs])),
                 "mean_final_fitness": float(np.mean([r[-1] for r in rs]))}
                for m, rs in hists.items()])
            ctx.save_csv(conv, "convergence")

    if quality_df is not None and "ga_wall_time_s" in quality_df:
        g = (quality_df.dropna(subset=["ga_wall_time_s"])
             .groupby(["method", "rate_bpp"])["ga_wall_time_s"]
             .agg(["mean", "std", "count"]).reset_index())
        ctx.save_csv(g, "ga_runtime")
    return df


# --------------------------------------------------------------------------- #
# comment 4 : significance testing
# --------------------------------------------------------------------------- #
def study_stats(ctx: RunContext, quality_df: pd.DataFrame,
                detection_df: Optional[pd.DataFrame] = None,
                proposed: str = "AMDT") -> pd.DataFrame:
    alpha = float(ctx.cfg.stats.alpha)
    rng = seeded_rng(ctx.seeds[0], "bootstrap")
    out: List[Dict] = []

    for rate, g in quality_df.groupby("rate_bpp"):
        piv = g.pivot_table(index=["image", "seed"], columns="method",
                            values=["psnr", "ssim", "mse", "embedding_efficiency"])
        if proposed not in piv["psnr"].columns:
            continue
        for metric in ("psnr", "ssim", "mse", "embedding_efficiency"):
            tab = piv[metric].dropna()
            base = {c: tab[c].to_numpy() for c in tab.columns if c != proposed}
            if not base:
                continue
            comps = compare_many(tab[proposed].to_numpy(), base, metric,
                                 name=proposed, alpha=alpha, rng=rng)
            for c in comps:
                d = c.as_dict()
                d["rate_bpp"] = rate
                d["power"] = power_estimate(c.cohens_dz, c.n_pairs, alpha, rng=rng)
                out.append(d)

    df = pd.DataFrame(out)
    ctx.save_csv(df, "significance")

    if not df.empty:
        ref = ctx.rates[len(ctx.rates) // 2]
        sub = df[(df.rate_bpp == ref) & (df.metric == "psnr")]

        class _Row:
            def __init__(self, r):
                self.__dict__.update(r)
        ctx.save_table(
            tables.significance_table(
                [_Row(r) for _, r in sub.iterrows()],
                caption=(f"Paired significance tests on PSNR at {ref} bpp "
                         f"({ctx.spec.n_images} covers $\\times$ {len(ctx.seeds)} seeds "
                         f"= {sub['n_pairs'].max() if not sub.empty else 0} paired "
                         r"observations)."),
                label="tab:significance", provenance=ctx.prov_line),
            "tab_significance")

        flagged = df[df.effect_below_noise & df.significant]
        if not flagged.empty:
            ctx.save_csv(flagged, "significance_flagged_small_effects")
    return df


# --------------------------------------------------------------------------- #
# comment 5 : reproducibility manifest
# --------------------------------------------------------------------------- #
def study_reproducibility(ctx: RunContext) -> Dict:
    shape = ctx.images[0].shape
    max_cap = capacity_bits(shape, int(ctx.cfg.ga.n_segments), mask=15)
    manifest = {
        "dataset": ctx.spec.as_dict(),
        "payloads": describe_payloads(shape, ctx.rates, max_cap),
        "max_capacity_bits": max_cap,
        "max_capacity_bpp": max_cap / (shape[0] * shape[1]),
        "ga": dict(ctx.cfg.ga),
        "seed_policy": SeedPolicy.describe(ctx.seeds[0], len(ctx.seeds), ctx.seeds),
        "hardware": hardware_string(ctx.provenance),
        # The thread setting is part of the reproducibility contract, not a
        # footnote: the same seed at a different thread count gives different
        # floats, because BLAS reduction order changes.
        "execution": ctx.target.as_dict(),
        "provenance": ctx.provenance,
        "feature_dimensions": {
            name: feature_dimension(name, shape[0])
            for name in ("spam686", "srm_subset")
        },
        "decomposition_spec": describe(),
        "secret_key_note": (
            "The 256-bit key drives T3/T4 only. It is never embedded; the header "
            "carries layer-enable flags and the block-size index (7 bits)."
        ),
    }
    ctx.save_json(manifest, "reproducibility_manifest")

    header = ["Item", "Value"]
    rows = [
        ["Dataset", f"{ctx.spec.name}, {ctx.spec.n_images} covers, "
                    f"{shape[0]}$\\times${shape[1]} 8-bit grayscale"],
        ["Image source", ctx.spec.source[:120] if ctx.spec.source else "--"],
        ["Payload sizes", ", ".join(
            f"{p['rate_bpp']} bpp = {int(p['payload_bits'])} bits "
            f"({p['percent_of_max_capacity']:.1f}\\% of capacity)"
            for p in manifest["payloads"])],
        ["Max capacity", f"{max_cap} bits ({max_cap/(shape[0]*shape[1]):.2f} bpp)"],
        ["GA", f"population {ctx.cfg.ga.population}, up to "
               f"{ctx.cfg.ga.generations} generations, tournament "
               f"{ctx.cfg.ga.tournament_size}, $p_c$={ctx.cfg.ga.crossover_prob}, "
               f"$p_m$={ctx.cfg.ga.mutation_prob}, patience {ctx.cfg.ga.patience}, "
               f"{ctx.cfg.ga.n_segments} segments"],
        ["Seeds", ", ".join(map(str, ctx.seeds))],
        ["Seed policy", SeedPolicy.text],
        ["Execution target", f"{ctx.target.name} / {ctx.target.device}, "
                             f"{ctx.target.threads} BLAS thread(s) "
                             f"({ctx.target.thread_mode}), n\\_jobs="
                             f"{ctx.target.n_jobs}, timings "
                             + ("reportable" if ctx.target.timing_valid
                                else f"NOT reportable ({ctx.target.timing_invalid_reason})")],
        ["Hardware", manifest["hardware"]],
        ["Software", f"Python {ctx.provenance['python'].split()[0]}, "
                     f"NumPy {ctx.provenance['packages'].get('numpy')}, "
                     f"SciPy {ctx.provenance['packages'].get('scipy')}, "
                     f"PyTorch {ctx.provenance['packages'].get('torch')}"],
        ["Git commit", str(ctx.provenance.get("git_commit"))],
    ]
    ctx.save_table(
        tables.latex_table([h for h in header],
                           [[a, b] for a, b in rows],
                           caption="Experimental setup and reproducibility manifest.",
                           label="tab:repro", column_spec="lp{0.72\\linewidth}",
                           provenance=ctx.prov_line),
        "tab_reproducibility")
    return manifest
