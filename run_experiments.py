#!/usr/bin/env python3
"""Entry point. Thin by design: read config, call into ``src/amdt``, write artifacts.

Every reviewer comment maps to one ``--study``::

    python run_experiments.py                                   # everything
    python run_experiments.py experiment.studies=[quality,stats]
    python run_experiments.py dataset=bossbase ga.generations=200
    python run_experiments.py steganalysis=cnn steganalysis.model=srnet
    python run_experiments.py -m ga.n_segments=1,2,4,8          # Hydra sweep

Reviewer comment -> artifact
    1 novelty justification   tables/tab_targeted.tex, tab_ablation.tex,
                              results/decomposition_spec.json
    2 decomposition formalism results/decomposition_spec.json (formulas +
                              parameters), src/amdt/stego/decomposition.py
    3 steganalysis            tables/tab_detection.tex, tab_detection_cnn.tex,
                              figures/fig10_roc.pdf, fig11_detectability.pdf
    4 statistics              tables/tab_significance.tex, results/significance.csv
    5 reproducibility         tables/tab_reproducibility.tex,
                              results/reproducibility_manifest.json
    6 runtime                 tables/tab_runtime.tex, figures/fig12_runtime.pdf,
                              fig13_convergence.pdf, results/complexity.csv
    7 baselines               tables/tab_quality.tex (WOW, S-UNIWARD, HILL,
                              MiPOD-lite alongside LSB/EA-LSB/GA-FT/PVD)
"""

from __future__ import annotations

# set_seed touches CUBLAS_WORKSPACE_CONFIG, which CUDA reads at init -- so it
# has to run before anything can import torch and create a context.
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from amdt.utils.seeding import set_seed  # noqa: E402

set_seed(int(os.environ.get("AMDT_BOOT_SEED", "0")))

import json  # noqa: E402
import logging  # noqa: E402

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from amdt.data.dataset import cover_wise_split, load_dataset  # noqa: E402
from amdt.experiments import studies  # noqa: E402
from amdt.experiments.registry import build_registry  # noqa: E402
from amdt.experiments.store import StegoStore  # noqa: E402
from amdt.utils.drive_sync import DriveSync, SyncConfig, runs_referenced_by_commits  # noqa: E402
from amdt.utils.execution import configure_execution  # noqa: E402
from amdt.utils.provenance import write as write_provenance  # noqa: E402
from amdt.utils.repo import GitRepo  # noqa: E402
from amdt.utils.seeding import seeded_rng  # noqa: E402
from amdt.utils.tracking import build_tracker, config_hash, preflight  # noqa: E402

log = logging.getLogger("amdt")


def _resolve_root(cfg_root: str) -> Path:
    p = Path(cfg_root)
    return p if p.is_absolute() else (ROOT / p).resolve()


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    log.info("run directory: %s", run_dir)

    # Start mirroring before anything else can fail. The artifacts most worth
    # having off-machine are the ones a crash would otherwise take with it, so
    # this does not wait for the run to finish and never prompts.
    sync = DriveSync(run_dir, SyncConfig(**OmegaConf.to_container(cfg.sync, resolve=True)),
                     run_id=run_dir.name).start()
    if sync.cfg.enabled:
        log.info("mirroring run folder to %s", sync.dest)

    seed0 = int(cfg.experiment.seeds[0])
    set_seed(seed0)

    # Declare the execution target and pin BLAS threads before any heavy numeric
    # work starts. Seeding alone is not enough on CPU: reduction order changes
    # with thread count, so the same seed gives different floats.
    target = configure_execution(cfg.env)
    if not target.timing_valid:
        log.warning("timings from this run are NOT reportable: %s",
                    target.timing_invalid_reason)

    OmegaConf.save(cfg, run_dir / "resolved_config.yaml")
    write_provenance(run_dir / "provenance.json",
                     {"config": OmegaConf.to_container(cfg),
                      "execution": target.as_dict(),
                      # Sync target and run ID sit next to the commit hash, so a
                      # table cell resolves to both a commit and a folder.
                      "sync": sync.describe()})

    # -- provenance: repo, tracker (skill sections 6 and 11) ----------------
    repo = None
    if bool(cfg.tracking.git.enabled):
        repo = GitRepo(ROOT, remote=(str(cfg.tracking.git.remote)
                                     if cfg.tracking.git.remote else None),
                       branch=str(cfg.tracking.git.branch))
        repo.init(lfs=bool(cfg.tracking.git.lfs))
        repo.assert_no_token_in_config()
        log.info("git repo at %s (HEAD=%s)", ROOT, repo.head())

    if bool(cfg.tracking.get("preflight", False)):
        pf = preflight(cfg)
        (run_dir / "tracking_preflight.json").write_text(json.dumps(pf, indent=2))
        log.info("W&B preflight: credential=%s (%s) entity=%s project=%s exists=%s",
                 pf["credential_found"], pf["credential_source"],
                 pf["resolved_entity"], pf["project"], pf["project_exists"])
        if pf.get("entity_note"):
            log.info(pf["entity_note"])
        if pf["hint"]:
            log.warning(pf["hint"])
        log.info("preflight %s", "OK" if pf["ok"] else "FAILED")
        if not pf["ok"]:
            # Non-zero exit so `set -e` in run_overnight.sh halts here rather
            # than starting an 8-hour job whose metrics go nowhere.
            raise SystemExit(1)
        return

    chash = config_hash(cfg)
    tracker = build_tracker(cfg, run_dir, seed0,
                            git_commit=(repo.head() if repo else None))
    log.info("config_hash=%s tracker=%s", chash, type(tracker).__name__)

    # Dataset acquisition. The source is configuration; the provenance is code:
    # whichever mirror the bytes come from, the archive is hashed and every
    # later run verifies against the pin.
    data_root = cfg.dataset.root
    dataset_pin = None
    acquire = str(cfg.dataset.get("acquire", "local"))

    if bool(cfg.dataset.get("probe_only", False)):
        from amdt.data.fetch import probe_mirrors
        status = probe_mirrors(list(cfg.dataset.get("mirrors", [])))
        (run_dir / "mirror_probe.json").write_text(json.dumps(status, indent=2))
        alive = [m["url"] for m in status if m["ok"]]
        log.info("live mirrors: %s", alive or "NONE -- find a working source "
                 "and add it to dataset.mirrors")
        return

    if acquire == "kaggle":
        from amdt.data.kaggle import (DEFAULT_CACHE, DatasetPin, download_competition,
                                      download_dataset)
        kag = cfg.dataset.kaggle
        if not kag.slug:
            raise ValueError(
                "dataset.source=kaggle but no slug is set. Several Kaggle "
                "re-uploads of BOSSBase exist and they are not the same corpus "
                "(some are resized or JPEG-recompressed, which would silently "
                "invalidate every spatial-domain result). Confirm the slug and "
                "the file count on the dataset page first."
            )
        expected = (DatasetPin(slug=str(kag.slug), version=kag.pin.version,
                               sha256=kag.pin.sha256)
                    if (kag.pin.version or kag.pin.sha256) else None)
        cache = Path(str(kag.cache_dir)) if kag.cache_dir else DEFAULT_CACHE
        fetch = download_competition if str(kag.kind) == "competition" else download_dataset
        extracted, dataset_pin = fetch(str(kag.slug), cache, expected)
        data_root = str(extracted)

    elif acquire == "url":
        from amdt.data.fetch import fetch_dataset
        from amdt.data.kaggle import DatasetPin
        expected = (DatasetPin(slug=str(cfg.dataset.name),
                               sha256=cfg.dataset.pin.sha256)
                    if cfg.dataset.pin.sha256 else None)
        extracted, dataset_pin = fetch_dataset(
            list(cfg.dataset.mirrors), str(cfg.dataset.name),
            expected=expected, license_note=str(cfg.dataset.license),
            pattern=str(cfg.dataset.pattern))
        data_root = str(extracted)

    if dataset_pin is not None:
        (run_dir / "dataset_pin.json").write_text(
            json.dumps(dataset_pin.as_dict(), indent=2))
        log.info("dataset pinned: sha256=%s n_files=%s",
                 (dataset_pin.sha256 or "")[:12], dataset_pin.n_files)
        if dataset_pin.drift:
            log.error("DATASET DRIFT: %s -- results are not comparable with the "
                      "pinned version", dataset_pin.drift_note)
        expected_n = cfg.dataset.get("pin", {}).get("n_files")
        if expected_n and dataset_pin.n_files != int(expected_n):
            log.error("expected %s files, found %s. A resized or recompressed "
                      "re-upload would silently invalidate every spatial-domain "
                      "result -- check the source before continuing.",
                      expected_n, dataset_pin.n_files)

    images, spec = load_dataset(
        _resolve_root(data_root), side=int(cfg.dataset.side),
        limit=cfg.dataset.limit, name=str(cfg.dataset.name),
        source=str(cfg.dataset.source), license=str(cfg.dataset.license),
        pattern=str(cfg.dataset.pattern),
    )
    log.info("loaded %d images (%dx%d)", len(images), spec.side, spec.side)
    if spec.steganalysis_warning():
        log.warning(spec.steganalysis_warning())

    key = bytes.fromhex(str(cfg.experiment.secret_key_hex))
    if len(key) != 32:
        raise ValueError("experiment.secret_key_hex must be 64 hex chars (256 bits)")

    ctx = studies.RunContext(run_dir, images, spec, cfg, key, tracker=tracker,
                             target=target)
    wanted = list(cfg.experiment.studies)
    quality_df = detection_df = None

    n_methods = (len(cfg.methods.proposed) + len(cfg.methods.classical)
                 + len(cfg.methods.adaptive))
    store = StegoStore(
        run_dir, backend=str(cfg.experiment.stego_cache),
        projected_images=len(images) * n_methods * len(cfg.experiment.payload_rates_bpp)
        * len(cfg.experiment.seeds),
        image_bytes=int(cfg.dataset.side) ** 2,
        max_memory_mb=float(cfg.experiment.max_cache_memory_mb),
        max_disk_mb=float(cfg.experiment.max_cache_disk_mb),
    )

    # reproducibility manifest first: if a run dies later, the manifest still
    # documents what was attempted.
    studies.study_reproducibility(ctx)

    if bool(cfg.search.enabled):
        log.info("study: Optuna hyper-parameter search (validation only)")
        from amdt.experiments.search import objective_ga, run_search

        # The search split is carved out of train+val; the test covers used for
        # the reported tables are never visible to Optuna.
        sp = cover_wise_split(len(images), seeded_rng(seed0, "searchsplit"),
                              train=1.0 - float(cfg.search.val_fraction)
                              - float(cfg.dataset.split.test),
                              val=float(cfg.search.val_fraction))
        result = run_search(
            lambda t: objective_ga(t, images, sp["train"], sp["val"], key,
                                   float(cfg.search.rate_bpp), seed0),
            n_trials=int(cfg.search.n_trials), study_name=str(cfg.search.study_name),
            storage=(str(cfg.search.storage) if cfg.search.storage else None),
            direction=str(cfg.search.direction), seed=seed0, tracker=tracker,
            timeout_s=cfg.search.timeout_s,
        )
        ctx.save_json(dict(result), "optuna_search")
        log.info("best trial %s: %.4f with %s", result["best_trial"],
                 result["best_value"], result["best_params"])

    if "quality" in wanted:
        log.info("study: quality (comments 4, 7)")
        reg = build_registry(ctx.ga_config(), key,
                             verify=bool(cfg.experiment.verify_extraction))
        quality_df = studies.study_quality(ctx, reg, store)
        bad = quality_df[(quality_df.extraction_ok == False)]  # noqa: E712
        if len(bad):
            log.error("EXTRACTION FAILED for %d rows -- results are invalid", len(bad))
            ctx.save_csv(bad, "extraction_failures")

    if "ablation" in wanted:
        log.info("study: ablation (comments 1, 2)")
        studies.study_ablation(ctx)

    if "targeted" in wanted:
        log.info("study: targeted attacks (comment 1)")
        studies.study_targeted(ctx)

    if "steganalysis" in wanted:
        log.info("study: SRM + ensemble steganalysis (comment 3)")
        detection_df = studies.study_steganalysis(ctx, store)

    if "cnn" in wanted:
        log.info("study: CNN steganalysis (comment 3)")
        studies.study_cnn(ctx, store)

    if "runtime" in wanted:
        log.info("study: runtime and complexity (comment 6)")
        studies.study_runtime(ctx, quality_df, store)

    if "stats" in wanted:
        log.info("study: significance testing (comment 4)")
        if quality_df is None:
            raise RuntimeError("the 'stats' study needs the 'quality' study")
        studies.study_stats(ctx, quality_df, detection_df)

    index = {
        "run_dir": str(run_dir),
        "studies": wanted,
        "stego_cache": store.describe(),
        "config_hash": chash,
        "tracker_run": getattr(tracker, "run_id", None),
        "tracker_url": getattr(tracker, "url", None),
        "git_commit": repo.head() if repo else None,
        "sync": sync.describe(),
        "figures": sorted(p.name for p in ctx.figures.glob("*")),
        "tables": sorted(p.name for p in ctx.tables.glob("*")),
        "results": sorted(p.name for p in ctx.results.glob("*")),
    }
    (run_dir / "index.json").write_text(json.dumps(index, indent=2))

    # End-of-run commit: the .tex tables, vector plots and profiling summary are
    # produced after everything stops, so they would otherwise never land.
    if repo is not None and bool(cfg.tracking.git.commit_artifacts):
        sha = repo.commit_artifacts(run_dir, seed0, chash,
                                    getattr(tracker, "run_id", None))
        if sha:
            log.info("committed artifacts as %s", sha[:8])
            if bool(cfg.tracking.git.push):
                repo.push()

    interventions = run_dir / "agent_interventions.jsonl"
    if interventions.exists():
        from amdt.experiments.supervisor import Supervisor
        sup = Supervisor(run_dir)
        altering = sup.method_altering_summary()
        if altering:
            log.warning("%d METHOD-ALTERING agent intervention(s) occurred. The "
                        "manuscript's methods section must be reconciled with "
                        "%s before submission.", len(altering), interventions)

    # Final flush, then retention. Runs named in a commit message are evidence
    # and are never pruned, however old.
    sync_status = sync.close(protected=runs_referenced_by_commits(repo))
    if sync_status.get("enabled"):
        log.info("mirrored %s files (%.1f MB) to %s",
                 sync_status["files_synced"], sync_status["megabytes_synced"],
                 sync_status["sync_target"])
        if sync_status.get("errors"):
            log.warning("drive sync had %d error(s): %s",
                        len(sync_status["errors"]), sync_status["errors"][0])

    tracker.log_artifact(ctx.tables, "tables", "results")
    tracker.summary({"n_figures": len(index["figures"]),
                     "n_tables": len(index["tables"]),
                     "config_hash": chash})
    tracker.finish()

    log.info("done: %d figures, %d tables, %d result files",
             len(index["figures"]), len(index["tables"]), len(index["results"]))


if __name__ == "__main__":
    main()
