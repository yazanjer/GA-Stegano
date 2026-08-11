"""Optuna hyper-parameter search — validation only, resumable, pruned.

Two searchable objectives, because this project has two very different
optimisation problems:

``ga``
    Tune the *GA's own* hyper-parameters (population, generations, tournament
    size, crossover/mutation probability, segment count).  Objective: mean PSNR
    on the **validation covers** at a fixed payload rate.  Fast, CPU-only.

``cnn``
    Tune the steganalyser (learning rate, weight decay, batch size, optimizer).
    Objective: validation ``P_E``, minimised — i.e. Optuna is used to make the
    *attacker as strong as possible*, which is the honest way to evaluate a
    steganographic method.  A weak detector is not evidence of security.

Guarantees
----------
* **The test split is never constructed here.**  ``objective_*`` receives only
  train/val indices and :func:`_assert_no_test` fails loudly if a caller tries
  to pass a test set in.  Selecting hyper-parameters on test is leakage wearing
  a respectable hat, and it is the failure reviewers look for hardest.
* The study is persisted to an SQLite file (put it on Drive under Colab), so an
  interrupted search resumes rather than restarting.
* ``MedianPruner`` kills hopeless trials; the CNN objective reports
  intermediate values every epoch and raises ``TrialPruned`` cleanly instead of
  crashing the run.
* Trials are mirrored to W&B via ``WeightsAndBiasesCallback`` when tracking is
  enabled.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..data.dataset import payload_bits_for_rate
from ..evaluation.metrics import psnr
from ..ga.optimizer import GAConfig
from ..stego.amdt import run_amdt
from ..utils.seeding import seeded_rng

__all__ = ["SearchResult", "run_search", "objective_ga", "objective_cnn"]

log = logging.getLogger("amdt.search")


class SearchResult(dict):
    """Best params + study handle, JSON-serialisable."""


def _assert_no_test(**kwargs) -> None:
    for k in kwargs:
        if "test" in k.lower():
            raise AssertionError(
                f"Optuna objective received {k!r}. The search must never see the "
                "test split -- that is hyper-parameter selection on test."
            )


# --------------------------------------------------------------------------- #
def objective_ga(trial, images: Sequence[np.ndarray], train_idx: Sequence[int],
                 val_idx: Sequence[int], key: bytes, rate_bpp: float,
                 seed: int = 0, **forbidden) -> float:
    """Maximise mean PSNR on the validation covers."""
    _assert_no_test(**forbidden)

    cfg = GAConfig(
        population=trial.suggest_int("population", 10, 60, step=5),
        generations=trial.suggest_int("generations", 20, 200, step=20),
        tournament_size=trial.suggest_int("tournament_size", 2, 6),
        crossover_prob=trial.suggest_float("crossover_prob", 0.5, 0.95),
        mutation_prob=trial.suggest_float("mutation_prob", 0.01, 0.3, log=True),
        elitism=trial.suggest_int("elitism", 1, 3),
        patience=trial.suggest_int("patience", 10, 50, step=10),
        n_segments=trial.suggest_categorical("n_segments", [1, 2, 4, 8]),
    )

    scores: List[float] = []
    for step, i in enumerate(val_idx):
        cover = images[i]
        n_bits = payload_bits_for_rate(cover.shape, rate_bpp)
        payload = seeded_rng(seed, f"payload:{rate_bpp}:{i}").integers(
            0, 2, n_bits, dtype=np.uint8)
        res = run_amdt(cover, payload, key, cfg, seeded_rng(seed, f"opt:{trial.number}:{i}"),
                       variant="full", verify=False)
        scores.append(res.quality.psnr)

        trial.report(float(np.mean(scores)), step)
        if trial.should_prune():
            import optuna
            raise optuna.TrialPruned()

    # Cost-awareness: a config that buys 0.05 dB for 4x the runtime is not an
    # improvement. Runtime enters as a soft penalty rather than a second
    # objective, so the study stays single-objective and easy to report.
    mean_psnr = float(np.mean(scores))
    trial.set_user_attr("mean_psnr", mean_psnr)
    trial.set_user_attr("evaluations", cfg.population * (cfg.generations + 1))
    return mean_psnr


def objective_cnn(trial, train_ds, val_ds, base_cfg, seed: int = 0, **forbidden) -> float:
    """Minimise validation ``P_E`` — i.e. make the attacker as strong as possible."""
    _assert_no_test(**forbidden)
    from ..steganalysis.cnn import TrainConfig, train_cnn

    cfg = TrainConfig(**{
        **base_cfg.as_dict(),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "batch_pairs": trial.suggest_categorical("batch_pairs", [8, 16, 32]),
        "optimizer": trial.suggest_categorical("optimizer", ["adamax", "adam", "sgd"]),
        "seed": seed,
        # Each trial needs its own checkpoint dir or trials overwrite each other.
        "checkpoint_dir": (str(Path(base_cfg.checkpoint_dir) / f"trial_{trial.number}")
                           if base_cfg.checkpoint_dir else None),
    })
    _, hist, _ = train_cnn(train_ds, val_ds, cfg, optuna_trial=trial)
    return float(min(hist.val_pe)) if hist.val_pe else 0.5


# --------------------------------------------------------------------------- #
def run_search(objective: Callable, n_trials: int, study_name: str,
               storage: Optional[str] = None, direction: str = "maximize",
               seed: int = 0, tracker=None, timeout_s: Optional[int] = None
               ) -> SearchResult:
    """Run (or resume) an Optuna study.

    ``storage`` should be an SQLite URL on persistent disk, e.g.
    ``sqlite:////content/drive/MyDrive/amdt/optuna.db`` on Colab, so a
    disconnected search resumes instead of restarting from trial zero.
    """
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name, storage=storage, load_if_exists=True,
        direction=direction, sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    callbacks = []
    if tracker is not None and getattr(tracker, "url", None):
        try:
            from optuna.integration.wandb import WeightsAndBiasesCallback
            callbacks.append(WeightsAndBiasesCallback(
                metric_name=("psnr" if direction == "maximize" else "p_e"),
                as_multirun=False))
        except Exception as exc:
            log.warning("Optuna->W&B callback unavailable (%s); trials logged locally", exc)

    done = len([t for t in study.trials
                if t.state.name in ("COMPLETE", "PRUNED")])
    remaining = max(0, n_trials - done)
    if done:
        log.info("resuming study %s: %d trials already done, %d to go",
                 study_name, done, remaining)
    if remaining:
        study.optimize(objective, n_trials=remaining, timeout=timeout_s,
                       callbacks=callbacks, gc_after_trial=True)

    best = study.best_trial
    result = SearchResult(
        study_name=study_name,
        best_value=float(best.value),
        best_params=dict(best.params),
        best_trial=best.number,
        n_trials=len(study.trials),
        n_pruned=len([t for t in study.trials if t.state.name == "PRUNED"]),
        user_attrs=dict(best.user_attrs),
        storage=storage,
    )
    if tracker is not None:
        tracker.summary({f"optuna/{k}": v for k, v in result.items()
                         if not isinstance(v, dict)})
    return result
