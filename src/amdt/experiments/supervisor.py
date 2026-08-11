"""Autonomous training supervision (§12).

Architecture
------------
The supervisor runs **outside** the Colab runtime.  It polls W&B for the live
run, decides whether a trip condition has fired, and — if so — pushes a patch to
the ``agent/patches`` branch.  A thin in-runtime stub (:class:`PatchStub`) polls
that branch between epochs and applies whatever it finds.

    watcher decides  ->  git branch  ->  stub applies

Nothing here reaches into the VM.  Anything that claims to restart a Colab
runtime from outside is fiction, and code written on that assumption fails
silently at 3 a.m.

Split isolation is structural, not advisory
-------------------------------------------
:class:`Supervisor` is constructed with validation metric names only, and
:meth:`Supervisor.observe` raises on any key containing ``test``.  The stub is
handed ``val_pe`` and ``train_loss`` by the trainer and never sees a test
loader.  An agent that iterates against a metric is a search procedure, and any
split it can observe becomes part of fitting rather than evaluation.

Budget
------
Three rounds, then halt and notify.  The counter lives **in the checkpoint**
(``supervisor_rounds_used``), not in memory, because a Colab reconnect would
otherwise reset it and the cap would mean nothing.

Every intervention appends one JSON line to ``agent_interventions.jsonl``, which
is committed with the run.  That file is the answer to "how did you arrive at
this architecture", and it stays accurate even when it is unflattering.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["TripConfig", "Intervention", "Supervisor", "PatchStub",
           "MECHANICAL_KEYS", "METHOD_ALTERING_KEYS"]

log = logging.getLogger("amdt.supervisor")

#: Applied silently.
MECHANICAL_KEYS = ("lr", "batch_pairs", "weight_decay", "optimizer", "scheduler",
                   "grad_clip", "momentum")
#: Applied automatically but tagged, and surfaced before the methods section is written.
METHOD_ALTERING_KEYS = ("loss", "architecture", "model", "tlu_threshold", "freeze_front")


@dataclass
class TripConfig:
    """Every trip condition is a number, not a judgement call."""

    # divergence
    loss_median_multiple: float = 10.0
    # plateau
    min_delta: float = 1e-3
    patience: int = 8
    # overfitting
    gap_threshold: float = 0.25          # (train - val) relative to train
    # underperformance vs a named baseline
    baseline_metric: Optional[float] = None
    warmup_epochs: int = 10
    # budget and thrash suppression
    max_rounds: int = 3
    min_epochs_between: int = 5
    poll_interval_s: int = 60

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Intervention:
    ts: str
    round: int
    trigger: str
    rule: str
    category: str
    files: List[str]
    diff_summary: str
    val_before: Optional[float] = None
    val_after: Optional[float] = None
    wandb_run: Optional[str] = None
    commit: Optional[str] = None
    patch: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(obj, default=str) + "\n")


# --------------------------------------------------------------------------- #
class Supervisor:
    """Out-of-runtime watcher. Decides; never applies."""

    def __init__(self, run_dir: str | Path, cfg: Optional[TripConfig] = None,
                 repo=None, wandb_run_path: Optional[str] = None,
                 metric: str = "val/p_e", lower_is_better: bool = True,
                 experiment_id: Optional[str] = None, registry=None,
                 target: str = "hosted", relaunch_fn=None) -> None:
        """
        Parameters
        ----------
        experiment_id, registry : when given, the revision budget is tracked
            against the *experiment* in a :class:`RunRegistry` rather than
            against this process, so it is shared across local and hosted
            targets and survives a migration or a supervisor restart.
        target : ``"local"`` or ``"hosted"``.  Only the delivery mechanism
            differs -- locally the supervisor patches and relaunches directly;
            on a hosted runtime it pushes to ``agent/patches`` for the stub.
        relaunch_fn : local-target callback invoked with the patch dict after it
            is written.  Absent, the patch is written and left for the next launch.
        """
        self.run_dir = Path(run_dir)
        self.cfg = cfg or TripConfig()
        self.repo = repo
        self.wandb_run_path = wandb_run_path
        self.metric = metric
        self.lower_is_better = lower_is_better
        self.experiment_id = experiment_id
        self.registry = registry
        self.target = target
        self.relaunch_fn = relaunch_fn

        self.log_path = self.run_dir / "agent_interventions.jsonl"
        self.patch_path = self.run_dir / "agent_patch.json"
        self.last_intervention_epoch = -10**9
        self._tried: set[str] = set()
        self._history: List[Dict[str, float]] = []

        # The counter lives in the registry when there is one: three rounds for
        # the experiment, wherever those rounds happen to run.
        if self.registry is not None and self.experiment_id:
            rec = self.registry.register(self.experiment_id, target=target)
            self._rounds_used = rec.rounds_used
        else:
            self._rounds_used = 0

    # -- budget (shared across targets) -----------------------------------
    @property
    def rounds_used(self) -> int:
        if self.registry is not None and self.experiment_id:
            rec = self.registry.get(self.experiment_id)
            return rec.rounds_used if rec else self._rounds_used
        return self._rounds_used

    @rounds_used.setter
    def rounds_used(self, value: int) -> None:
        self._rounds_used = int(value)

    def _consume_round(self) -> bool:
        if self.registry is not None and self.experiment_id:
            return self.registry.consume_round(self.experiment_id, self.cfg.max_rounds)
        if self._rounds_used >= self.cfg.max_rounds:
            return False
        self._rounds_used += 1
        return True

    # -- observation ------------------------------------------------------
    def observe(self, epoch: int, **metrics: float) -> None:
        """Record one epoch of *validation* metrics."""
        for k in metrics:
            if "test" in k.lower():
                raise AssertionError(
                    f"supervisor received {k!r}; it must never observe the test split"
                )
        self._history.append({"epoch": float(epoch), **metrics})

    # -- trip conditions --------------------------------------------------
    def check(self) -> Optional[Dict[str, str]]:
        """Return ``{trigger, rule}`` for the first condition that fires."""
        if not self._history:
            return None
        h = self._history
        last = h[-1]
        epoch = int(last["epoch"])
        losses = [r["train_loss"] for r in h if "train_loss" in r]
        vals = [r["val"] for r in h if "val" in r]

        # 1. divergence -- fire immediately
        if losses:
            cur = losses[-1]
            if cur != cur or cur in (float("inf"), float("-inf")):
                return {"trigger": "divergence", "rule": "train loss is NaN or infinite"}
            if len(losses) >= 5:
                med = sorted(losses)[len(losses) // 2]
                if med > 0 and cur > self.cfg.loss_median_multiple * med:
                    return {"trigger": "divergence",
                            "rule": f"train loss > {self.cfg.loss_median_multiple}x "
                                    f"running median ({cur:.4g} vs {med:.4g})"}

        # 2. plateau
        if len(vals) > self.cfg.patience:
            window = vals[-(self.cfg.patience + 1):]
            best_before = (min if self.lower_is_better else max)(window[:-1])
            gain = (best_before - min(window)) if self.lower_is_better \
                else (max(window) - best_before)
            if gain < self.cfg.min_delta:
                return {"trigger": "plateau",
                        "rule": f"no {self.metric} gain > {self.cfg.min_delta} "
                                f"for {self.cfg.patience} epochs"}

        # 3. overfitting
        if "train_metric" in last and "val" in last:
            tr, va = last["train_metric"], last["val"]
            if tr > 0 and abs(tr - va) / abs(tr) > self.cfg.gap_threshold:
                return {"trigger": "overfitting",
                        "rule": f"train-val gap {(abs(tr-va)/abs(tr)):.3f} > "
                                f"{self.cfg.gap_threshold}"}

        # 4. underperformance vs a named baseline
        if (self.cfg.baseline_metric is not None and epoch >= self.cfg.warmup_epochs
                and vals):
            cur = vals[-1]
            worse = cur > self.cfg.baseline_metric if self.lower_is_better \
                else cur < self.cfg.baseline_metric
            if worse:
                return {"trigger": "underperformance",
                        "rule": f"{self.metric} {cur:.4f} worse than baseline "
                                f"{self.cfg.baseline_metric:.4f} after "
                                f"{self.cfg.warmup_epochs} warmup epochs"}
        return None

    # -- decision ---------------------------------------------------------
    def propose(self, trip: Dict[str, str], current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Map a trip condition to a concrete parameter patch."""
        t = trip["trigger"]
        if t == "divergence":
            return {"lr": float(current.get("lr", 1e-3)) * 0.2, "grad_clip": 1.0}
        if t == "plateau":
            # alternate between an LR cut and a regularisation bump so two
            # consecutive plateaus do not produce the same patch twice
            if self.rounds_used % 2 == 0:
                return {"lr": float(current.get("lr", 1e-3)) * 0.3}
            return {"weight_decay": float(current.get("weight_decay", 5e-4)) * 3.0}
        if t == "overfitting":
            return {"weight_decay": float(current.get("weight_decay", 5e-4)) * 5.0}
        if t == "underperformance":
            return {"lr": float(current.get("lr", 1e-3)) * 2.0,
                    "optimizer": "adamax"}
        return None

    @staticmethod
    def categorise(patch: Dict[str, Any]) -> str:
        if any(k in METHOD_ALTERING_KEYS for k in patch):
            return "method_altering"
        return "mechanical"

    @staticmethod
    def _fingerprint(patch: Dict[str, Any]) -> str:
        return hashlib.blake2b(json.dumps(patch, sort_keys=True, default=str).encode(),
                               digest_size=8).hexdigest()

    # -- action -----------------------------------------------------------
    def maybe_intervene(self, current: Dict[str, Any], seed: int = 0,
                        config_hash: str = "", run_id: Optional[str] = None
                        ) -> Optional[Intervention]:
        """Check, decide, write the patch, log it. Returns ``None`` if nothing fired."""
        if self.rounds_used >= self.cfg.max_rounds:
            log.warning("supervisor budget exhausted (%d/%d rounds) -- halting and "
                        "notifying instead of intervening again",
                        self.rounds_used, self.cfg.max_rounds)
            return None
        if not self._history:
            return None
        epoch = int(self._history[-1]["epoch"])
        if epoch - self.last_intervention_epoch < self.cfg.min_epochs_between:
            return None

        trip = self.check()
        if trip is None:
            return None
        patch = self.propose(trip, current)
        if not patch:
            return None

        fp = self._fingerprint(patch)
        if fp in self._tried:
            log.info("suppressing patch %s: equivalent to one already tried", fp)
            return None

        if not self._consume_round():
            return None
        self._tried.add(fp)
        self.last_intervention_epoch = epoch
        category = self.categorise(patch)
        iv = Intervention(
            ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            round=self.rounds_used, trigger=trip["trigger"], rule=trip["rule"],
            category=category, files=["configs/steganalysis/cnn.yaml"],
            diff_summary=", ".join(f"{k} -> {v}" for k, v in patch.items()),
            val_before=self._history[-1].get("val"), wandb_run=run_id, patch=patch,
        )

        self.patch_path.write_text(json.dumps(
            {"round": self.rounds_used, "epoch": epoch, "patch": patch,
             "category": category, "trigger": trip["trigger"],
             "target": self.target, "experiment_id": self.experiment_id}, indent=2))
        _append_jsonl(self.log_path, iv.as_dict())

        # Shared decision logic; only the delivery channel depends on the target.
        if self.target == "local":
            # Same machine as training: patch and relaunch directly. No branch,
            # no polling stub.
            if self.relaunch_fn is not None:
                self.relaunch_fn(patch)
                log.info("local target: patch applied and run relaunched")
            else:
                log.info("local target: patch written to %s; no relaunch callback "
                         "provided, it will be picked up on the next launch",
                         self.patch_path)
        elif self.repo is not None:
            from ..utils.repo import provenance_message
            msg = provenance_message(
                f"agent: {trip['trigger']} round {self.rounds_used}",
                seed=seed, config_hash=config_hash, run_id=run_id,
                trigger=trip["trigger"],
                agent_revision=f"{self.rounds_used}/{self.cfg.max_rounds}",
            )
            self.repo.push_patch_branch(self.patch_path, message=msg)

        log.warning("supervisor round %d/%d fired (%s) on %s: %s",
                    self.rounds_used, self.cfg.max_rounds, trip["trigger"],
                    self.target, iv.diff_summary)
        if category == "method_altering":
            log.warning("this patch is METHOD-ALTERING; the manuscript's methods "
                        "section must be reconciled with it before submission")
        return iv

    # -- reporting --------------------------------------------------------
    def interventions(self) -> List[Dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(l) for l in self.log_path.read_text().splitlines() if l.strip()]

    def method_altering_summary(self) -> List[Dict[str, Any]]:
        """Consolidated diff of everything that changed the method, for §12 review.

        Call this before writing the methods section: the manuscript describes
        what the author designed, and after autonomous edits it may not describe
        what actually ran.
        """
        return [i for i in self.interventions() if i.get("category") == "method_altering"]

    def watch(self, poll_fn, current_fn, max_polls: Optional[int] = None,
              seed: int = 0, config_hash: str = "", run_id: Optional[str] = None,
              pressure: Optional["PressureMonitor"] = None,
              migrate_fn=None) -> List[Intervention]:
        """Polling loop for out-of-runtime use.

        ``poll_fn()`` returns ``{"epoch": int, "train_loss": float, "val": float}``
        for the latest logged epoch (typically read from the W&B API), or ``None``
        when nothing new has arrived.

        ``pressure`` / ``migrate_fn``: when both are given, sustained local
        pressure (a numeric threshold breached for N consecutive samples)
        triggers ``migrate_fn(reason)`` and the run is marked timing-invalid in
        the registry.  The shared budget is *not* reset by the move.
        """
        out: List[Intervention] = []
        polls = 0
        while max_polls is None or polls < max_polls:
            polls += 1

            if pressure is not None and migrate_fn is not None and self.target == "local":
                reason = pressure.sample()
                if reason:
                    migrate_fn(reason)
                    if self.registry is not None and self.experiment_id:
                        self.registry.mark_migrated(self.experiment_id, "hosted", reason)
                    self.target = "hosted"

            rec = poll_fn()
            if rec:
                self.observe(rec.pop("epoch"), **rec)
                iv = self.maybe_intervene(current_fn(), seed, config_hash, run_id)
                if iv:
                    out.append(iv)
                if self.rounds_used >= self.cfg.max_rounds:
                    log.warning("shared budget spent (%d/%d rounds across all "
                                "targets); halting", self.rounds_used,
                                self.cfg.max_rounds)
                    break
            if max_polls is None:
                time.sleep(self.cfg.poll_interval_s)
        return out


def wandb_poller(run_path: str, metric: str = "val/p_e"):
    """``poll_fn`` backed by the W&B public API. ``run_path`` = 'entity/project/run_id'."""
    import wandb

    api = wandb.Api()
    seen = {"epoch": -1}

    def poll() -> Optional[Dict[str, float]]:
        run = api.run(run_path)
        hist = run.history(keys=[metric, "train/loss"], pandas=False)
        if not hist:
            return None
        last = hist[-1]
        epoch = int(last.get("_step", len(hist) - 1))
        if epoch <= seen["epoch"]:
            return None
        seen["epoch"] = epoch
        return {"epoch": epoch, "val": float(last.get(metric, float("nan"))),
                "train_loss": float(last.get("train/loss", float("nan")))}

    return poll


# --------------------------------------------------------------------------- #
class PatchStub:
    """In-runtime applier. Polls the patch branch between epochs and applies.

    Deliberately dumb: it does not decide anything, it only applies what the
    watcher published.  Keeping the decision outside the VM is what makes the
    budget and the intervention log trustworthy.
    """

    def __init__(self, patch_dir: str | Path, run_dir: str | Path,
                 max_rounds: int = 3, repo=None, branch: str = "agent/patches") -> None:
        self.patch_dir = Path(patch_dir)
        self.run_dir = Path(run_dir)
        self.max_rounds = max_rounds
        self.repo = repo
        self.branch = branch
        self.rounds_used = 0
        self.applied: List[Dict[str, Any]] = []
        self.log_path = self.run_dir / "agent_interventions.jsonl"

    def _fetch(self) -> Optional[Dict[str, Any]]:
        if self.repo is not None:
            self.repo._run("fetch", "origin", self.branch, check=False)
            r = self.repo._run("show", f"origin/{self.branch}:agent_patch.json",
                               check=False)
            if r.returncode == 0 and r.stdout.strip():
                return json.loads(r.stdout)
        p = self.patch_dir / "agent_patch.json"
        if p.exists():
            return json.loads(p.read_text())
        return None

    def poll_and_apply(self, epoch: int, val_pe: float, train_loss: float,
                       model=None, optimizer=None) -> Optional[Dict[str, Any]]:
        """Apply a pending patch, if any. Returns the applied patch or ``None``."""
        if self.rounds_used >= self.max_rounds:
            return None
        rec = self._fetch()
        if not rec:
            return None
        rnd = int(rec.get("round", 0))
        if rnd <= self.rounds_used:
            return None                      # already applied

        patch = rec.get("patch", {})
        applied: Dict[str, Any] = {}
        if optimizer is not None and "lr" in patch:
            for g in optimizer.param_groups:
                g["lr"] = float(patch["lr"])
            applied["lr"] = float(patch["lr"])
        if optimizer is not None and "weight_decay" in patch:
            for g in optimizer.param_groups:
                g["weight_decay"] = float(patch["weight_decay"])
            applied["weight_decay"] = float(patch["weight_decay"])
        # Anything the stub cannot apply in-place (optimizer swap, architecture)
        # is recorded and left for the next launch, rather than half-applied.
        deferred = {k: v for k, v in patch.items() if k not in applied}

        self.rounds_used = rnd
        entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "round": rnd, "epoch": epoch, "applied": applied,
                 "deferred": deferred, "val_after": val_pe,
                 "train_loss": train_loss, "source": "stub"}
        self.applied.append(entry)
        _append_jsonl(self.log_path, entry)
        log.warning("stub applied agent patch round %d at epoch %d: %s%s",
                    rnd, epoch, applied,
                    f" (deferred: {deferred})" if deferred else "")
        return entry
