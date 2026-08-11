"""Run registry and migration for dual-target supervision (§12).

One supervisor process manages both local and hosted runs.  The decision logic
is shared; only the delivery mechanism differs:

* ``local``  — the supervisor is a process on the same machine as training, so
  it patches the config and relaunches directly.  No branch, no polling stub.
* ``hosted`` — the supervisor cannot reach into the VM, so it pushes to
  ``agent/patches`` and the in-notebook stub applies between epochs.

The registry is keyed by **experiment ID**, not by process, because the
revision budget is shared across targets: three rounds for the experiment,
wherever those rounds happen to run.  Tracking the counter against the process
means a migration silently resets it, and the cap stops meaning anything.

Migration
---------
A run may move local → hosted when the local machine is under sustained
pressure.  "Sustained" and "pressure" are numbers here
(:class:`MigrationPolicy`), never a subjective judgement: load per core or
memory-used fraction above a threshold for N consecutive samples.

Migration transfers the full checkpoint including RNG state, so **metrics
remain valid**.  Timings do not: a run that spent part of its life on one
machine and the rest on another has a wall-clock figure that describes neither.
Migrated runs are marked ``timing_invalid: true`` in the provenance record and
their latency and epoch-time are excluded from results tables.  Metrics from a
migrated run are reportable; its speed numbers are not.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["RunRecord", "RunRegistry", "MigrationPolicy", "PressureMonitor",
           "migrate_checkpoint"]

log = logging.getLogger("amdt.runs")


# --------------------------------------------------------------------------- #
@dataclass
class RunRecord:
    """One experiment, wherever it currently runs."""

    experiment_id: str
    target: str = "local"                 # local | hosted
    status: str = "active"                # active | migrated | finished | halted
    rounds_used: int = 0                  # shared across targets
    checkpoint_path: Optional[str] = None
    wandb_run: Optional[str] = None
    timing_valid: bool = True
    timing_invalid_reason: Optional[str] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    created: str = field(default_factory=
                         lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    updated: str = field(default_factory=
                         lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def touch(self) -> None:
        self.updated = datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunRegistry:
    """Persistent map ``experiment_id -> RunRecord``.

    Persisted to JSON so the counter survives the supervisor process itself
    being restarted — the budget belongs to the experiment, not to any process
    that happens to be watching it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.runs: Dict[str, RunRecord] = {}
        if self.path.exists():
            for k, v in json.loads(self.path.read_text()).items():
                self.runs[k] = RunRecord(**v)

    # -- persistence ------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v.as_dict() for k, v in self.runs.items()},
                                  indent=2, default=str))
        tmp.replace(self.path)

    # -- access -----------------------------------------------------------
    def register(self, experiment_id: str, target: str = "local", **kw) -> RunRecord:
        rec = self.runs.get(experiment_id)
        if rec is None:
            rec = RunRecord(experiment_id=experiment_id, target=target, **kw)
            self.runs[experiment_id] = rec
            log.info("registered experiment %s on %s", experiment_id, target)
        else:
            # Re-registering after a restart must not reset the budget.
            rec.target = target
            for k, v in kw.items():
                if v is not None:
                    setattr(rec, k, v)
            rec.touch()
            log.info("re-attached experiment %s on %s (%d/%s rounds already used)",
                     experiment_id, target, rec.rounds_used, "N")
        self.save()
        return rec

    def get(self, experiment_id: str) -> Optional[RunRecord]:
        return self.runs.get(experiment_id)

    def active(self) -> List[RunRecord]:
        return [r for r in self.runs.values() if r.status == "active"]

    def budget_remaining(self, experiment_id: str, max_rounds: int) -> int:
        rec = self.runs.get(experiment_id)
        return max_rounds - (rec.rounds_used if rec else 0)

    def consume_round(self, experiment_id: str, max_rounds: int) -> bool:
        """Spend one revision round. ``False`` when the shared budget is gone."""
        rec = self.runs.get(experiment_id) or self.register(experiment_id)
        if rec.rounds_used >= max_rounds:
            log.warning("experiment %s has spent its %d-round budget across all "
                        "targets; halting and notifying", experiment_id, max_rounds)
            rec.status = "halted"
            rec.touch()
            self.save()
            return False
        rec.rounds_used += 1
        rec.touch()
        self.save()
        return True

    # -- migration --------------------------------------------------------
    def mark_migrated(self, experiment_id: str, to_target: str, reason: str,
                      new_checkpoint: Optional[str] = None) -> RunRecord:
        rec = self.runs.get(experiment_id) or self.register(experiment_id)
        rec.history.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "from": rec.target, "to": to_target, "reason": reason,
                            "rounds_used_at_migration": rec.rounds_used})
        rec.target = to_target
        if new_checkpoint:
            rec.checkpoint_path = new_checkpoint
        # Metrics survive a migration; wall-clock does not.
        rec.timing_valid = False
        rec.timing_invalid_reason = (
            f"run migrated {rec.history[-1]['from']} -> {to_target}: wall-clock "
            "spans two machines and describes neither")
        rec.touch()
        self.save()
        log.warning("experiment %s migrated to %s (%s). Metrics stay valid; "
                    "latency and epoch-time are excluded from results tables.",
                    experiment_id, to_target, reason)
        return rec


# --------------------------------------------------------------------------- #
@dataclass
class MigrationPolicy:
    """Numeric migration trigger. Never a subjective 'the machine is busy'."""

    load_per_core: float = 2.0
    memory_used_fraction: float = 0.90
    consecutive_samples: int = 5          # N consecutive breaches before firing
    sample_interval_s: int = 60
    enabled: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PressureMonitor:
    """Counts consecutive threshold breaches so a transient spike cannot migrate a run."""

    def __init__(self, policy: Optional[MigrationPolicy] = None) -> None:
        self.policy = policy or MigrationPolicy()
        self.streak = 0
        self.last: Dict[str, float] = {}

    def sample(self, reading: Optional[Dict[str, float]] = None) -> Optional[str]:
        """Take one reading. Returns a reason string once the streak is met."""
        from ..utils.execution import memory_pressure

        r = reading if reading is not None else memory_pressure()
        self.last = r
        if not self.policy.enabled:
            return None

        breaches = []
        if r.get("load_per_core", 0.0) > self.policy.load_per_core:
            breaches.append(f"load/core {r['load_per_core']:.2f} > "
                            f"{self.policy.load_per_core}")
        if r.get("memory_used_fraction", 0.0) > self.policy.memory_used_fraction:
            breaches.append(f"memory used {r['memory_used_fraction']:.0%} > "
                            f"{self.policy.memory_used_fraction:.0%}")

        if not breaches:
            self.streak = 0
            return None

        self.streak += 1
        if self.streak < self.policy.consecutive_samples:
            log.info("pressure breach %d/%d: %s", self.streak,
                     self.policy.consecutive_samples, "; ".join(breaches))
            return None
        return (f"{'; '.join(breaches)} for {self.streak} consecutive samples "
                f"({self.streak * self.policy.sample_interval_s}s)")


# --------------------------------------------------------------------------- #
def migrate_checkpoint(src: str | Path, dst: str | Path) -> Path:
    """Copy a full-state checkpoint to the new target.

    Verifies the RNG state travelled: without it the resumed run is not
    bit-identical, and "migrated" would quietly mean "restarted with different
    randomness", which invalidates the metrics too — not just the timings.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"checkpoint not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    shutil.copy2(src, tmp)
    tmp.replace(dst)

    try:
        import torch
        st = torch.load(dst, map_location="cpu", weights_only=False)
        missing = [k for k in ("model_state_dict", "optimizer_state_dict",
                               "rng_state", "numpy_rng_state", "epoch")
                   if k not in st]
        if missing:
            raise ValueError(
                f"migrated checkpoint is missing {missing}; a resume from it "
                "would not be bit-identical, so the metrics would be invalid too"
            )
    except ImportError:
        log.warning("torch unavailable; could not verify the migrated checkpoint "
                    "carries RNG state")
    log.info("checkpoint migrated: %s -> %s", src, dst)
    return dst
