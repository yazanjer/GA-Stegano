"""Experiment tracking (Weights & Biases) with mandatory provenance.

A metric without provenance cannot be defended, so :class:`Tracker` refuses to
log anything until it has the config, the git commit and the seed.  If W&B is
unavailable or disabled the tracker degrades to a JSONL file in the run
directory -- the *record* is never optional, only the dashboard is.

Design notes
------------
* ``wandb`` is imported lazily.  A missing dependency must not break the
  classical half of the pipeline, which has no training loop to watch.
* ``run_id`` is deterministic from ``(experiment, seed, config hash)``, so a
  resumed Colab session reattaches to the same W&B run instead of creating a
  duplicate that splits the loss curve across two charts.
* ``config_hash`` is the short BLAKE2b of the resolved config.  It goes into
  every commit message (see :mod:`amdt.utils.repo`) and every W&B run, which is
  what lets a number in the manuscript be traced to an exact state.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = ["config_hash", "Tracker", "NullTracker", "build_tracker", "preflight"]

log = logging.getLogger("amdt.tracking")


def config_hash(cfg: Any, length: int = 7) -> str:
    """Short, stable hash of a resolved config.

    Keys are sorted before hashing so that two configs differing only in
    declaration order hash identically -- otherwise a harmless YAML reshuffle
    would invalidate every commit-message ``config_hash`` in the history.
    """
    obj = cfg
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            obj = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        pass
    payload = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()[:length]


@dataclass
class NullTracker:
    """Records to JSONL. Used when W&B is off, unavailable, or in tests."""

    run_dir: Path
    # None, not "local": an absent tracking run should leave the field out of
    # commit messages rather than stamping them with a fake run id.
    run_id: Optional[str] = None
    config_hash: str = ""
    _fh: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.run_dir / "metrics.jsonl"

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        rec = dict(data)
        if step is not None:
            rec["_step"] = step
        with self._path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def summary(self, data: Dict[str, Any]) -> None:
        (self.run_dir / "summary.json").write_text(json.dumps(data, indent=2, default=str))

    def log_artifact(self, path: str | Path, name: str, type_: str = "results") -> None:
        self.log({"artifact": str(path), "artifact_name": name, "artifact_type": type_})

    def finish(self) -> None:
        pass

    @property
    def url(self) -> Optional[str]:
        return None


@dataclass
class Tracker:
    """Weights & Biases wrapper.

    Every run carries ``seed``, ``git_commit`` and ``config_hash`` in its config
    so a chart can always be traced back to code.
    """

    project: str
    run_dir: Path
    cfg: Any
    seed: int
    git_commit: Optional[str] = None
    entity: Optional[str] = None
    group: Optional[str] = None
    tags: tuple = ()
    mode: str = "online"          # online | offline | disabled
    _run: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import wandb
        from omegaconf import OmegaConf

        self.run_dir = Path(self.run_dir)
        self.config_hash = config_hash(self.cfg)
        # Deterministic id -> a resumed Colab session reattaches instead of
        # forking the loss curve into a second run.
        self.run_id = hashlib.blake2b(
            f"{self.project}:{self.seed}:{self.config_hash}".encode(), digest_size=8
        ).hexdigest()

        conf = OmegaConf.to_container(self.cfg, resolve=True) if hasattr(self.cfg, "keys") else self.cfg
        self._run = wandb.init(
            project=self.project, entity=self.entity, group=self.group,
            id=self.run_id, resume="allow", mode=self.mode,
            dir=str(self.run_dir), tags=list(self.tags),
            config={"seed": self.seed, "git_commit": self.git_commit,
                    "config_hash": self.config_hash, **(conf or {})},
        )
        log.info("wandb run %s (%s)", self.run_id, self.url)

    def log(self, data: Dict[str, Any], step: Optional[int] = None) -> None:
        self._run.log(data, step=step)

    def summary(self, data: Dict[str, Any]) -> None:
        for k, v in data.items():
            self._run.summary[k] = v

    def log_artifact(self, path: str | Path, name: str, type_: str = "results") -> None:
        import wandb
        art = wandb.Artifact(name, type=type_)
        p = Path(path)
        art.add_dir(str(p)) if p.is_dir() else art.add_file(str(p))
        self._run.log_artifact(art)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()

    @property
    def url(self) -> Optional[str]:
        try:
            return self._run.get_url()
        except Exception:
            return None


def preflight(cfg) -> Dict[str, Any]:
    """Verify W&B auth and resolve the entity, without ever revealing the key.

    Answers the two questions that otherwise surface eight hours into an
    untracked overnight run: *is a credential present*, and *which workspace
    will the runs land in*.  The entity is the slug in
    ``wandb.ai/<entity>/<project>`` — it is not your email address, and leaving
    it ``null`` resolves to the default entity of whoever is logged in.

    The key itself is never returned, logged or printed; only whether one was
    found and where it came from.
    """
    tcfg = getattr(cfg, "tracking", None)
    out: Dict[str, Any] = {
        "enabled": bool(getattr(tcfg, "enabled", False)) if tcfg else False,
        "project": str(getattr(tcfg, "project", "")) if tcfg else "",
        "configured_entity": (str(tcfg.entity) if tcfg and tcfg.entity else None),
        "wandb_installed": False,
        "credential_found": False,
        "credential_source": None,
        "resolved_entity": None,
        "default_entity": None,
        "available_entities": None,
        "entity_note": None,
        "project_exists": None,
        "ok": False,
        "hint": None,
    }

    try:
        import wandb
    except ImportError:
        out["hint"] = ("wandb is not installed. `pip install wandb`, then "
                       "`wandb login` (the key goes to ~/.netrc, never into the repo).")
        return out
    out["wandb_installed"] = True

    if os.environ.get("WANDB_API_KEY"):
        out["credential_found"], out["credential_source"] = True, "environment"
    else:
        try:
            key = wandb.api.api_key           # reads ~/.netrc; value never stored
            if key:
                out["credential_found"], out["credential_source"] = True, "netrc"
        except Exception:
            pass

    if not out["credential_found"]:
        out["hint"] = ("no W&B credential found. Locally: `wandb login`. On Colab: "
                       "add WANDB_API_KEY to Secrets and export it. Do not paste "
                       "the key into a config, a notebook cell or a chat — it is "
                       "account-wide and would have to be rotated, not just removed.")
        return out

    try:
        api = wandb.Api()

        # An organization is not a valid run target. W&B only rejects it at
        # wandb.init() time, which is 8 hours too late, so check here: the
        # viewer's team list is the set of entities that can actually hold runs.
        teams = []
        try:
            teams = list(getattr(api.viewer, "teams", []) or [])
        except Exception:
            pass
        out["available_entities"] = teams

        default = api.default_entity
        out["default_entity"] = default
        out["resolved_entity"] = out["configured_entity"] or default
        cfg_ent = out["configured_entity"]
        if cfg_ent and teams and cfg_ent not in teams:
            likely = cfg_ent[:-4] if cfg_ent.endswith("-org") else None
            out["ok"] = False
            out["hint"] = (
                f"entity {cfg_ent!r} is not one of your run-capable entities "
                f"{teams}. "
                + (f"It looks like an organization; the team entity is usually "
                   f"the same name without the '-org' suffix, i.e. {likely!r}. "
                   if likely else "")
                + "W&B refuses runs addressed to an organization.")
            return out

        if out["configured_entity"] and default and out["configured_entity"] != default:
            # Not an error -- pinning a team entity while your personal one is
            # the default is the normal case -- but worth saying out loud, since
            # the alternative is discovering it when a run is missing.
            out["entity_note"] = (
                f"runs will go to the pinned team entity {out['configured_entity']!r}, "
                f"not your default {default!r}")
        try:
            api.project(name=out["project"], entity=out["resolved_entity"])
            out["project_exists"] = True
        except Exception:
            out["project_exists"] = False
            out["hint"] = (f"project {out['resolved_entity']}/{out['project']} does "
                           "not exist yet; W&B will create it on the first run. "
                           "Check the spelling now — a typo silently starts a "
                           "second, half-full project.")
        out["ok"] = True
    except Exception as exc:
        out["hint"] = f"credential present but the W&B API rejected it: {exc}"
    return out


def build_tracker(cfg, run_dir: str | Path, seed: int, git_commit: Optional[str] = None):
    """Return a :class:`Tracker`, falling back to :class:`NullTracker`.

    The fallback is logged loudly: a silent downgrade to no tracking is how a
    run ends up with no provenance at all.
    """
    tcfg = getattr(cfg, "tracking", None)
    if tcfg is None or not bool(getattr(tcfg, "enabled", False)):
        return NullTracker(Path(run_dir), config_hash=config_hash(cfg))
    try:
        return Tracker(
            project=str(tcfg.project), run_dir=Path(run_dir), cfg=cfg, seed=seed,
            git_commit=git_commit, entity=(str(tcfg.entity) if tcfg.entity else None),
            group=(str(tcfg.group) if tcfg.group else None),
            tags=tuple(tcfg.tags), mode=str(tcfg.mode),
        )
    except ImportError:
        log.warning("tracking.enabled=true but wandb is not installed; "
                    "falling back to metrics.jsonl. `pip install wandb` to enable.")
    except Exception as exc:  # network, auth, quota
        log.warning("W&B init failed (%s); falling back to metrics.jsonl", exc)
    return NullTracker(Path(run_dir), config_hash=config_hash(cfg))
