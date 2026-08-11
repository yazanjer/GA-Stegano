"""Automatic run-folder mirroring to Google Drive (§15).

Every run's output folder is mirrored as part of the run — no per-run prompt,
and **not** deferred to the end. A crashed run's logs are the ones most worth
having, so the mirror runs incrementally on a background thread from the moment
the run directory exists.

Two environments, two mechanisms
--------------------------------
*Hosted (Colab).* Mount once at notebook start and treat the mounted path as
the sync target. The single OAuth approval per session is Google's consent
step, not something to engineer around; after it, syncing is silent.

*Local.* Do **not** use the Drive API. Google Drive for Desktop already exposes
a synced folder — on macOS under
``~/Library/CloudStorage/GoogleDrive-<account>/My Drive`` — and writing into it
*is* the integration. :func:`resolve_drive_root` finds it at startup and raises
if it is absent, because a sync that quietly does nothing is worse than no sync.

Why nothing is written directly into the synced folder
------------------------------------------------------
Drive for Desktop uploads files as they change. A checkpoint still being
written gets partially uploaded; a log being appended uploads mid-line. So the
local run directory is the staging area, and only complete snapshots cross:
copy to a temp file *inside the destination directory*, then ``os.replace``.
The temp lives in the destination so the rename is same-filesystem and
therefore atomic — ``os.replace`` across filesystems raises ``EXDEV``, which is
the bug this ordering avoids.

Retention
---------
Unbounded mirroring fills a Drive quota fast once checkpoints are included.
:meth:`DriveSync.prune` keeps the last N runs *plus* every run referenced by a
commit message. A run whose ID appears in the provenance record of a reported
result is evidence and is never pruned.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = ["SyncConfig", "DriveSync", "resolve_drive_root", "DEFAULT_EXCLUDES",
           "DriveNotFound"]

log = logging.getLogger("amdt.sync")

#: Stated here and echoed in configs/sync/drive.yaml so the exclusion list is
#: visible rather than buried. Large, re-derivable, or versioned elsewhere.
DEFAULT_EXCLUDES: tuple = (
    "*.tmp", "*.part", "*.lock",
    ".git/*", ".git", "wandb/*", "wandb",           # versioned / cached elsewhere
    "stego/*",                                       # re-derivable stego images
    "data/*", "*.pgm", "*.zip", "*.tar.bz2",        # raw datasets
    ".venv/*", "__pycache__/*", "*.pyc",
)


class DriveNotFound(RuntimeError):
    """Raised when sync is enabled but no Drive target can be resolved."""


# --------------------------------------------------------------------------- #
def resolve_drive_root(explicit: Optional[str] = None,
                       subdir: str = "amdt-runs") -> Path:
    """Locate the Drive sync target. Fails loudly rather than silently no-op'ing.

    Order: explicit config path, Colab mount, Drive for Desktop (macOS), then
    the Linux/Windows conventional locations.
    """
    if explicit:
        p = Path(explicit).expanduser()
        if not p.parent.exists():
            raise DriveNotFound(
                f"sync.root={p} but its parent does not exist. Either create it, "
                "point sync.root somewhere real, or set sync.enabled=false — a "
                "sync that quietly does nothing is worse than no sync."
            )
        p.mkdir(parents=True, exist_ok=True)
        return p

    # Hosted runtime: mounted once at notebook start.
    colab = Path("/content/drive/MyDrive")
    if colab.exists():
        return _ensure(colab / subdir)
    if "google.colab" in sys.modules:
        raise DriveNotFound(
            "running on Colab but /content/drive/MyDrive is not mounted. Mount "
            "once at the top of the notebook:\n"
            "    from google.colab import drive; drive.mount('/content/drive')\n"
            "One OAuth approval per session is Google's consent step; after it "
            "syncing is automatic."
        )

    # Local: Google Drive for Desktop exposes a plain folder. No API involved.
    candidates: List[Path] = []
    home = Path.home()
    if sys.platform == "darwin":
        candidates += sorted((home / "Library" / "CloudStorage").glob("GoogleDrive-*/My Drive"))
        candidates.append(home / "Google Drive" / "My Drive")
        candidates.append(home / "Google Drive")
    else:
        candidates += [home / "GoogleDrive", home / "Google Drive",
                       home / "gdrive", Path("G:/My Drive")]
    for c in candidates:
        if c.exists():
            return _ensure(c / subdir)

    raise DriveNotFound(
        "no Google Drive folder found. Install Google Drive for Desktop and "
        "sign in (macOS puts it under ~/Library/CloudStorage/GoogleDrive-<account>/"
        "My Drive), or set sync.root explicitly, or sync.enabled=false. "
        "Do not fall back to the Drive API — the synced folder is the integration."
    )


def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
@dataclass
class SyncConfig:
    enabled: bool = False
    root: Optional[str] = None
    subdir: str = "amdt-runs"
    interval_s: int = 60
    excludes: Sequence[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    include_checkpoints: bool = True
    max_file_mb: float = 512.0
    keep_last_n: int = 20
    prune: bool = True

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["excludes"] = list(self.excludes)
        return d


class DriveSync:
    """Incremental one-way mirror: local run directory -> Drive folder."""

    def __init__(self, run_dir: str | Path, cfg: Optional[SyncConfig] = None,
                 run_id: Optional[str] = None) -> None:
        self.cfg = cfg or SyncConfig()
        self.run_dir = Path(run_dir)
        self.run_id = run_id or self.run_dir.name
        self.root: Optional[Path] = None
        self.dest: Optional[Path] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._seen: Dict[str, tuple] = {}      # relpath -> (mtime, size)
        self.files_synced = 0
        self.bytes_synced = 0
        self.errors: List[str] = []

        if not self.cfg.enabled:
            log.info("drive sync disabled")
            return
        self.root = resolve_drive_root(self.cfg.root, self.cfg.subdir)
        self.dest = _ensure(self.root / self.run_id)
        log.info("drive sync target: %s", self.dest)

    # -- filtering --------------------------------------------------------
    def _excluded(self, rel: Path) -> bool:
        s = str(rel)
        for pat in self.cfg.excludes:
            if fnmatch.fnmatch(s, pat) or fnmatch.fnmatch(rel.name, pat):
                return True
            if any(fnmatch.fnmatch(part, pat.rstrip("/*")) for part in rel.parts[:-1]
                   if pat.endswith("/*")):
                return True
        if not self.cfg.include_checkpoints and rel.suffix in (".pt", ".ckpt"):
            return True
        return False

    # -- the one operation that matters -----------------------------------
    def _copy_atomic(self, src: Path, dst: Path) -> None:
        """Stage inside the destination directory, then rename.

        The temp file lives beside the target so the rename is same-filesystem
        (``os.replace`` across filesystems raises ``EXDEV``).  Drive for Desktop
        therefore only ever sees a complete file appear, never a half-written
        checkpoint or a log truncated mid-line.
        """
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".amdt-part")
        try:
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    # -- sync passes ------------------------------------------------------
    def sync_once(self) -> int:
        """Mirror everything that changed since the last pass. Returns file count."""
        if not self.cfg.enabled or self.dest is None or not self.run_dir.exists():
            return 0
        n = 0
        for src in self.run_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(self.run_dir)
            if self._excluded(rel):
                continue
            try:
                st = src.stat()
            except FileNotFoundError:
                continue
            if st.st_size > self.cfg.max_file_mb * 1024**2:
                continue
            key = str(rel)
            sig = (st.st_mtime_ns, st.st_size)
            if self._seen.get(key) == sig:
                continue
            try:
                self._copy_atomic(src, self.dest / rel)
                self._seen[key] = sig
                self.files_synced += 1
                self.bytes_synced += st.st_size
                n += 1
            except Exception as exc:            # a sync failure must not kill a run
                msg = f"{rel}: {exc}"
                if msg not in self.errors:
                    self.errors.append(msg)
                    log.warning("drive sync failed for %s", msg)
        return n

    # -- background operation ---------------------------------------------
    def start(self) -> "DriveSync":
        """Begin mirroring immediately, on a daemon thread.

        Deliberately not deferred to the end of the run: the artifacts most
        worth having off-machine are the ones a crash would otherwise take
        with it.
        """
        if not self.cfg.enabled or self._thread is not None:
            return self

        def loop() -> None:
            while not self._stop.wait(self.cfg.interval_s):
                try:
                    self.sync_once()
                except Exception as exc:        # pragma: no cover
                    log.warning("drive sync pass failed: %s", exc)

        self.sync_once()                        # first pass now, not in 60 s
        self._thread = threading.Thread(target=loop, name="drive-sync", daemon=True)
        self._thread.start()
        return self

    def close(self, prune: Optional[bool] = None,
              protected: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        """Final flush, then optional retention pass."""
        if not self.cfg.enabled:
            return {"enabled": False}
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.sync_once()
        if prune if prune is not None else self.cfg.prune:
            self.prune(protected=protected)
        return self.describe()

    # -- retention --------------------------------------------------------
    def prune(self, keep_last_n: Optional[int] = None,
              protected: Optional[Iterable[str]] = None) -> List[str]:
        """Keep the last N runs plus every protected run. Returns pruned IDs.

        A run referenced by a commit — i.e. one whose ID appears in the
        provenance record of a reported result — is evidence, and is never
        removed regardless of age.
        """
        if not self.cfg.enabled or self.root is None:
            return []
        keep_n = keep_last_n if keep_last_n is not None else self.cfg.keep_last_n
        keep_ids = set(protected or ()) | {self.run_id}

        runs = sorted((p for p in self.root.iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        removed = []
        for p in runs[keep_n:]:
            if p.name in keep_ids:
                log.info("keeping %s: referenced by a commit", p.name)
                continue
            try:
                shutil.rmtree(p)
                removed.append(p.name)
            except Exception as exc:
                log.warning("could not prune %s: %s", p.name, exc)
        if removed:
            log.info("pruned %d old run folder(s) from Drive: %s",
                     len(removed), ", ".join(removed))
        return removed

    # -- provenance -------------------------------------------------------
    def describe(self) -> Dict[str, Any]:
        """Recorded next to the commit hash, so a table cell resolves to both."""
        return {
            "enabled": self.cfg.enabled,
            "sync_root": str(self.root) if self.root else None,
            "sync_target": str(self.dest) if self.dest else None,
            "run_id": self.run_id,
            "files_synced": self.files_synced,
            "megabytes_synced": round(self.bytes_synced / 1024**2, 2),
            "excludes": list(self.cfg.excludes),
            "errors": self.errors[:10],
        }


# --------------------------------------------------------------------------- #
def runs_referenced_by_commits(repo, limit: int = 500) -> List[str]:
    """Run IDs mentioned in commit messages — the folders that are evidence."""
    if repo is None or not getattr(repo, "initialised", False):
        return []
    try:
        out = repo._run("log", f"-{limit}", "--format=%B", check=False).stdout
    except Exception:
        return []
    ids = set()
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("run_dir:"):
            ids.add(Path(line.split(":", 1)[1].strip()).name)
        elif line.startswith("results:"):
            ids.add(line.split(":", 1)[1].strip())
    return sorted(ids)
