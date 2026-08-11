"""Kaggle dataset and model acquisition (§14).

Relevant here because BOSSBase and BOWS2 — the corpora Reviewer 1's comment 3
really needs — are mirrored on Kaggle, and pulling them by hand is where the
provenance chain usually breaks.

Credentials
-----------
Never in the repository, never in a chat.  On a hosted runtime they come from
Colab Secrets; locally from ``~/.kaggle/kaggle.json`` with mode ``600`` (the
client refuses looser permissions).  Which path applies is detected at runtime
rather than branching on a manual flag.  ``kaggle.json`` and ``.env`` are in
``.gitignore`` *before* the first commit — a key that has ever been committed
must be rotated, because history retains it.

The 403 that is not an auth failure
-----------------------------------
Competition data requires accepting the rules on the website first, and until
you do the API returns a 403 that looks exactly like bad credentials.  That is
the single most common false alarm in this path, so
:func:`download_competition` checks rule acceptance before it lets you debug
credentials.

Version pinning
---------------
The dataset slug, its version number and a SHA-256 of the archive go into the
provenance record.  Each run compares the live version against the pinned one
and warns loudly on drift — but never auto-upgrades.  A dataset silently
gaining rows between your ablation and your final table is a reproducibility
failure no seed will catch.  The hash is verified on every load; a cached file
that fails the check is re-downloaded, not used.

The leakage trap specific to Kaggle
-----------------------------------
Competition data ships as labeled ``train`` plus unlabeled ``test`` scored by
the leaderboard.  That split is **not** a held-out test set in the academic
sense: you cannot compute metrics on it locally, and a leaderboard score is a
measurement on a set thousands of submissions have already probed.  For a
manuscript, carve your own test split from the labeled data before any
preprocessing (see :func:`amdt.data.dataset.cover_wise_split`) and keep it
isolated.  :func:`warn_leaderboard_split` exists to say so at the point of use.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import stat
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["KaggleCredentials", "DatasetPin", "load_credentials", "sha256_file",
           "download_dataset", "download_competition", "load_model",
           "verify_pin", "warn_leaderboard_split", "DEFAULT_CACHE"]

log = logging.getLogger("amdt.kaggle")

#: Cache lives outside the repository -- downloaded data is never committed.
DEFAULT_CACHE = Path(os.environ.get("AMDT_DATA_CACHE",
                                    Path.home() / ".cache" / "amdt" / "kaggle"))


# --------------------------------------------------------------------------- #
@dataclass
class KaggleCredentials:
    source: str                      # colab_secrets | kaggle_json | environment
    username: str
    path: Optional[str] = None

    def describe(self) -> Dict[str, str]:
        # The key itself never appears in a record or a log line.
        return {"source": self.source, "username": self.username,
                "path": self.path or ""}


def load_credentials() -> KaggleCredentials:
    """Detect and install credentials into the environment the client expects."""
    # 1. hosted runtime
    try:
        from google.colab import userdata  # type: ignore
        user = userdata.get("KAGGLE_USERNAME")
        keyv = userdata.get("KAGGLE_KEY")
        if user and keyv:
            os.environ["KAGGLE_USERNAME"] = user
            os.environ["KAGGLE_KEY"] = keyv
            return KaggleCredentials("colab_secrets", user)
    except Exception:
        pass

    # 2. already in the environment
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return KaggleCredentials("environment", os.environ["KAGGLE_USERNAME"])

    # 3. local kaggle.json
    p = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if p.exists():
        mode = stat.S_IMODE(p.stat().st_mode)
        if mode & 0o077:
            raise PermissionError(
                f"{p} has mode {oct(mode)}; the Kaggle client refuses anything "
                f"looser than 600. Run: chmod 600 {p}"
            )
        data = json.loads(p.read_text())
        os.environ["KAGGLE_USERNAME"] = data["username"]
        os.environ["KAGGLE_KEY"] = data["key"]
        return KaggleCredentials("kaggle_json", data["username"], str(p))

    raise RuntimeError(
        "No Kaggle credentials found. On Colab add KAGGLE_USERNAME and "
        "KAGGLE_KEY to Secrets; locally place kaggle.json at ~/.kaggle/ with "
        "chmod 600. Do not paste the key into a notebook, a config or a chat."
    )


# --------------------------------------------------------------------------- #
@dataclass
class DatasetPin:
    """Everything needed to prove which bytes produced a result."""

    slug: str
    kind: str = "dataset"            # dataset | competition | model
    version: Optional[str] = None
    sha256: Optional[str] = None
    archive: Optional[str] = None
    license: Optional[str] = None
    n_files: Optional[int] = None
    size_bytes: Optional[int] = None
    downloaded: Optional[str] = None
    drift: bool = False
    drift_note: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "DatasetPin":
        return cls(**json.loads(Path(path).read_text()))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _live_version(slug: str, kind: str = "dataset") -> Optional[str]:
    """Current version reported by Kaggle, or ``None`` if it cannot be read."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        if kind == "dataset":
            owner, name = slug.split("/", 1)
            for d in api.dataset_list(search=name, user=owner):
                if str(d.ref).lower() == slug.lower():
                    return str(getattr(d, "currentVersionNumber", None)
                               or getattr(d, "lastUpdated", ""))
    except Exception as exc:
        log.debug("could not read live version for %s: %s", slug, exc)
    return None


def verify_pin(pin: DatasetPin, expected: Optional[DatasetPin] = None) -> DatasetPin:
    """Compare against the pinned version/hash. Warns on drift; never upgrades."""
    if expected is None:
        return pin
    if expected.version and pin.version and expected.version != pin.version:
        pin.drift = True
        pin.drift_note = (f"pinned version {expected.version}, live {pin.version}. "
                          "NOT auto-upgraded -- a dataset gaining rows between "
                          "your ablation and your final table is a "
                          "reproducibility failure no seed will catch.")
        log.warning("DATASET DRIFT for %s: %s", pin.slug, pin.drift_note)
    if expected.sha256 and pin.sha256 and expected.sha256 != pin.sha256:
        pin.drift = True
        pin.drift_note = ((pin.drift_note or "") +
                          f" Archive SHA-256 differs (pinned {expected.sha256[:12]}, "
                          f"got {pin.sha256[:12]}).")
        log.warning("ARCHIVE HASH MISMATCH for %s", pin.slug)
    return pin


# --------------------------------------------------------------------------- #
def download_dataset(slug: str, cache_dir: str | Path = DEFAULT_CACHE,
                     expected: Optional[DatasetPin] = None,
                     license_note: str = "",
                     force: bool = False) -> tuple[Path, DatasetPin]:
    """Download (or reuse) a Kaggle dataset, pinned and hash-verified.

    Returns ``(extracted_dir, pin)``.  Stage to fast local disk per §3 before
    training reads from it.
    """
    from datetime import datetime, timezone

    creds = load_credentials()
    log.info("kaggle credentials: %s (%s)", creds.source, creds.username)

    cache = Path(cache_dir) / slug.replace("/", "__")
    cache.mkdir(parents=True, exist_ok=True)
    extracted = cache / "extracted"
    archive = cache / "archive.zip"

    # Reuse the cache only if it still matches the pinned hash.
    if archive.exists() and not force:
        digest = sha256_file(archive)
        if expected and expected.sha256 and digest != expected.sha256:
            log.warning("cached archive for %s fails the pinned hash check; "
                        "re-downloading rather than using it", slug)
            archive.unlink()
        else:
            log.info("using cached archive %s", archive)

    if not archive.exists():
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        log.info("downloading %s ...", slug)
        api.dataset_download_files(slug, path=str(cache), unzip=False, quiet=False)
        zips = sorted(cache.glob("*.zip"))
        if not zips:
            raise FileNotFoundError(f"no archive downloaded for {slug}")
        zips[0].replace(archive)

    if not extracted.exists() or force:
        import shutil
        extracted.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(archive), str(extracted))

    files = [p for p in extracted.rglob("*") if p.is_file()]
    pin = DatasetPin(
        slug=slug, kind="dataset", version=_live_version(slug),
        sha256=sha256_file(archive), archive=str(archive),
        license=license_note or "check the dataset page; many Kaggle datasets "
                                "forbid redistribution",
        n_files=len(files), size_bytes=sum(p.stat().st_size for p in files),
        downloaded=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    pin = verify_pin(pin, expected)
    pin.save(cache / "pin.json")
    log.info("%s: %d files, %.1f MB, sha256=%s", slug, pin.n_files,
             (pin.size_bytes or 0) / 1024**2, (pin.sha256 or "")[:12])
    return extracted, pin


def download_competition(slug: str, cache_dir: str | Path = DEFAULT_CACHE,
                         expected: Optional[DatasetPin] = None
                         ) -> tuple[Path, DatasetPin]:
    """Competition data. Checks rule acceptance *before* blaming credentials."""
    from datetime import datetime, timezone

    load_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    cache = Path(cache_dir) / f"competition__{slug}"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_files(slug, path=str(cache), quiet=False)
    except Exception as exc:
        if "403" in str(exc) or "Forbidden" in str(exc):
            raise PermissionError(
                f"403 from Kaggle for competition {slug!r}. Before debugging "
                "credentials: have you accepted the competition rules on the "
                f"website? https://www.kaggle.com/c/{slug}/rules -- the API "
                "returns a 403 that looks exactly like an auth failure until "
                "you do. This is the most common false alarm in this path."
            ) from exc
        raise

    zips = sorted(cache.glob("*.zip"))
    archive = zips[0] if zips else None
    pin = DatasetPin(
        slug=slug, kind="competition",
        sha256=sha256_file(archive) if archive else None,
        archive=str(archive) if archive else None,
        downloaded=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        license="competition rules apply; check redistribution terms",
    )
    pin = verify_pin(pin, expected)
    pin.save(cache / "pin.json")
    warn_leaderboard_split()
    return cache, pin


def load_model(handle: str, cache_dir: str | Path = DEFAULT_CACHE) -> tuple[Path, DatasetPin]:
    """Pretrained weights via kagglehub, pinned like a dataset.

    Pretrained weights are an experimental variable, not infrastructure: the
    exact handle and version belong in the provenance record.
    """
    from datetime import datetime, timezone
    import kagglehub

    load_credentials()
    path = Path(kagglehub.model_download(handle))
    files = [p for p in path.rglob("*") if p.is_file()]
    pin = DatasetPin(
        slug=handle, kind="model", version=path.name,
        n_files=len(files), size_bytes=sum(p.stat().st_size for p in files),
        downloaded=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    pin.save(Path(cache_dir) / f"model__{handle.replace('/', '__')}.json")
    return path, pin


def warn_leaderboard_split() -> str:
    """State the Kaggle-specific leakage trap at the point of use."""
    msg = (
        "Kaggle competition data ships as labeled train + unlabeled test scored "
        "by the leaderboard. That is NOT a held-out test set for a manuscript: "
        "you cannot compute metrics on it locally, and the leaderboard has "
        "already been probed by thousands of submissions. Carve your own test "
        "split from the labeled data before any preprocessing and keep it "
        "isolated; report leaderboard standing separately as context, never as "
        "your held-out evaluation."
    )
    log.warning(msg)
    return msg
