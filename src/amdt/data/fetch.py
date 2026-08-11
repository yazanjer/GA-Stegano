"""Mirror-agnostic dataset acquisition with the same pinning discipline as §14.

Why this exists alongside ``kaggle.py``: BOSSBase has outlived several of its
download locations.  The original BOSS site, the CVUT ``stegodata`` index, the
Binghamton DDE mirror and assorted Kaggle re-uploads have each been the
canonical source at some point, and each has been dead at some point.  Hard-
coding one of them into the pipeline guarantees a broken run later.

So the *source* is configuration and the *provenance* is code: whichever mirror
you use, the archive is hashed, the hash is pinned, and every later run
verifies it.  A corpus that changed between your ablation and your final table
is a reproducibility failure no seed will catch — and that risk is identical
whether the bytes came from Kaggle or a university web server.

``mirrors`` is a list, tried in order, so a dead link falls through to the next
instead of failing the run.  The one that succeeded is recorded in the pin.
"""

from __future__ import annotations

import logging
import shutil
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .kaggle import DEFAULT_CACHE, DatasetPin, sha256_file, verify_pin

__all__ = ["fetch_archive", "fetch_dataset", "probe_mirrors"]

log = logging.getLogger("amdt.fetch")

_UA = "amdt-pipeline/1.0 (academic research; contact via repository)"


def probe_mirrors(mirrors: Sequence[str], timeout: int = 15) -> List[dict]:
    """HEAD each mirror and report which are alive, without downloading.

    Run this before committing to an overnight download — a 404 discovered at
    2 a.m. costs a night.
    """
    out = []
    for url in mirrors:
        rec = {"url": url, "ok": False, "status": None, "size_bytes": None,
               "error": None}
        try:
            req = urllib.request.Request(url, method="HEAD",
                                         headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                rec["ok"] = 200 <= r.status < 300
                rec["status"] = r.status
                cl = r.headers.get("Content-Length")
                rec["size_bytes"] = int(cl) if cl else None
        except urllib.error.HTTPError as e:
            rec["status"] = e.code
            rec["error"] = str(e)
        except Exception as e:
            rec["error"] = str(e)
        log.info("mirror %s -> %s", url, "OK" if rec["ok"] else rec["error"] or rec["status"])
        out.append(rec)
    return out


def fetch_archive(mirrors: Sequence[str], dest: str | Path,
                  expected_sha256: Optional[str] = None,
                  timeout: int = 60, force: bool = False) -> Tuple[Path, str]:
    """Download the first mirror that works. Returns ``(path, sha256)``.

    Writes to a temp path and renames, so an interrupted download can never be
    mistaken for a complete one.  A cached file that fails the pinned hash is
    re-downloaded, never used.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        digest = sha256_file(dest)
        if expected_sha256 and digest != expected_sha256:
            log.warning("cached %s fails the pinned hash; re-downloading", dest.name)
            dest.unlink()
        else:
            log.info("using cached %s (%.1f MB)", dest.name,
                     dest.stat().st_size / 1024**2)
            return dest, digest

    errors = []
    for url in mirrors:
        try:
            log.info("downloading %s", url)
            tmp = dest.with_suffix(dest.suffix + ".part")
            req = urllib.request.Request(url, headers={"User-Agent": _UA})
            with urllib.request.urlopen(req, timeout=timeout) as r, tmp.open("wb") as fh:
                shutil.copyfileobj(r, fh, length=1 << 20)
            tmp.replace(dest)
            digest = sha256_file(dest)
            if expected_sha256 and digest != expected_sha256:
                raise ValueError(
                    f"archive from {url} has sha256 {digest[:12]}, pinned "
                    f"{expected_sha256[:12]}. Refusing to use it: this is a "
                    "different corpus from the one your earlier results used."
                )
            log.info("downloaded %.1f MB, sha256=%s",
                     dest.stat().st_size / 1024**2, digest[:12])
            return dest, digest
        except Exception as exc:
            log.warning("mirror failed (%s): %s", url, exc)
            errors.append(f"{url}: {exc}")

    raise RuntimeError(
        "every mirror failed:\n  " + "\n  ".join(errors) +
        "\n\nBOSSBase has outlived several download locations. Find a live "
        "mirror, add it to dataset.mirrors, and record the SHA-256 in the pin."
    )


def fetch_dataset(mirrors: Sequence[str], name: str,
                  cache_dir: str | Path = DEFAULT_CACHE,
                  expected: Optional[DatasetPin] = None,
                  license_note: str = "", pattern: str = "*",
                  force: bool = False) -> Tuple[Path, DatasetPin]:
    """Download, extract and pin a dataset from a plain URL.

    Returns ``(extracted_dir, pin)``.  The pin is the artifact to paste into the
    dataset config; from then on every run verifies against it.
    """
    cache = Path(cache_dir) / name
    cache.mkdir(parents=True, exist_ok=True)
    suffix = Path(mirrors[0]).suffix or ".zip"
    archive = cache / f"archive{suffix}"
    extracted = cache / "extracted"

    archive, digest = fetch_archive(
        mirrors, archive,
        expected_sha256=(expected.sha256 if expected else None), force=force)

    if not extracted.exists() or force:
        extracted.mkdir(parents=True, exist_ok=True)
        log.info("extracting %s ...", archive.name)
        shutil.unpack_archive(str(archive), str(extracted))

    files = [p for p in extracted.rglob(pattern) if p.is_file()]
    pin = DatasetPin(
        slug=name, kind="url", version=None, sha256=digest,
        archive=str(archive), license=license_note or "check the source's terms",
        n_files=len(files), size_bytes=sum(p.stat().st_size for p in files),
        downloaded=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    pin = verify_pin(pin, expected)
    pin.save(cache / "pin.json")
    log.info("%s: %d files matching %r, %.1f MB", name, pin.n_files, pattern,
             (pin.size_bytes or 0) / 1024**2)
    if pin.n_files == 0:
        log.error("no files matched %r under %s -- check the archive layout "
                  "and the dataset.pattern setting", pattern, extracted)
    return extracted, pin
