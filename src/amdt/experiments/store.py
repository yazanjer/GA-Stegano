"""Bounded storage for stego images produced during a run.

The naive design -- keep every stego image in a dict so later studies can reuse
them -- costs ``n_images x n_methods x n_rates x n_seeds x H x W`` bytes.  On the
29-cover set that is a few hundred MB; on BOSSBase (10,000 covers, 10 methods,
4 rates) it is over 100 GB and the process is killed long before the
steganalysis study runs.  This module makes the trade-off explicit and picks a
safe default instead of failing at hour three of a sweep.

Backends
--------
``memory``  dict in RAM. Fast; used when the projected footprint is under
            ``max_memory_mb``.
``disk``    lossless PNG per image under ``<run_dir>/stego/``. Bounded RAM, and
            the stego images themselves become a reviewable artifact.
``none``    store nothing; downstream studies re-embed on demand, holding one
            method-rate set at a time.  This is what makes a full BOSSBase run
            possible on a normal machine.
``auto``    pick ``memory`` if it fits, else ``disk`` if the disk budget fits,
            else ``none``.

Downstream code must therefore treat :meth:`StegoStore.get` as a *cache lookup
that may miss*, never as a guaranteed dictionary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

__all__ = ["StegoStore"]

log = logging.getLogger("amdt.store")

Key = Tuple[Any, ...]


class StegoStore:
    def __init__(self, run_dir: str | Path, backend: str = "auto",
                 projected_images: int = 0, image_bytes: int = 512 * 512,
                 max_memory_mb: float = 1024.0, max_disk_mb: float = 20_000.0) -> None:
        self.root = Path(run_dir) / "stego"
        self.projected_mb = projected_images * image_bytes / 1024 ** 2
        self.max_memory_mb = max_memory_mb
        self.max_disk_mb = max_disk_mb
        self.backend = self._choose(backend)
        self._mem: Dict[Key, np.ndarray] = {}
        self._meta: Dict[Key, Any] = {}          # small objects (GA histories)
        if self.backend == "disk":
            self.root.mkdir(parents=True, exist_ok=True)
        log.info("stego store: backend=%s projected=%.0f MB", self.backend, self.projected_mb)

    # ------------------------------------------------------------------ #
    def _choose(self, backend: str) -> str:
        if backend != "auto":
            return backend
        if self.projected_mb <= self.max_memory_mb:
            return "memory"
        if self.projected_mb <= self.max_disk_mb:
            log.warning("projected %.0f MB of stego images exceeds the %.0f MB RAM "
                        "budget; spilling to disk", self.projected_mb, self.max_memory_mb)
            return "disk"
        log.warning("projected %.0f MB of stego images exceeds both budgets; caching "
                    "disabled, later studies will re-embed on demand", self.projected_mb)
        return "none"

    # ------------------------------------------------------------------ #
    @staticmethod
    def _fname(key: Key) -> str:
        return "__".join(str(k).replace("/", "-") for k in key) + ".png"

    def put(self, key: Key, image: np.ndarray) -> None:
        if self.backend == "memory":
            self._mem[key] = image
        elif self.backend == "disk":
            from PIL import Image
            Image.fromarray(image, mode="L").save(self.root / self._fname(key),
                                                  optimize=True)

    def get(self, key: Key) -> Optional[np.ndarray]:
        """Cache lookup. ``None`` means "not stored" -- recompute, do not assume."""
        if self.backend == "memory":
            return self._mem.get(key)
        if self.backend == "disk":
            p = self.root / self._fname(key)
            if not p.exists():
                return None
            from PIL import Image
            with Image.open(p) as im:
                return np.asarray(im.convert("L"), dtype=np.uint8)
        return None

    # -- small side objects (GA convergence histories) ------------------ #
    def put_meta(self, key: Key, obj: Any) -> None:
        self._meta[key] = obj

    def meta_items(self) -> Iterable[Tuple[Key, Any]]:
        return self._meta.items()

    # ------------------------------------------------------------------ #
    def keys(self) -> Iterable[Key]:
        if self.backend == "memory":
            return list(self._mem.keys())
        if self.backend == "disk":
            return [tuple(p.stem.split("__")) for p in self.root.glob("*.png")]
        return []

    def __len__(self) -> int:
        return len(list(self.keys()))

    @property
    def enabled(self) -> bool:
        return self.backend != "none"

    def describe(self) -> Dict[str, Any]:
        return {"backend": self.backend, "projected_mb": self.projected_mb,
                "stored": len(self), "path": str(self.root) if self.backend == "disk" else None}
