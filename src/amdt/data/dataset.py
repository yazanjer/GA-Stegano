"""Dataset loading, description and splitting.

Reviewer 1, comment 5: "Complete dataset description. Image sources. Message
sizes in bits."  Every loader returns a :class:`DatasetSpec` that is serialised
into the run directory, so the description in the paper is generated from the
data actually used rather than written from memory.

Two sources are supported:

``local``
    The 29 JPEG images shipped with the MATLAB benchmark
    (``baseline code matlab/Images``).  Sufficient for imperceptibility and
    runtime experiments, and it keeps continuity with the published benchmark
    numbers -- but **too small for a credible steganalysis claim**, and
    :func:`DatasetSpec.steganalysis_warning` says so in the emitted metadata so
    the limitation cannot be lost between code and manuscript.

``bossbase``
    BOSSBase 1.01 (10,000 512x512 8-bit grayscale PGM covers) and/or BOWS2.
    This is the standard corpus for SRM/CNN steganalysis; point
    ``cfg.dataset.root`` at the extracted directory and the same pipeline runs
    unchanged.

Images are converted to 8-bit grayscale and centre-cropped/resized to a fixed
side (default 512) so that the ``x_off``/``y_off`` genes keep their 9-bit range.

Splitting is **cover-wise**: a cover and its stego version always land in the
same split.  Putting the stego of a training cover into the test set is the
classic steganalysis leak -- the detector then learns the cover, not the
embedding -- and it inflates accuracy by tens of points.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["ImageRecord", "DatasetSpec", "load_dataset", "cover_wise_split",
           "payload_bits_for_rate", "describe_payloads"]

_EXT = (".pgm", ".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg", ".JPG", ".PNG")


@dataclass
class ImageRecord:
    path: str
    name: str
    sha256: str
    height: int
    width: int
    original_mode: str
    original_size: Tuple[int, int]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class DatasetSpec:
    name: str
    root: str
    n_images: int
    side: int
    grayscale: bool = True
    bit_depth: int = 8
    source: str = ""
    license: str = ""
    records: List[ImageRecord] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def pixels_per_image(self) -> int:
        return self.side * self.side

    def steganalysis_warning(self) -> Optional[str]:
        if self.n_images < 500:
            return (
                f"{self.n_images} images is far below the >=10,000-cover corpora "
                "(BOSSBase 1.01 / BOWS2) that SRM+ensemble and CNN steganalysis "
                "require; detection numbers computed on this set are indicative "
                "only and are reported with cover-wise cross-validation and "
                "confidence intervals rather than a single split."
            )
        return None

    def as_dict(self) -> Dict[str, object]:
        d = asdict(self)
        d["pixels_per_image"] = self.pixels_per_image
        d["steganalysis_warning"] = self.steganalysis_warning()
        return d


def _to_gray_uint8(path: Path, side: Optional[int]) -> Tuple[np.ndarray, str, Tuple[int, int]]:
    from PIL import Image

    with Image.open(path) as im:
        mode, size = im.mode, im.size
        g = im.convert("L")
        if side is not None and (g.size[0] != side or g.size[1] != side):
            # centre-crop to square, then resize -- avoids anisotropic scaling,
            # which would create directional artefacts a steganalyser can latch
            # onto and confound the comparison between methods.
            w, h = g.size
            s = min(w, h)
            g = g.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
            g = g.resize((side, side), Image.LANCZOS)
    return np.asarray(g, dtype=np.uint8), mode, size


def load_dataset(
    root: str | Path,
    side: Optional[int] = 512,
    limit: Optional[int] = None,
    name: str = "local",
    source: str = "",
    license: str = "",
    pattern: str = "*",
) -> Tuple[List[np.ndarray], DatasetSpec]:
    """Load every image under ``root`` as 8-bit grayscale ``side x side``."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")
    files = sorted(
        p for p in root.rglob(pattern)
        if p.is_file() and p.suffix in _EXT
    )
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no images matching {pattern} under {root}")

    images, records = [], []
    for p in files:
        arr, mode, size = _to_gray_uint8(p, side)
        images.append(arr)
        records.append(ImageRecord(
            path=str(p), name=p.name,
            sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            height=arr.shape[0], width=arr.shape[1],
            original_mode=mode, original_size=size,
        ))

    notes = []
    if any(r.path.lower().endswith((".jpg", ".jpeg")) for r in records):
        notes.append(
            "Source images are JPEG. JPEG covers carry blocking artefacts that "
            "some steganalysers exploit; spatial-domain results on JPEG-sourced "
            "covers are not directly comparable with BOSSBase (uncompressed PGM)."
        )
    spec = DatasetSpec(
        name=name, root=str(root), n_images=len(images), side=side or records[0].height,
        source=source, license=license, records=records, notes=notes,
    )
    return images, spec


def cover_wise_split(
    n_images: int,
    rng: np.random.Generator,
    train: float = 0.6,
    val: float = 0.2,
) -> Dict[str, np.ndarray]:
    """Disjoint index sets. Stego images inherit their cover's split."""
    idx = rng.permutation(n_images)
    n_tr = int(round(train * n_images))
    n_va = int(round(val * n_images))
    return {
        "train": np.sort(idx[:n_tr]),
        "val": np.sort(idx[n_tr:n_tr + n_va]),
        "test": np.sort(idx[n_tr + n_va:]),
    }


def payload_bits_for_rate(shape: Tuple[int, int], rate_bpp: float) -> int:
    """Payload length in bits for a bits-per-pixel rate.

    Steganalysis papers quote payload as **bpp** (bits per cover pixel); the
    manuscript quotes it as a percentage of maximum capacity.  Both are emitted
    by :func:`describe_payloads` so the two literatures can be compared.
    """
    return int(round(rate_bpp * shape[0] * shape[1]))


def describe_payloads(shape: Tuple[int, int], rates_bpp: Sequence[float],
                      max_capacity_bits: int) -> List[Dict[str, float]]:
    n = shape[0] * shape[1]
    out = []
    for r in rates_bpp:
        bits = payload_bits_for_rate(shape, r)
        out.append({
            "rate_bpp": r,
            "payload_bits": bits,
            "payload_bytes": bits / 8.0,
            "percent_of_max_capacity": 100.0 * bits / max_capacity_bits
            if max_capacity_bits else float("nan"),
            "cover_pixels": n,
        })
    return out
