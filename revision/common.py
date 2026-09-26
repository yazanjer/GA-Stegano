"""Shared helpers for the revision-2 experiment scripts (see revision/README.md)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amdt.utils.seeding import derive_seed  # noqa: E402

KEY = bytes.fromhex("2a7f4c9e1b6d80f3a5c2e7091d4b8fa63e5c1908d7b2a4f60c3e819d5a7b2f4c")
BOSS = Path(os.environ.get("BOSS_DIR", "/workspace/data/BOSSbase_1.01"))
SPLIT_SEED = 20260926


def boss_files() -> List[Path]:
    fs = sorted(BOSS.glob("*.pgm"), key=lambda p: int(p.stem))
    if len(fs) != 10000:
        raise RuntimeError(f"expected 10,000 BOSSBase covers, found {len(fs)}")
    return fs


def cover_sets() -> Dict[str, List[Path]]:
    """Deterministic, disjoint cover sets drawn from BOSSBase 1.01.

    quality : 100 covers (imperceptibility, ablation, significance, runtime)
    stego   : 2,000 covers (steganalysis; the first 1,000 are used for the
              SRM ensemble, all 2,000 for SRNet)
    """
    fs = boss_files()
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(fs))
    return {"quality": [fs[i] for i in perm[:100]],
            "stego": [fs[i] for i in perm[100:2100]]}


def load(p: Path) -> np.ndarray:
    from PIL import Image
    x = np.array(Image.open(p))
    assert x.shape == (512, 512) and x.dtype == np.uint8
    return x


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def payload_bits(cover_id: str, rate: float, seed: int) -> np.ndarray:
    """Payload shared by every method for a given (cover, rate, seed): paired design."""
    n = int(round(rate * 512 * 512))
    rng = np.random.default_rng(derive_seed(seed, f"payload:{cover_id}:{rate}"))
    return rng.integers(0, 2, n, dtype=np.uint8)


def method_rng(method: str, cover_id: str, rate: float, seed: int) -> np.random.Generator:
    return np.random.default_rng(derive_seed(seed, f"{method}:{cover_id}:{rate}"))
