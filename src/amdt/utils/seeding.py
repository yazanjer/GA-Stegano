"""Deterministic seed locking for every source of randomness.

Reviewer 1, comment 5 (reproducibility): "Random seed policy".

Policy
------
1. A single integer ``seed`` is the root of *all* randomness in the pipeline.
2. Every stochastic component receives an explicitly *derived* sub-seed
   (``derive_seed(root, tag)``) rather than sharing a global stream.  This makes
   a component's randomness independent of the order in which components run,
   so adding a baseline to the sweep cannot change the GA's trajectory.
3. ``set_seed`` must be called at the very top of the entry point, before any
   CUDA context is created (``CUBLAS_WORKSPACE_CONFIG`` is read at CUDA init).
4. Every reported number is produced under ``seed in cfg.experiment.seeds``
   and the seed is recorded next to the number in the results CSV.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Optional

import numpy as np

__all__ = ["set_seed", "derive_seed", "seeded_rng", "SeedPolicy"]

SEED_POLICY_TEXT = (
    "Single root seed; per-component sub-seeds derived via BLAKE2b(root||tag); "
    "torch/cuDNN forced deterministic; DataLoader workers reseeded from root; "
    "all reported results averaged over >=5 root seeds."
)


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Lock every RNG the pipeline can touch.

    Call this *first* in the entry point.  ``CUBLAS_WORKSPACE_CONFIG`` has to be
    in the environment before CUDA initialises, otherwise deterministic cuBLAS
    matmuls are silently unavailable.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional for the classical half of the pipeline
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Strict by default: an op with no deterministic kernel should raise, so
        # the exception is a considered decision rather than an oversight.
        #
        # Known exception in this repository: none. Both steganalysers
        # (Yedroudj-Net, SRNet) use only Conv2d / BatchNorm2d / AvgPool2d /
        # AdaptiveAvgPool2d / Linear / ReLU, all of which have deterministic CUDA
        # kernels. `adaptive_avg_pool2d_backward_cuda` is non-deterministic for
        # *non-integer* output ratios; SRNet's global pool is implemented as a
        # spatial `mean`, and Yedroudj-Net's `AdaptiveAvgPool2d(1)` reduces the
        # full map, so neither hits that path.
        #
        # Set AMDT_ALLOW_NONDETERMINISTIC=1 to downgrade to warn_only if you add
        # a layer that trips this; record the op name in README.md when you do.
        strict = os.environ.get("AMDT_ALLOW_NONDETERMINISTIC", "0") != "1"
        torch.use_deterministic_algorithms(True, warn_only=not strict)


def derive_seed(root: int, tag: str) -> int:
    """Derive a stable 32-bit sub-seed for a named component.

    Deterministic across processes and Python versions (``hash()`` is not).
    """
    h = hashlib.blake2b(f"{root}:{tag}".encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h[:4], "little")


def seeded_rng(root: int, tag: str) -> np.random.Generator:
    """A PCG64 generator private to ``tag``."""
    return np.random.default_rng(derive_seed(root, tag))


def torch_dataloader_kwargs(root: int, tag: str = "dataloader") -> dict:
    """Generator + worker_init_fn so DataLoader workers stay deterministic."""
    import torch

    base = derive_seed(root, tag)
    g = torch.Generator()
    g.manual_seed(base)

    def _worker_init(worker_id: int) -> None:
        np.random.seed((base + worker_id) % (2**32))
        random.seed(base + worker_id)

    return {"generator": g, "worker_init_fn": _worker_init}


class SeedPolicy:
    """Documentation object emitted into every run directory."""

    text: str = SEED_POLICY_TEXT

    @staticmethod
    def describe(root: int, n_seeds: int, seeds: Optional[list] = None) -> dict:
        return {
            "policy": SEED_POLICY_TEXT,
            "root_seed": root,
            "n_seeds": n_seeds,
            "seeds": seeds or [],
        }
