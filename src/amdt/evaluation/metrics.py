"""Imperceptibility and capacity metrics.

Self-contained (NumPy only) so the fitness function has no heavyweight import
and the GA stays fast.  SSIM follows Wang et al. (2004) exactly: 11x11
Gaussian window, sigma=1.5, K1=0.01, K2=0.03, L=255 -- the same defaults as
MATLAB's ``ssim`` and ``skimage.metrics.structural_similarity(gaussian_weights=
True, use_sample_covariance=False)``, so numbers stay comparable with the
MATLAB benchmark results in the manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

__all__ = ["mse", "psnr", "ssim", "correlation", "embedding_efficiency",
           "fused_measure_1", "fused_measure_2", "QualityReport", "evaluate_pair"]


def _as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def mse(cover: np.ndarray, stego: np.ndarray) -> float:
    d = _as_float(cover) - _as_float(stego)
    return float(np.mean(d * d))


def psnr(cover: np.ndarray, stego: np.ndarray, peak: float = 255.0) -> float:
    m = mse(cover, stego)
    if m == 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak * peak / m))


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    k = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _filter2_valid(img: np.ndarray, k1d: np.ndarray) -> np.ndarray:
    """Separable 'valid' convolution with a symmetric 1-D kernel."""
    n = k1d.size
    h, w = img.shape
    if h < n or w < n:
        raise ValueError("image smaller than the SSIM window")
    # rows
    cols = np.lib.stride_tricks.sliding_window_view(img, n, axis=1)
    tmp = np.einsum("ijk,k->ij", cols, k1d)
    rows = np.lib.stride_tricks.sliding_window_view(tmp, n, axis=0)
    return np.einsum("ijk,k->ij", rows, k1d)


def ssim(cover: np.ndarray, stego: np.ndarray, peak: float = 255.0,
         k1: float = 0.01, k2: float = 0.03) -> float:
    x, y = _as_float(cover), _as_float(stego)
    k = _gaussian_kernel()
    c1, c2 = (k1 * peak) ** 2, (k2 * peak) ** 2

    mu_x, mu_y = _filter2_valid(x, k), _filter2_valid(y, k)
    xx, yy, xy = _filter2_valid(x * x, k), _filter2_valid(y * y, k), _filter2_valid(x * y, k)
    sxx = xx - mu_x * mu_x
    syy = yy - mu_y * mu_y
    sxy = xy - mu_x * mu_y

    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return float(np.mean(num / den))


def correlation(cover: np.ndarray, stego: np.ndarray) -> float:
    a, b = _as_float(cover).ravel(), _as_float(stego).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den else 1.0


def embedding_efficiency(payload_bits: int, n_changes: int) -> float:
    """Bits embedded per embedding change (the standard definition).

    The manuscript's Figure 6 uses "bits per unit distortion"; both are
    reported -- see :func:`evaluate_pair` -- because the change-based version is
    the one steganography reviewers expect and is comparable across papers.
    """
    return float(payload_bits) / float(n_changes) if n_changes else float("inf")


def fused_measure_1(corr: float, ssim_val: float) -> float:
    """corr x SSIM, as defined in the manuscript."""
    return float(corr * ssim_val)


def fused_measure_2(corr: float, ssim_val: float, mse_val: float) -> float:
    """(corr x SSIM) / MSE, as defined in the manuscript."""
    return float(corr * ssim_val / mse_val) if mse_val > 0 else float("inf")


@dataclass
class QualityReport:
    mse: float
    psnr: float
    ssim: float
    correlation: float
    payload_bits: int
    n_changes: int
    change_rate: float
    embedding_efficiency: float
    bits_per_pixel: float
    bits_per_unit_distortion: float
    fused_measure_1: float
    fused_measure_2: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def evaluate_pair(cover: np.ndarray, stego: np.ndarray, payload_bits: int,
                  n_changes: int | None = None) -> QualityReport:
    if n_changes is None:
        n_changes = int(np.count_nonzero(np.asarray(cover) != np.asarray(stego)))
    m = mse(cover, stego)
    s = ssim(cover, stego)
    c = correlation(cover, stego)
    n_px = int(np.asarray(cover).size)
    return QualityReport(
        mse=m,
        psnr=psnr(cover, stego),
        ssim=s,
        correlation=c,
        payload_bits=int(payload_bits),
        n_changes=int(n_changes),
        change_rate=n_changes / n_px,
        embedding_efficiency=embedding_efficiency(payload_bits, n_changes),
        bits_per_pixel=payload_bits / n_px,
        bits_per_unit_distortion=(payload_bits / m) if m > 0 else float("inf"),
        fused_measure_1=fused_measure_1(c, s),
        fused_measure_2=fused_measure_2(c, s, m),
    )
