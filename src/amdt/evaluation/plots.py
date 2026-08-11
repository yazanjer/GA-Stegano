"""Publication-grade figures: vector output, serif fonts, journal column widths.

Every figure is written as **PDF and SVG** (300 dpi rasters only where a bitmap
is unavoidable, e.g. stego previews).  Font sizes are set for a two-column
journal page, so the text in a figure matches the body text after LaTeX scales
it to ``\\columnwidth``.

Figures produced (mapping to the manuscript):

    Fig. 2  PSNR by method            :func:`metric_boxplot`
    Fig. 3  SSIM by method            :func:`metric_boxplot`
    Fig. 4  MSE by method             :func:`metric_boxplot`
    Fig. 5  Laplacian edge histogram  :func:`laplacian_histogram`
    Fig. 6  Embedding efficiency      :func:`metric_boxplot`
    Fig. 7  Quality vs payload rate   :func:`metric_vs_rate`
    Fig. 8  Correlation               :func:`metric_boxplot`
    new     GA convergence            :func:`convergence_curve`
    new     Detector ROC              :func:`roc_panel`
    new     Runtime vs payload        :func:`runtime_plot`
    new     Detectability vs payload  :func:`detectability_curve`
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["apply_style", "save", "metric_boxplot", "laplacian_histogram",
           "convergence_curve", "roc_panel", "metric_vs_rate", "runtime_plot",
           "detectability_curve"]

#: Single-column and double-column widths in inches for a typical IEEE/Elsevier page.
COL_WIDTH = 3.5
FULL_WIDTH = 7.16


def apply_style(base_font: float = 8.0) -> None:
    """Set rcParams once; every figure inherits them."""
    import logging

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fontTools logs one INFO line per glyph table on every PDF save; at this
    # volume it buries the actual run log.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_font,
        "axes.labelsize": base_font,
        "axes.titlesize": base_font,
        "xtick.labelsize": base_font - 1,
        "ytick.labelsize": base_font - 1,
        "legend.fontsize": base_font - 1,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.0,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,          # embed TrueType -- required by most journals
        "ps.fonttype": 42,
        "figure.dpi": 300,
    })


def save(fig, out_dir: str | Path, name: str, formats: Sequence[str] = ("pdf", "svg")) -> List[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        p = out / f"{name}.{f}"
        fig.savefig(p, format=f)
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
def metric_boxplot(data: Dict[str, Sequence[float]], ylabel: str,
                   title: str = "", highlight: Optional[str] = None,
                   width: float = COL_WIDTH, height: float = 2.2,
                   annotate_median: bool = True):
    """Boxplot with per-method scatter overlay (n is small: show every point)."""
    import matplotlib.pyplot as plt

    labels = list(data.keys())
    vals = [np.asarray(data[k], dtype=float) for k in labels]
    fig, ax = plt.subplots(figsize=(width, height))
    bp = ax.boxplot(vals, labels=labels, widths=0.6, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.0},
                    boxprops={"linewidth": 0.6}, whiskerprops={"linewidth": 0.6},
                    capprops={"linewidth": 0.6})
    rng = np.random.default_rng(0)
    for i, v in enumerate(vals, start=1):
        v = v[np.isfinite(v)]
        ax.plot(i + rng.uniform(-0.12, 0.12, v.size), v, ".", ms=2.0, alpha=0.55,
                color="tab:blue" if labels[i - 1] != highlight else "tab:red")
    if highlight and highlight in labels:
        i = labels.index(highlight)
        bp["boxes"][i].set_color("tab:red")
        bp["boxes"][i].set_linewidth(1.1)
    if annotate_median:
        for i, v in enumerate(vals, start=1):
            v = v[np.isfinite(v)]
            if v.size:
                ax.text(i, np.median(v), f"{np.median(v):.3g}", ha="center",
                        va="bottom", fontsize=5.5, color="black")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def laplacian_histogram(cover: np.ndarray, stegos: Dict[str, np.ndarray],
                        bins: int = 61, span: int = 15,
                        width: float = COL_WIDTH, height: float = 2.2):
    """Laplacian edge-response histograms, cover vs. stego variants (Fig. 5)."""
    import matplotlib.pyplot as plt

    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)

    def lap(img):
        p = np.pad(img.astype(np.float64), 1, mode="symmetric")
        win = np.lib.stride_tricks.sliding_window_view(p, (3, 3))
        return np.einsum("ijkl,kl->ij", win, k).ravel()

    edges = np.linspace(-span, span, bins + 1)
    fig, ax = plt.subplots(figsize=(width, height))
    hc, _ = np.histogram(lap(cover), bins=edges, density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(centres, hc, "k-", lw=1.2, label="cover")
    for name, st in stegos.items():
        hs, _ = np.histogram(lap(st), bins=edges, density=True)
        ax.plot(centres, hs, lw=0.8, alpha=0.85, label=name)
    ax.set_xlabel("Laplacian edge response")
    ax.set_ylabel("density")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize=5.5)
    fig.tight_layout()
    return fig


def convergence_curve(histories: Dict[str, Sequence[Sequence[float]]],
                      xlabel: str = "generation", ylabel: str = r"best fitness ($-$MSE)",
                      width: float = COL_WIDTH, height: float = 2.2):
    """Mean +- SD convergence across seeds/images for each configuration."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    for name, runs in histories.items():
        n = min(len(r) for r in runs)
        m = np.stack([np.asarray(r[:n], dtype=float) for r in runs])
        x = np.arange(n)
        mu, sd = m.mean(0), m.std(0)
        ax.plot(x, mu, label=name)
        ax.fill_between(x, mu - sd, mu + sd, alpha=0.18, linewidth=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig


def roc_panel(curves: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
              width: float = COL_WIDTH, height: float = 2.4):
    """ROC per method; AUC in the legend. Chance diagonal drawn for reference."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    for name, (fpr, tpr, a) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.6, label="chance")
    ax.set_xlabel("false-alarm rate")
    ax.set_ylabel("detection rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=5.5)
    fig.tight_layout()
    return fig


def metric_vs_rate(series: Dict[str, Tuple[Sequence[float], Sequence[float], Sequence[float]]],
                   xlabel: str = "payload rate (bpp)", ylabel: str = "PSNR (dB)",
                   width: float = COL_WIDTH, height: float = 2.2):
    """Mean +- SD of a metric against payload rate, one line per method."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    for name, (x, mu, sd) in series.items():
        x = np.asarray(x, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sd = np.asarray(sd, dtype=float)
        ax.errorbar(x, mu, yerr=sd, marker="o", ms=2.5, capsize=1.5, lw=0.9, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=5.5)
    fig.tight_layout()
    return fig


def runtime_plot(rows: Sequence[Dict[str, float]], width: float = COL_WIDTH,
                 height: float = 2.2):
    """Horizontal bar chart of mean wall-clock per stage, with SD error bars."""
    import matplotlib.pyplot as plt

    rows = list(rows)[::-1]
    names = [r["stage"] for r in rows]
    mu = [r["wall_mean_s"] for r in rows]
    sd = [r.get("wall_std_s", 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(width, height))
    ax.barh(names, mu, xerr=sd, height=0.6, capsize=1.5)
    ax.set_xlabel("wall-clock per call (s)")
    ax.set_xscale("log")
    fig.tight_layout()
    return fig


def detectability_curve(series: Dict[str, Tuple[Sequence[float], Sequence[float], Sequence[float]]],
                        width: float = COL_WIDTH, height: float = 2.2):
    """Detector P_E vs payload rate. 0.5 = undetectable; the axis says so."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(width, height))
    for name, (x, mu, sd) in series.items():
        ax.errorbar(np.asarray(x, float), np.asarray(mu, float), yerr=np.asarray(sd, float),
                    marker="o", ms=2.5, capsize=1.5, lw=0.9, label=name)
    ax.axhline(0.5, color="k", ls="--", lw=0.6)
    ax.text(0.99, 0.505, "undetectable", ha="right", va="bottom",
            transform=ax.get_yaxis_transform(), fontsize=5.5)
    ax.set_xlabel("payload rate (bpp)")
    ax.set_ylabel(r"detector $P_E$ (higher = safer)")
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=5.5)
    fig.tight_layout()
    return fig
