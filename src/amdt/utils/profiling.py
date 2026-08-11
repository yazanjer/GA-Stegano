"""Runtime and computational-complexity instrumentation.

Reviewer 1, comment 6: "Average runtime. Computational complexity. Convergence
behaviour. Comparison with baseline methods."

Two things are measured and one is derived:

* wall-clock and CPU time per stage (:class:`Timer`, :class:`StageProfiler`),
  reported as mean +- std over images and seeds;
* peak resident memory (``resource.getrusage``) and, when torch is present,
  peak VRAM;
* the analytic cost model in :data:`COMPLEXITY`, which is what actually answers
  "computational complexity" -- a measured second count is not a complexity
  claim, and reviewers notice when the two are conflated.
"""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

__all__ = ["Timer", "StageProfiler", "COMPLEXITY", "peak_memory_mb", "peak_vram_mb",
           "complexity_table", "count_parameters", "count_flops",
           "inference_latency", "profile_model"]


#: Analytic per-image cost of each stage.
#: N = pixels in the cover, L = payload bits, k = popcount(mask) in 1..4,
#: P = GA population, G = generations, S = segments, B = scrambling block size.
COMPLEXITY: Dict[str, Dict[str, str]] = {
    "traversal_order": {
        "time": "O(N)", "space": "O(N)",
        "note": "one pass to materialise the visiting order per segment",
    },
    "decomposition_T1_T2": {"time": "O(L)", "space": "O(L)", "note": "elementwise"},
    "decomposition_T3": {
        "time": "O(L + (L/B) log(L/B))", "space": "O(L)",
        "note": "Fisher-Yates over L/B blocks + one gather",
    },
    "decomposition_T4": {"time": "O(L)", "space": "O(L)", "note": "prefix XOR"},
    "embed_one_candidate": {
        "time": "O(N + L)", "space": "O(N)",
        "note": "dominated by the cover copy and the MSE evaluation",
    },
    "ga_search": {
        "time": "O(P*G*(N + L))", "space": "O(P*S + N)",
        "note": "fitness evaluation dominates; memoised on the genotype",
    },
    "extraction": {"time": "O(N + L)", "space": "O(L)", "note": "single pass"},
    "spam_features": {"time": "O(N)", "space": "O(1)", "note": "686-D, 4 directions"},
    "srm_subset": {"time": "O(N * R)", "space": "O(1)", "note": "R residual types"},
    "fld_ensemble": {
        "time": "O(Ltrain * d_sub^2 + d_sub^3) per base learner",
        "space": "O(d_sub^2)",
        "note": "Kodovsky-Fridrich; d_sub << full feature dimension",
    },
}


def peak_memory_mb() -> Optional[float]:
    try:
        import resource
        import sys
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 1024.0 if sys.platform == "linux" else v / 1024.0 / 1024.0
    except Exception:
        return None


def peak_vram_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return None


@dataclass
class Timer:
    """Context manager measuring wall-clock and CPU time."""

    label: str = ""
    wall_s: float = 0.0
    cpu_s: float = 0.0

    def __enter__(self) -> "Timer":
        self._w0 = time.perf_counter()
        self._c0 = time.process_time()
        return self

    def __exit__(self, *exc) -> None:
        self.wall_s = time.perf_counter() - self._w0
        self.cpu_s = time.process_time() - self._c0


@dataclass
class StageProfiler:
    """Accumulates repeated timings per named stage.

    Carries the :class:`~amdt.utils.execution.ExecutionTarget` so every timing
    row is stamped with the hardware that produced it.  A results table where
    the baseline ran locally and the proposed method ran on a T4 is not a
    comparison, and a reviewer reads the speedup as careless or deliberate.
    """

    target: Any = None
    wall: Dict[str, List[float]] = field(default_factory=dict)
    cpu: Dict[str, List[float]] = field(default_factory=dict)
    #: stage -> hardware label, so a mixed-device run is detectable
    hardware: Dict[str, str] = field(default_factory=dict)

    @property
    def _label(self) -> str:
        return getattr(self.target, "label", "unspecified")

    @property
    def _timing_valid(self) -> bool:
        return bool(getattr(self.target, "timing_valid", True))

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        w0, c0 = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            self.wall.setdefault(name, []).append(time.perf_counter() - w0)
            self.cpu.setdefault(name, []).append(time.process_time() - c0)
            prev = self.hardware.setdefault(name, self._label)
            if prev != self._label:
                # Same stage timed on two devices -> the mean is meaningless.
                self.hardware[name] = "MIXED"

    def record(self, name: str, wall_s: float, cpu_s: float = 0.0,
               hardware: Optional[str] = None) -> None:
        self.wall.setdefault(name, []).append(wall_s)
        self.cpu.setdefault(name, []).append(cpu_s)
        self.hardware.setdefault(name, hardware or self._label)

    @property
    def spans_devices(self) -> bool:
        """True when timings came from more than one device."""
        labels = set(self.hardware.values())
        return "MIXED" in labels or len(labels) > 1

    def summary(self) -> List[Dict[str, Any]]:
        rows = []
        for name, vals in self.wall.items():
            cpu = self.cpu.get(name, [])
            hw = self.hardware.get(name, self._label)
            rows.append({
                "stage": name,
                "n": len(vals),
                "wall_mean_s": statistics.fmean(vals),
                "wall_std_s": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "wall_median_s": statistics.median(vals),
                "wall_total_s": sum(vals),
                "cpu_mean_s": statistics.fmean(cpu) if cpu else float("nan"),
                "hardware": hw,
                "threads": getattr(self.target, "threads", None),
                "thread_mode": getattr(self.target, "thread_mode", None),
                # A row that is not timing-valid still reports its number, but
                # flagged, so it can be excluded from the manuscript rather than
                # silently averaged in.
                "timing_valid": self._timing_valid and hw != "MIXED",
                "timing_invalid_reason": (
                    "stage timed on more than one device" if hw == "MIXED"
                    else getattr(self.target, "timing_invalid_reason", None)),
                "complexity_time": COMPLEXITY.get(name, {}).get("time", ""),
                "complexity_space": COMPLEXITY.get(name, {}).get("space", ""),
            })
        return sorted(rows, key=lambda r: -r["wall_total_s"])


def complexity_table() -> List[Dict[str, str]]:
    return [{"stage": k, **v} for k, v in COMPLEXITY.items()]


# --------------------------------------------------------------------------- #
# model-level profiling (CNN steganalysers)
# --------------------------------------------------------------------------- #
def count_parameters(model) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable),
            "frozen_parameters": int(total - trainable)}


def count_flops(model, input_shape: tuple = (1, 1, 512, 512)) -> Optional[float]:
    """FLOPs for one forward pass.

    Tries ``torch.utils.flop_counter`` (no extra dependency) and falls back to
    ``fvcore``.  Returns ``None`` rather than a guess if neither is available --
    a fabricated FLOP count is worse than an absent one.
    """
    try:
        import torch
    except ImportError:
        return None
    x = torch.zeros(*input_shape)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        with FlopCounterMode(display=False) as fc:
            model(x)
        return float(fc.get_total_flops())
    except Exception:
        pass
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(model, x).total())
    except Exception:
        return None


def inference_latency(model, input_shape: tuple = (1, 1, 512, 512),
                      warmup: int = 20, iters: int = 100,
                      device: Optional[str] = None) -> Dict[str, float]:
    """Per-sample latency, measured properly.

    Two things that are routinely got wrong and then reported anyway:
    the first iterations include cuDNN autotuning and lazy kernel loading, so
    they are discarded; and CUDA is asynchronous, so ``torch.cuda.synchronize()``
    must be called before stopping the clock -- otherwise the number is the time
    to *enqueue* the work, not to do it, and it will be absurdly low.
    """
    import torch

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()
    x = torch.zeros(*input_shape, device=dev)

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()

        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            model(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.perf_counter() - t0)

    per_sample = [s / max(input_shape[0], 1) for s in samples]
    return {
        "latency_mean_ms": 1000 * statistics.fmean(per_sample),
        "latency_std_ms": 1000 * (statistics.pstdev(per_sample) if len(per_sample) > 1 else 0.0),
        "latency_p50_ms": 1000 * statistics.median(per_sample),
        "latency_p95_ms": 1000 * sorted(per_sample)[int(0.95 * (len(per_sample) - 1))],
        "throughput_img_s": input_shape[0] / statistics.fmean(samples),
        "device": str(dev),
        "batch_size": int(input_shape[0]),
        "warmup_iters": warmup,
        "timed_iters": iters,
    }


def profile_model(model, input_shape: tuple = (1, 1, 512, 512),
                  device: Optional[str] = None) -> Dict[str, object]:
    """Everything reviewers ask about cost, in one record."""
    rec: Dict[str, object] = {"input_shape": list(input_shape)}
    rec.update(count_parameters(model))
    flops = count_flops(model, input_shape)
    rec["flops_forward"] = flops
    rec["gflops_forward"] = (flops / 1e9) if flops else None
    try:
        rec.update(inference_latency(model, input_shape, device=device))
    except Exception as exc:  # pragma: no cover
        rec["latency_error"] = str(exc)
    rec["peak_vram_mb"] = peak_vram_mb()
    rec["peak_rss_mb"] = peak_memory_mb()
    return rec
