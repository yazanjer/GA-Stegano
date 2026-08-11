"""Execution target, thread pinning and hardware fingerprinting (§13).

Most of this pipeline is CPU work — the GA, the SRM/SPAM feature extraction and
the FLD ensemble never touch a GPU — so the local machine is the default target
and the CPU determinism rules apply to nearly every reported number.

Why the target is declared, never inferred
------------------------------------------
A hosted runtime allocates a different VM each session, so the CPU model varies
run to run and wall-clock numbers stop being comparable.  Numerics survive once
threads are pinned; timings do not.  Any measurement that will appear in a
table belongs on fixed hardware, which means the target has to be a *decision*
recorded in the config, not whatever the process happened to land on.

``env=local``       fixed hardware, threads pinned, timings reportable
``env=colab_cpu``   overflow for memory-heavy exploration; timings NOT reportable
``env=colab_gpu``   CNN steganalysis; GPU timings reportable, CPU ones are not

Thread pinning is a *mode*, not a global
----------------------------------------
Seeding alone is insufficient on CPU: BLAS reduction order changes with thread
count, so the same seed at a different thread setting produces different floats.
Pinning costs real parallelism, so :func:`configure_execution` pins only when
``reported=true`` and leaves exploration multithreaded — and records which mode
produced each result, because a timing measured multithreaded and a metric
measured single-threaded must never share a table row.

Apple Silicon
-------------
MPS is refused for reported runs.  Operator coverage is incomplete and
unsupported ops fall back to CPU silently, which makes latency figures
meaningless, and its determinism guarantees are weaker than CUDA's.  CPU is the
defensible target on a Mac.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

__all__ = ["ExecutionTarget", "configure_execution", "hardware_fingerprint",
           "resolve_n_jobs", "peak_rss_mb", "memory_pressure", "MPSRefused"]

log = logging.getLogger("amdt.execution")

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


class MPSRefused(RuntimeError):
    """Raised when a reported run is pointed at Apple's MPS backend."""


# --------------------------------------------------------------------------- #
@dataclass
class ExecutionTarget:
    """The declared execution environment for this run."""

    name: str = "local"                 # local | colab_cpu | colab_gpu
    device: str = "cpu"                 # cpu | cuda | mps
    reported: bool = True               # will these numbers appear in a table?
    threads: int = 1                    # pinned thread count when reported
    n_jobs: int = 1                     # explicit; never -1
    timing_valid: bool = True
    timing_invalid_reason: Optional[str] = None
    hardware: Dict[str, Any] = field(default_factory=dict)
    thread_mode: str = "pinned"         # pinned | multithreaded

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def label(self) -> str:
        """Short hardware label for the results table's hardware column."""
        hw = self.hardware
        if self.device == "cuda":
            return f"GPU: {hw.get('gpu_name', 'CUDA')}"
        cpu = str(hw.get("cpu_model") or platform.machine())
        cpu = cpu.replace("(R)", "").replace("(TM)", "").strip()
        return f"CPU: {cpu} ({self.threads}t, {self.thread_mode})"

    def invalidate_timings(self, reason: str) -> None:
        self.timing_valid = False
        self.timing_invalid_reason = reason
        log.warning("timings for this run are now invalid: %s", reason)


# --------------------------------------------------------------------------- #
def hardware_fingerprint() -> Dict[str, Any]:
    """Enough hardware detail to tell two machines apart in a results table."""
    rec: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "python": sys.version.split()[0],
        "is_colab": "google.colab" in sys.modules,
    }
    try:
        if sys.platform == "linux":
            for line in open("/proc/cpuinfo"):
                if line.lower().startswith("model name"):
                    rec["cpu_model"] = line.split(":", 1)[1].strip()
                    break
            for line in open("/proc/meminfo"):
                if line.startswith("MemTotal"):
                    rec["memory_total_gb"] = round(int(line.split()[1]) / 1024**2, 2)
                    break
        elif sys.platform == "darwin":
            rec["cpu_model"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            b = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                               text=True, timeout=5).stdout.strip()
            rec["memory_total_gb"] = round(int(b) / 1024**3, 2)
            rec["apple_silicon"] = platform.machine() == "arm64"
    except Exception:
        pass

    try:
        import torch
        rec["torch"] = torch.__version__
        rec["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            rec["gpu_name"] = torch.cuda.get_device_name(0)
            rec["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        rec["mps_available"] = bool(getattr(torch.backends, "mps", None)
                                    and torch.backends.mps.is_available())
    except ImportError:
        rec["torch"] = None
    return rec


def peak_rss_mb() -> Optional[float]:
    """Peak resident memory. The CPU analogue of peak VRAM."""
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 1024.0 if sys.platform == "linux" else v / 1024.0 / 1024.0
    except Exception:
        return None


def memory_pressure() -> Dict[str, float]:
    """Numeric pressure signals for the migration trigger — never a subjective 'busy'."""
    out: Dict[str, float] = {}
    try:
        out["load_avg_1m"] = os.getloadavg()[0]
        out["load_per_core"] = out["load_avg_1m"] / max(os.cpu_count() or 1, 1)
    except Exception:
        pass
    try:
        import psutil
        vm = psutil.virtual_memory()
        out["memory_used_fraction"] = vm.percent / 100.0
        out["memory_available_gb"] = vm.available / 1024**3
    except ImportError:
        if sys.platform == "linux":
            try:
                info = {l.split(":")[0]: int(l.split()[1])
                        for l in open("/proc/meminfo") if ":" in l}
                total, avail = info.get("MemTotal", 0), info.get("MemAvailable", 0)
                if total:
                    out["memory_used_fraction"] = 1.0 - avail / total
                    out["memory_available_gb"] = avail / 1024**2
            except Exception:
                pass
    rss = peak_rss_mb()
    if rss is not None:
        out["peak_rss_mb"] = rss
    return out


# --------------------------------------------------------------------------- #
def resolve_n_jobs(requested: Any, target: "ExecutionTarget",
                   low_ram_gb: float = 16.0) -> int:
    """Turn a config value into an explicit worker count.

    ``-1`` is rejected: it makes results depend on the machine's core count, and
    joblib copies the dataset into every worker, so on a 16 GB host a 1 GB frame
    at ``n_jobs=6`` costs 6 GB before the estimator allocates anything.  That
    failure shows up as swap thrash, not a clean error.
    """
    n = int(requested) if requested is not None else 1
    if n == -1:
        log.warning("n_jobs=-1 makes results machine-dependent; using %d "
                    "(target.n_jobs). Set it explicitly in the config.",
                    target.n_jobs)
        n = target.n_jobs
    if target.reported and target.thread_mode == "pinned" and n != 1:
        log.warning("reported run with pinned threads: forcing n_jobs=1 "
                    "(was %d) so BLAS reduction order is fixed", n)
        n = 1
    mem = target.hardware.get("memory_total_gb")
    if mem and mem <= low_ram_gb and n > 2:
        log.warning("%.0f GB host with n_jobs=%d: joblib copies the data per "
                    "worker. Capping at 2.", mem, n)
        n = 2
    return max(1, n)


# --------------------------------------------------------------------------- #
def configure_execution(cfg=None, name: str = "local", device: str = "cpu",
                        reported: bool = True, threads: int = 1,
                        n_jobs: int = 1, allow_mps: bool = False
                        ) -> ExecutionTarget:
    """Declare the target, pin threads if the numbers will be reported.

    Must run **before** heavy numeric imports do their thread-count discovery,
    i.e. right after ``set_seed`` in the entry point.
    """
    if cfg is not None:
        name = str(getattr(cfg, "name", name))
        device = str(getattr(cfg, "device", device))
        reported = bool(getattr(cfg, "reported", reported))
        threads = int(getattr(cfg, "threads", threads))
        n_jobs = int(getattr(cfg, "n_jobs", n_jobs))
        allow_mps = bool(getattr(cfg, "allow_mps", allow_mps))

    hw = hardware_fingerprint()
    mode = "pinned" if reported else "multithreaded"

    if reported:
        for var in _THREAD_ENV:
            os.environ[var] = str(threads)
        try:
            import torch
            torch.set_num_threads(threads)
            torch.set_num_interop_threads(1)
        except ImportError:
            pass
        except RuntimeError:
            # interop threads can only be set once per process; harmless on resume
            pass
    else:
        log.info("exploration mode: threads left unpinned, timings from this run "
                 "must not appear in a results table")

    target = ExecutionTarget(
        name=name, device=device, reported=reported,
        threads=threads if reported else (os.cpu_count() or 1),
        n_jobs=n_jobs, hardware=hw, thread_mode=mode,
    )

    # --- MPS ---------------------------------------------------------------
    if device == "mps":
        if reported and not allow_mps:
            raise MPSRefused(
                "MPS is refused for reported runs: unsupported ops fall back to "
                "CPU silently, so latency figures are meaningless, and its "
                "determinism guarantees are weaker than CUDA's. Use env=local "
                "(CPU) for anything that will appear in a table, or set "
                "env.allow_mps=true and label the rows explicitly."
            )
        target.invalidate_timings("MPS backend: silent CPU fallback")

    # --- hosted runtimes ---------------------------------------------------
    if name.startswith("colab") and device != "cuda":
        target.invalidate_timings(
            "hosted CPU runtime: the VM's CPU model varies per session, so "
            "wall-clock is not comparable across runs")
    if hw.get("is_colab") and name == "local":
        log.warning("env=local but this process is running inside Colab. The "
                    "declared target and the actual hardware disagree; set "
                    "env=colab_cpu or env=colab_gpu.")
        target.invalidate_timings("declared env=local while running on a hosted VM")

    if device == "cuda" and not hw.get("cuda_available"):
        raise RuntimeError("env.device=cuda but no CUDA device is visible")

    log.info("execution target: %s | %s | threads=%s (%s) | n_jobs=%d | "
             "timings %s", name, target.label, target.threads, mode, n_jobs,
             "valid" if target.timing_valid else
             f"INVALID ({target.timing_invalid_reason})")
    return target
