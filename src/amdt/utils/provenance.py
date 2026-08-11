"""Environment and hardware provenance capture.

Reviewer 1, comment 5: "Hardware specifications. MATLAB version. Random seed
policy."  The manuscript is being ported to Python, so the MATLAB version is
replaced by the full interpreter + library manifest.  ``capture()`` is called
once per run and its output is written to ``<run_dir>/provenance.json`` and
rendered into the reproducibility table of the rebuttal.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

__all__ = ["capture", "write", "hardware_string", "git_commit"]

_TRACKED = (
    "numpy", "scipy", "scikit-learn", "scikit-image", "opencv-python",
    "PyWavelets", "matplotlib", "pandas", "torch", "torchvision",
    "hydra-core", "omegaconf", "wandb", "optuna", "Pillow",
)


def git_commit(root: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root or Path(__file__).resolve().parents[3]),
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _package_versions() -> Dict[str, Optional[str]]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover
        return {}
    out: Dict[str, Optional[str]] = {}
    for name in _TRACKED:
        try:
            out[name] = version(name)
        except Exception:
            out[name] = None
    return out


def _cpu_model() -> Optional[str]:
    try:
        if sys.platform == "linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    return platform.processor() or None


def _memory_gb() -> Optional[float]:
    try:
        if sys.platform == "linux":
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal"):
                    return round(int(line.split()[1]) / 1024 / 1024, 2)
        elif sys.platform == "darwin":
            b = subprocess.run(["sysctl", "-n", "hw.memsize"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
            return round(int(b) / 1024**3, 2)
    except Exception:
        pass
    return None


def _gpu_info() -> Dict[str, object]:
    info: Dict[str, object] = {"available": False}
    try:
        import torch
    except ImportError:
        info["torch"] = None
        return info
    info["torch"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        info["available"] = True
        info["devices"] = [
            {
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(i))),
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


def capture(extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Everything a reviewer needs to reproduce or price a run."""
    rec: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "memory_total_gb": _memory_gb(),
        "gpu": _gpu_info(),
        "packages": _package_versions(),
        "git_commit": git_commit(),
        "env": {
            k: os.environ.get(k)
            for k in ("PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG", "OMP_NUM_THREADS")
        },
    }
    if extra:
        rec.update(extra)
    return rec


def hardware_string(rec: Optional[Dict[str, object]] = None) -> str:
    """One-line hardware description for the manuscript's setup paragraph."""
    r = rec or capture()
    gpu = r.get("gpu", {})
    gpu_txt = "no GPU"
    if isinstance(gpu, dict) and gpu.get("available"):
        d = gpu.get("devices", [{}])[0]
        gpu_txt = f"{d.get('name')} ({d.get('total_memory_gb')} GB)"
    return (
        f"{r.get('cpu_model')} x{r.get('cpu_count_logical')} logical cores, "
        f"{r.get('memory_total_gb')} GB RAM, {gpu_txt}; "
        f"{r.get('implementation')} {r.get('python','').split()[0]} on {r.get('platform')}"
    )


def write(path: str | Path, extra: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    rec = capture(extra)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=2, default=str))
    return rec
