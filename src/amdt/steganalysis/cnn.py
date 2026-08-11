"""CNN steganalysers: Yedroudj-Net and SRNet (PyTorch).

Reviewer 1, comment 3: "CNN-based steganalyzers / Deep-learning steganalysis
methods".

Both architectures follow their papers:

* **Yedroudj-Net** (Yedroudj, Comby & Chaumont, ICASSP 2018) -- 30 fixed SRM
  high-pass kernels, TLU (truncated linear unit) after the preprocessing layer,
  five convolutional blocks with BN and absolute-value / scaling layers, three
  fully connected layers.  ~500k parameters; trains on a single mid-range GPU.
* **SRNet** (Boroumand, Chen & Fridrich, TIFS 2019) -- fully learned front end
  (no fixed kernels), 12 layers in four types with residual shortcuts and no
  pooling in the first seven layers so the stego signal is not smoothed away.
  ~4.8M parameters; needs a GPU and a curriculum (train at high payload, then
  fine-tune down) to converge.

Both are trained on **paired** cover/stego batches with cover-wise splits
(:func:`amdt.data.dataset.cover_wise_split`), which is what prevents the model
from memorising covers.  Selection is on validation ``P_E``; the test set is
touched once.

torch is a hard requirement of this module -- importing it without torch raises
immediately rather than silently degrading, so a missing GPU can never be
mistaken for a completed CNN experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "CNN steganalysis requires PyTorch. Install a CUDA build matching your "
        "driver, e.g. `pip install torch --index-url "
        "https://download.pytorch.org/whl/cu121`, then re-run."
    ) from exc

from ..utils.seeding import derive_seed, torch_dataloader_kwargs
from .metrics import DetectionMetrics, evaluate_scores

__all__ = ["SRM_KERNELS", "YedroudjNet", "SRNet", "PairedStegoDataset",
           "TrainConfig", "train_cnn", "evaluate_cnn", "build_model"]


# --------------------------------------------------------------------------- #
# fixed high-pass front end (Yedroudj-Net)
# --------------------------------------------------------------------------- #
def _srm_basis() -> np.ndarray:
    """30 5x5 SRM high-pass kernels (the standard SRM filter bank)."""
    k = np.zeros((30, 5, 5), dtype=np.float32)
    i = 0

    def put(mat, norm):
        nonlocal i
        m = np.zeros((5, 5), dtype=np.float32)
        a = np.asarray(mat, dtype=np.float32)
        o = (5 - a.shape[0]) // 2
        m[o:o + a.shape[0], o:o + a.shape[1]] = a
        k[i] = m / norm
        i += 1

    # 1st order, 8 directions
    for d in range(4):
        base = np.rot90(np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]), d)
        put(base, 1.0)
        put(-base, 1.0)
    # 2nd order, 4 orientations
    for d in range(4):
        put(np.rot90(np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]]), d), 2.0)
    # 3rd order, 4 orientations
    for d in range(4):
        put(np.rot90(np.array([[0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [1, -3, 3, -1, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0]]), d), 3.0)
    # square / edge 3x3, rotations
    sq3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
    put(sq3, 4.0)
    edge3 = np.array([[0, 0, 0], [2, -4, 2], [-1, 2, -1]])
    for d in range(4):
        put(np.rot90(edge3, d), 4.0)
    # square 5x5
    put(np.array([[-1, 2, -2, 2, -1],
                  [2, -6, 8, -6, 2],
                  [-2, 8, -12, 8, -2],
                  [2, -6, 8, -6, 2],
                  [-1, 2, -2, 2, -1]]), 12.0)
    # edge 5x5, rotations
    e5 = np.array([[-1, 2, -2, 0, 0],
                   [2, -6, 8, -2, 0],
                   [-2, 8, -12, 0, 0],
                   [0, -2, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
    for d in range(4):
        if i >= 30:
            break
        put(np.rot90(e5, d), 12.0)
    while i < 30:                       # pad with rotations of sq3 if short
        put(np.rot90(sq3, i % 4), 4.0)
    return k


SRM_KERNELS = _srm_basis()


class _TLU(nn.Module):
    """Truncated linear unit: clamp to [-T, T]. Keeps the stego residual in range."""

    def __init__(self, t: float = 3.0) -> None:
        super().__init__()
        self.t = t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, -self.t, self.t)


class YedroudjNet(nn.Module):
    def __init__(self, tlu_threshold: float = 3.0, freeze_front: bool = True) -> None:
        super().__init__()
        self.pre = nn.Conv2d(1, 30, 5, padding=2, bias=False)
        with torch.no_grad():
            self.pre.weight.copy_(torch.from_numpy(SRM_KERNELS).unsqueeze(1))
        self.pre.weight.requires_grad = not freeze_front
        self.tlu = _TLU(tlu_threshold)

        self.b1 = nn.Sequential(nn.Conv2d(30, 30, 5, padding=2), nn.BatchNorm2d(30))
        self.b2 = nn.Sequential(nn.Conv2d(30, 30, 5, padding=2), nn.BatchNorm2d(30),
                                nn.ReLU(inplace=True), nn.AvgPool2d(5, 2, padding=2))
        self.b3 = nn.Sequential(nn.Conv2d(30, 32, 3, padding=1), nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True), nn.AvgPool2d(5, 2, padding=2))
        self.b4 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True), nn.AvgPool2d(5, 2, padding=2))
        self.b5 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
                                nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128, 256), nn.ReLU(inplace=True),
                                nn.Linear(256, 1024), nn.ReLU(inplace=True),
                                nn.Linear(1024, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.tlu(self.pre(x))
        x = torch.abs(self.b1(x))
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        return self.fc(x)


class _SRNetT1(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.c = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return F.relu(self.bn(self.c(x)))


class _SRNetT2(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.a = _SRNetT1(c, c)
        self.c = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        return x + self.bn(self.c(self.a(x)))


class _SRNetT3(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.a = _SRNetT1(cin, cout)
        self.c = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.pool = nn.AvgPool2d(3, 2, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1, stride=2, bias=False)
        self.bns = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bns(self.skip(x)) + self.pool(self.bn(self.c(self.a(x))))


class _SRNetT4(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.a = _SRNetT1(cin, cout)
        self.c = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        return self.bn(self.c(self.a(x))).mean(dim=(2, 3))


class SRNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l1 = _SRNetT1(1, 64)
        self.l2 = _SRNetT1(64, 16)
        self.l3 = nn.Sequential(*[_SRNetT2(16) for _ in range(5)])
        self.l8 = _SRNetT3(16, 16)
        self.l9 = _SRNetT3(16, 64)
        self.l10 = _SRNetT3(64, 128)
        self.l11 = _SRNetT3(128, 256)
        self.l12 = _SRNetT4(256, 512)
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.l3(self.l2(self.l1(x)))
        x = self.l11(self.l10(self.l9(self.l8(x))))
        return self.fc(self.l12(x))


def build_model(name: str) -> nn.Module:
    return {"yedroudj": YedroudjNet, "srnet": SRNet}[name.lower()]()


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
class PairedStegoDataset(torch.utils.data.Dataset):
    """Yields (cover, stego) pairs so every batch is class-balanced by design."""

    def __init__(self, covers: Sequence[np.ndarray], stegos: Sequence[np.ndarray]) -> None:
        if len(covers) != len(stegos):
            raise ValueError("covers and stegos must be paired")
        self.covers = covers
        self.stegos = stegos

    def __len__(self) -> int:
        return len(self.covers)

    def __getitem__(self, i: int):
        c = torch.from_numpy(self.covers[i].astype(np.float32)).unsqueeze(0)
        s = torch.from_numpy(self.stegos[i].astype(np.float32)).unsqueeze(0)
        return c, s


def _collate(batch):
    c = torch.stack([b[0] for b in batch])
    s = torch.stack([b[1] for b in batch])
    x = torch.cat([c, s], dim=0)
    y = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                   torch.ones(len(batch), dtype=torch.long)])
    return x, y


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    model: str = "yedroudj"
    epochs: int = 100
    batch_pairs: int = 16
    lr: float = 1e-3
    weight_decay: float = 5e-4
    momentum: float = 0.9
    optimizer: str = "adamax"
    scheduler: str = "cosine"
    seed: int = 0
    num_workers: int = 2
    amp: bool = True
    checkpoint_dir: Optional[str] = None       # point at Drive on Colab
    checkpoint_every: int = 1
    early_stop_patience: int = 20
    curriculum_from: Optional[str] = None      # path to weights from a higher payload

    def as_dict(self) -> Dict[str, object]:
        return dict(self.__dict__)


@dataclass
class TrainHistory:
    epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_pe: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    epoch_time_s: List[float] = field(default_factory=list)


def _make_loader(ds: PairedStegoDataset, cfg: TrainConfig, shuffle: bool):
    kw = torch_dataloader_kwargs(cfg.seed)
    return torch.utils.data.DataLoader(
        ds, batch_size=cfg.batch_pairs, shuffle=shuffle, collate_fn=_collate,
        num_workers=cfg.num_workers, drop_last=shuffle, pin_memory=True, **kw
    )


def _ckpt_path(cfg: TrainConfig, tag: str) -> Optional[Path]:
    if not cfg.checkpoint_dir:
        return None
    p = Path(cfg.checkpoint_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{cfg.model}_{tag}.pt"


def _save_atomic(state: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)               # a disconnect mid-write cannot corrupt the ckpt


def train_cnn(train_ds: PairedStegoDataset, val_ds: PairedStegoDataset,
              cfg: TrainConfig, device: Optional[str] = None,
              tracker=None, optuna_trial=None, supervisor_stub=None
              ) -> Tuple[nn.Module, TrainHistory, Dict[str, object]]:
    """Train with full-state checkpointing and auto-resume (Colab-safe).

    Parameters
    ----------
    tracker : optional :class:`amdt.utils.tracking.Tracker`; receives per-epoch
        loss, val ``P_E``, learning rate and epoch wall-clock.
    optuna_trial : optional Optuna trial; intermediate ``P_E`` is reported each
        epoch and ``TrialPruned`` is raised cleanly so a pruned trial exits
        rather than crashing the study.
    supervisor_stub : optional :class:`amdt.experiments.supervisor.PatchStub`;
        polled between epochs for a patch pushed by the out-of-runtime watcher.
        The stub sees validation metrics only -- ``val_ds``, never a test set.
    """
    import time

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(cfg.model).to(dev)

    if cfg.curriculum_from:
        model.load_state_dict(torch.load(cfg.curriculum_from, map_location=dev)["model_state_dict"])

    if cfg.optimizer == "adamax":
        opt = torch.optim.Adamax(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                              weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs) \
        if cfg.scheduler == "cosine" else torch.optim.lr_scheduler.ConstantLR(opt, 1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and dev.type == "cuda")

    hist = TrainHistory()
    start_epoch, best_pe, stale = 0, float("inf"), 0
    last = _ckpt_path(cfg, "last")
    best = _ckpt_path(cfg, "best")

    if last is not None and last.exists():                       # auto-resume
        st = torch.load(last, map_location=dev)
        model.load_state_dict(st["model_state_dict"])
        opt.load_state_dict(st["optimizer_state_dict"])
        sched.load_state_dict(st["scheduler_state_dict"])
        if st.get("scaler_state_dict"):
            scaler.load_state_dict(st["scaler_state_dict"])
        torch.set_rng_state(st["rng_state"].cpu())
        np.random.set_state(st["numpy_rng_state"])
        start_epoch = int(st["epoch"]) + 1
        best_pe = float(st.get("best_pe", float("inf")))
        if supervisor_stub is not None:
            supervisor_stub.rounds_used = int(st.get("supervisor_rounds_used", 0))

    tl = _make_loader(train_ds, cfg, True)
    vl = _make_loader(val_ds, cfg, False)
    crit = nn.CrossEntropyLoss()

    for ep in range(start_epoch, cfg.epochs):
        t0 = time.perf_counter()
        model.train()
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * y.size(0)
            n += y.size(0)
        sched.step()

        m = evaluate_cnn(model, vl, dev)
        hist.epoch.append(ep)
        hist.train_loss.append(tot / max(n, 1))
        hist.val_pe.append(m.p_e)
        hist.lr.append(opt.param_groups[0]["lr"])
        hist.epoch_time_s.append(time.perf_counter() - t0)

        if tracker is not None:
            tracker.log({"train/loss": hist.train_loss[-1], "val/p_e": m.p_e,
                         "val/accuracy": m.accuracy, "val/auc": m.auc,
                         "lr": hist.lr[-1], "epoch_time_s": hist.epoch_time_s[-1]},
                        step=ep)

        if optuna_trial is not None:
            optuna_trial.report(m.p_e, ep)
            if optuna_trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        state = {
            "epoch": ep, "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler.is_enabled() else None,
            "rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "val_pe": m.p_e, "best_pe": min(best_pe, m.p_e), "config": cfg.as_dict(),
            "optuna_trial_id": optuna_trial.number if optuna_trial is not None else None,
            # Persisted in the checkpoint, not in memory: a Colab reconnect must
            # not silently reset the supervisor's revision budget.
            "supervisor_rounds_used": (supervisor_stub.rounds_used
                                       if supervisor_stub is not None else 0),
        }
        if last is not None and ep % cfg.checkpoint_every == 0:
            _save_atomic(state, last)
        if m.p_e < best_pe:
            best_pe, stale = m.p_e, 0
            if best is not None:
                _save_atomic(state, best)
        else:
            stale += 1
            if cfg.early_stop_patience and stale >= cfg.early_stop_patience:
                break

        # Watcher decides, stub applies. Nothing outside the VM can reach in.
        if supervisor_stub is not None:
            supervisor_stub.poll_and_apply(epoch=ep, val_pe=m.p_e,
                                           train_loss=hist.train_loss[-1],
                                           model=model, optimizer=opt)

    if best is not None and best.exists():
        model.load_state_dict(torch.load(best, map_location=dev)["model_state_dict"])

    prof = {
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "peak_vram_mb": (torch.cuda.max_memory_allocated() / 1024**2) if dev.type == "cuda" else None,
        "device": str(dev),
        "epochs_run": len(hist.epoch),
        "mean_epoch_s": float(np.mean(hist.epoch_time_s)) if hist.epoch_time_s else float("nan"),
    }
    return model, hist, prof


@torch.no_grad()
def evaluate_cnn(model: nn.Module, loader, device) -> DetectionMetrics:
    model.eval()
    scores, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, 1]
        scores.append(p.cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    l = np.concatenate(labels)
    return evaluate_scores(l, s, threshold=0.5)
