#!/usr/bin/env python
"""SRNet steganalysis (Boroumand, Chen & Fridrich, IEEE TIFS 2019) at 0.4 bpp.

    python revision/srnet_v2.py --run /workspace/runs/stego --methods AMDT,AMDT-D,...

Protocol: 2,000 BOSSBase covers (the 1,000 SRM covers plus 1,000 more from the
same deterministic stego pool), split cover-wise into 1,400 / 200 / 400 for
training / validation / test (the split is shared by every method).  A cover
and its stego always fall in the same split and are presented in the same
mini-batch (16 pairs = 32 images, 512 x 512, no resizing).  Training uses
random D4 augmentation (rotations by 90 degrees and flips, applied identically
to both members of a pair), Adamax with cosine decay, bfloat16 autocast.
Curriculum: with only 1,400 training pairs SRNet does not leave the chance
plateau from random initialisation, so one network is first pre-trained for
30 epochs (lr 1e-3) on the training covers against synthetic +-1 embedding at
0.5 bpp (``srnet_pretrained.pt``; shipped as models/srnet_pretrained_lsbm05.pt)
and then fine-tuned for 60 epochs (lr 5e-4) for each method.  The epoch with
the lowest validation P_E is evaluated once on the test split.  Output: ``steganalysis_srnet.csv`` and ``srnet_history.csv``.
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import cover_sets, load  # noqa: E402
from amdt.steganalysis.cnn import SRNet  # noqa: E402
from amdt.steganalysis.metrics import evaluate_scores  # noqa: E402


def _png(p):
    from PIL import Image
    return np.array(Image.open(p))


def _aug_gpu(x, g):
    """x: (B, 2, H, W) pairs; the same random D4 transform per pair."""
    out = torch.empty_like(x)
    ks = torch.randint(0, 4, (x.shape[0],), generator=g).tolist()
    fs = torch.randint(0, 2, (x.shape[0],), generator=g).tolist()
    for i, (k, f) in enumerate(zip(ks, fs)):
        y = torch.rot90(x[i], k, dims=(-2, -1))
        out[i] = torch.flip(y, dims=(-1,)) if f else y
    return out


def _to_input(pairs):
    # (B,2,H,W) -> (2B,1,H,W) ordered [covers..., stegos...]
    B = pairs.shape[0]
    x = torch.cat([pairs[:, 0], pairs[:, 1]])[:, None].float()
    y = torch.cat([torch.zeros(B), torch.ones(B)]).long().to(pairs.device)
    return x.contiguous(memory_format=torch.channels_last), y


def _lsbm(c, rate, gen):
    """Synthetic +-1 embedding (LSB matching) at ``rate`` changes-per-pixel/2."""
    u = torch.rand(c.shape, device=c.device, generator=gen)
    s = torch.where(torch.rand(c.shape, device=c.device, generator=gen) < 0.5, -1, 1).to(torch.int16)
    d = torch.where(u < rate / 2, s, torch.zeros_like(s))
    x = c.to(torch.int16) + d
    x = torch.where(x < 0, 1, torch.where(x > 255, 254, x))
    return x.to(torch.uint8)


class Trainer:
    def __init__(self, a, dev, covers_gpu, split):
        self.a, self.dev, self.C, self.split = a, dev, covers_gpu, split
        self.g = torch.Generator().manual_seed(a.seed)
        self.gg = torch.Generator(device=dev).manual_seed(a.seed)

    def evaluate(self, net, S, idx):
        net.eval(); sc, yy = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for j in range(0, len(idx), 16):
                ii = torch.as_tensor(idx[j:j + 16], device=self.dev)
                x, y = _to_input(torch.stack([self.C[ii], S[ii]], 1))
                sc.append(torch.softmax(net(x).float(), 1)[:, 1].cpu().numpy()); yy.append(y.cpu().numpy())
        return evaluate_scores(np.concatenate(yy), np.concatenate(sc), 0.5)

    def fit(self, net, stego_fn, epochs, lr, tag, log_hist, val_S=None):
        a = self.a
        opt = torch.optim.Adamax(net.parameters(), lr=lr, weight_decay=2e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        tr, va, _ = self.split
        best, best_state, t0 = 1.0, None, time.time()
        for ep in range(1, epochs + 1):
            net.train()
            perm = tr[torch.randperm(len(tr), generator=self.g).numpy()]
            tot, nb = 0.0, 0
            for j in range(0, len(perm) - a.pairs + 1, a.pairs):
                ii = torch.as_tensor(perm[j:j + a.pairs], device=self.dev)
                pairs = _aug_gpu(torch.stack([self.C[ii], stego_fn(ii)], 1), self.g)
                x, y = _to_input(pairs)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = F.cross_entropy(net(x).float(), y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tot += float(loss); nb += 1
            sched.step()
            vS = val_S if val_S is not None else None
            if vS is None:   # synthetic validation for pre-training
                vS = torch.zeros_like(self.C)
                vi = torch.as_tensor(va, device=self.dev)
                g2 = torch.Generator(device=self.dev).manual_seed(999)
                vS[vi] = _lsbm(self.C[vi], a.pre_rate, g2)
            v = self.evaluate(net, vS, va)
            if v.p_e < best:
                best, best_state = v.p_e, {k: t.detach().clone() for k, t in net.state_dict().items()}
            log_hist.writerow({"method": tag, "epoch": ep, "train_loss": tot / max(1, nb), "val_pe": v.p_e,
                               "val_acc": v.accuracy, "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0})
            print(f"{tag} ep{ep} loss={tot/max(1,nb):.4f} valPE={v.p_e:.3f} best={best:.3f} {time.time()-t0:.0f}s", flush=True)
        net.load_state_dict(best_state)
        return best


def load_stegos(a, method, ids, dev):
    sdir = Path(a.run) / "stego" / f"{method.replace(':', '_')}@{a.rate}"
    return torch.from_numpy(np.stack([_png(sdir / f"{i}.png") for i in ids])).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--methods", required=True)
    ap.add_argument("--rate", type=float, default=0.4)
    ap.add_argument("--pre-epochs", type=int, default=30)
    ap.add_argument("--pre-rate", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    covers = cover_sets()["stego"][:2000]
    ids = [p.stem for p in covers]
    C = torch.from_numpy(np.stack([load(p) for p in covers])).to(dev)
    perm = np.random.default_rng(2026).permutation(2000)
    split = (perm[:1400], perm[1400:1600], perm[1600:])
    run = Path(a.run)
    hist_f = open(run / f"srnet_history{a.tag}.csv", "a", newline="")
    hist = csv.DictWriter(hist_f, fieldnames=["method", "epoch", "train_loss", "val_pe", "val_acc", "lr", "elapsed_s"])
    if hist_f.tell() == 0:
        hist.writeheader()
    T = Trainer(a, dev, C, split)
    pre_path = run / f"srnet_pretrained{a.tag}.pt"
    net = SRNet().to(dev).to(memory_format=torch.channels_last)
    if pre_path.exists():
        net.load_state_dict(torch.load(pre_path, map_location=dev))
    else:
        gpre = torch.Generator(device=dev).manual_seed(a.seed + 1)
        T.fit(net, lambda ii: _lsbm(C[ii], a.pre_rate, gpre), a.pre_epochs, a.lr, "pretrain-LSBM", hist)
        torch.save(net.state_dict(), pre_path)
    base = {k: v.clone() for k, v in net.state_dict().items()}
    out = run / f"steganalysis_srnet{a.tag}.csv"
    for method in a.methods.split(","):
        t0 = time.time()
        S = load_stegos(a, method, ids, dev)
        net.load_state_dict(base)
        bval = T.fit(net, lambda ii: S[ii], a.epochs, a.ft_lr, method, hist, val_S=S)
        hist_f.flush()
        m = T.evaluate(net, S, split[2])
        row = {"method": method, "rate": a.rate, **m.as_dict(), "best_val_pe": bval, "epochs": a.epochs,
               "pre_epochs": a.pre_epochs, "train_pairs": 1400, "val_pairs": 200, "test_pairs": 400,
               "wall_min": (time.time() - t0) / 60, "detector": "SRNet (curriculum from LSBM 0.5 bpp)"}
        new = not out.exists()
        with open(out, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new: w.writeheader()
            w.writerow(row)
        print("RESULT", json.dumps(row), flush=True)
        del S


if __name__ == "__main__":
    main()
