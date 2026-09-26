#!/usr/bin/env python
"""Payload-statistics, keystream-reuse and header-signature study (revision 2).

    python revision/targeted_v2.py --out runs/targeted --workers 30

A. Payload statistics of each decomposition variant (bias, lag-1
   autocorrelation, binary entropy of the transformed payload) and chi-square /
   RS / WS attacks on the stego image (plus a sequential chi-square and a direct
   LSB-bias z-test on the embedded region), for uniform, ASCII-text and all-zero
   payloads, 100 BOSSBase covers, 0.1 bpp.  The embedding path is fixed
   (raster, LSB plane, four segments) so that only the decomposition varies.
B. Keystream reuse across messages: two different ASCII payloads embedded
   under the same key, with (v1 behaviour) and without (v2) nonce reuse.  The
   statistic is the bias of d_i = (s_i ^ s_{i-1}) ^ (s'_i ^ s'_{i-1}), which
   equals the XOR of the two scrambled plaintexts when the keystream repeats.
C. Header signature: fraction of stego/cover images whose last 8 LSBs equal
   the v1 magic byte 0xA7, and the per-position mean of the v2 header bits.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, csv, json, multiprocessing as mp, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KEY, cover_sets, load, payload_bits  # noqa: E402
from amdt.stego.codec import Chromosome, embed, header_bit_count  # noqa: E402
from amdt.stego.decomposition import decompose, derive_subkey, segment_message  # noqa: E402
from amdt.steganalysis.targeted import chi_square_attack, payload_uniformity, rs_analysis, weighted_stego_estimate  # noqa: E402

TEXT = (b"It is a truth universally acknowledged, that a single man in possession of a good fortune, "
        b"must be in want of a wife. However little known the feelings or views of such a man may be on his "
        b"first entering a neighbourhood, this truth is so well fixed in the minds of the surrounding families, "
        b"that he is considered the rightful property of some one or other of their daughters. ")
VARIANTS = {"no_decomp": {}, "T1T2_only": {"alpha": 1, "beta": 1}, "T3_only": {"sigma": 1, "block_idx": 1},
            "T4_only": {"delta": 1}, "full": {"sigma": 1, "block_idx": 1, "delta": 1}}
ATTACKED = ("no_decomp", "full")


def _payload(kind, n, cid, off=0):
    if kind == "uniform":
        return payload_bits(cid, 0.1, 0)[:n]
    if kind == "all_zeros":
        return np.zeros(n, dtype=np.uint8)
    t = np.unpackbits(np.frombuffer(TEXT, dtype=np.uint8))
    return np.resize(np.roll(t, off), n).astype(np.uint8)


def task_a(args):
    path, kind = args
    cid = Path(path).stem
    x = load(Path(path))
    n = int(round(0.1 * x.size))
    m = _payload(kind, n, cid)
    nonce = int(cid).to_bytes(8, "big")
    rows = []
    for v, genes in VARIANTS.items():
        chroms = [Chromosome(direction=0, mask=1, **genes) for _ in range(4)]
        t = np.concatenate([decompose(seg, c.decomposition, derive_subkey(KEY, nonce, s))
                            for s, (c, seg) in enumerate(zip(chroms, segment_message(m, 4)))])
        pu = payload_uniformity(t)
        row = {"cover": cid, "payload": kind, "variant": v, "bias": pu["bias"], "rho1": pu["autocorr_lag1"],
               "entropy": pu["shannon_entropy"], "chi2_p": "", "rs_rate": "", "ws_rate": ""}
        if v in ATTACKED:
            res = embed(x, m, chroms, KEY, nonce)
            st = res.stego
            # attacks on the embedded region of segment 0 (raster path, known here
            # because the path is fixed for this study): sequential chi-square
            # (Westfeld-Pfitzmann; p_embedded near 1 = consistent with embedding of
            # random bits, near 0 = not) and a direct LSB-plane bias test
            k = segment_message(m, 4)[0].size
            region = st.reshape(-1)[:k]
            ones = int((region & 1).sum())
            z = (ones - k / 2) / np.sqrt(k / 4)
            row.update(chi2_p=chi_square_attack(region)["p_embedded"], lsb_bias_z=abs(z),
                       rs_rate=rs_analysis(st)["estimated_rate"],
                       ws_rate=weighted_stego_estimate(st)["estimated_rate"])
            if kind == "uniform" and v == "full":
                hb = st.reshape(-1)[-header_bit_count(4):] & 1
                row["header_bits"] = "".join(map(str, hb.tolist()))
                row["magic_hit_stego"] = int(np.packbits(st.reshape(-1)[::-1][:8] & 1)[0] == 0xA7)
                row["magic_hit_cover"] = int(np.packbits(x.reshape(-1)[::-1][:8] & 1)[0] == 0xA7)
        rows.append(row)
    # the same attacks on the untouched cover, for reference
    k = segment_message(m, 4)[0].size
    region = x.reshape(-1)[:k]
    z = (int((region & 1).sum()) - k / 2) / np.sqrt(k / 4)
    rows.append({"cover": cid, "payload": kind, "variant": "cover", "bias": "", "rho1": "", "entropy": "",
                 "chi2_p": chi_square_attack(region)["p_embedded"], "lsb_bias_z": abs(z),
                 "rs_rate": rs_analysis(x)["estimated_rate"], "ws_rate": weighted_stego_estimate(x)["estimated_rate"]})
    return rows


def task_b(args):
    path, reuse = args
    cid = Path(path).stem
    n = int(round(0.1 * 512 * 512)) // 4
    m1, m2 = _payload("ascii", n, cid, 0), _payload("ascii", n, cid, 811)
    c = Chromosome(sigma=1, block_idx=1, delta=1)
    n1 = int(cid).to_bytes(8, "big")
    n2 = n1 if reuse else (int(cid) + 10**6).to_bytes(8, "big")
    s1 = decompose(m1, c.decomposition, derive_subkey(KEY, n1, 0))
    s2 = decompose(m2, c.decomposition, derive_subkey(KEY, n2, 0))
    d = (s1[1:] ^ s1[:-1]) ^ (s2[1:] ^ s2[:-1])
    ref = decompose(m1, c.decomposition.__class__(sigma=1, block_idx=1), derive_subkey(KEY, n1, 0)) ^ \
        decompose(m2, c.decomposition.__class__(sigma=1, block_idx=1), derive_subkey(KEY, n1, 0))
    return {"cover": cid, "keystream": "reused (v1)" if reuse else "fresh nonce (v2)",
            "d_bias": abs(float(d.mean()) - 0.5), "match_plaintext_xor": float(np.mean(d == ref[1:]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=30)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    covers = [str(p) for p in cover_sets()["quality"]]
    with mp.get_context("fork").Pool(a.workers) as pool:
        ra = [r for rs in pool.map(task_a, [(c, k) for c in covers for k in ("uniform", "ascii_text", "all_zeros")]) for r in rs]
        rb = pool.map(task_b, [(c, r) for c in covers for r in (True, False)])
    keys = sorted({k for r in ra for k in r})
    with open(out / "payload_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(ra)
    with open(out / "keystream_reuse.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rb[0])); w.writeheader(); w.writerows(rb)
    print("done", len(ra), len(rb))


if __name__ == "__main__":
    main()
