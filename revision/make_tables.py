#!/usr/bin/env python
"""Regenerate every table/figure input of the revised manuscript from raw CSVs.

    python revision/make_tables.py --runs /workspace/runs --out /workspace/runs/tables

Inputs (raw, per-image / per-seed):  quality/quality.csv, ablation/ablation.csv,
stego/stego.csv, stego/steganalysis_srm.csv, stego/steganalysis_srnet.csv,
runtime/runtime.csv, targeted/payload_stats.csv, targeted/keystream_reuse.csv.

Conventions (Reviewer 1, comments 6, 8 and 11):
* "mean +- SD" is the mean and standard deviation over the n = covers x seeds
  observations (image-to-image spread).  The seed-to-seed SD of the per-seed
  means is reported in its own column.
* Paired comparisons are on the per-(cover, seed) differences d:  SD_d is the
  standard deviation of the differences, SE = SD_d / sqrt(n), Cohen's d_z =
  mean(d) / SD_d.  These, not the marginal SDs, are what the paired tests use.
* PSNR is reported to three decimals in the ablation table.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amdt.evaluation.significance import cliffs_delta, holm_bonferroni, power_estimate  # noqa: E402

COMPLETE = ["AMDT", "AMDT-D", "GA-FT", "LSB", "LSB-M", "EA-LSB", "PVD", "FM-PSO-LSB",
            "HILL-STC", "S-UNIWARD-STC", "WOW-STC", "MiPOD-STC", "EvoHILL-STC"]
SIMS = ["HILL-SIM", "S-UNIWARD-SIM", "WOW-SIM", "MiPOD-SIM"]


def boot_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, (n_boot, d.size))
    m = d[idx].mean(1)
    return float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def paired(a, b):
    from scipy import stats
    d = a - b
    n = d.size
    sd = float(d.std(ddof=1))
    sh = float(stats.shapiro(d).pvalue) if 3 <= n <= 5000 and sd > 0 else float("nan")
    normal = bool(np.isnan(sh) or sh > 0.05)
    if sd == 0:
        p, test = 1.0, "none (identical)"
    elif normal:
        p, test = float(stats.ttest_rel(a, b).pvalue), "paired t"
    else:
        p, test = float(stats.wilcoxon(a, b).pvalue), "Wilcoxon"
    lo, hi = boot_ci(d)
    dz = float(d.mean() / sd) if sd > 0 else float("nan")
    return {"n": n, "mean_diff": float(d.mean()), "sd_diff": sd, "se_diff": sd / np.sqrt(n),
            "ci_low": lo, "ci_high": hi, "test": test, "p": p, "shapiro_p": sh, "dz": dz,
            "cliffs_delta": float(cliffs_delta(a, b)), "power": power_estimate(dz, n) if np.isfinite(dz) else float("nan")}


def summarise(df, by=("method", "rate")):
    rows = []
    for key, g in df.groupby(list(by)):
        seedm = g.groupby("seed")["psnr"].mean()
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        r.update({"n": len(g), "psnr_mean": g.psnr.mean(), "psnr_sd": g.psnr.std(ddof=1),
                  "psnr_seed_sd": seedm.std(ddof=1) if len(seedm) > 1 else np.nan,
                  "ssim_mean": g.ssim.mean(), "ssim_sd": g.ssim.std(ddof=1),
                  "mse_mean": g.mse.mean(), "mse_sd": g.mse.std(ddof=1),
                  "eff_mean": g.embedding_efficiency.mean(), "eff_sd": g.embedding_efficiency.std(ddof=1),
                  "changes_mean": g.n_changes.mean(),
                  "hill_median": g.hill_distortion.median(), "hill_mean": g.hill_distortion.mean(),
                  "loghill_mean": np.log10(g.hill_distortion.clip(lower=1e-12)).mean(),
                  "extract_ok": int((g.extraction_ok.astype(str) == "True").sum()),
                  "extract_checked": int(g.extraction_ok.astype(str).isin(["True", "False"]).sum()),
                  "blind": str(g.blind_extractable.iloc[0]),
                  "wall_mean": g.wall_s.mean(), "wall_sd": g.wall_s.std(ddof=1),
                  "evals_mean": pd.to_numeric(g.ga_evaluations, errors="coerce").mean(),
                  "gens_mean": pd.to_numeric(g.ga_generations, errors="coerce").mean()})
        rows.append(r)
    return pd.DataFrame(rows)


def significance(df, proposed, baselines, metric="psnr"):
    out = []
    for rate, g in df.groupby("rate"):
        piv = g.pivot_table(index=["cover", "seed"], columns="method", values=metric)
        fam = []
        for b in baselines:
            if b == proposed or b not in piv or proposed not in piv:
                continue
            ab = piv[[proposed, b]].dropna()
            r = paired(ab[proposed].to_numpy(), ab[b].to_numpy())
            r.update({"proposed": proposed, "baseline": b, "rate": rate, "metric": metric})
            fam.append(r)
        if fam:
            adj, _ = holm_bonferroni([r["p"] for r in fam])
            for r, a in zip(fam, adj):
                r["p_holm"] = float(a)
            out += fam
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R, O = Path(a.runs), Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    facts = {}
    q = pd.read_csv(R / "quality/quality.csv")
    qs = summarise(q); qs.to_csv(O / "quality_summary.csv", index=False)
    sig = pd.concat([significance(q, "AMDT", COMPLETE + SIMS, m) for m in ("psnr", "hill_distortion")] +
                    [significance(q, "AMDT-D", COMPLETE + SIMS, m) for m in ("psnr", "hill_distortion")])
    sig.to_csv(O / "significance.csv", index=False)
    # convergence curves
    conv = {}
    for m in ("AMDT", "AMDT-D", "GA-FT"):
        hs = [json.loads(e).get("hist") for e in q[(q.method == m) & (q.rate == 0.1)].extra]
        hs = [h for h in hs if h]
        if hs:
            L = max(map(len, hs))
            M = np.array([h + [h[-1]] * (L - len(h)) for h in hs], dtype=float)
            conv[m] = {"mean": M.mean(0).tolist(), "q25": np.quantile(M, .25, 0).tolist(),
                       "q75": np.quantile(M, .75, 0).tolist(), "n": len(hs),
                       "gens_mean": float(np.mean([len(h) - 1 for h in hs]))}
    (O / "convergence.json").write_text(json.dumps(conv))
    # ablation (full = the canonical AMDT / AMDT-D rows of the quality run)
    ab_path = R / "ablation/ablation.csv"
    if ab_path.exists():
        ab = pd.read_csv(ab_path)
        full = q[(q.rate == 0.1) & (q.method.isin(["AMDT", "AMDT-D"]))].copy()
        full["method"] = full.method + ":full"
        ab = pd.concat([full, ab])
        s = summarise(ab, ("method",))
        rows = []
        for fam in ("AMDT", "AMDT-D"):
            base = ab[ab.method == f"{fam}:full"].set_index(["cover", "seed"])
            for m in s.method:
                if not m.startswith(fam + ":") or m == f"{fam}:full":
                    continue
                v = ab[ab.method == m].set_index(["cover", "seed"])
                j = base.join(v, lsuffix="_f", rsuffix="_v", how="inner")
                for metric in ("psnr", "hill_distortion"):
                    r = paired(j[f"{metric}_v"].to_numpy(), j[f"{metric}_f"].to_numpy())
                    r.update({"variant": m, "metric": metric})
                    rows.append(r)
        abs_ = pd.DataFrame(rows)
        for fam in ("AMDT", "AMDT-D"):
            for metric in ("psnr", "hill_distortion"):
                msk = abs_.variant.str.startswith(fam + ":") & (abs_.metric == metric)
                if msk.any():
                    abs_.loc[msk, "p_holm"] = holm_bonferroni(abs_.loc[msk, "p"].to_numpy())[0]
        s.to_csv(O / "ablation_summary.csv", index=False)
        abs_.to_csv(O / "ablation_tests.csv", index=False)
    st_path = R / "stego/stego.csv"
    if st_path.exists():
        st = pd.read_csv(st_path)
        summarise(st).to_csv(O / "stego_quality_summary.csv", index=False)
    for f in ("stego/steganalysis_srm.csv", "stego/steganalysis_srnet.csv", "targeted/payload_stats.csv",
              "targeted/keystream_reuse.csv", "stego/rsws.csv"):
        p = R / f
        if p.exists():
            df = pd.read_csv(p)
            if f.endswith("payload_stats.csv"):
                num = ["bias", "rho1", "entropy", "chi2_p", "lsb_bias_z", "rs_rate", "ws_rate"]
                for c in num:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df.groupby(["payload", "variant"])[num].mean().reset_index().to_csv(O / "payload_stats_summary.csv", index=False)
                if "header_bits" in df:
                    hb = df.header_bits.dropna().astype(str)
                    M = np.array([[int(c) for c in s] for s in hb])
                    facts["header_bit_mean_min"] = float(M.mean(0).min()); facts["header_bit_mean_max"] = float(M.mean(0).max())
                    facts["header_n"] = int(M.shape[0])
                    facts["magic_hit_stego"] = float(pd.to_numeric(df.magic_hit_stego, errors="coerce").mean())
                    facts["magic_hit_cover"] = float(pd.to_numeric(df.magic_hit_cover, errors="coerce").mean())
            elif f.endswith("keystream_reuse.csv"):
                df.groupby("keystream")[["d_bias", "match_plaintext_xor"]].agg(["mean", "std"]).to_csv(O / "keystream_reuse_summary.csv")
            else:
                df.to_csv(O / Path(f).name, index=False)
    rt = R / "runtime/runtime.csv"
    if rt.exists():
        d = pd.read_csv(rt)
        d.groupby(["method"]).wall_s.agg(["count", "mean", "std", "median"]).reset_index().to_csv(O / "runtime_summary.csv", index=False)
        d.groupby(["method", "rate"]).wall_s.agg(["count", "mean", "std"]).reset_index().to_csv(O / "runtime_by_rate.csv", index=False)
    (O / "facts.json").write_text(json.dumps(facts, indent=1))
    print("tables written to", O)


if __name__ == "__main__":
    main()
