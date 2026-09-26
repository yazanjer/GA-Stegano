# Revision 2 (IJIES paper ID 20265582): reproduction scripts

This folder regenerates every table and figure in the revised manuscript
("Adaptive Multi-Directional Traversal Path Optimization for Image Steganography
Using Genetic Algorithms with Enhanced Message Decomposition Strategy") from raw
per-image measurements. Nothing in the manuscript is typed in by hand. Each number
comes from the CSVs these scripts write.

## Data

BOSSBase 1.01 (10,000 grayscale 512 x 512 PGM images). The archive is checked by
SHA-256 (`9823e888d4cfa36e94a58676416ea840de3edf1eb663f605451abae3ba8c8524`). The
covers used are listed in `cover_manifest.json`, with a per-image SHA-256, in every
run directory.

| Subset | Covers | Seeds | Used for |
|---|---|---|---|
| Q | 100 | 5 (0-4) | imperceptibility, ablation, statistics (n = 500 paired observations) |
| S | 1,000 (+1,000 for SRNet) | 1 | full-SRM ensemble and SRNet steganalysis |

Payloads are 0.05, 0.1, 0.2 and 0.4 bpp. Subsets are drawn with split seed 20260926.
Per-run seeds come from the cover index and the seed. The message does not depend
on the method, so every method embeds the same message in the same cover
(see `revision/common.py`).

## Commands (in the order they were run)

```bash
export BOSS_DIR=/workspace/data/BOSSbase_1.01      # unpacked PGMs (default path)
python revision/run_v2.py quality  --out runs/quality  --workers 30      # Tables 6, 7, 13; Fig. 2
python revision/run_v2.py ablation --out runs/ablation --workers 30      # Tables 8, 9; Fig. 4
python revision/targeted_v2.py     --out runs/targeted --workers 30      # Table 10 (payload statistics, keystream reuse)
python revision/run_v2.py stego    --out runs/stego --workers 30 --n-covers 1000   # stego sets for steganalysis
python revision/rsws_v2.py                                                 # header-signature statistics (Section 4.4)
python revision/features_v2.py     --run runs/stego --workers 30           # full 34,671-D SRM features
python revision/fix_srm_layout.py   # only for feature files written before the key-sorting fix (see below)
python revision/classify_v2.py     --run runs/stego                        # FLD ensemble, 5-fold cover-wise CV (Table 11, Fig. 3)
python revision/run_v2.py srnet    --out runs/stego --workers 30           # extra covers at 0.4 bpp for SRNet
python revision/srnet_v2.py        --run runs/stego --methods AMDT,AMDT-D,LSB-M,HILL-STC,FM-PSO-LSB,EvoHILL-STC,S-UNIWARD-STC   # Table 12
python revision/run_v2.py runtime  --out runs/runtime --workers 1 --n-covers 10   # Table 14 (single thread, idle node)
NO_GA_BUDGET=2015 python revision/run_v2.py runtime --out runs/runtime --workers 1 --n-covers 10 --rates 0.1 \
       --methods AMDT:no_ga,AMDT:no_search,AMDT:single_seg,AMDT-D:no_search
python revision/make_tables.py     --runs runs --out runs/tables           # every summary / significance table
```

All runs can be resumed. Finished (method, rate, cover, seed) keys are skipped.
`run_v2.py` writes `<study>_meta.json` with the platform, CPU model, package versions,
GA settings and command line.

## Conventions

* "mean ± SD" means the mean and the SD over the n = covers x seeds observations. The
  seed-to-seed SD of the per-seed means is a separate column (`psnr_seed_sd`).
* Paired tests: a Shapiro-Wilk test picks between the paired t-test and the Wilcoxon
  test. CIs come from a 10,000-sample bootstrap. p-values are Holm-corrected within each
  (proposed method x payload) family. Effect sizes are Cohen's d_z and Cliff's delta.
  `sd_diff` is the SD of the paired differences. `se_diff = sd_diff / sqrt(n)`.
* `no_ga` is a matched-budget random search. It evaluates as many distinct chromosomes
  as the GA did on the same (cover, seed). `no_search` is a single random chromosome.
* P_E = min over thresholds of (P_FA + P_MD)/2 on the majority-vote score.
  Its 95 % CI comes from 2,000 pairwise bootstrap resamples.

## Methods

| Name | Where | Notes |
|---|---|---|
| AMDT | `src/amdt/stego/amdt.py` | 33-bit chromosome per segment. Searched space 2.01e9 (sigma = delta = 1); unconstrained space 8.05e9 |
| AMDT-D | `src/amdt/stego/amdt_d.py` | HILL-cost fitness, receiver-recomputable gating, carry-safe +-1, header version 3 |
| LSB, LSB-M, EA-LSB, PVD, GA-FT | `src/amdt/baselines/classical.py` | EA-LSB ranks by the Sobel magnitude of `x & 0xFE`, so the receiver can recompute it |
| HILL/S-UNIWARD/WOW/MiPOD -SIM and -STC | `src/amdt/baselines/reference.py` | costs from conseal 2025.11. STC: `src/amdt/stego/stc.py` (h = 10) |
| EvoHILL-STC | `src/amdt/baselines/reference.py` | Wang, Yi & Wu (2026) evolved cost + STC |
| FM-PSO-LSB | `src/amdt/baselines/reference.py` | Aljughaiman & Alrawashdeh, Sci. Rep. 16:4922 (2026) |
| Full SRM | `src/amdt/steganalysis/srm_full.py` | sealwatch 2025.9, with a bit-identical numba co-occurrence counter |

## Keyed layers and header

T3/T4 use an HMAC-SHA256 counter-mode keystream. The per-segment subkey is
K_s = HMAC(K, "AMDT-seg" || N || s), with a fresh 64-bit nonce N for each message.
The header is `nonce(64) | ENC(version 4 | nseg 8 | len 24 | genes 33*S) | tag(16)`,
which is 116 + 33*S bits. The fixed magic byte and CRC of the first version have been
removed. A wrong key or a tampered header is rejected by the tag.

## Note on the SRM feature layout

sealwatch 2025.9 fills its last few submodels from a Python set. Their order in the
concatenated vector therefore changed with `PYTHONHASHSEED` from one
`features_v2.py` invocation to the next. Within a file the order was constant,
because the workers are forked. The feature files of this run came from several
invocations on three machines, so `fix_srm_layout.py` put every file into the
sorted-name layout before classification. After the fix, cover features computed
independently on the three machines are identical, and randomly chosen stego rows
match a fresh computation. `srm_full` now always sorts the names; this is checked by
`test_srm_layout_does_not_depend_on_hash_seed`.

## Released results (`results/revision2/`)

Raw per-image CSVs: `quality.csv.gz`, `ablation.csv.gz`, `stego.csv.gz`,
`srnet_embedding.csv.gz`, `runtime.csv`, `payload_stats.csv`,
`keystream_reuse.csv`. Steganalysis: `steganalysis_srm.csv` (+ `roc_srm_part*.csv.gz`,
fold assignment `srm_folds.csv`), `steganalysis_srnet.csv`, `srnet_history.csv`.
Summaries: `tables/*.csv`, `runtime_summary.csv`, `runtime_by_rate.csv`,
`runtime_facts.json`, `rsws.csv`, `header_stats.json`. Manifests and environment:
`cover_manifest_*.json` (per-image SHA-256), `*_meta.json`. The pretrained SRNet is in
`models/srnet_pretrained_lsbm05.pt`. That file and `stego.csv.gz` are stored in 3 MB
parts; `models/README.md` shows how to join them and the SHA-256 of each joined file.
