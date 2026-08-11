# AMDT — Python rebuttal pipeline

Python re-implementation and experimental pipeline for
**"Adaptive Multi-Directional Traversal Path Optimization for Image
Steganography Using Genetic Algorithms with Enhanced Message Decomposition"**
(JST-6070-2025.R1), built to answer every point in Reviewer 1's decision letter
with a generated artifact rather than prose.

It replaces the MATLAB benchmark (`baseline code matlab/`) — the traversal,
bit-plane and GA logic are ports of `CreateHostPixelSeq.m`, `EmbeddingTheMessage.m`
and `GApart.m`, with three bugs in the original traversal fixed and documented
in `src/amdt/stego/traversal.py`.

---

## Reviewer comment → artifact

| # | Reviewer 1 comment | What the code produces | Where |
|---|---|---|---|
| 1 | Strengthen the novelty justification | Formal security argument for T3/T4, key-space accounting, and payload-statistics attacks (χ², RS, WS) run against every decomposition variant and three payload types including the adversarial all-zero case | `tables/tab_targeted.tex`, `tables/tab_ablation.tex`, `results/decomposition_spec.json`, module docstring of `stego/decomposition.py` |
| 2 | Table 2 names scrambling / diffusion with no formalism | Full mathematical definition, algorithm, parameters and header bit-budget for **T1 complement, T2 reversal, T3 keyed block scrambling, T4 keyed diffusion, S segmentation** — all invertibility-tested | `stego/decomposition.py`, `results/decomposition_spec.json` |
| 3 | No recognised steganalysis framework | SPAM-686 and an SRM-style rich model + Kodovský–Fridrich FLD ensemble; Yedroudj-Net and SRNet in PyTorch. Accuracy, precision, recall, F1, ROC/AUC, plus `P_E` and MD@FA5% | `tables/tab_detection.tex`, `tab_detection_cnn.tex`, `figures/fig10_roc.pdf`, `fig11_detectability.pdf` |
| 4 | Only averages and boxplots | Paired t-test **and** Wilcoxon with a Shapiro–Wilk selector, bootstrap CIs, SDs, Holm–Bonferroni correction, Cohen's *d_z*, Cliff's δ, and simulated power | `tables/tab_significance.tex`, `results/significance.csv` |
| 5 | Reproducibility | Dataset manifest with per-image SHA-256, payload sizes in bits **and** as % of capacity, GA iteration counts, hardware, full package manifest, git commit, seed policy | `tables/tab_reproducibility.tex`, `results/reproducibility_manifest.json`, `provenance.json` |
| 6 | Runtime analysis | Per-stage wall-clock (mean ± SD), analytic time/space complexity per stage, GA convergence curves, and matched-budget random-search control | `tables/tab_runtime.tex`, `figures/fig12_runtime.pdf`, `fig13_convergence.pdf`, `results/complexity.csv` |
| 7 | Baselines are only classical | LSB, LSB-M, EA-LSB, GA-FT, PVD **plus** WOW, S-UNIWARD, HILL and MiPOD-lite under a payload-limited optimal simulator | `tables/tab_quality.tex` |

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate

# Local Mac / any CPU box (the default target — GA, SRM, ensemble, stats):
pip install -r requirements-local.txt      # CPU torch wheel

# Colab GPU (adds the CUDA build for CNN steganalysis):
# pip install -r requirements.txt

pytest -q                      # 210 correctness tests, ~1 s
./run_overnight.sh             # the full rebuttal run, ~6-9 h
```

`run_overnight.sh` is the intended entry point: it runs the test suite first,
then the benchmark-cover studies, then probes for a live BOSSBase mirror, and
prints which result files to read first. Each stage writes its own timestamped
run directory, so a failure late in the night does not cost the earlier work.

Outputs land in `outputs/amdt_rebuttal/<timestamp>/` with `figures/`,
`tables/`, `results/`, `resolved_config.yaml`, `provenance.json` and
`index.json`.

### Common invocations

```bash
# One study at a time
python run_experiments.py 'experiment.studies=[quality,stats]'

# Check a BOSSBase mirror is alive before committing to the download
python run_experiments.py dataset=bossbase dataset.probe_only=true

# The real steganalysis corpus (acquire=local | url | kaggle)
python run_experiments.py dataset=bossbase dataset.acquire=local \
    dataset.root=/data/BOSSbase_1.01 dataset.limit=2000

# CNN steganalysis (needs a GPU)
python run_experiments.py steganalysis=cnn steganalysis.model=srnet \
    'experiment.studies=[quality,cnn]'

# Sweep the segment count (Hydra multirun)
python run_experiments.py -m ga.n_segments=1,2,4,8
```

Nothing is hard-coded: every hyper-parameter, path and switch lives in
`configs/`. A reviewer asking "what settings produced Table 3?" is answered by
`resolved_config.yaml` in that run directory.

---

## What the method actually is

A chromosome (33 bits **per segment**) encodes:

```
direction 4b | x_off 9b | y_off 9b | mask 4b | alpha 1b | beta 1b
          | bp_dir 1b | sigma 1b | block_idx 2b | delta 1b
```

giving a search space of 4.0 × 10⁹ per segment. The GA (population 25,
tournament 3, two-point crossover 0.8, mutation 0.1, elitism 1) minimises MSE
**subject to the security constraint** `sigma = delta = 1` — see
`SECURITY_LOCKS` in `stego/amdt.py`. Dropping that constraint (the
`unconstrained` ablation) makes the GA switch the security layers off, because
a randomised payload agrees with the cover's LSB plane slightly less often than
a structured one. That is reported explicitly rather than hidden in a default.

Extraction is **blind given the key**: a 60 + 33·S bit header (magic, version,
segment count, payload length, per-segment genes, CRC-16) is written into the
reserved bottom rows. The 256-bit secret is never embedded.

---

## Honest findings you should know before writing the rebuttal

These come out of the pipeline and are *not* what the current manuscript says.

1. **T1 and T2 are not security mechanisms.** Complementation and reversal are
   distortion controls — they let the GA pick the bit-ordering that best agrees
   with the cover LSB plane. The code labels them `distortion_layers`. Claiming
   them as a security contribution is what drew comment 1; the defensible claim
   is T3 ∘ T4, and it is a claim about *payload-statistics* attacks only.
2. **T3/T4 do not defeat SRM or a CNN.** They make the embedded stream
   indistinguishable from uniform, which kills χ², pairs-of-values and
   histogram attacks. Residual-based detectors see embedding *changes*, not
   payload content, and are unaffected. `study_targeted` measures exactly this
   and the module docstring states the boundary.
3. **The modern adaptive baselines beat AMDT on distortion, by a lot.**
   Measured on all 29 covers at 0.1 bpp (`ga.generations=10`, seed 0):

   | Method | PSNR (dB) | Emb. eff. (bits/change) |
   |---|---|---|
   | S-UNIWARD | **66.56** | **7.02** |
   | WOW | 65.75 | 5.85 |
   | HILL | 65.68 | 5.76 |
   | MiPOD-lite | 64.70 | 4.59 |
   | LSB / LSB-M / EA-LSB | 61.13–61.15 | ~2.00 |
   | GA-FT | 60.90 | 2.06 |
   | AMDT | 58.70 | 2.21 |
   | PVD | 52.35 | 2.03 |

   AMDT's ceiling is LSB-level distortion, because it minimises MSE by
   *choosing where and in which planes* to do ±LSB writes; the cost-based
   schemes minimise a **detectability-aware** cost and change 3–4× fewer
   pixels for the same payload. A longer GA run closes the gap to LSB (the GA
   converges towards `mask=1`) but cannot cross it. This is the substance
   behind comment 7, and adding more classical baselines will not address it.
   The defensible framing is that AMDT provides *blind, exactly decodable*
   extraction along a key-dependent path — which the cost-based simulators do
   not provide at all — at LSB-comparable distortion.
4. **Detection numbers on 29 covers are indicative only.** Every steganalysis
   artifact carries the warning from `DatasetSpec.steganalysis_warning()`, and
   `results/steganalysis_caveat.json` records it. Use BOSSBase for the table
   that goes into the paper.
5. **The MATLAB traversal had three defects** (unreachable last row, asymmetric
   upward wrap, duplicate cells at serpentine turns). Duplicates silently
   overwrite earlier message bits. If any published number came from a
   configuration that hit them, it needs re-running — the Python version is
   verified duplicate-free for all 16 patterns.

---

## Layout

```
configs/            Hydra YAML: env/, dataset/, ga/, steganalysis/, tracking/,
                    search/, supervisor/, sync/, main.yaml
src/amdt/
  stego/            traversal, bit-planes, decomposition, codec, AMDT method
  ga/               GA + matched-budget random-search control
  baselines/        classical.py (LSB, LSB-M, EA-LSB, GA-FT, PVD)
                    adaptive.py  (WOW, S-UNIWARD, HILL, MiPOD-lite)
  steganalysis/     features (SPAM/SRM), FLD ensemble, CNNs, targeted attacks
  evaluation/       metrics, significance, vector plots, booktabs tables
  experiments/      method registry, the seven studies, bounded stego store,
                    Optuna search, dual-target supervisor, run registry
  utils/            seeding, provenance, profiling, W&B tracking, git repo,
                    execution target + thread pinning, Drive run-folder sync
  data/             dataset loading/splitting, Kaggle acquisition
tests/              210 tests (see below)
notebooks/demo.ipynb  one-click Colab reproduction
run_experiments.py  thin Hydra entry point
watch_training.py   out-of-runtime supervisor (run outside Colab)
```

## Reproducibility policy

* **Seeds.** One root seed; every component draws a sub-seed from
  `BLAKE2b(root ‖ tag)`, so adding a baseline cannot perturb the GA's
  trajectory. Results are averaged over ≥5 root seeds and each number carries
  its seed in the CSV.
* **Determinism.** `set_seed()` runs before any CUDA context exists;
  cuDNN deterministic, `torch.use_deterministic_algorithms(True)`, DataLoader
  workers reseeded from the root.
* **Splits.** Cover-wise — a cover and its stego always land in the same split.
  Putting the stego of a training cover in the test set is the classic
  steganalysis leak and inflates accuracy by tens of points.
* **Selection.** The FLD ensemble picks `d_sub` and its learner count by
  out-of-bag error on training data; CNNs select on validation `P_E`. The test
  set is touched once.
* **Checkpointing.** Full state (model, optimizer, scheduler, AMP scaler, torch
  and NumPy RNG state) written atomically, with auto-resume — a Colab
  disconnect costs nothing and a resumed run is bit-identical.

## Tracking, search, provenance and supervision

All four are **off by default** so a fresh clone runs offline with no accounts.

```bash
# Confirm auth and the target workspace first -- never prints the key
python run_experiments.py tracking=wandb tracking.preflight=true

# W&B + git provenance commits (project defaults to GA-Stegno)
python run_experiments.py tracking=wandb \
    tracking.git.enabled=true tracking.git.remote=https://github.com/<owner>/<repo>.git

# Optuna: tune the GA on validation covers (never on test)
python run_experiments.py search.enabled=true search.n_trials=40 \
    search.storage=sqlite:////content/drive/MyDrive/amdt/optuna.db

# Tune the *attacker* instead — a weak detector is not evidence of security
python run_experiments.py search.enabled=true search.objective=cnn \
    search.direction=minimize
```

With tracking off there is still a record: metrics go to `metrics.jsonl` in the
run directory. The dashboard is optional, the provenance is not.

**Git provenance.** `tracking.git.enabled=true` initialises the repo, writes the
ignore/LFS policy and commits it, then makes one commit per run carrying the
tables, vector plots and profiling summary. Artifacts are copied out of the
gitignored `outputs/<timestamp>/` into a stable `paper/` path, so the manuscript
can `\input{paper/tables/tab_quality.tex}` and a re-run updates it in place.
Commit messages carry `seed`, `config_hash`, `wandb_run` and `trigger`.

**The W&B key never reaches this repo.** `wandb login` writes it to `~/.netrc`;
on Colab it lives in Secrets as `WANDB_API_KEY`. The *entity* is the workspace
slug in `wandb.ai/<entity>/<project>` — not your email — and leaving it `null`
resolves to the default entity of whoever is logged in. `tracking.preflight=true`
confirms both without printing the credential.

**Tokens.** Never paste a PAT into a chat, a notebook or a config. Store it in
Colab Secrets as `GITHUB_PAT`; `repo.load_pat()` reads it at runtime and
`repo.push()` passes it through a per-invocation credential helper. Scope it to
the one repository, Contents read/write only, expiry ≤ 90 days.
`assert_no_token_in_config()` fails the run if a token ever lands in
`.git/config` — which is what `git remote set-url https://<token>@...` does, and
why that shortcut is not used here.

**Autonomous supervision** applies to the CNN steganalysis study only (the GA
study has no epochs and no loss curve to watch). The watcher runs outside Colab:

```bash
python watch_training.py --run entity/project/run_id \
    --remote https://github.com/<owner>/<repo>.git --baseline 0.45
```

It polls W&B, and on divergence / plateau / overfitting / underperformance it
delivers a patch. **One supervisor handles both targets**: locally it patches
and relaunches directly (same machine as training, no branch needed); on a
hosted runtime it pushes to `agent/patches` and the in-notebook `PatchStub`
applies it between epochs. The decision logic is identical — only delivery
differs.

Three rounds maximum, and the budget is **shared across targets**: it is tracked
against the experiment ID in `RunRegistry`, not against the process, so a
migration or a supervisor restart cannot hand the run a fresh three. Every
intervention appends a line to `agent_interventions.jsonl`, committed with the
run.

**Migration.** A run can move local → hosted when the local machine is under
sustained pressure, defined numerically (load per core or memory-used fraction
above a threshold for N *consecutive* samples — a transient spike resets the
streak). The full checkpoint travels, RNG state included, and
`migrate_checkpoint` refuses to proceed without it: a resume that is not
bit-identical invalidates the metrics too, not just the timings. Migrated runs
are marked `timing_invalid` and their latency and epoch-time are excluded from
every table — a wall-clock figure spanning two machines describes neither.

Two things to know before you rely on it:

* The supervisor sees validation metrics only — `Supervisor.observe` raises on
  any key containing `test`, and the stub is handed `val_pe` and `train_loss`
  by the trainer. An agent that iterates against a metric is a search
  procedure, and any split it can see stops being evaluation.
* Patches touching loss or architecture are tagged `method_altering` and
  surfaced at end of run. After autonomous edits the manuscript may no longer
  describe what actually ran, and that mismatch is caught at writing time or
  not at all.

## Execution target — read this before quoting a runtime

Most of this pipeline is CPU work (the GA, SRM/SPAM features, the FLD ensemble),
so the CPU determinism rules apply to nearly every reported number.

```bash
python run_experiments.py env=local          # default: threads pinned, timings reportable
python run_experiments.py env=local_explore  # multithreaded; timings NOT reportable
python run_experiments.py env=colab_cpu      # overflow; timings NOT reportable
python run_experiments.py env=colab_gpu      # CNN steganalysis
```

The target is **declared, never inferred**. A hosted runtime hands you a
different VM each session, so the CPU model varies and wall-clock stops being
comparable. Numerics survive thread pinning; timings do not.

* **Threads are pinned only for reported runs.** `OMP/MKL/OPENBLAS_NUM_THREADS`
  and `torch.set_num_threads` go to 1, because BLAS reduction order changes with
  thread count and the same seed then gives different floats. Pinning costs
  real parallelism, so exploration runs stay multithreaded — and are marked so
  their timings never reach a table.
* **`n_jobs=-1` is rejected.** It ties results to the host's core count.
  `resolve_n_jobs` warns, substitutes the explicit value, forces 1 on a pinned
  reported run, and caps at 2 on a ≤16 GB host (joblib copies the dataset into
  every worker, and that fails as swap thrash rather than a clean error).
* **MPS is refused for reported runs.** Unsupported ops fall back to CPU
  silently, which makes latency meaningless. On your Mac, `env=local` (CPU) is
  the defensible target; `env.allow_mps=true` is available but marks the run
  timing-invalid.
* **Timings are labelled, not silently mixed.** Every stage row carries its
  hardware, thread count and mode. `tab_runtime.tex` grows a Hardware column
  automatically when rows span devices, and flags invalid rows with a dagger
  and a stated reason rather than dropping them.

## Kaggle data (BOSSBase mirror)

```bash
python run_experiments.py dataset=bossbase_kaggle
```

Credentials come from Colab Secrets (`KAGGLE_USERNAME` / `KAGGLE_KEY`) or
`~/.kaggle/kaggle.json` at mode 600 — detected at runtime, never from a config
or a chat. `kaggle.json`, `.kaggle/` and `.env` are the first entries in
`.gitignore`, committed before anything else: a key that has ever been
committed must be rotated, because history retains it.

The slug, version and archive SHA-256 land in `dataset_pin.json`. Copy them into
`configs/dataset/bossbase_kaggle.yaml` after the first download; every later run
compares against the pin, **warns loudly on drift and never auto-upgrades**. A
dataset that gains rows between your ablation and your final table is a
reproducibility failure no seed will catch.

Two traps the module handles explicitly: a competition 403 usually means you
have not accepted the rules on the website, not that your key is wrong; and a
competition leaderboard split is **not** a held-out test set for a manuscript —
carve your own from the labeled data before preprocessing.

## Run-folder sync to Drive

```bash
python run_experiments.py sync=drive        # mirrors while the run is going
```

Off by default; turn it on for anything whose numbers you intend to report. The
mirror starts before the first study and runs on a background thread, because
the artifacts most worth having off-machine are the ones a crash would take
with it. There is no per-run prompt.

* **Locally this is Drive for Desktop, not the Drive API.** On macOS the folder
  is auto-detected under `~/Library/CloudStorage/GoogleDrive-<account>/My Drive`.
  If sync is enabled and no Drive folder can be found, the run **fails** — a
  sync that quietly does nothing is worse than no sync.
* **On Colab**, mount once at the top of the notebook; the mounted path becomes
  the target. The single OAuth approval is Google's consent step.
* **Nothing is written straight into the synced folder.** Files are staged in
  the local run directory and moved across with an atomic rename whose temp sits
  *beside the target* (a cross-filesystem `os.replace` raises `EXDEV`). Drive
  never uploads a half-written checkpoint or a log truncated mid-line.
* **Exclusions live in `configs/sync/drive.yaml`**, not in code: `.git/`,
  `wandb/`, cached stego images, raw datasets, virtualenvs.
* **Retention** keeps the last N runs plus every run named in a commit message.
  A folder referenced by a reported result is evidence and is never pruned.

The sync target and run ID go into `provenance.json` next to the commit hash, so
a table cell resolves to both a commit and a folder.

## Tests

```bash
pytest -q            # 209 passed, 1 skipped
```

They guard properties, not lines. Steganography: all 16 traversal patterns are bijections;
decomposition is invertible for all 128 flag/block combinations at 8 payload
lengths; the codec round-trips over every direction × mask; a wrong key yields
noise, not a near-miss; unused bit-planes are never zeroed (a clean tail leaks
the payload length); RS and WS track known embedding rates on real covers;
Holm correction is monotone; the adaptive simulator hits its payload to within
5%. Sync: the staging temp is always on the destination filesystem, no `.part`
files survive a pass, excluded paths never cross, and pruning never removes a
run named by a commit. Operations: a token in `.git/config` is detected and raises; `push` without
a PAT declines instead of guessing; the supervisor refuses to observe anything
named `test`, stops at exactly three rounds, suppresses repeat patches, and
respects `min_epochs_between`; the stub applies a patch once and defers what it
cannot change in place; Optuna objectives reject a test split.

## Limitations, stated up front

* `srm_subset` is a documented **subset** of SRM (≈1.7k features vs 34,671),
  named as such everywhere. It is a *weaker* attacker than full SRM, so
  resistance results from it are conservative.
* `MiPOD-lite` uses a local-variance proxy for MiPOD's Fisher-information
  model. Labelled "-lite" wherever it appears.
* The adaptive baselines use the payload-limited **simulator**, not STC coding,
  and therefore produce statistically correct but non-decodable stego objects.
  They are excluded from the round-trip test and flagged
  `blind_extractable: false`.
* No learned (GAN-based) cost map is included. `register_cost_map()` accepts
  one; faking it would be worse than omitting it.
