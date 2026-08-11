#!/usr/bin/env bash
# Overnight run for the JST-6070-2025.R1 rebuttal — local Mac (env=local).
#
#     chmod +x run_overnight.sh && ./run_overnight.sh
#
# Everything here is CPU work with BLAS threads pinned, so the timings are
# reportable. CNN steganalysis is NOT included: run notebooks/demo.ipynb on
# Colab with a GPU for that (Reviewer 1, comment 3).
#
# The run folder is mirrored to Google Drive as it goes (sync=drive), so a
# crash at hour 7 does not lose hours 1-6. Requires Google Drive for Desktop;
# drop `sync=drive` if you do not have it.
#
# Expect roughly 6–9 h on an M-series Mac. Each stage writes to its own
# timestamped run directory, so a crash in stage 3 does not cost stages 1–2.

set -euo pipefail
cd "$(dirname "$0")"

# Sync mode, overridable by setup_and_run.sh (sync=drive | sync=none).
SYNC="${1:-sync=drive}"

SEEDS="[0,1,2,3,4]"
RATES="[0.05,0.1,0.2,0.4]"
LOG="overnight_$(date +%Y%m%d_%H%M%S).log"

echo "logging to $LOG"
exec > >(tee -a "$LOG") 2>&1

echo "=== 0. environment ==========================================="
# Fail here, in one readable line, rather than 40 lines into a pip resolver
# error. numpy>=2.1 and scipy>=1.15 both require Python 3.10+; on 3.9 pip
# silently offers numpy 2.0.2 and the pinned versions look "unavailable".
PYV=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'; then
  echo "ERROR: this venv runs Python $PYV; the pipeline needs 3.10 or newer."
  echo
  echo "  Find a newer interpreter:"
  echo "    ls /opt/homebrew/bin/python3.1* /usr/local/bin/python3.1*"
  echo "  Then rebuild the venv with it:"
  echo "    rm -rf .venv && /opt/homebrew/bin/python3.12 -m venv .venv"
  echo "    source .venv/bin/activate && python -m pip install --upgrade pip"
  echo "    pip install -r requirements-local.txt"
  echo
  echo "  No 3.10+ installed?  brew install python@3.12"
  exit 1
fi
echo "python $PYV: ok"

for mod in numpy scipy pandas sklearn matplotlib hydra pytest; do
  python -c "import $mod" 2>/dev/null || {
    echo "ERROR: '$mod' is missing. Run: pip install -r requirements-local.txt"
    echo "(check you are in the project venv -- 'which python' should end in"
    echo " amdt_python/.venv/bin/python)"
    exit 1
  }
done
echo "dependencies: ok  ($(which python))"

echo
echo "=== 0b. sanity: tests, then W&B auth ========================="
pytest -q

# Confirm the credential and the workspace BEFORE committing 8 hours to a run.
# Prints the resolved entity/project; never prints the key.
python run_experiments.py tracking=wandb tracking.preflight=true

echo
echo "=== 1. benchmark covers: quality, ablation, targeted, stats ==="
# The 29 MATLAB covers. Keeps continuity with the published numbers and
# produces every figure except the steganalysis panels.
python run_experiments.py \
  env=local "$SYNC" tracking=wandb \
  experiment.seeds="$SEEDS" \
  experiment.payload_rates_bpp="$RATES" \
  ga.population=25 ga.generations=100 ga.patience=25 ga.n_segments=4 \
  'experiment.studies=[quality,ablation,targeted,runtime,stats]'

echo
echo "=== 2. is a BOSSBase mirror alive? (no download) ============="
# If this reports NONE, fix dataset.mirrors before stage 3 wastes the night.
python run_experiments.py dataset=bossbase dataset.probe_only=true || true

echo
echo "=== 3. BOSSBase steganalysis, 2000-cover subsample ==========="
echo "Skipping unless dataset.acquire is configured — edit configs/dataset/bossbase.yaml"
echo "then re-run just this stage:"
cat <<'EOF'

  python run_experiments.py \
    env=local dataset=bossbase dataset.acquire=url dataset.limit=2000 \
    experiment.seeds=[0,1,2,3,4] \
    experiment.payload_rates_bpp=[0.05,0.1,0.2,0.4] \
    steganalysis.features=srm_subset \
    'experiment.studies=[quality,steganalysis,stats]'

EOF

echo
echo "=== done ====================================================="
echo "Newest run:"
ls -dt outputs/amdt_rebuttal/*/ | head -1
echo
echo "Read first:"
echo "  results/quality.csv        — does AMDT hold up at 5 seeds?"
echo "  results/significance.csv   — check 'effect_below_noise' before claiming anything"
echo "  results/extraction_failures.csv — must NOT exist; if it does, results are void"
echo "  tables/tab_runtime.tex     — daggered rows are timings you cannot quote"
