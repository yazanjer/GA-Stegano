#!/usr/bin/env bash
# BOSSBase steganalysis — Reviewer 1, comment 3.
#
#     ./run_steganalysis.sh              # defaults below, ~7-9 h
#     COVERS=2000 RATES='[0.05,0.1,0.2,0.4]' ./run_steganalysis.sh
#     COVERS=200 ./run_steganalysis.sh   # ~45 min smoke test, NOT reportable
#
# Produces the SRM-subset + FLD-ensemble detection results, and packages the
# cover/stego pairs so the CNN half can run on a Colab GPU without re-embedding.
#
# WHY THE DEFAULTS ARE WHAT THEY ARE
#   AMDT costs ~10.5 s per embedding (measured, pinned M1 Pro), and that single
#   number sets the whole budget:
#       covers x rates x 10.5 s = AMDT wall-clock
#       1000 x 2 =  5.8 h        <- default
#       2000 x 4 = 23.3 h
#      10000 x 4 = 116 h         <- the full corpus; do not start this casually
#   Feature extraction is cheap by comparison (~0.14 s/image).
#
#   Detection experiments do not need multiple seeds per cover the way the
#   quality study does: the corpus itself supplies the variance, and the FLD
#   ensemble is cross-validated cover-wise. One seed, more covers, is the better
#   trade at fixed compute.

set -euo pipefail
cd "$(dirname "$0")"

COVERS="${COVERS:-1000}"
RATES="${RATES:-[0.1,0.4]}"
SEEDS="${SEEDS:-[0]}"
FEATURES="${FEATURES:-srm_subset}"
LOG="steganalysis_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1
echo "logging to $LOG"

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { printf '\n\033[31mSTOP\033[0m %s\n\n' "$*"; exit 1; }

[ -x .venv/bin/python ] || die "no .venv — run ./setup_and_run.sh first"
# shellcheck disable=SC1091
source .venv/bin/activate

python - <<PY
covers, rates = $COVERS, len("$RATES".split(","))
h = covers * rates * 10.5 / 3600
print(f"\n  {covers} covers x {rates} payload rate(s)")
print(f"  AMDT embedding alone: ~{h:.1f} h  (+ ~1 h features and classifier)")
if h > 24:
    print("  WARNING: this is more than a day. Reduce COVERS or RATES.")
PY

# --------------------------------------------------------------------------- #
say "1/4  is a mirror alive?"
# Checked before downloading 1.6 GB, and before an overnight job discovers at
# 3 a.m. that the corpus never arrived.
python run_experiments.py dataset=bossbase dataset.probe_only=true
LIVE=$(python - <<'PY'
import json, glob, os
runs = sorted(glob.glob("outputs/amdt_rebuttal/*/mirror_probe.json"), key=os.path.getmtime)
print(sum(1 for m in json.load(open(runs[-1])) if m["ok"]) if runs else 0)
PY
)
[ "$LIVE" -gt 0 ] || die "no live BOSSBase mirror. Find one, add it to
      configs/dataset/bossbase.yaml under 'mirrors:', and re-run. Do NOT
      substitute a different corpus without saying so in the paper."

# --------------------------------------------------------------------------- #
say "2/4  fetch, verify and pin the corpus"
# First run downloads ~1.6 GB into ~/.cache/amdt (outside the repo) and writes
# dataset_pin.json. Later runs verify against the pin and refuse a changed archive.
python run_experiments.py \
  env=local sync=drive tracking=wandb \
  dataset=bossbase dataset.acquire=url "dataset.limit=$COVERS" \
  'experiment.studies=[]' \
  experiment.seeds="$SEEDS"

PIN=$(ls -t outputs/amdt_rebuttal/*/dataset_pin.json | head -1)
python - "$PIN" <<'PY'
import json, sys, pathlib
pin = json.load(open(sys.argv[1]))
n, sha = pin.get("n_files"), (pin.get("sha256") or "")
print(f"  files={n}  sha256={sha[:16]}")
if pin.get("drift"):
    print(f"  DRIFT: {pin.get('drift_note')}")
if n and n != 10000:
    print(f"""
  !! Expected 10000 files in BOSSbase 1.01, found {n}.
     Several re-uploads are resized or JPEG-recompressed. A recompressed cover
     invalidates every spatial-domain result -- the embedding changes you are
     trying to detect get swamped by compression artefacts.
     Verify the source before continuing.""")
    raise SystemExit(1)
print(f"""
  Paste into configs/dataset/bossbase.yaml so later runs verify against it:
      pin:
        sha256: {sha}
        n_files: {n}""")
PY

# --------------------------------------------------------------------------- #
say "3/4  embed + SRM-subset features + FLD ensemble"
# Cover-wise k-fold: a cover and its stego always share a fold, so the detector
# can never have seen a test cover during training.
python run_experiments.py \
  env=local sync=drive tracking=wandb \
  dataset=bossbase dataset.acquire=url "dataset.limit=$COVERS" \
  experiment.seeds="$SEEDS" \
  experiment.payload_rates_bpp="$RATES" \
  experiment.stego_cache=disk \
  "steganalysis.features=$FEATURES" \
  steganalysis.cross_validation.n_folds=5 \
  'experiment.studies=[quality,steganalysis,stats]'

RUN=$(ls -dt outputs/amdt_rebuttal/*/ | head -1)
echo "  run: $RUN"

[ -f "$RUN/results/extraction_failures.csv" ] && die "extraction failed — every
      number in this run is void. Send me results/extraction_failures.csv."

# --------------------------------------------------------------------------- #
say "4/4  package cover/stego pairs for the Colab GPU stage"
# The Drive mirror deliberately excludes stego/ (re-derivable, and it would
# dominate the quota). But re-embedding on Colab would cost hours of GPU time
# doing CPU work, so ship the pairs across once as a zip.
python - "$RUN" <<'PY'
import sys, zipfile, shutil
from pathlib import Path
import numpy as np

run = Path(sys.argv[1])
stego = run / "stego"
if not stego.exists():
    print("  no stego cache (experiment.stego_cache=disk not honoured?) — the")
    print("  Colab stage will have to re-embed. That is correct but slow.")
    raise SystemExit(0)

out = run / "cnn_payload.zip"
covers = sorted((Path("~/.cache/amdt").expanduser()).rglob("*.pgm"))
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:   # PNGs already compressed
    for p in sorted(stego.glob("*.png")):
        z.write(p, f"stego/{p.name}")
    for p in covers:
        z.write(p, f"cover/{p.name}")
print(f"  wrote {out}  ({out.stat().st_size / 1024**3:.2f} GB)")

drive = sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*/My Drive"))
if drive:
    dest = drive[0] / "amdt-runs" / run.name / "cnn_payload.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    shutil.copy2(out, tmp); tmp.replace(dest)      # atomic: Drive never sees a partial
    print(f"  copied to Drive: {dest}")
    print("  Wait for Drive to finish uploading before starting the Colab stage.")
else:
    print("  no Drive folder — upload cnn_payload.zip to Colab manually.")
PY

# --------------------------------------------------------------------------- #
say "done"
echo "  $RUN"
echo
echo "Read first:"
echo "  results/steganalysis_classical.csv  — accuracy, P_E, AUC per method"
echo "  tables/tab_detection.tex            — the table for the manuscript"
echo "  figures/fig10_roc.pdf, fig11_detectability.pdf"
echo
echo "Reading it: LOWER detector accuracy is better for the steganographic"
echo "method. P_E = 0.5 means undetectable, 0.0 means perfectly detected."
echo
echo "Next: notebooks/cnn_colab.ipynb on a GPU runtime for the CNN half."
