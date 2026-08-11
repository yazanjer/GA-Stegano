#!/usr/bin/env bash
# One-command setup + launch.
#
#     ./setup_and_run.sh
#
# Does everything that can be automated, and stops with a clear instruction at
# the two points that genuinely need you: pasting the W&B key, and signing in
# to Google Drive. Safe to re-run — every step is idempotent.

set -uo pipefail
cd "$(dirname "$0")"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
ok()   { printf '  \033[32mok\033[0m  %s\n' "$*"; }
warn() { printf '  \033[33m!!\033[0m  %s\n' "$*"; }
die()  { printf '\n\033[31mSTOP\033[0m %s\n\n' "$*"; exit 1; }

# --------------------------------------------------------------------------- #
say "1/5  virtual environment"
if [ ! -x .venv/bin/python ]; then
  command -v python3.12 >/dev/null 2>&1 \
    || die "python3.12 not found. Install it from python.org (3.12.x, macOS
      64-bit universal2 installer), open a NEW Terminal, then re-run this."
  python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

PYV=$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')
python -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' \
  || die "the venv runs Python $PYV; 3.10+ is required.
      Delete it and start over:  rm -rf .venv && ./setup_and_run.sh"
ok "python $PYV  ($(which python))"

# --------------------------------------------------------------------------- #
say "2/5  dependencies"
if python -c 'import numpy, scipy, pandas, sklearn, matplotlib, hydra, pytest, wandb' 2>/dev/null; then
  ok "all present"
else
  echo "  installing (a few minutes on first run)..."
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements-local.txt \
    || die "install failed. Scroll up for the reason; usually a Python version
      mismatch or no network."
  ok "installed"
fi

# --------------------------------------------------------------------------- #
say "3/5  Weights & Biases"
# The key is never handled by this script -- `wandb login` prompts for it and
# stores it in ~/.netrc. It is account-wide, so it must not go into a config,
# a repo, or a chat.
if python -c 'import wandb,sys; sys.exit(0 if wandb.api.api_key else 1)' 2>/dev/null; then
  ok "already logged in"
else
  warn "not logged in yet"
  cat <<'EOF'

      Open  https://wandb.ai/authorize  and copy the 40-character key.
      It is lowercase letters and digits only.

      NOT the hyphenated 'yazan-aljeroudi-rachis-systems-org' -- that is your
      entity (the workspace), and its hyphens are what caused the earlier
      "API key may only contain the letters A-Z, digits and underscores".

      Paste it at the prompt below. Nothing appears on screen while you paste;
      that is normal. Press Return when done.

EOF
  wandb login --relogin || die "login failed. Re-run this script to try again."
  ok "logged in"
fi

# --------------------------------------------------------------------------- #
say "4/5  Google Drive mirroring"
SYNC="sync=drive"
DRIVE=$(ls -d "$HOME"/Library/CloudStorage/GoogleDrive-*/"My Drive" 2>/dev/null | head -1)
if [ -n "$DRIVE" ]; then
  ok "found: $DRIVE"
else
  warn "no Drive folder yet"
  cat <<'EOF'

      Google Drive for Desktop is installed but has not finished signing in --
      the "My Drive" folder only appears afterwards.

      Open Google Drive from the menu bar, sign in, and let it finish setup.
      Then re-run this script and the run folder will be mirrored as it goes.

      Or continue now without mirroring: everything still lands in outputs/,
      you just lose the off-machine copy if the Mac crashes mid-run.

EOF
  printf "      Continue without mirroring? [y/N] "
  read -r reply
  case "$reply" in
    [yY]*) SYNC="sync=none"; warn "continuing unmirrored" ;;
    *)     die "stopped. Sign in to Drive, then re-run." ;;
  esac
fi

# --------------------------------------------------------------------------- #
say "5/5  starting the overnight run"
echo "      6-9 hours. Safe to close the lid? No -- keep the Mac awake."
echo "      Tip: run 'caffeinate -i -w \$\$' in another tab to prevent sleep."
echo

exec ./run_overnight.sh "$SYNC"
