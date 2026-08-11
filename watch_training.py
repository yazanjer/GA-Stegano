#!/usr/bin/env python3
"""Out-of-runtime training supervisor (skill §12).

Run this **on your own machine**, not inside the Colab notebook — the whole
point is that it survives the runtime it is watching. It polls W&B for the live
CNN steganalysis run, and when a trip condition fires it pushes a patch to the
``agent/patches`` branch. The in-notebook stub picks it up between epochs.

    watcher (here)  ->  git branch  ->  stub (in Colab)

Usage
-----
    python watch_training.py --run entity/project/run_id \\
        --run-dir outputs/amdt_rebuttal/2026-01-01_12-00-00 \\
        --remote https://github.com/<owner>/<repo>.git

    # dry run against a finished run, no pushes, no waiting
    python watch_training.py --run entity/project/run_id --max-polls 1 --no-push

The supervisor sees validation metrics only. It is not given, and cannot
construct, a test loader — see the assertion in ``Supervisor.observe``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from amdt.experiments.supervisor import Supervisor, TripConfig, wandb_poller  # noqa: E402
from amdt.utils.repo import GitRepo  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("watch")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="W&B run path: entity/project/run_id")
    p.add_argument("--run-dir", default="outputs/watch",
                   help="where to write agent_patch.json and the intervention log")
    p.add_argument("--metric", default="val/p_e")
    p.add_argument("--higher-is-better", action="store_true",
                   help="default is lower-is-better, correct for P_E")
    p.add_argument("--remote", default=None, help="git remote for the patch branch")
    p.add_argument("--branch", default="agent/patches")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-3)
    p.add_argument("--baseline", type=float, default=None,
                   help="named baseline for the underperformance trip")
    p.add_argument("--poll-interval", type=int, default=60)
    p.add_argument("--max-polls", type=int, default=None,
                   help="stop after N polls (omit to run until the budget is spent)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config-hash", default="")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    repo = None
    if args.remote and not args.no_push:
        repo = GitRepo(ROOT, remote=args.remote)
        repo.init()
        repo.assert_no_token_in_config()

    cfg = TripConfig(max_rounds=args.max_rounds, patience=args.patience,
                     min_delta=args.min_delta, baseline_metric=args.baseline,
                     poll_interval_s=args.poll_interval)
    sup = Supervisor(run_dir, cfg, repo=repo, wandb_run_path=args.run,
                     metric=args.metric, lower_is_better=not args.higher_is_better)

    poll = wandb_poller(args.run, args.metric)

    # The watcher only knows the hyper-parameters it can read back from the
    # tracked config; it never reaches into the running process.
    def current():
        try:
            import wandb
            c = wandb.Api().run(args.run).config
            return {"lr": c.get("lr", 1e-3), "weight_decay": c.get("weight_decay", 5e-4)}
        except Exception:
            return {"lr": 1e-3, "weight_decay": 5e-4}

    log.info("watching %s on %s (budget %d rounds)", args.run, args.metric,
             cfg.max_rounds)
    ivs = sup.watch(poll, current, max_polls=args.max_polls, seed=args.seed,
                    config_hash=args.config_hash)

    print(json.dumps({"interventions": [i.as_dict() for i in ivs],
                      "rounds_used": sup.rounds_used,
                      "log": str(sup.log_path)}, indent=2, default=str))

    altering = sup.method_altering_summary()
    if altering:
        log.warning("%d method-altering intervention(s). Reconcile the methods "
                    "section with %s before writing it up.", len(altering), sup.log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
