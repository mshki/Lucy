"""Train an OpenSpiel External-Sampling MCCFR solver on HU NLHE FCPA.

Usage:
    python -m bench.train_openspiel --iters 100000 --seed 1 --out path/to/model.pkl
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyspiel  # type: ignore

# Ensure ``bench`` package is importable when this script is invoked directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench.common import HU_NLHE_FCPA_GAMEDEF  # noqa: E402
from bench.openspiel_bot import save_solver  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, required=True,
                   help="Number of MCCFR iterations to run.")
    p.add_argument("--seed", type=int, default=0,
                   help="PRNG seed (0 = nondeterministic).")
    p.add_argument("--out", required=True,
                   help="Output path for pickled MCCFR solver.")
    p.add_argument("--algo", default="outcome",
                   choices=["outcome", "external"],
                   help="MCCFR sampling scheme. 'external' enumerates the "
                        "traverser's actions and samples opponent+chance; "
                        "'outcome' samples a single root-to-leaf path with "
                        "importance weighting. 'external' is intractable on "
                        "full HU NLHE FCPA 100bb due to chance-node fan-out, "
                        "so we default to 'outcome' for this benchmark. Both "
                        "converge to the same Nash equilibrium asymptotically.")
    p.add_argument("--avg-type", default="full", choices=["full", "simple"],
                   help="MCCFR average type for external-sampling.")
    args = p.parse_args()

    if args.seed != 0:
        np.random.seed(args.seed)
        try:
            pyspiel.seed_all(args.seed)
        except AttributeError:
            pass

    game = pyspiel.load_game(HU_NLHE_FCPA_GAMEDEF)

    if args.algo == "external":
        avg_type = (pyspiel.MCCFRAverageType.FULL if args.avg_type == "full"
                    else pyspiel.MCCFRAverageType.SIMPLE)
        solver = pyspiel.ExternalSamplingMCCFRSolver(game, avg_type=avg_type)
    else:
        solver = pyspiel.OutcomeSamplingMCCFRSolver(game)

    print(f"[openspiel] training MCCFR HU NLHE FCPA algo={args.algo} "
          f"iters={args.iters} seed={args.seed}", flush=True)
    log_every = max(1, args.iters // 20)
    t0 = time.time()
    for i in range(args.iters):
        solver.run_iteration()
        if i == 0 or (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"[openspiel] iter {i + 1}/{args.iters} "
                  f"elapsed={elapsed:.1f}s rate={rate:.0f}it/s", flush=True)

    elapsed = time.time() - t0
    print(f"[openspiel] training complete in {elapsed:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save_solver(solver, args.out)
    print(f"[openspiel] saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
