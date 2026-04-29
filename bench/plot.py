"""Plot mbb/hand vs training iterations for each Lucy variant.

Reads the CSV produced by ``aggregate.py`` and produces a PNG with one curve
per Lucy variant (legacy, cuda, …) on a log-x training-iteration axis. Error
bars use the seed-pooled 95% CI when multiple seeds are present, otherwise
the per-match CI.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Aggregated CSV from aggregate.py")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--baseline", default="openspiel",
                   help="Substring identifying the baseline solver in the "
                        "matchup label (e.g. 'openspiel'). Curves are "
                        "Lucy-vs-baseline at matched iterations.")
    p.add_argument("--matched-iters", action="store_true", default=True,
                   help="Only plot points where lucy_iters == baseline_iters.")
    args = p.parse_args()

    # matchup name convention: '<lucy_variant>-vs-<baseline>'.
    by_variant = defaultdict(list)  # variant -> [(iters, mean, low, high)]
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            if row["seed"] != "__SEED_POOL__":
                continue
            mu = row["matchup"]
            if args.baseline not in mu:
                continue
            lit = int(row["lucy_iters"])
            bit = int(row["baseline_iters"])
            if args.matched_iters and lit != bit:
                continue
            variant = mu.split("-vs-")[0]
            try:
                mean = float(row["mbb_per_hand_a"])
                low = float(row["ci_low"])
                high = float(row["ci_high"])
            except (TypeError, ValueError):
                continue
            by_variant[variant].append((lit, mean, low, high))

    if not by_variant:
        print("[plot] no rows matched filters; nothing to plot", file=sys.stderr)
        sys.exit(1)

    # Lazy import so the module imports without matplotlib for non-plot
    # callers.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for variant in sorted(by_variant):
        pts = sorted(by_variant[variant])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lo = [p[1] - p[2] for p in pts]
        hi = [p[3] - p[1] for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3,
                    label=variant)

    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("Training iterations (matched both bots)")
    ax.set_ylabel("Lucy edge over OpenSpiel  (mbb / hand)")
    ax.set_title("Lucy MCCFR variants vs OpenSpiel External-Sampling MCCFR\n"
                 "Heads-Up NLHE, FCPA abstraction (Fold/Call/Pot/All-in)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[plot] saved -> {args.out}")


if __name__ == "__main__":
    main()
