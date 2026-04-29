"""Walk a results/ directory tree and produce a tidy CSV for plotting.

The tree convention is::

    results/<matchup>/lucy_iters=<N>/baseline_iters=<M>/seed=<S>.json

Each JSON is a ``MatchResult`` from ``play_match.py``. The aggregator emits
a CSV with columns:

    matchup, lucy_iters, baseline_iters, seed, num_hands,
    mbb_per_hand_a, mbb_se, ci_low, ci_high, seconds

Plus per-(matchup, lucy_iters, baseline_iters) summary rows that pool seeds
and report mean ± SE across seeds (label='__SEED_POOL__' for the seed col).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True,
                   help="Top-level results directory.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    args = p.parse_args()

    rows = []
    pattern = os.path.join(args.results_dir, "*",
                           "lucy_iters=*", "baseline_iters=*", "seed=*.json")
    for path in sorted(glob.glob(pattern)):
        # Parse path components.
        parts = path.split(os.sep)
        try:
            matchup = parts[-4]
            lucy_iters = int(parts[-3].split("=")[1])
            baseline_iters = int(parts[-2].split("=")[1])
            seed = int(parts[-1].split("=")[1].rsplit(".", 1)[0])
        except Exception:
            print(f"[aggregate] skip unparseable path: {path}",
                  file=sys.stderr)
            continue

        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"[aggregate] skip unreadable {path}: {e}",
                  file=sys.stderr)
            continue

        rows.append({
            "matchup": matchup,
            "lucy_iters": lucy_iters,
            "baseline_iters": baseline_iters,
            "seed": seed,
            "num_hands": d.get("num_hands"),
            "mbb_per_hand_a": d.get("mbb_per_hand_a"),
            "mbb_se": d.get("mbb_per_hand_a_se"),
            "ci_low": d.get("ci95_low"),
            "ci_high": d.get("ci95_high"),
            "seconds": d.get("seconds"),
        })

    # Pool across seeds.
    grouped = defaultdict(list)
    for r in rows:
        key = (r["matchup"], r["lucy_iters"], r["baseline_iters"])
        grouped[key].append(r)

    pool_rows = []
    for (matchup, lit, bit), seed_rows in grouped.items():
        means = [r["mbb_per_hand_a"] for r in seed_rows
                 if r["mbb_per_hand_a"] is not None]
        ns = [r["num_hands"] for r in seed_rows if r["num_hands"]]
        if not means:
            continue
        n_seeds = len(means)
        mean = sum(means) / n_seeds
        if n_seeds > 1:
            var = sum((m - mean) ** 2 for m in means) / (n_seeds - 1)
            se = (var / n_seeds) ** 0.5
        else:
            # Fallback: use the per-match SE from the single seed.
            se = seed_rows[0].get("mbb_se") or 0.0
        pool_rows.append({
            "matchup": matchup,
            "lucy_iters": lit,
            "baseline_iters": bit,
            "seed": "__SEED_POOL__",
            "num_hands": sum(ns) if ns else None,
            "mbb_per_hand_a": mean,
            "mbb_se": se,
            "ci_low": mean - 1.96 * se,
            "ci_high": mean + 1.96 * se,
            "seconds": sum(r.get("seconds") or 0 for r in seed_rows) or None,
        })

    out_rows = rows + pool_rows
    fieldnames = ["matchup", "lucy_iters", "baseline_iters", "seed",
                  "num_hands", "mbb_per_hand_a", "mbb_se",
                  "ci_low", "ci_high", "seconds"]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"[aggregate] wrote {len(out_rows)} rows ({len(rows)} per-seed, "
          f"{len(pool_rows)} pooled) -> {args.out}")


if __name__ == "__main__":
    main()
