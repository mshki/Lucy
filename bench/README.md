# Lucy CFR Poker Benchmark

Head-to-head benchmark of Lucy MCCFR variants vs an OpenSpiel baseline on
**Heads-Up No-Limit Texas Hold'em with the FCPA action abstraction**
(Fold / Check-Call / Pot-sized bet / All-in). The benchmark plots Lucy's
win rate (mbb / hand) against OpenSpiel's external-implementation MCCFR over
matched training-iteration budgets.

## Pieces

```
bench/
  common.py          # Game definition, card encoding, OS state extractor
  lucy_bot.py        # pyspiel.Bot wrapping a Lucy --serve subprocess
  openspiel_bot.py   # pyspiel.Bot wrapping a trained OpenSpiel policy
                     # + RandomBot / AlwaysFoldBot floor baselines
  train_openspiel.py # Train pyspiel.OutcomeSamplingMCCFRSolver, pickle to disk
  play_match.py      # Duplicate-pair seat-swap match harness
  aggregate.py       # results/ tree -> tidy CSV with seed-pooled rows
  plot.py            # mbb/hand vs iters curve per Lucy variant
  print_results.py   # Quick stdout dump of per-match JSON

slurm/
  setup_env.sh       # Builds venv, pip-installs open_spiel, builds both Lucy variants
  setup.sbatch       # Submits setup_env.sh as a SLURM job (avoids login-node SSH disconnect)
  train_array.sbatch # Job array over (variant x iters x seed)
  eval_array.sbatch  # Job array over (lucy_variant x iters x seed)
  submit_all.sh      # Driver: train array + dependent eval array
  deploy.sh          # Local -> Unity rsync helper
```

## Variants benchmarked

- **`lucy-legacy`** — pre-`NodeMatrix` Lucy at commit `a8deeee`, `Node` class,
  unbatched per-infoset regret updates. The implementation we had at end of
  last semester.
- **`lucy-cuda`** — current `bench/v1` head, `NodeMatrix` + batched flush via
  CUDA kernel (CPU fallback used during this benchmark since `cpu-preempt`
  partition has no GPUs).
- **`openspiel`** — `pyspiel.OutcomeSamplingMCCFRSolver` on the same
  `universal_poker(... bettingAbstraction=fcpa)` game.

> **Why outcome-sampling for OpenSpiel?** External-sampling MCCFR is
> intractable on full HU NLHE FCPA at 100bb stacks because it enumerates
> every chance outcome at the per-iteration root (~50–thousands per node);
> a single iteration takes >30 s. Outcome-sampling MCCFR samples a single
> root-to-leaf trajectory per iteration with importance weighting and runs
> at ~50 k iters / second on the same game. Both algorithms converge to
> the same Nash equilibrium asymptotically.

## Game configuration (`bench/common.py:HU_NLHE_FCPA_GAMEDEF`)

```
universal_poker(betting=nolimit,
  numPlayers=2, numRounds=4,
  blind=2 1, firstPlayer=2 1 1 1,
  numSuits=4, numRanks=13,
  numHoleCards=2, numBoardCards=0 3 1 1,
  stack=200 200, bettingAbstraction=fcpa)
```

Heads-Up NLHE, 100 BB stacks, 1/2 blinds, FCPA action abstraction. OpenSpiel's
FCPA action IDs are stable: `0=fold, 1=check/call, 2=pot, 3=allin`. Lucy's
serve mode is patched to emit the same IDs (commit `225297a`).

## Methodology

- Metric: **mbb / hand** (milli-big-blinds per hand).
- Variance reduction: **duplicate poker** with seat swap. Each unique deck
  (controlled by a seeded RNG) is played twice with bots in swapped seats.
  Cuts variance ~30–50% for free.
- Per-pair returns are averaged across both hands; the mean across pairs is
  reported with a 95% CI from the per-pair standard error.
- Sample-size note: σ ≈ 6 BB/hand for HU NLHE → ~55 k hands needed to detect
  a 50 mbb / hand edge at p < 0.05 raw. We default to 5 000 hands per match
  (2 500 pairs); this resolves edges of order 200 mbb / hand at 95% CI. Use
  `PAIRS=...` env override for tighter resolution at higher cost.

## Running

```bash
# Local (Mac/Linux), one-off smoke
mkdir build && cmake -S . -B build && cmake --build build -j
pip install open_spiel numpy
python -m bench.train_openspiel --iters 1000000 --seed 1 --out os.pkl
./build/bin/PokerBotMAIF --train 100000 --players 2 --abstraction fcpa --seed 1 --out lucy.dat
python -m bench.play_match \
  --bot-a "lucy:bin=$PWD/build/bin/PokerBotMAIF,model=lucy.dat,abstraction=fcpa" \
  --bot-b "openspiel:os.pkl" --pairs 2500 --seed 11 --out result.json
```

```bash
# Unity HPC
bash slurm/deploy.sh                                          # rsync
ssh unity
cd /work/pi_machta_umass_edu/$USER/lucy-bench
sbatch -p cpu-preempt -t 1:00:00 slurm/setup.sbatch           # one-shot
ITERS="1000 10000 100000 1000000" SEEDS="1 2 3" PAIRS=2500 \
  bash slurm/submit_all.sh

# Wait, then aggregate
python -m bench.aggregate --results-dir results --out results/summary.csv
python -m bench.plot --csv results/summary.csv --out results/plot.png
```

## Adding a new Lucy variant (e.g., `cuda-v2` with retuned training loop)

1. Create a new git worktree at the desired commit:
   ```
   git worktree add ../Lucy-worktrees/cuda-v2 <commit-or-branch>
   ```
2. Apply the same FCPA + JSON-serve patch (see commits `225297a` for cuda,
   `b10a885` for legacy) to expose the same CLI surface.
3. Update the variant list in `slurm/submit_all.sh` and the dispatch in
   `slurm/train_array.sbatch` / `slurm/eval_array.sbatch`.
4. Re-run `bash slurm/setup_env.sh` to build the new binary.
5. Re-submit the sweep — the harness produces a new variant curve in the plot.
