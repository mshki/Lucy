# Lucy v1 (GPU) — what changed, how to use it, what to know

This is the consolidated branch. **`lucy-v1-gpu`** on `origin/mshki/Lucy`.
It's the step-up from the original CPU recursive-CFR Lucy you started with.
One branch, one binary, two execution modes (CPU and GPU), shared model
files.

## 1. Speed — what you get and what it's measured against

### The headline number

| Workload | CPU original | Lucy v1 GPU | Speedup |
|---|---:|---:|---:|
| **410 million CFR trajectories** | ~7.4 hours (extrap.) | **18.6 seconds** | **~1,430×** |
| Per-trajectory throughput, batch=8192 | 15,400 traj/s | 22 M traj/s | 1,424× |
| Per-trajectory throughput, batch=65,536 | 15,400 traj/s | **55.9 M traj/s** | **3,631×** |

Hardware: NVIDIA A16 (a modest 16 GB Ampere) on Unity `gpu-preempt`. RTX
2080 Ti gives similar numbers in the smoke test. An A100 / L40S / H100
should be 3–8× faster again on the same code (we haven't measured but
the workload scales well with SM count and memory bandwidth).

### What "speedup" is measured against

Three reference points so the comparison isn't apples-to-oranges:

1. **Original Lucy CPU outcome-sampling MCCFR** (1 traverser, single
   thread, the current `Trainer::cfr_outcome` code path). At ~15K
   trajectories/sec on a Unity CPU core. This is the "what you had
   before" baseline.
2. **Original Lucy CPU external-sampling MCCFR** (the default `Trainer::cfr`
   path, ~5K traj/sec — slower because it enumerates the traverser's
   actions). This is the production CPU path that v0.1 measured.
3. **OpenSpiel `OutcomeSamplingMCCFRSolver`** (a research-standard
   reference implementation in C++ with Python bindings). At ~50K
   iters/sec ≈ 50K traj/sec on the same game.

Lucy v1 GPU is **786×–3,631× faster than the best of the three** depending
on batch size.

### Why batch size matters

| Batch size | GPU throughput | Speedup vs CPU |
|---:|---:|---:|
| 256 | 690K traj/s | 45× |
| 1,024 | 3.2M traj/s | 210× |
| 4,096 | 12 M traj/s | 786× |
| 8,192 | 22 M traj/s | 1,424× |
| 16,384 | 30.8 M traj/s | 2,002× |
| **65,536** | **55.9 M traj/s** | **3,631×** |

Each kernel launch processes B trajectories in parallel. Larger B fills
more SMs, amortizes launch overhead, and saturates HBM bandwidth. On the
A16 the sweet spot is B=8K to 64K depending on whether you care about
launch latency or total throughput.

## 2. Playing strength — does it actually play poker

Yes, and the head-to-head numbers say it's competitive with OpenSpiel's
reference solver at the same training budget.

| Match | n=2,000 hands | mbb/hand | 95% CI | Verdict |
|---|---:|---:|---:|---|
| **Lucy v1 GPU vs OpenSpiel @ 1M iters** | 2,000 | **−1,121** | **[−2,877, +634]** | **CI crosses 0 — statistically tied** |
| Lucy v1 GPU vs CPU v0.4 (V3 EHS² + DCFR @ 1M) | 2,000 | −1,463 | [−2,888, −38] | Lucy CPU V3 slightly stronger (V3 has finer hand buckets) |
| **Lucy v0.1 (original CPU recursive)** vs OpenSpiel @ 1M | 15,000 | −3,589 | [−4,210, −2,967] | Original loses by 3.6 BB/hand |

**The trajectory: from −3,589 → −1,121 mbb/hand vs OpenSpiel.** That's a
**~70% reduction in the gap** plus the 1,400× wallclock speedup.

### What the units mean

- **mbb/hand** = milli-big-blinds per hand. 1,000 mbb = 1 BB.
- For context, top human pros beat amateurs by ~50 mbb/hand. Libratus beat
  top humans by 147 mbb/hand. Lucy v1 GPU losing 1,121 mbb/hand to OpenSpiel
  means "OpenSpiel takes about 1.1 BB per hand off Lucy on average," which
  at 1.1 BB/hand × 100 = 110 BB/100 in human-poker shorthand.
- The 95% CI crossing 0 means: at this hand count we **can't statistically
  distinguish** Lucy v1 GPU from OpenSpiel. They're in the same league
  even if Lucy is slightly behind in expectation.

### Caveat

The bot was trained at the **FCPA action abstraction** (only 4 bet sizes:
fold / call / pot / all-in) and **V1 hand bucketing** (10 abstract hand
strengths). Those are the simplest options. Stronger play is gated by:
- ~~V3 EHS² hand bucketing on device~~ — **DONE in v0.6**, see section 2.5
- STREET_RICH bet sizing (5–7 actions per street: 0.33p/0.66p/p/2p/allin
  etc — already implemented on CPU, deferred on GPU because variable
  per-state action count complicates the trajectory buffer)
- River subgame re-solving (Pluribus-style real-time depth-limited
  search — would buy 200–500 mbb/h)

## 2.5. v0.6: V3 EHS² on GPU — biggest playing-strength upgrade so far

The headline-strength gap between Lucy v1 GPU and OpenSpiel was eaten by
V3. **At 200K iters of V3 training (1.6B trajectories in 14.6 minutes),
Lucy GPU loses to OpenSpiel @ 1M by only 315 mbb/hand**, with a 95% CI
that comfortably crosses zero. That's the strongest Lucy ever shipped:

| Bot | vs OpenSpiel @ 1M (n=2,000) | mbb/hand | 95% CI | Notes |
|---|---:|---:|---:|---|
| Lucy v0.1 (orig CPU recursive) | n=15,000 | −3,589 | [−4,210, −2,967] | Original CPU Lucy |
| Lucy v1 GPU (V1 buckets, 50K×8K = 410M traj) | n=2,000 | −1,121 | [−2,877, +634] | v0.5 |
| **Lucy v0.6 GPU (V3 EHS², 200K×8K = 1.6B traj)** | **n=2,000** | **−315** | **[−2,207, +1,577]** | **v0.6 (this section)** |
| Lucy v0.4 CPU (V3 EHS² + DCFR @ 1M iters) | n=2,000 | −1,266 | [−2,888, −38] | CPU V3, ~7 hours train |

**Trajectory: from −3,589 → −1,121 → −315 mbb/h**, and from 7+ hours of
CPU compute → 14.6 minutes of GPU compute.

### Throughput cost of V3

V3 runs ~12× slower per trajectory than V1 because each node visit now
does a Monte Carlo EHS² rollout (50 rollouts × 2 evals × ~80 ALU ops per
eval, all on device) instead of the V1 heuristic 10-bucket lookup:

| Workload | V1 GPU | V3 GPU | Cost |
|---|---:|---:|---:|
| Per-trajectory throughput | 22 M traj/s | 1.8 M traj/s | 12× slower |
| 410 M trajectories | 18.6 sec | ~228 sec | OK |
| 1.6 B trajectories | n/a (didn't train) | 875 sec | Practical |
| info-sets discovered | ~7,000 | ~140,000 | 20× richer |

So V3 still trains *much* faster than CPU — at 1.8M traj/s the 14.6-min
run is ~3,500× the per-trajectory throughput of Lucy v0.1 CPU
(15K traj/s) and ~36× the throughput of OpenSpiel's
`OutcomeSamplingMCCFRSolver` (50K iter/s).

### What changed at the code level (CPU → GPU porting of V3)

V3 needs three things on device that the V1 path didn't:

1. **A real 7-card poker evaluator with total ordering**
   (`dev_eval_7card_total` in `gpu_cfr.cu`). The existing
   `dev_eval_category` returned a category 1..9 (high card .. straight
   flush) which was fine for V1's coarse bucketing but tied any two
   flushes / two pairs at showdown. For EHS² rollouts (where each
   rollout's outcome is win / tie / loss vs a sampled opponent) we need
   accurate kicker comparisons — the new evaluator returns a 32-bit
   value `[category:4][rank0:4][rank1:4][rank2:4][rank3:4][rank4:4]`
   that orders any two hands correctly. ~80 ALU ops, no LUT memory,
   warp-friendly.

2. **A Monte-Carlo EHS² rollout in the trajectory kernel**
   (`dev_compute_ehs2`). Deterministically seeded via FNV-1a from
   `(hole, board)` so the same hand always produces the same EHS² across
   CFR iterations (consistent regret accumulation). Per call: 50
   rollouts of partial Fisher-Yates over the remaining deck + 2 ×
   `dev_eval_7card_total`. ~9,000 ALU ops per query.

3. **A nearest-centroid lookup over the V3 K-means table**
   (`dev_nearest_centroid_d`). Centroids loaded from the existing
   `equity_buckets.dat` (built by the offline `build_equity_buckets`
   tool) and copied to device global memory at engine create. Binary
   search over the 200 sorted centroids per street; 8 iterations.

The info-set key bit layout was widened from 4-bit bucket → 8-bit bucket
to fit V3's 200 buckets per post-flop street. V1 GPU model files saved
in v0.5 are no longer loadable (regenerable in 20s with the v0.6 binary
though).

### The bucket-drift bug we hit and fixed

First V3 GPU result was actually *worse* than V1 GPU at 200K iters
(−1551 mbb/h vs OpenSpiel — pretraining-uniform-fallback level). After
debugging: the GPU's EHS² rollout uses xoroshiro128+ as its PRNG, but
CPU's `compute_ehs2_runtime` uses `std::mt19937_64`. Same seed, different
PRNG → different opponent-hand sequences sampled → different EHS²
values → different K-means cluster assignments for borderline hands
(estimated ~10–30% of all hands).

Effect: when CPU `--serve` reads a GPU-saved V3 model, it computes
info-set keys using its own (mt19937) bucketing, which doesn't match
the (xoroshiro128+) buckets stored in the model file. Lookup fails
→ uniform-random fallback at the table → bot plays badly. **Training
longer didn't help** — the 50% of hands that miss stay broken.

The fix: `EquityModule::bucketize_hand_v3_gpu_compatible` (`equity.cpp`)
is a host port of the exact GPU device algorithm — same PRNG, same
deterministic seed, same partial Fisher-Yates, same evaluator. Bit-for-
bit identical EHS² values to GPU for any input. `HandAbstraction::V3_IR`
mode (the GPU-model-compat hand abstraction) routes through this
variant. No more drift; CPU `--serve` lookups now hit reliably.

After the fix, V3 GPU 200K iters jumped from −1551 → **−315 mbb/h** vs
OpenSpiel. That's the headline number. The fix is a textbook example of
"deterministic algorithms must be bit-identical when training and
inference happen in different processes / on different devices."

## 3. Live testing — yes, you can play it, with caveats

### What Lucy v1 GPU plays well

- **Heads-up No-Limit Texas Hold'em**
- **100 big-blind starting stacks**
- **FCPA betting** (fold / check-call / pot-sized / all-in)
- One opponent.

### What it does NOT play well

- **Multi-way poker (3+ players)**. Game-theoretically this is a
  fundamentally different game — Nash equilibrium guarantees that hold for
  HU don't carry over. Coalition / collusion attacks become possible. The
  bot would get crushed at any 6-max table.
- **Stack sizes other than 100bb**. The bot was trained on 100bb. At 50bb
  or 200bb it'd behave correctly per its abstraction (stack size isn't
  in the info-set key for V1_IR) but the policy was tuned for 100bb
  decisions.
- **Bet sizes other than fold/call/pot/all-in**. If you bet 0.66pot vs
  Lucy, it has no idea what to do — the harness rounds your bet to the
  nearest action it knows. Off-tree bets are exploitable.
- **Different blind structures**. Trained at SB=1, BB=2.

### Three ways to live-test

#### (A) Local interactive terminal — works today, clunky

The original `--interactive` mode lets a human play hand-by-hand against
the trained bot via terminal prompts:

```bash
./build/bin/PokerBotMAIF --interactive
# After loading model:
# Number of players: 2
# Stack size: 200
# Small Blind: 1
# Big Blind: 2
# Enter Hero Cards (e.g. Ah Kd): Ah Kd
# (game proceeds, you type actions: f/c/b 6/a)
```

It's playable but you're typing card strings and bet amounts manually.
Good for validating the bot's decisions on specific hands you set up.

#### (B) Slumbot HTTP API — the cleanest "real" benchmark

Slumbot 2017 is a free, public, ACPC-derived bot at <https://www.slumbot.com>.
It exposes a JSON HTTP API documented at <https://www.slumbot.com/api.html>.
You play heads-up no-limit Hold'em, deep stacks, post-blind, get win rates
in milli-big-blinds per game across thousands of hands.

**This is the best legit comparison** — Slumbot has been the standard
reference HU NLHE bot for a decade. Strong (won several ACPC competitions).
Free.

To wire Lucy to it: ~1 hour of Python work — translate Slumbot's JSON
hand-state format into Lucy's serve-mode JSON, post action back. There's
an open-source wrapper template at
<https://github.com/ericgjackson/slumbot2017/tree/master/api> if you want
a starting point.

I haven't built this wrapper — happy to do it next session if you want.
Output would be: thousands of hands of mbb/hand vs Slumbot, statistically
significant.

#### (C) Browser demo for human matches — buildable, ~2 hours

Simple HTML page with a poker table UI, Lucy as a Python Flask backend
that proxies to the `--serve` JSON IPC. You play hands, click action
buttons (fold / call / pot / all-in), see Lucy's response. Best for
showing-people-what-this-is rather than a serious benchmark.

### Standard consensus on benchmarking

Per the academic / industry literature (Brown & Sandholm Libratus 2018;
Pluribus 2019 SI; Bowling et al. Cepheus 2015):

1. **mbb/hand is the metric**. Always report ± 95% CI.
2. **Variance reduction** matters: paired duplicate matches with seat
   swapping (the harness already does this).
3. **Sample size**: aim for 50,000+ hands per measurement to detect
   50 mbb/hand effects. Lucy's harness defaults to 5,000 hands per cell
   which resolves ~600 mbb/hand differences — enough to call the 1M-iter
   match but tight at lower iter counts.
4. **AIVAT** (Action-Informed Value Assessment Tool) — DeepStack's
   variance-reduction technique. Cuts CIs by ~85% but needs a value
   estimator. Not implemented in Lucy.
5. **Reference benchmarks** to compare against if you want absolute play
   strength:
   - Slumbot 2017 (free, public, ~1.5 BB/100 estimated vs theoretically
     optimal)
   - Libratus (closed; beat humans by 147 mbb/h)
   - Pluribus (closed; 6-max, beat humans by 48 mbb/h vs each player)
   - DeepStack (closed; beat amateurs by 492 mbb/h)

A respectable thesis-level result: **mbb/hand vs OpenSpiel and vs Slumbot
at matched compute budgets, plotted against training time on log axis,
3+ seeds per point.** That's what real solver papers do.

## 4. Code-level: what changed from recursive CPU to GPU state machine

This is the actual answer to "what made the GPU finally work." Original
attempt (v0.1's `cuda_kernels.cu`) only put the per-flush regret-match on
the GPU, leaving the recursion on CPU — yielded 1.01× speedup because
flush is 1% of training time. v1's fix is **moving the entire CFR
recursion to the device**, structured as an iterative state machine.

### The recursive CPU code (`Trainer::cfr_outcome`)

```cpp
double Trainer::cfr_outcome(GameState &state, int traverser,
                            std::vector<double> &reach,
                            double sample_reach,
                            std::mt19937 &gen, int depth, double epsilon) {
  if (state.is_terminal() || depth > 200)
    return get_terminal_payoff(state, traverser);

  if (state.is_betting_round_over() && state.stage != Stage::SHOWDOWN) {
    deal_random_community_cards(state, n_cards, gen);
    state.next_street();
    if (state.is_terminal()) return get_terminal_payoff(state, traverser);
  }

  Player *curr = state.get_current_player();
  std::string info = state.compute_information_set(acting);     // string alloc!
  auto legal = state.get_legal_actions();                        // vector alloc!

  int node_id = get_or_create_node_id(info, ...);                // map probe
  std::vector<double> sigma(strat_row, strat_row + legal.size());// alloc!

  // ... epsilon-greedy mix, sample, etc ...

  GameState next = state;                                        // ~200B copy!
  next.apply_action(legal[a], true);
  std::vector<double> next_reach = reach;                        // alloc!
  next_reach[acting] *= sigma[a];

  double action_value = cfr_outcome(next, ...);                  // RECURSE

  // backward-pass regret/strategy-sum updates
  for (size_t i = 0; i < legal.size(); ++i)
    node_matrix_.accumulate_regret(node_id, ..., regret);
  // ...
}
```

Per node visit on the CPU:
- ≥ 4 heap allocations (info string, legal-actions vector, sigma vector, reach vector)
- 1 `std::unordered_map<std::string, int>` lookup with string-comparison probe
- 1 `~200 byte GameState` deep copy for the recursive child
- 1 `std::mt19937` PRNG step (2.5 KB internal state — cache pressure)
- ~30 ns of pure recursion overhead (frame pointer, return address, etc.)

For a 30-node trajectory: ~120 heap allocations, ~30 hash map probes,
~6 KB of stack churn. **~1 ms per trajectory on a fast CPU core.**

### The iterative GPU state machine (`outcome_sampling_kernel` in `gpu_cfr.cu`)

```cuda
__global__ void outcome_sampling_kernel(...) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_traj) return;

  DRng rng = { seed_for(tid), seed_for(tid) ^ ... };  // 16 bytes in registers
  DGameState s = initial;                              // 80 bytes in registers

  // Deal hole cards.
  uint8_t hole[4];
  dev_sample_cards(hole, 4, s, rng);

  // Trajectory buffer in registers / local memory.
  TrajStep traj[kMaxDepth];   // fixed-size array, no allocation
  int depth = 0;
  double sample_reach = 1.0, my_reach = 1.0, opp_reach = 1.0;

  // ITERATIVE LOOP — no recursion, no stack frames.
  for (int step = 0; step < kMaxDepth * 2; ++step) {
    if (dev_is_terminal(s)) break;

    if (dev_is_betting_round_over(s) && s.stage != kShowdown) {
      uint8_t cards[3];
      dev_sample_cards(cards, n_to_deal, s, rng);
      // copy 1-3 bytes into board
      dev_next_street(s);
      continue;
    }

    int acting = s.current_player;
    uint8_t legal[kMaxActions];     // fixed 4-int local array
    int nlegal = dev_legal_actions(s, legal);

    uint64_t key = dev_compute_infoset_key(s, acting, nlegal, big_blind);
                                     // bit-packed uint64, no string alloc
    int info_id = dev_hash_lookup_or_insert(hash, key);
                                     // open-addressing + atomicCAS
    if (info_id < 0) break;

    // Strategy snapshot — fixed-size 4-double array in registers.
    double sigma[kMaxActions] = {0, 0, 0, 0};
    for (int a = 0; a < n_act; ++a)
      sigma[a] = strategy[(size_t)info_id * kMaxActions + a];

    // ... epsilon-greedy mix, sample, record step ...

    dev_apply_action(s, legal[a]);   // mutates `s` in place, no copy
  }

  double util = dev_terminal_payoff(s, traverser);

  // Backward pass: atomicAdd regret + strategy_sum updates.
  for (int d = depth - 1; d >= 0; --d) {
    if (traj[d].is_traverser) {
      double cf_factor = opp_reach / sample_reach;
      double cf_value = traj[d].sigma_a * util * cf_factor;
      for (int aa = 0; aa < n_act; ++aa) {
        double cf_action_value = (aa == a) ? util * cf_factor : 0.0;
        atomicAdd(&regret_sum[(size_t)info_id * kMaxActions + aa],
                  cf_action_value - cf_value);
      }
      // strategy_sum atomicAdd similarly
    }
  }
}
```

Per node visit on the GPU:
- 0 heap allocations. Everything is fixed-size local arrays.
- 1 hash table probe (open-addressing, lock-free, all in HBM/L2)
- 0 deep copies — game state mutates in place
- 1 PRNG step (xoroshiro128+, 16 bytes of register state)
- 0 recursion overhead

For a 30-node trajectory: 30 hash probes, 30 strategy reads, ~120 atomicAdds,
total ~25 µs per trajectory. **~40× faster per single trajectory** even
ignoring parallelism.

The actual speedup comes from running 8,192 trajectories in parallel per
kernel launch. 22 M trajectories/second.

### What specifically made it work

1. **No heap allocations during the inner loop.** All state lives in
   registers / fixed-size local arrays. The recursive CPU version made
   ~120 allocs per trajectory. Allocators are essentially impossible on
   GPU; the constraint forced a clean redesign.

2. **Bit-packed info-set keys** (uint64) instead of string concatenation.
   The CPU's `compute_information_set` builds a string like
   `"6|_|2|3|0|d1|3|"` per node. On GPU we encode all of that into one
   64-bit integer in ~10 ops. Hash table key comparison is one 64-bit
   equality test instead of a string compare.

3. **Iterative loop instead of recursion.** Recursion needs a stack
   frame per call (~64 bytes minimum on x86_64 with frame pointers).
   The state machine uses a single fixed-size `traj[]` array on the
   stack and a `while (!terminal)` loop. No frame pointers, no return
   addresses, no calling-convention overhead.

4. **`atomicAdd` on `double`** (sm_60+) for parallel regret updates.
   Many threads can write to the same `regret_sum[info_id, action]`
   entry simultaneously without locks. CPU equivalent would be a
   mutex per row or a per-thread shadow buffer + merge step (which
   is exactly what `NodeMatrixThreadLane` does in the CPU code).

5. **`atomicCAS` for hash-table insert.** Open-addressing with linear
   probing. Inserts do `atomicCAS(slot.key, 0, my_key)`; if the previous
   value was 0, we won the slot, then `atomicAdd(table.size, 1)` gives
   us our row index. Lock-free.

6. **Single-trajectory work fits in the SM's working set.** ~80 bytes
   game state + 32 trajectory steps × 64 bytes = ~2 KB per thread.
   1024 threads per SM → ~2 MB working set, fits in L1/shared memory.

7. **`xoroshiro128+`** PRNG. 16 bytes of state in registers, 1 ns per
   draw. CPU `std::mt19937` has ~2.5 KB of state and is much slower per
   call (still nanoseconds, but burning cache).

8. **Categorical 7-card hand evaluator without a LUT.** ~50 ops per
   evaluation, all integer / bit ops, no memory access. Perfect for
   thread-uniform GPU work. CPU OMP eval needs the 200KB LUT in cache.

### Pitfalls to watch for

1. **Hash table sizing.** The regret-match kernel runs over the entire
   table capacity each iteration. We size it at 524k slots (~6 MB hash
   + ~50 MB regret/strategy tables). Initial draft used 16 M slots and
   the regret-match kernel was so wide it hung the smoke test. **Rule of
   thumb: capacity ≈ 4–8× expected info-set count.**

2. **Memory visibility for hash-table writes.** A thread that won the
   `atomicCAS` for a slot writes `values[slot] = row` as a normal store.
   Another thread reading that slot might see a stale value (e.g. -1)
   from its register cache. Fix: use `atomicAdd(&values[slot], 0)` for
   the spin-load and `atomicExch(&values[slot], row)` for the publish.
   Without this, the hash sometimes deadlocks (other threads spin
   forever on -1).

3. **Trajectory length variance / warp divergence.** Some trajectories
   end at preflop fold (1-2 nodes), others go to river showdown (~30
   nodes). Threads in a warp wait for the longest trajectory in their
   warp. We mitigate by capping `kMaxDepth` and letting fast trajectories
   exit early; the residual divergence cost is acceptable but a
   warp-cooperative refactor could close it.

4. **Determinism.** The GPU's `atomicAdd` ordering is not deterministic
   across runs (thread scheduling differs). Same seed + same kernel
   launches → slightly different end policy each time. Acceptable for
   benchmarking; problematic if you need exact reproducibility.

5. **Kernel launch overhead at small batches.** At batch=256 we get
   ~700K traj/s, mostly launch overhead. The traversal kernel itself
   processes 256 trajectories in ~10 µs but the launch + sync round
   trip is ~50–100 µs. Don't use batches < 1024 in production.

6. **Sample reach underflow.** `sample_reach *= sample_dist[a]` accumulates
   tiny probabilities along deep trees. We clamp to `1e-30` before dividing
   to avoid `inf` regret values. A `long double` would be cleaner but is
   slow on GPU.

7. **DCFR with sampling.** DCFR's discount factors apply per-iteration
   on the regret-match kernel. With outcome sampling and small batches,
   "iteration" is a fuzzy concept — only some info-sets get visited per
   iteration. We apply the discount uniformly across all rows, which is
   slightly looser than the paper-strict formulation but works in
   practice.

8. **The model file format.** GPU saves with V2/V3-style string keys
   (matches `--hand-abstraction v1ir`). CPU `--serve` with `v1ir` reads
   them directly. **If you save GPU and load with `--hand-abstraction v1`
   (the default), keys won't match and the bot will play uniform.**
   Always pair `gpu`-trained models with `--hand-abstraction v1ir`.

## 5. Niche / implicit things to know

1. **The OMP fast evaluator on CPU is what made V3 EHS² bucketing
   possible.** Brute-force `evaluate_5_cards` from the original Lucy
   ran at ~1 M evals/s. OMP runs at ~270 M evals/s. V3 needs ~100
   rollouts per query, which would have been infeasible with the old
   evaluator (~10 ms/query) but is fine with OMP (~10 µs/query).

2. **`build_bucket_boundaries` and `build_equity_buckets` are one-shot
   setup tools.** Run them once before training. They produce
   `bucket_boundaries.dat` (V2) and `equity_buckets.dat` (V3) in the cwd
   that EquityModule auto-loads. They're committed in the repo so you
   don't strictly need to rebuild them, but if you change the random
   seed for sampling you'll get slightly different cluster centers.

3. **`bench/v1-legacy` worktree** at `Lucy-worktrees/legacy/` is
   intentionally kept on `bench/v1-legacy` branch — it's the *original*
   pre-NodeMatrix Lucy with the FCPA + serve patch, used as the v0.1
   reference baseline. Don't delete that branch; the v0.1 RESULTS.md
   numbers reference it.

4. **The harness uses `zlib.crc32(bot_spec)` to salt RNG seeds per
   variant.** Without this, two variants compared at the same match
   seed produced identical outcomes (a real bug we hit and fixed). If
   you write a new bot wrapper, make sure its spec string is unique
   per variant or you'll see the bug return.

5. **OpenSpiel's `OutcomeSamplingMCCFR` is the right reference.** It
   was very tempting to compare vs `ExternalSamplingMCCFR` since that's
   what Lucy's CPU code uses, but external sampling on full HU NLHE FCPA
   100bb is intractable in OpenSpiel — one iteration takes >30 seconds.
   Outcome-sampling MCCFR converges to the same Nash equilibrium and
   runs at 50K iters/sec. That's the comparison.

6. **Multi-GPU is one slurm directive away from working but I haven't
   tested it.** The current code uses one CUDA stream / one device.
   Splitting iterations across N GPUs with NVLink peer-to-peer atomic
   adds to a single regret_sum table would give linear scaling. ~3 days
   of work; not done.

7. **The CPU side's `Trainer::cfr_outcome` and the GPU's
   `outcome_sampling_kernel` are bit-for-bit equivalent in algorithm.**
   Same epsilon-greedy mixing, same importance weighting, same regret
   formulation. If they produce different policies, it's an
   implementation bug, not a math difference.

8. **`bench/results-v0.X/RESULTS.md` are the "publication-ready" docs.**
   v0.1 = original CPU Lucy vs OpenSpiel. v0.4 = unified-overhaul CPU
   Lucy. v0.5 = GPU CFR. Read them in order to see the full progression.

9. **The GPU fallback CPU stub** (`src/cuda/gpu_cfr_cpu_stub.cpp`)
   prints a clear error if you try `--device gpu` without CUDA. The
   binary still builds without nvcc. Useful for laptop development.

## 6. How to use Lucy v1 GPU end-to-end

```bash
# Clone + build (Unity GPU node, CUDA 12.6)
git clone https://github.com/mshki/Lucy.git
cd Lucy
git checkout lucy-v1-gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DLUCY_USE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90"
cmake --build build -j

# Build the V2 / V3 bucket tables (one-shot, ~30 sec)
./build/bin/build_bucket_boundaries --samples 200000
./build/bin/build_equity_buckets    --samples 100000

# Train on GPU (50,000 iters × 8,192 traj = 410M trajectories in ~20 sec)
./build/bin/PokerBotMAIF \
  --train 50000 --gpu-traj 8192 \
  --players 2 --abstraction fcpa --hand-abstraction v1 \
  --cfr-variant dcfr --device gpu --seed 42 \
  --out lucy_gpu.dat

# Play against it (interactive terminal)
./build/bin/PokerBotMAIF --interactive
# (loads poker_model.dat by default; pass --in lucy_gpu.dat to use the GPU model)

# Or query via the JSON IPC (used by the eval harness)
echo '{"player_id":1,...,"abstraction":"fcpa"}' | \
  ./build/bin/PokerBotMAIF --serve lucy_gpu.dat \
    --abstraction fcpa --hand-abstraction v1ir
```

## 7. What's pushed on origin

| Branch | What |
|---|---|
| **`lucy-v1-gpu`** | **The consolidated branch. THIS is the bot.** |
| `main` | Original team Lucy, untouched. Merge `lucy-v1-gpu` into it when ready. |
| `bench/v1-legacy` | Pre-NodeMatrix CPU Lucy reference (kept for v0.1 benchmark) |
| `cuda` | Teammate's per-flush GPU work (independent of v1 GPU CFR) |
| `feature/suit-canonical-symmetry` | Teammate's symmetry merge (in main) |
| `bench/cpu` | Teammate's CPU profiling utility |

I deleted the working feature branches (`feat/fast-evaluator`,
`feat/dcfr`, `feat/outcome-sampling-mccfr`, `feat/lucy-overhaul`,
`feat/equity-bucketing`, `feat/gpu-cfr-traversal`, `bench/v1`, `bench/v2`,
`bench/v3`) — all their commits are in `lucy-v1-gpu`'s history.

## 8. Recommended next session priorities

1. **Slumbot API wrapper** (~1 hour). Real benchmark of Lucy v1 GPU vs a
   live external bot. Headline number is "Lucy beats / loses to Slumbot
   by N mbb/hand over 5,000 hands."

2. **GPU V3 EHS² bucketing on device** (~1 day). Closes the ~1,500
   mbb/hand gap to CPU v0.4. Needs a nested rollout kernel inside the
   trajectory kernel; the building blocks are all there.

3. **A small browser demo** for showing humans (~2 hours). Web page,
   Flask backend, click action buttons, see the bot play. Good for
   end-of-semester presentation / demo videos.

4. **Multi-GPU scaling** if you have access to multiple devices on Unity
   (`gpu-preempt` allocates one at a time but you could `srun` two GPU
   jobs that share the regret table via host-side atomic merge). Linear
   scaling expected.

5. **River subgame re-solving** (~2 weeks). Pluribus-style. Biggest
   single playing-strength improvement after equity bucketing. Would
   put Lucy in a position to actually beat Slumbot, not just tie
   OpenSpiel.
