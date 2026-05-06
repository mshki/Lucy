#ifndef GPU_CFR_H
#define GPU_CFR_H

// GPU-resident CFR engine for HU NLHE FCPA.
//
// Unlike cuda_kernels.h (which only does the per-flush regret-match on the
// device while leaving cfr() recursion on CPU), this engine puts the **entire
// outcome-sampling MCCFR traversal** on the GPU. Each thread runs one
// trajectory: chance-sample, walk the game tree, accumulate trajectory steps
// into a per-thread buffer, evaluate terminal payoff via a device-side hand
// evaluator, then backward-pass the regret + strategy_sum updates with
// atomicAdd into device-resident tables. A separate regret-match kernel
// refreshes the strategy cache between batches.
//
// The engine maintains its own info-set hash table on the device — open
// addressing with linear probing, ~16M slots, atomic-CAS insert. Info-set
// keys are encoded as 64-bit integers packing (bucket, stage, pot_bucket,
// per-street raise summary). For HU NLHE FCPA + V1 hand abstraction +
// imperfect-recall keys this is well under the 16M slot capacity.
//
// Memory budget:
//   regret_sum:   capacity * MAX_ACTIONS * 8 bytes  (~512 MB at 16M, K=4)
//   strategy_sum: capacity * MAX_ACTIONS * 8 bytes  (~512 MB)
//   strategy:     capacity * MAX_ACTIONS * 8 bytes  (~512 MB)
//   hash table:   capacity * (8 + 4) bytes            (~192 MB)
//   total:        ~1.7 GB device memory at full capacity
//
// All public functions are host-side and serialise to one CUDA stream. The
// engine is NOT thread-safe (one process / one engine instance assumed).
//
// Design notes (more in src/cuda/gpu_cfr.cu):
//   * V1 hand bucketing only for v0.5 — keeps the device evaluator simple
//     and the bucket-key encoding compact. V2/V3 require more LUT memory
//     or per-query rollouts; deferred.
//   * Outcome sampling only for v0.5 — external sampling has variable
//     branching per node which would hurt warp efficiency. OS gives
//     thread-uniform per-trajectory work.
//   * FCPA only for v0.5 — 4 stable action IDs per node simplify the
//     trajectory buffer (fixed-size arrays).
//   * Save/load to disk uses a custom binary format (`gpu_model.dat`)
//     that round-trips device memory. Cross-compat with the CPU
//     poker_model.dat is handled via separate gpu_cfr_export_to_cpu_format
//     which materialises host-side string keys.

#include <cstdint>
#include <string>
#include <vector>

namespace gpu_cfr {

// FCPA: 4 stable action IDs (0=fold, 1=check/call, 2=pot, 3=allin).
constexpr int kMaxActions = 4;

// Trajectory-step buffer depth. Cap at 32 — HU NLHE preflop+flop+turn+river
// with 6 raises per street caps at ~24 player actions plus ~4 chance nodes.
constexpr int kMaxDepth = 32;

// Hash-table capacity. Power of 2. 1<<19 = 524,288 slots is plenty for HU
// NLHE FCPA + V1 hand abstraction + imperfect-recall keys (typical info-set
// count under this abstraction is well under 100k). For V3 (200 buckets per
// post-flop street) the count grows ~20x; we bump to 1<<20 = 1M slots.
// Memory: ~12 MB hash + ~100 MB regret/strategy tables at MAX_ACTIONS=4.
// The regret-match kernel only loops over this many rows per iteration so
// the cost is bounded.
constexpr int kHashCapacityLog2 = 20;
constexpr int kHashCapacity = 1 << kHashCapacityLog2;

// Hand abstraction selector.
//   V1: 10-bucket heuristic (no rollouts, fast). Existing default.
//   V3: EHS²-clustered buckets (169 preflop canonical, 200/200/200 post-flop
//       K-means centroids). Requires equity_buckets.dat at engine create.
//       Per node visit: O(rollouts × 2 × 80) extra ops for the EHS²
//       Monte Carlo rollouts on device. ~5-8x slower per trajectory than
//       V1 but ~1500 mbb/h stronger play (matches CPU V3 quality).
enum class HandAbstraction : uint8_t {
  V1 = 0,
  V3 = 1,
};

// V3 cluster counts per street (must match build_equity_buckets defaults
// and the on-disk equity_buckets.dat layout).
constexpr int kV3PreflopBuckets = 169;
constexpr int kV3PostflopBuckets = 200;

// Configuration for one training run.
struct GpuCfrConfig {
  int num_players       = 2;
  double small_blind    = 1.0;
  double big_blind      = 2.0;
  double starting_stack = 200.0;
  // Outcome-sampling exploration mixing (OpenSpiel default).
  double epsilon        = 0.6;
  // DCFR (alpha, beta, gamma). (1e30, 1e30, 0.0) = vanilla.
  double dcfr_alpha     = 1.5;
  double dcfr_beta      = 0.0;
  double dcfr_gamma     = 2.0;
  // Trajectories per kernel launch. Larger = better GPU utilisation but
  // more device memory for trajectory buffers (each trajectory keeps
  // up to kMaxDepth * (info_id + sigma vector) bytes in shared/registers).
  int batch_size        = 4096;

  // Hand abstraction. V1 (default) uses the 10-bucket heuristic and is the
  // existing fast path. V3 uses EHS² K-means clusters and produces ~1500
  // mbb/h stronger play at the cost of per-trajectory device work.
  HandAbstraction hand_abstraction = HandAbstraction::V1;
  // Path to equity_buckets.dat (V3 only; ignored for V1). Empty string
  // searches cwd. The file is produced by `build_equity_buckets`.
  std::string equity_buckets_path  = "equity_buckets.dat";
  // Number of MC rollouts per EHS² query (V3 only). 50 matches the
  // build_equity_buckets default. Increasing improves bucket assignment
  // accuracy at a linear wallclock cost; 100 is "production".
  int ehs2_rollouts                = 50;
};

// Lifetime of the device engine: create -> train -> save / query -> destroy.
struct GpuCfrEngine;

GpuCfrEngine *gpu_cfr_create(const GpuCfrConfig &cfg);
void gpu_cfr_destroy(GpuCfrEngine *eng);

// Run `num_iterations` * `batch_size` trajectory samples. Each iteration
// is one batched kernel launch + one regret-match kernel launch. Returns
// the wall-clock seconds spent on the device.
double gpu_cfr_train(GpuCfrEngine *eng, int num_iterations,
                     uint64_t base_seed);

// Number of distinct info-sets discovered so far.
int gpu_cfr_num_infosets(const GpuCfrEngine *eng);

// Save the device state to a binary file. Format described in gpu_cfr.cu.
// Loads can resume training or be used at eval time.
bool gpu_cfr_save(GpuCfrEngine *eng, const std::string &path);
bool gpu_cfr_load(GpuCfrEngine *eng, const std::string &path);

// Query the average strategy at a given state — used at eval time. Returns
// the FCPA-action probability vector aligned with [F, C, P, A] (zero for
// illegal slots). The query is reasonably cheap (one hash lookup + one
// strategy_sum read), but cumulative cost across thousands of eval queries
// can exceed kernel-launch overhead — for evaluation, prefer downloading
// the whole strategy table once via gpu_cfr_export_strategy_sum and
// querying host-side.
struct DeviceQueryRequest {
  int player_id;
  int dealer;       // 0 or 1 for HU
  uint8_t hole_cards[2]; // OMP-encoded card IDs (0..51)
  uint8_t board_cards[5];
  uint8_t num_board;
  uint8_t stage;     // 1=preflop ... 4=river
  // History (FCPA action IDs in [0,3]) and per-action acting players.
  int num_history;
  uint8_t history_action[kMaxDepth];
  uint8_t history_player[kMaxDepth];
};
struct DeviceQueryResponse {
  bool found;             // true iff the info-set was visited during training
  uint64_t infoset_key;
  double probabilities[kMaxActions];
  int num_legal;
  int legal_actions[kMaxActions]; // FCPA IDs in [0,3]
};
DeviceQueryResponse gpu_cfr_query(GpuCfrEngine *eng,
                                  const DeviceQueryRequest &req);

// Profile counters dumped at gpu_cfr_destroy (or via this getter). All
// times are nanoseconds.
struct GpuCfrProfile {
  long long total_ns        = 0;
  long long traversal_ns    = 0; // the big trajectory kernel
  long long match_ns        = 0; // regret-match refresh
  long long strategy_dl_ns  = 0; // strategy snapshot pulled to host (rare)
  long long sync_ns         = 0;
  long long iters           = 0;
  long long trajectories    = 0;
};
GpuCfrProfile gpu_cfr_profile(const GpuCfrEngine *eng);

} // namespace gpu_cfr

#endif
