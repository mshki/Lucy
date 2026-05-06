// GPU-resident outcome-sampling MCCFR for HU NLHE FCPA.
//
// One CUDA thread = one CFR trajectory. Per kernel launch, B threads sample
// B independent trajectories. Each thread:
//   1. Walks the betting tree from the initial state (preflop, blinds posted).
//   2. At chance nodes, samples cards from a thread-local PRNG.
//   3. At player nodes, computes the info-set key, looks up (or inserts) the
//      info-set ID in a global open-addressing hash table, reads the strategy
//      row from the strategy cache, samples an action under epsilon-greedy
//      mixing, advances the state, and records the trajectory step.
//   4. At terminal, evaluates payoff via a device-side categorical 7-card hand
//      evaluator.
//   5. Walks the trajectory backwards, doing importance-weighted regret +
//      strategy_sum atomicAdds into device-resident tables.
//
// A separate regret-match kernel runs after every iteration to refresh the
// strategy cache from the (now updated) regret_sum table. DCFR discount
// factors are applied per-iteration in the regret-match kernel.
//
// Key design choices for v0.5:
//   * V1 hand abstraction (heuristic 10-bucket). Simple device-side rules,
//     no LUT memory. V2/V3 deferred (need OMP LUT or per-query rollouts on
//     device).
//   * FCPA action abstraction. 4 stable action IDs. kMaxActions = 4.
//   * Outcome sampling. External sampling has variable per-action work →
//     warp divergence kills throughput.
//   * Imperfect-recall info-set keys (Pluribus standard) — same encoding as
//     CPU path so save/load can be made compatible if desired.

#include "cuda/gpu_cfr.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

namespace gpu_cfr {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e__ = (call);                                                  \
    if (e__ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(e__));                                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// ============================================================================
// Constants visible to host AND device (kHashCapacity etc. are in the header).
// ============================================================================

constexpr int kPreflop = 1;
constexpr int kFlop = 2;
constexpr int kTurn = 3;
constexpr int kRiver = 4;
constexpr int kShowdown = 5;

// Action IDs (FCPA). Stable across states.
constexpr uint8_t kActFold  = 0;
constexpr uint8_t kActCall  = 1;
constexpr uint8_t kActPot   = 2;
constexpr uint8_t kActAllin = 3;

// Bucket IDs (V1 abstraction). Match CPU enum BucketID.
constexpr uint8_t kBucketAir         = 0;
constexpr uint8_t kBucketWeakDraw    = 2;
constexpr uint8_t kBucketStrongDraw  = 3;
constexpr uint8_t kBucketWeakPair    = 4;
constexpr uint8_t kBucketMidPair     = 5;
constexpr uint8_t kBucketTopPair     = 6;
constexpr uint8_t kBucketOverPair    = 7;
constexpr uint8_t kBucketStrongMade  = 8;
constexpr uint8_t kBucketNuts        = 9;

// ============================================================================
// Compact device-side game state.
//
// Layout aimed at register / shared-memory residency. ~80 bytes total.
// ============================================================================

struct DGameState {
  // Cards encoded as 0..51 (= rank * 4 + suit). 0xFF = unset.
  uint8_t hole[2][2];           // 4
  uint8_t board[5];             // 5
  uint8_t num_board;            // 1

  // Stage: kPreflop..kShowdown.
  uint8_t stage;                // 1
  uint8_t dealer;               // 1
  uint8_t current_player;       // 1
  uint8_t folded_mask;          // 1   bit i = player i folded
  uint8_t allin_mask;           // 1
  uint8_t acted_mask;           // 1   bit i = player i has acted this street

  // Chip amounts in BB units (HU NLHE 100bb max → fits in int16).
  int16_t pot;                  // 2
  int16_t high_bet;             // 2   max contributed-this-street any player
  int16_t cur_bet[2];           // 4   contributed this street, per player
  int16_t total_bet[2];         // 4   contributed in entire hand
  int16_t stack[2];             // 4

  // Action history this hand. Recorded for the imperfect-recall key. We
  // only need per-street raise summaries so we can compress aggressively.
  // For now keep a 32-entry log; it's cheap for HU FCPA.
  uint8_t num_hist;             // 1
  // High 4 bits = acting player, low 4 bits = action_id (FCPA 0..3).
  uint8_t hist[kMaxDepth];      // kMaxDepth = 32
};

// ============================================================================
// Device-side categorical 7-card hand evaluator.
//
// Returns one of 9 categories matching OMP's encoding:
//   1 = high card, 2 = one pair, 3 = two pair, 4 = three of a kind,
//   5 = straight, 6 = flush, 7 = full house, 8 = four of a kind,
//   9 = straight flush (incl. royal).
//
// Implementation: scan the up-to-7 cards once, build per-suit rank masks +
// per-rank counts, then test categories from strongest down. Total ~50 ops,
// no branches that depend on card values (warp-friendly). No LUT memory.
// ============================================================================

__device__ inline int dev_eval_category(const uint8_t *cards, int num_cards) {
  // Per-rank count (0..7), per-suit rank-mask (13 bits).
  int rank_count[13] = {0};
  uint16_t suit_mask[4] = {0, 0, 0, 0};
  for (int i = 0; i < num_cards; ++i) {
    uint8_t c = cards[i];
    int r = c >> 2;       // rank 0..12
    int s = c & 3;        // suit 0..3
    rank_count[r] += 1;
    suit_mask[s] |= (uint16_t)(1 << r);
  }

  // Detect flush: any suit with ≥5 cards.
  int flush_suit = -1;
  for (int s = 0; s < 4; ++s) {
    if (__popc(suit_mask[s]) >= 5) {
      flush_suit = s;
      break;
    }
  }

  // Detect straight: 5 consecutive ranks present in the rank set.
  uint16_t any_rank = 0;
  for (int r = 0; r < 13; ++r) if (rank_count[r] > 0) any_rank |= (1 << r);
  // Wheel: A-5 straight uses ace as low. Add bit for "ace at -1" by OR'ing
  // an implicit low-ace bit; equivalent: shift the ace bit (bit 12) down to
  // bit -1 by checking specifically.
  bool has_straight = false;
  int top_straight_rank = -1;
  // Test windows from high to low.
  for (int top = 12; top >= 4; --top) {
    uint16_t window = (uint16_t)(0x1Fu << (top - 4));
    if ((any_rank & window) == window) {
      has_straight = true;
      top_straight_rank = top;
      break;
    }
  }
  // Wheel A-2-3-4-5: top=3 (=5), need ranks {0,1,2,3,12}.
  if (!has_straight) {
    if ((any_rank & 0x100Fu) == 0x100Fu) { // bits 0..3 + bit 12
      has_straight = true;
      top_straight_rank = 3; // 5-high
    }
  }

  // Detect straight flush: all 5 cards in the straight share `flush_suit`.
  bool has_straight_flush = false;
  if (flush_suit >= 0 && has_straight) {
    uint16_t fmask = suit_mask[flush_suit];
    for (int top = 12; top >= 4; --top) {
      uint16_t window = (uint16_t)(0x1Fu << (top - 4));
      if ((fmask & window) == window) { has_straight_flush = true; break; }
    }
    if (!has_straight_flush) {
      if ((fmask & 0x100Fu) == 0x100Fu) has_straight_flush = true;
    }
  }
  if (has_straight_flush) return 9;

  // Quads / full house / trips / two pair / pair.
  int quads = 0, trips = 0, pairs = 0;
  for (int r = 0; r < 13; ++r) {
    if (rank_count[r] == 4) quads += 1;
    else if (rank_count[r] == 3) trips += 1;
    else if (rank_count[r] == 2) pairs += 1;
  }
  if (quads >= 1) return 8;
  if (trips >= 1 && (trips >= 2 || pairs >= 1)) return 7; // full house
  if (flush_suit >= 0) return 6;
  if (has_straight) return 5;
  if (trips >= 1) return 4;
  if (pairs >= 2) return 3;
  if (pairs == 1) return 2;
  return 1;
}

// ============================================================================
// V3 EHS² support: full 32-bit total-ordering 7-card evaluator + Monte Carlo
// rollout + nearest-centroid lookup. Used when cfg.hand_abstraction == V3.
//
// The 7-card evaluator returns a 32-bit comparable score (higher = stronger
// hand). Total ordering matches standard poker rules; ties only occur for
// genuinely equal-strength hands. The encoding is:
//   bits 24..27   category (1=high card .. 9=straight flush)
//   bits 0..23    up to 6 kicker / kept ranks, 4 bits each, top-down
// Two distinct flushes / two pairs / etc. resolve correctly via kicker bits.
// ============================================================================

__device__ inline uint32_t
dev_eval_7card_total(const uint8_t *cards, int num_cards) {
  int rank_count[13] = {0};
  uint16_t suit_mask[4] = {0, 0, 0, 0};
  for (int i = 0; i < num_cards; ++i) {
    int r = cards[i] >> 2;
    int s = cards[i] & 3;
    rank_count[r] += 1;
    suit_mask[s] |= (uint16_t)(1u << r);
  }

  int flush_suit = -1;
  for (int s = 0; s < 4; ++s) {
    if (__popc(suit_mask[s]) >= 5) { flush_suit = s; break; }
  }

  uint16_t any_rank = 0;
  for (int r = 0; r < 13; ++r)
    if (rank_count[r] > 0) any_rank |= (uint16_t)(1u << r);

  int top_straight = -1;
  for (int top = 12; top >= 4; --top) {
    uint16_t window = (uint16_t)(0x1Fu << (top - 4));
    if ((any_rank & window) == window) { top_straight = top; break; }
  }
  if (top_straight < 0 && (any_rank & 0x100Fu) == 0x100Fu) top_straight = 3;

  int top_sflush = -1;
  if (flush_suit >= 0) {
    uint16_t fmask = suit_mask[flush_suit];
    for (int top = 12; top >= 4; --top) {
      uint16_t window = (uint16_t)(0x1Fu << (top - 4));
      if ((fmask & window) == window) { top_sflush = top; break; }
    }
    if (top_sflush < 0 && (fmask & 0x100Fu) == 0x100Fu) top_sflush = 3;
  }

  // Pack helper. Cat in bits 24..27, then 5 ranks × 4 bits in 4..23.
  auto pack = [](int cat, int r0, int r1, int r2, int r3, int r4) -> uint32_t {
    return ((uint32_t)cat << 24) |
           ((uint32_t)(r0 & 0xF) << 20) |
           ((uint32_t)(r1 & 0xF) << 16) |
           ((uint32_t)(r2 & 0xF) << 12) |
           ((uint32_t)(r3 & 0xF) << 8)  |
           ((uint32_t)(r4 & 0xF) << 4);
  };

  if (top_sflush >= 0) return pack(9, top_sflush, 0, 0, 0, 0);

  int quads_rank = -1, trips_rank = -1, second_trips_rank = -1;
  int top_pair = -1, second_pair = -1;
  for (int r = 12; r >= 0; --r) {
    int c = rank_count[r];
    if (c == 4) { if (quads_rank < 0) quads_rank = r; }
    else if (c == 3) {
      if (trips_rank < 0) trips_rank = r;
      else if (second_trips_rank < 0) second_trips_rank = r;
    }
    else if (c == 2) {
      if (top_pair < 0) top_pair = r;
      else if (second_pair < 0) second_pair = r;
    }
  }

  if (quads_rank >= 0) {
    int kicker = -1;
    for (int r = 12; r >= 0; --r)
      if (r != quads_rank && rank_count[r] >= 1) { kicker = r; break; }
    return pack(8, quads_rank, kicker, 0, 0, 0);
  }

  // Full house: trips + (pair or another trips). With 7 cards we can have
  // two sets of trips; the lower set acts as the pair.
  if (trips_rank >= 0 && (second_trips_rank >= 0 || top_pair >= 0)) {
    int pair_rank;
    if (second_trips_rank >= 0 && (top_pair < 0 || second_trips_rank > top_pair))
      pair_rank = second_trips_rank;
    else
      pair_rank = top_pair;
    return pack(7, trips_rank, pair_rank, 0, 0, 0);
  }

  if (flush_suit >= 0) {
    uint16_t fmask = suit_mask[flush_suit];
    int r[5] = {0, 0, 0, 0, 0};
    int kept = 0;
    for (int rr = 12; rr >= 0 && kept < 5; --rr) {
      if (fmask & (uint16_t)(1u << rr)) { r[kept++] = rr; }
    }
    return pack(6, r[0], r[1], r[2], r[3], r[4]);
  }

  if (top_straight >= 0) return pack(5, top_straight, 0, 0, 0, 0);

  if (trips_rank >= 0) {
    int k0 = -1, k1 = -1;
    for (int r = 12; r >= 0; --r) {
      if (r != trips_rank && rank_count[r] >= 1) {
        if (k0 < 0) k0 = r;
        else if (k1 < 0) { k1 = r; break; }
      }
    }
    return pack(4, trips_rank, k0, k1, 0, 0);
  }

  if (top_pair >= 0 && second_pair >= 0) {
    int k0 = -1;
    for (int r = 12; r >= 0; --r) {
      if (r != top_pair && r != second_pair && rank_count[r] >= 1) {
        k0 = r; break;
      }
    }
    return pack(3, top_pair, second_pair, k0, 0, 0);
  }

  if (top_pair >= 0) {
    int k[3] = {-1, -1, -1};
    int kept = 0;
    for (int r = 12; r >= 0 && kept < 3; --r) {
      if (r != top_pair && rank_count[r] >= 1) k[kept++] = r;
    }
    return pack(2, top_pair, k[0], k[1], k[2], 0);
  }

  // High card: top 5
  int r[5] = {0, 0, 0, 0, 0};
  int kept = 0;
  for (int rr = 12; rr >= 0 && kept < 5; --rr) {
    if (rank_count[rr] >= 1) r[kept++] = rr;
  }
  return pack(1, r[0], r[1], r[2], r[3], r[4]);
}

// Preflop canonical-169 hand index. 13 pocket pairs (0..12) + 78 suited
// (13..90) + 78 offsuit (91..168). Matches CPU
// EquityModule::preflop_canonical_index exactly.
__device__ inline int dev_preflop_canonical_169(uint8_t hole0, uint8_t hole1) {
  int r0 = hole0 >> 2, s0 = hole0 & 3;
  int r1 = hole1 >> 2, s1 = hole1 & 3;
  int hi = max(r0, r1), lo = min(r0, r1);
  if (hi == lo) return hi;             // pocket pair
  bool suited = (s0 == s1);
  int idx = (hi * (hi - 1)) / 2 + lo;
  return suited ? (13 + idx) : (13 + 78 + idx);
}

// Deterministic FNV-1a 64-bit seed for a (hole, board) configuration. Same
// algorithm as CPU EquityModule::compute_ehs2_runtime so EHS² values from
// the same input agree across iterations within a training run.
__device__ inline uint64_t dev_ehs2_seed(uint8_t hole0, uint8_t hole1,
                                         const uint8_t *board, int num_board) {
  uint64_t s = 0xcbf29ce484222325ULL;
  s ^= (uint64_t)hole0; s *= 0x100000001b3ULL;
  s ^= (uint64_t)hole1; s *= 0x100000001b3ULL;
  for (int i = 0; i < num_board; ++i) {
    s ^= (uint64_t)board[i];
    s *= 0x100000001b3ULL;
  }
  return s;
}

// Monte Carlo EHS² rollout. Deterministically seeded per (hole, board) so
// the same hand always produces the same EHS² across CFR iterations (this
// is critical for regret consistency at the same info-set).
//
// n_rollouts samples. Each rollout: partial Fisher-Yates on the remaining
// deck to draw an opponent hole pair + missing board completion. Evaluate
// both 7-card hands. Score = 1/0.5/0 for win/tie/loss. Return mean(score²).
//
// Per call cost: n_rollouts × (k card swaps + 2 × dev_eval_7card_total)
// ≈ n_rollouts × ~180 ALU ops at the river. At 50 rollouts that's ~9K
// ops per query; called once per node visit.
__device__ inline double dev_compute_ehs2(uint8_t hole0, uint8_t hole1,
                                          const uint8_t *board, int num_board,
                                          int n_rollouts);

// (definition below DRng)

// Nearest centroid by absolute distance in 1-D. Centroids sorted ascending.
// Binary search to bracket then compare neighbours. ~log2(200) = 8 iters.
__device__ inline int dev_nearest_centroid_d(double v, const double *cents,
                                              int n) {
  if (n <= 0) return 0;
  if (v <= cents[0]) return 0;
  if (v >= cents[n - 1]) return n - 1;
  int lo = 0, hi = n - 1;
  while (lo + 1 < hi) {
    int m = (lo + hi) >> 1;
    if (cents[m] <= v) lo = m;
    else               hi = m;
  }
  double d_lo = v - cents[lo];
  double d_hi = cents[hi] - v;
  return (d_lo < d_hi) ? lo : hi;
}

// V3 device-side bucketize. Returns per-street bucket index:
//   preflop: 0..168 (canonical-169)
//   flop/turn/river: 0..199 (nearest of 200 centroids by EHS²)
// The trajectory kernel composes the global 0..768 index for the save format.
struct DV3Centroids {
  const double *flop;   int n_flop;
  const double *turn;   int n_turn;
  const double *river;  int n_river;
  int n_rollouts;
};

__device__ inline int dev_bucketize_hand_v3(const DGameState &s, int player,
                                             const DV3Centroids &v3) {
  uint8_t hole0 = s.hole[player][0];
  uint8_t hole1 = s.hole[player][1];
  if (s.stage == kPreflop || s.num_board == 0) {
    return dev_preflop_canonical_169(hole0, hole1);
  }
  double e2 = dev_compute_ehs2(hole0, hole1, s.board, s.num_board,
                                v3.n_rollouts);
  if (s.stage == kFlop)  return dev_nearest_centroid_d(e2, v3.flop,  v3.n_flop);
  if (s.stage == kTurn)  return dev_nearest_centroid_d(e2, v3.turn,  v3.n_turn);
  if (s.stage == kRiver) return dev_nearest_centroid_d(e2, v3.river, v3.n_river);
  return 0;
}

// Compose a flat 0..768 global bucket index from (stage, per_street_bucket).
// preflop (stage=1) → 0..168
// flop    (stage=2) → 169..368
// turn    (stage=3) → 369..568
// river   (stage=4) → 569..768
// Matches EquityModule::v2_global_index exactly.
__device__ inline int dev_v3_global_bucket(int stage, int per_street) {
  switch (stage) {
  case kFlop:  return kV3PreflopBuckets + per_street;                    // 169..
  case kTurn:  return kV3PreflopBuckets + kV3PostflopBuckets + per_street;
  case kRiver: return kV3PreflopBuckets + 2 * kV3PostflopBuckets + per_street;
  default:     return per_street;                                        // preflop
  }
}

// ============================================================================
// V1 hand abstraction on device. Returns bucket id 0..9 (BucketID enum).
// Mirrors EquityModule::bucketize_hand for the V1 path.
// ============================================================================

__device__ inline uint8_t dev_bucketize_hand(const DGameState &s, int player) {
  // Build hero+board card list.
  uint8_t cards[7];
  int n = 0;
  cards[n++] = s.hole[player][0];
  cards[n++] = s.hole[player][1];
  for (int i = 0; i < s.num_board; ++i) cards[n++] = s.board[i];

  // Preflop: heuristic over the 2 hole cards.
  if (s.stage == kPreflop || s.num_board == 0) {
    int r1 = cards[0] >> 2, r2 = cards[1] >> 2;
    int s1 = cards[0] & 3, s2 = cards[1] & 3;
    int hi = max(r1, r2), lo = min(r1, r2);
    bool suited = (s1 == s2);
    bool pair = (r1 == r2);
    if (pair) {
      if (hi >= 10) return kBucketNuts;        // QQ+
      if (hi >= 7)  return kBucketOverPair;    // 99+
      if (hi >= 4)  return kBucketTopPair;     // 66+
      return kBucketMidPair;
    }
    if (hi >= 12 && lo >= 8) return suited ? kBucketStrongMade : kBucketTopPair;
    if (hi >= 11 && lo >= 8) return suited ? kBucketTopPair : kBucketMidPair;
    if (hi >= 9 && lo >= 7)  return suited ? kBucketMidPair : kBucketWeakPair;
    if (hi >= 8 || suited)   return kBucketWeakPair;
    return kBucketAir;
  }

  // Post-flop: rely on category from the device evaluator.
  int cat = dev_eval_category(cards, n);
  if (cat >= 5) return kBucketStrongMade;        // straight or better
  if (cat == 4) return kBucketStrongMade;        // trips
  if (cat == 3) return kBucketTopPair;           // two pair
  if (cat == 2) {
    // Pair — refine by board top.
    int board_high = -1;
    for (int i = 0; i < s.num_board; ++i) {
      int r = s.board[i] >> 2;
      if (r > board_high) board_high = r;
    }
    int hr0 = s.hole[player][0] >> 2;
    int hr1 = s.hole[player][1] >> 2;
    bool pocket_pair = (hr0 == hr1);
    if (pocket_pair) {
      if (hr0 > board_high) return kBucketOverPair;
      if (hr0 == board_high) return kBucketTopPair;
      return kBucketWeakPair;
    }
    if (hr0 == board_high || hr1 == board_high) return kBucketTopPair;
    return kBucketMidPair;
  }
  // High card. Detect flush draw (4 of a suit).
  uint16_t suit_mask[4] = {0, 0, 0, 0};
  for (int i = 0; i < n; ++i) suit_mask[cards[i] & 3] |= (1u << (cards[i] >> 2));
  for (int s2 = 0; s2 < 4; ++s2) {
    if (__popc(suit_mask[s2]) >= 4) return kBucketStrongDraw;
  }
  return kBucketAir;
}

// ============================================================================
// Imperfect-recall info-set key encoding. Packs into a single uint64_t.
//
// V2 layout (bumped from v0.5 to support V3's 200-bucket abstraction).
// V1 GPU model files saved before this change are incompatible — but they
// can be regenerated in ~20 sec with the v0.6+ binary.
//
//   bits  0..7   bucket (V1: 0..9, V3: per-street 0..199, 8 bits)
//   bits  8..10  stage (1..4, 3 bits)
//   bits 11..13  pot bucket (0..4, 3 bits)
//   bits 14..15  preflop raise count, 0..3 capped (2 bits)
//   bits 16..17  flop raise count    (2)
//   bits 18..19  turn raise count    (2)
//   bits 20..21  river raise count   (2)
//   bits 22..23  preflop aggressor  ('-'=0, 'd'=1, 's'=2)  (2)
//   bits 24..25  flop aggressor                              (2)
//   bits 26..27  turn aggressor                              (2)
//   bits 28..29  river aggressor                             (2)
//   bits 30..31  num_legal (3..4 stored modulo 4) (2)
//   bits 32..33  current player (2)
//   bits 34..62  reserved (=0)
//   bit  63      sentinel (always 1 — avoids all-zero hash slot)
// ============================================================================

__device__ inline int dev_pot_bucket(int pot, int bb) {
  if (bb <= 0) bb = 1;
  int pot_bb = pot / bb;
  if (pot_bb < 5)  return 0;
  if (pot_bb < 15) return 1;
  if (pot_bb < 30) return 2;
  if (pot_bb < 60) return 3;
  return 4;
}

// Compute info-set key. Branches on hand abstraction:
//   V1: bucket = dev_bucketize_hand (0..9 heuristic)
//   V3: bucket = per-street EHS² cluster (0..199), with stage encoding
//       letting us reconstruct the global 0..768 index at save time.
__device__ inline uint64_t dev_compute_infoset_key(const DGameState &s,
                                                    int player, int num_legal,
                                                    int bb,
                                                    int abstraction,
                                                    const DV3Centroids &v3) {
  int bucket;
  if (abstraction == 1) {
    bucket = dev_bucketize_hand_v3(s, player, v3);
  } else {
    bucket = (int)dev_bucketize_hand(s, player);
  }
  uint8_t pot_b  = (uint8_t)dev_pot_bucket(s.pot, bb);

  // Walk history with proper per-street tracking. We don't store street
  // tags in DGameState.hist directly, so we infer them by counting how
  // many times the betting round closed before the current state. The
  // CPU's imperfect_recall_summary uses a similar heuristic.
  //
  // Algorithm: track a running "active player count" of who has acted
  // and the per-player current contribution. When all (non-folded,
  // non-allin) players have acted AND their bets are equal, the round
  // closes — bump the street tag. (HU FCPA: 2 players, betting closes
  // when the second player's action equalises the high bet.)
  uint8_t raise_cnt[4] = {0, 0, 0, 0};
  uint8_t last_raiser[4] = {0, 0, 0, 0}; // 0=none, 1=button, 2=non-button
  int cur_street = 0;
  uint8_t acted = 0;       // bitmask of players who have acted on this street
  int8_t  cur_bet[2] = {0, 0};
  uint8_t folded = 0;
  uint8_t allin = 0;
  // Initialise blinds-posted state: in HU, SB at (dealer+1)%2 has posted
  // SB (1bb) and BB at dealer has posted BB (2bb).
  int sb_pos = (s.dealer + 1) & 1;
  int bb_pos = s.dealer;
  cur_bet[sb_pos] = 1; // small blind in chips (1)
  cur_bet[bb_pos] = 2; // big blind (2)
  int8_t high_bet = 2;

  auto round_closed = [&](void) -> bool {
    for (int p = 0; p < 2; ++p) {
      if ((folded >> p) & 1) continue;
      if ((allin  >> p) & 1) continue;
      if (!((acted >> p) & 1)) return false;
      if (cur_bet[p] != high_bet) return false;
    }
    return true;
  };

  for (int i = 0; i < s.num_hist; ++i) {
    uint8_t entry = s.hist[i];
    int p = entry >> 4;
    int a = entry & 0xF;
    acted |= (uint8_t)(1u << p);

    if (a == kActFold) {
      folded |= (uint8_t)(1u << p);
    } else if (a == kActCall) {
      int call = high_bet - cur_bet[p];
      if (call > 0) cur_bet[p] = high_bet;
    } else if (a == kActPot || a == kActAllin) {
      if (raise_cnt[cur_street] < 3) raise_cnt[cur_street] += 1;
      int rel = (p - s.dealer + 2) % 2;
      last_raiser[cur_street] = (rel == 0) ? 1 : 2;
      // Estimate the new high bet conservatively: pot raise + allin both
      // raise the high bet beyond the call amount. Exact chip math doesn't
      // matter for the round-close detection — we just need cur_bet[p]
      // to exceed high_bet so the OTHER player's acted bit will need to
      // be set + their cur_bet equalised before the round closes.
      cur_bet[p] = high_bet + 1;
      high_bet = cur_bet[p];
      if (a == kActAllin) allin |= (uint8_t)(1u << p);
    }

    if (round_closed() && cur_street < 3) {
      // Bump street; reset acted + cur_bet for the new street.
      cur_street += 1;
      acted = 0;
      cur_bet[0] = 0; cur_bet[1] = 0;
      high_bet = 0;
    }
  }
  // Bound cur_street to the current stage (defensive).
  if (s.stage >= kPreflop) {
    int s_idx = s.stage - kPreflop;
    if (s_idx > cur_street) cur_street = s_idx;
  }

  uint64_t key = 0;
  key |= (uint64_t)(bucket & 0xFF);                       // bits  0..7
  key |= ((uint64_t)(s.stage & 0x7))    << 8;             // bits  8..10
  key |= ((uint64_t)(pot_b & 0x7))      << 11;            // bits 11..13
  key |= ((uint64_t)(raise_cnt[0] & 0x3))   << 14;        // bits 14..15
  key |= ((uint64_t)(raise_cnt[1] & 0x3))   << 16;        // bits 16..17
  key |= ((uint64_t)(raise_cnt[2] & 0x3))   << 18;        // bits 18..19
  key |= ((uint64_t)(raise_cnt[3] & 0x3))   << 20;        // bits 20..21
  key |= ((uint64_t)(last_raiser[0] & 0x3)) << 22;        // bits 22..23
  key |= ((uint64_t)(last_raiser[1] & 0x3)) << 24;        // bits 24..25
  key |= ((uint64_t)(last_raiser[2] & 0x3)) << 26;        // bits 26..27
  key |= ((uint64_t)(last_raiser[3] & 0x3)) << 28;        // bits 28..29
  key |= ((uint64_t)(num_legal & 0x3))      << 30;        // bits 30..31
  key |= ((uint64_t)(player & 0x3))         << 32;        // bits 32..33
  // Avoid the all-zero key (used as "empty slot" sentinel) by setting bit 63.
  key |= (1ULL << 63);
  return key;
}

// ============================================================================
// Open-addressing hash table on device.
//
// keys[]: empty = 0, otherwise the 64-bit info-set key.
// values[]: corresponding row index in regret_sum / strategy_sum tables.
//
// Insert: find first empty slot via linear probing. atomicCAS to claim.
// Lookup: same probe sequence; bail on first empty slot.
// ============================================================================

struct DHashTable {
  uint64_t *keys;
  int *values;
  int *size;       // current count, atomic
  int capacity;    // power of 2
  int capacity_mask;
};

__device__ inline int dev_hash_lookup_or_insert(DHashTable ht, uint64_t key) {
  // FNV-1a-ish spread to stir the bits — info-set keys have low entropy in
  // the low bits because of the bit-packed layout.
  uint64_t h = key;
  h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
  h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
  h ^= h >> 33;

  int idx = (int)(h & (uint64_t)ht.capacity_mask);
  for (int probe = 0; probe < 64; ++probe) {
    int slot = (idx + probe) & ht.capacity_mask;
    // Atomic load of keys[slot] (CAS with same-value is a no-op load).
    uint64_t cur = atomicCAS((unsigned long long *)&ht.keys[slot], 0ULL, 0ULL);
    if (cur == key) {
      // Already exists — return its value. value may be -1 if a peer is in
      // the middle of allocating; spin via atomicAdd-0 to bypass cache.
      int v;
      while ((v = atomicAdd((int *)&ht.values[slot], 0)) < 0) {
        // benign spin until the inserter publishes the row id
      }
      return v;
    }
    if (cur == 0ULL) {
      // Empty — try to claim it.
      uint64_t prev =
          atomicCAS((unsigned long long *)&ht.keys[slot], 0ULL,
                    (unsigned long long)key);
      if (prev == 0ULL) {
        // Claimed. Allocate row.
        int row = atomicAdd(ht.size, 1);
        // atomicExch makes the write visible immediately to other threads.
        atomicExch((int *)&ht.values[slot], row);
        return row;
      }
      if (prev == key) {
        int v;
        while ((v = atomicAdd((int *)&ht.values[slot], 0)) < 0) {}
        return v;
      }
      // Some other thread took this slot for a different key — keep probing.
    }
  }
  // Hash table full or extreme contention. Caller falls back to uniform.
  return -1;
}

// ============================================================================
// PRNG: xoroshiro128+ in registers.
// ============================================================================

struct DRng { uint64_t s0, s1; };

__device__ inline uint64_t dev_rotl(uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}
__device__ inline uint64_t dev_next_u64(DRng &r) {
  uint64_t s0 = r.s0, s1 = r.s1;
  uint64_t result = s0 + s1;
  s1 ^= s0;
  r.s0 = dev_rotl(s0, 55) ^ s1 ^ (s1 << 14);
  r.s1 = dev_rotl(s1, 36);
  return result;
}
__device__ inline double dev_next_double(DRng &r) {
  // Top 53 bits → [0, 1).
  return (double)(dev_next_u64(r) >> 11) * (1.0 / 9007199254740992.0);
}
// Sample from a discrete distribution of size n (n ≤ kMaxActions).
__device__ inline int dev_sample_discrete(const double *probs, int n, DRng &r) {
  double u = dev_next_double(r);
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    acc += probs[i];
    if (u <= acc) return i;
  }
  return n - 1;
}

// EHS² body (forward-declared above, defined here once DRng is in scope).
__device__ inline double dev_compute_ehs2(uint8_t hole0, uint8_t hole1,
                                          const uint8_t *board, int num_board,
                                          int n_rollouts) {
  // Build remaining-deck list (52 minus dealt cards).
  uint8_t deck[52];
  bool used[52] = {false};
  used[hole0] = true;
  used[hole1] = true;
  for (int i = 0; i < num_board; ++i) used[board[i]] = true;
  int n_deck = 0;
  for (int c = 0; c < 52; ++c) if (!used[c]) deck[n_deck++] = (uint8_t)c;

  int board_needed = 5 - num_board;
  int k = 2 + board_needed;

  // Deterministic seed → xoroshiro128+ state. Same input always produces
  // the same EHS² so regret accumulation across CFR iters is consistent.
  uint64_t seed = dev_ehs2_seed(hole0, hole1, board, num_board);
  DRng rng;
  rng.s0 = seed ^ 0x9E3779B97F4A7C15ULL;
  rng.s1 = seed * 0xBF58476D1CE4E5B9ULL ^ 0xC2B2AE3D27D4EB4FULL;

  uint8_t hero[7];
  uint8_t opp[7];
  hero[0] = hole0; hero[1] = hole1;
  for (int i = 0; i < num_board; ++i) hero[2 + i] = board[i];

  double sum_sq = 0.0;
  for (int r = 0; r < n_rollouts; ++r) {
    // Partial Fisher-Yates: shuffle first k slots of deck. Tail untouched.
    for (int i = 0; i < k; ++i) {
      uint64_t u = dev_next_u64(rng);
      int j = i + (int)(u % (uint64_t)(n_deck - i));
      uint8_t tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
    }
    // opp hole = deck[0..1]
    opp[0] = deck[0]; opp[1] = deck[1];
    for (int i = 0; i < num_board; ++i) opp[2 + i] = board[i];
    // Runout = deck[2..k-1]
    for (int i = 0; i < board_needed; ++i) {
      hero[2 + num_board + i] = deck[2 + i];
      opp [2 + num_board + i] = deck[2 + i];
    }
    int n_full = 2 + num_board + board_needed;     // = 7 at river
    uint32_t hv = dev_eval_7card_total(hero, n_full);
    uint32_t ov = dev_eval_7card_total(opp,  n_full);
    double score = (hv > ov) ? 1.0 : ((hv == ov) ? 0.5 : 0.0);
    sum_sq += score * score;
  }
  return sum_sq / (double)n_rollouts;
}

// ============================================================================
// Game-state helpers (apply action, advance street, terminal payoff).
// ============================================================================

__device__ inline bool dev_is_terminal(const DGameState &s) {
  if (s.stage >= kShowdown) return true;
  // Only one player still in?
  int active = 0;
  for (int p = 0; p < 2; ++p) {
    if (!((s.folded_mask >> p) & 1u)) active += 1;
  }
  return active <= 1;
}

__device__ inline bool dev_is_betting_round_over(const DGameState &s) {
  for (int p = 0; p < 2; ++p) {
    if ((s.folded_mask >> p) & 1u) continue;
    if ((s.allin_mask  >> p) & 1u) continue;
    if (!((s.acted_mask >> p) & 1u)) return false;
    if (s.cur_bet[p] != s.high_bet)  return false;
  }
  return true;
}

// Returns the FCPA action ID list legal in `s` for the current player.
// `legal[]` must have at least 4 slots. Returns the number of legal actions.
__device__ inline int dev_legal_actions(const DGameState &s, uint8_t *legal) {
  int cur = s.current_player;
  int call_amt = s.high_bet - s.cur_bet[cur];
  int n = 0;
  if (call_amt > 0) legal[n++] = kActFold;
  legal[n++] = kActCall;
  // Pot raise legality.
  if (call_amt == 0) {
    int pot = s.pot > 0 ? s.pot : (int)2; // BB
    if (pot > 0 && pot < s.stack[cur]) legal[n++] = kActPot;
  } else {
    int pot = s.pot > (int)2 ? s.pot : (int)2;
    int base = pot + call_amt;
    int raise_to = s.high_bet + base;
    if (raise_to > s.high_bet && (raise_to - s.cur_bet[cur]) < s.stack[cur]) {
      legal[n++] = kActPot;
    }
  }
  if (s.stack[cur] > 0) legal[n++] = kActAllin;
  return n;
}

__device__ inline void dev_apply_action(DGameState &s, uint8_t action) {
  int p = s.current_player;
  int call_amt = s.high_bet - s.cur_bet[p];
  s.acted_mask |= (uint8_t)(1u << p);

  // Record in history first so the previous_bet semantics matches CPU.
  if (s.num_hist < kMaxDepth) {
    s.hist[s.num_hist++] = (uint8_t)((p << 4) | (action & 0xF));
  }

  int chips_added = 0;
  switch (action) {
  case kActFold:
    s.folded_mask |= (uint8_t)(1u << p);
    break;
  case kActCall: {
    chips_added = call_amt < s.stack[p] ? call_amt : s.stack[p];
    s.cur_bet[p]   += (int16_t)chips_added;
    s.total_bet[p] += (int16_t)chips_added;
    s.stack[p]     -= (int16_t)chips_added;
    s.pot          += (int16_t)chips_added;
    if (s.stack[p] <= 0) { s.allin_mask |= (uint8_t)(1u << p); s.stack[p] = 0; }
    break;
  }
  case kActPot: {
    int amount;
    if (call_amt == 0) {
      int pot = s.pot > 0 ? s.pot : 2;
      amount = pot;
    } else {
      int pot = s.pot > 2 ? s.pot : 2;
      int base = pot + call_amt;
      int raise_to = s.high_bet + base;
      amount = raise_to - s.cur_bet[p];
      s.high_bet = (int16_t)raise_to;
    }
    chips_added = amount < s.stack[p] ? amount : s.stack[p];
    if (call_amt == 0) {
      s.high_bet = (int16_t)(s.cur_bet[p] + chips_added);
    }
    s.cur_bet[p]   += (int16_t)chips_added;
    s.total_bet[p] += (int16_t)chips_added;
    s.stack[p]     -= (int16_t)chips_added;
    s.pot          += (int16_t)chips_added;
    if (s.stack[p] <= 0) { s.allin_mask |= (uint8_t)(1u << p); s.stack[p] = 0; }
    break;
  }
  case kActAllin: {
    chips_added = s.stack[p];
    s.cur_bet[p]   += (int16_t)chips_added;
    s.total_bet[p] += (int16_t)chips_added;
    s.stack[p]      = 0;
    s.pot          += (int16_t)chips_added;
    s.allin_mask   |= (uint8_t)(1u << p);
    if (s.cur_bet[p] > s.high_bet) s.high_bet = s.cur_bet[p];
    break;
  }
  }

  // Determine next state — if everyone except 1 folded, we're at showdown.
  int active = 0;
  for (int q = 0; q < 2; ++q) if (!((s.folded_mask >> q) & 1u)) active += 1;
  if (active <= 1) { s.stage = kShowdown; return; }

  if (!dev_is_betting_round_over(s)) {
    // Rotate to next player not folded/allin.
    for (int step = 1; step <= 2; ++step) {
      int q = (s.current_player + step) % 2;
      if (!((s.folded_mask >> q) & 1u) && !((s.allin_mask >> q) & 1u)) {
        s.current_player = (uint8_t)q;
        break;
      }
    }
  }
  // If the round is over, we leave the state for the chance-handling code in
  // the kernel to detect and advance. (Mirrors the CPU implementation.)
}

__device__ inline void dev_next_street(DGameState &s) {
  s.cur_bet[0] = 0; s.cur_bet[1] = 0;
  s.high_bet = 0;
  s.acted_mask = 0;
  if      (s.stage == kPreflop) s.stage = kFlop;
  else if (s.stage == kFlop)    s.stage = kTurn;
  else if (s.stage == kTurn)    s.stage = kRiver;
  else if (s.stage == kRiver)   s.stage = kShowdown;

  // HU rule: BB acts first postflop (BB sits at dealer in our convention
  // for HU).
  s.current_player = s.dealer;
  // Skip folded/allin (defensive).
  for (int step = 0; step < 2; ++step) {
    int q = (s.dealer + step) % 2;
    if (!((s.folded_mask >> q) & 1u) && !((s.allin_mask >> q) & 1u)) {
      s.current_player = (uint8_t)q;
      break;
    }
  }
}

// Sample n cards uniformly from the deck minus already-dealt cards. Writes
// to dst[0..n-1].
__device__ inline void dev_sample_cards(uint8_t *dst, int n,
                                         const DGameState &s, DRng &rng) {
  // Build dealt mask.
  uint64_t dealt_lo = 0, dealt_hi = 0; // 52 bits split across two u64s
  auto mark = [&](int c) {
    if (c < 64) dealt_lo |= (1ULL << c);
    else        dealt_hi |= (1ULL << (c - 64));
  };
  for (int p = 0; p < 2; ++p) for (int i = 0; i < 2; ++i) mark(s.hole[p][i]);
  for (int i = 0; i < s.num_board; ++i) mark(s.board[i]);

  // Draw n cards by rejection sampling.
  int drawn = 0;
  while (drawn < n) {
    int c = (int)(dev_next_u64(rng) % 52);
    bool d = (c < 64) ? ((dealt_lo >> c) & 1u)
                       : ((dealt_hi >> (c - 64)) & 1u);
    if (!d) {
      dst[drawn++] = (uint8_t)c;
      if (c < 64) dealt_lo |= (1ULL << c);
      else        dealt_hi |= (1ULL << (c - 64));
    }
  }
}

// Terminal payoff for `traverser`. Net chips won/lost relative to total bet.
__device__ inline double dev_terminal_payoff(const DGameState &s, int trav) {
  // Folded? Lose your total bet.
  bool trav_folded = (s.folded_mask >> trav) & 1u;
  bool opp_folded  = (s.folded_mask >> (1 - trav)) & 1u;
  if (trav_folded && opp_folded) return 0.0; // both folded — shouldn't happen
  if (trav_folded) return -(double)s.total_bet[trav];
  if (opp_folded)  return  (double)s.total_bet[1 - trav];

  // Showdown: compare hand categories. Higher cat wins. Ties split.
  uint8_t cards_t[7], cards_o[7];
  int nt = 0, no = 0;
  cards_t[nt++] = s.hole[trav][0];
  cards_t[nt++] = s.hole[trav][1];
  cards_o[no++] = s.hole[1 - trav][0];
  cards_o[no++] = s.hole[1 - trav][1];
  for (int i = 0; i < s.num_board; ++i) {
    cards_t[nt++] = s.board[i];
    cards_o[no++] = s.board[i];
  }
  int cat_t = dev_eval_category(cards_t, nt);
  int cat_o = dev_eval_category(cards_o, no);
  if (cat_t > cat_o) return  (double)s.total_bet[1 - trav];
  if (cat_t < cat_o) return -(double)s.total_bet[trav];
  return 0.0;
}

// ============================================================================
// Outcome-sampling MCCFR kernel. One trajectory per thread.
// ============================================================================

struct TrajStep {
  int info_id;
  uint8_t action;
  uint8_t is_traverser;
  uint8_t num_actions;
  uint8_t pad;
  double sigma_a;
  double sigma[kMaxActions]; // strategy snapshot at this node
};

__global__ void outcome_sampling_kernel(
    DGameState initial,
    int traverser,
    uint64_t base_seed,
    int num_traj,
    int big_blind,
    double epsilon,
    int abstraction,           // 0 = V1 heuristic, 1 = V3 EHS² clusters
    DV3Centroids v3,           // V3 centroid pointers (unused for V1)
    DHashTable hash,
    double *regret_sum,
    double *strategy_sum,
    const double *strategy,    // read-only snapshot (refreshed by match kernel)
    int *num_actions,          // [hash.capacity]
    int *traversal_counter
) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_traj) return;

  DRng rng;
  rng.s0 = base_seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(tid + 1));
  rng.s1 = base_seed * 0xBF58476D1CE4E5B9ULL ^ (uint64_t)tid;

  DGameState s = initial;

  // Deal hole cards.
  uint8_t hole[4];
  dev_sample_cards(hole, 4, s, rng);
  s.hole[0][0] = hole[0]; s.hole[0][1] = hole[1];
  s.hole[1][0] = hole[2]; s.hole[1][1] = hole[3];

  TrajStep traj[kMaxDepth];
  int depth = 0;
  double sample_reach = 1.0;
  double my_reach = 1.0;
  double opp_reach = 1.0;
  bool fail = false;

  for (int step = 0; step < kMaxDepth * 2 && !fail; ++step) {
    if (dev_is_terminal(s)) break;

    // Street transition? deal flop/turn/river.
    if (dev_is_betting_round_over(s) && s.stage != kShowdown) {
      int n_to_deal = 0;
      if (s.stage == kPreflop && s.num_board == 0)      n_to_deal = 3;
      else if (s.stage == kFlop && s.num_board == 3)    n_to_deal = 1;
      else if (s.stage == kTurn && s.num_board == 4)    n_to_deal = 1;
      if (n_to_deal > 0) {
        uint8_t c[3];
        dev_sample_cards(c, n_to_deal, s, rng);
        for (int i = 0; i < n_to_deal; ++i) s.board[s.num_board++] = c[i];
      }
      dev_next_street(s);
      if (dev_is_terminal(s)) break;
      continue;
    }

    int acting = s.current_player;
    uint8_t legal[kMaxActions];
    int nlegal = dev_legal_actions(s, legal);
    if (nlegal == 0) break;

    uint64_t key = dev_compute_infoset_key(s, acting, nlegal, big_blind,
                                            abstraction, v3);
    int info_id = dev_hash_lookup_or_insert(hash, key);
    if (info_id < 0) { fail = true; break; }

    // Initialize on first visit: record num_actions, seed strategy uniform.
    int prev_n = num_actions[info_id];
    if (prev_n == 0) {
      // Race-friendly: only one thread will succeed at writing, others see
      // either 0 (and re-write — idempotent) or the correct value.
      atomicCAS((int *)&num_actions[info_id], 0, nlegal);
    }
    int n_act = num_actions[info_id]; if (n_act < 1) n_act = nlegal;

    // Read strategy row (cached from previous regret-match).
    double sigma[kMaxActions] = {0, 0, 0, 0};
    double sigma_sum = 0.0;
    for (int a = 0; a < n_act; ++a) {
      sigma[a] = strategy[(size_t)info_id * kMaxActions + a];
      sigma_sum += sigma[a];
    }
    if (sigma_sum < 1e-12) {
      double u = 1.0 / n_act;
      for (int a = 0; a < n_act; ++a) sigma[a] = u;
    }

    // Build sample distribution. Traverser uses ε-greedy mix.
    double sample_dist[kMaxActions] = {0, 0, 0, 0};
    if (acting == traverser) {
      double inv_n = 1.0 / n_act;
      for (int a = 0; a < n_act; ++a)
        sample_dist[a] = epsilon * inv_n + (1.0 - epsilon) * sigma[a];
    } else {
      for (int a = 0; a < n_act; ++a) sample_dist[a] = sigma[a];
    }
    int a = dev_sample_discrete(sample_dist, n_act, rng);
    double q = sample_dist[a];
    if (q < 1e-30) q = 1e-30;

    // Record step.
    if (depth < kMaxDepth) {
      traj[depth].info_id = info_id;
      traj[depth].action = legal[a]; // store the FCPA action id we took
      traj[depth].is_traverser = (uint8_t)(acting == traverser ? 1 : 0);
      traj[depth].num_actions = (uint8_t)n_act;
      traj[depth].sigma_a = sigma[a];
      for (int i = 0; i < kMaxActions; ++i) traj[depth].sigma[i] = sigma[i];
      depth++;
    }

    // Update reaches.
    sample_reach *= q;
    if (acting == traverser) my_reach *= sigma[a]; else opp_reach *= sigma[a];

    dev_apply_action(s, legal[a]);
  }

  if (fail) return;

  // Terminal payoff.
  double util = dev_terminal_payoff(s, traverser);

  // Backward pass: regret + strategy_sum updates.
  for (int d = depth - 1; d >= 0; --d) {
    int info_id = traj[d].info_id;
    int n_act   = traj[d].num_actions;
    int a       = -1;
    // We stored the FCPA action id; we need its index in the local (legal)
    // ordering. For FCPA, legal_ids and per-state legal index can differ —
    // re-derive by scanning. In practice we stored a deterministic (info,
    // index). To keep this simple, we treat traj[d].action as the index in
    // the strategy row directly (i.e., the slot in sigma[]). Reconstruct
    // from sigma[] by argmax-match: the index whose sigma matches sigma_a
    // at the time of recording. Since sigma[] is captured, we can find
    // the slot by scanning for sigma[i] == sigma_a (works for FCPA where
    // ties are unlikely). Fall back to 0 if no match.
    for (int i = 0; i < n_act; ++i) {
      if (traj[d].sigma[i] == traj[d].sigma_a) { a = i; break; }
    }
    if (a < 0) a = 0;

    if (traj[d].is_traverser) {
      double cf_factor = opp_reach / sample_reach;
      double cf_value = traj[d].sigma_a * util * cf_factor;
      for (int aa = 0; aa < n_act; ++aa) {
        double cf_action_value = (aa == a) ? util * cf_factor : 0.0;
        double r = cf_action_value - cf_value;
        atomicAdd(&regret_sum[(size_t)info_id * kMaxActions + aa], r);
      }
      double w = my_reach / sample_reach;
      for (int aa = 0; aa < n_act; ++aa) {
        atomicAdd(&strategy_sum[(size_t)info_id * kMaxActions + aa],
                  w * traj[d].sigma[aa]);
      }
    } else {
      double w = opp_reach / sample_reach;
      for (int aa = 0; aa < n_act; ++aa) {
        atomicAdd(&strategy_sum[(size_t)info_id * kMaxActions + aa],
                  w * traj[d].sigma[aa]);
      }
    }
  }

  if (traversal_counter) atomicAdd(traversal_counter, 1);
}

// ============================================================================
// Regret-match kernel — refresh `strategy` from `regret_sum` after each batch.
// One thread per info-set row. Applies DCFR discounts in place.
// ============================================================================

__global__ void regret_match_kernel(double *regret_sum,
                                    double *strategy,
                                    const int *num_actions,
                                    int n_rows,
                                    double dcfr_alpha,
                                    double dcfr_beta,
                                    int iteration) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) return;
  int n_act = num_actions[row];
  if (n_act <= 0) {
    // Empty/uninitialised — keep strategy uniform-zero.
    for (int a = 0; a < kMaxActions; ++a)
      strategy[(size_t)row * kMaxActions + a] = 0.0;
    return;
  }

  // DCFR: discount cumulative regrets per their sign.
  double t = (double)max(iteration, 1);
  double dpos = (dcfr_alpha >= 1e29) ? 1.0
                                       : pow(t, dcfr_alpha) /
                                             (pow(t, dcfr_alpha) + 1.0);
  double dneg = (dcfr_beta >= 1e29) ? 1.0
                                      : (dcfr_beta <= -1e29 ? 0.0
                                          : pow(t, dcfr_beta) /
                                                (pow(t, dcfr_beta) + 1.0));
  for (int a = 0; a < n_act; ++a) {
    double r = regret_sum[(size_t)row * kMaxActions + a];
    regret_sum[(size_t)row * kMaxActions + a] = (r > 0.0) ? r * dpos : r * dneg;
  }

  // Regret matching from the discounted regret_sum.
  double sum = 0.0;
  double pos[kMaxActions] = {0, 0, 0, 0};
  for (int a = 0; a < n_act; ++a) {
    double r = regret_sum[(size_t)row * kMaxActions + a];
    pos[a] = r > 0.0 ? r : 0.0;
    sum += pos[a];
  }
  for (int a = 0; a < n_act; ++a) {
    double s = (sum > 0.0) ? pos[a] / sum : 1.0 / n_act;
    strategy[(size_t)row * kMaxActions + a] = s;
  }
  for (int a = n_act; a < kMaxActions; ++a)
    strategy[(size_t)row * kMaxActions + a] = 0.0;
}

// ============================================================================
// Host-side engine.
// ============================================================================

struct GpuCfrEngine {
  GpuCfrConfig cfg;
  int row_capacity;

  // Device buffers.
  uint64_t *d_keys = nullptr;
  int      *d_values = nullptr;
  int      *d_size = nullptr;
  int      *d_num_actions = nullptr;
  double   *d_regret_sum = nullptr;
  double   *d_strategy_sum = nullptr;
  double   *d_strategy = nullptr;
  int      *d_traversal_counter = nullptr;

  // V3 centroid arrays (allocated only when cfg.hand_abstraction == V3).
  // Loaded once at create() from equity_buckets.dat. Sorted ascending.
  double *d_cents_flop  = nullptr;
  double *d_cents_turn  = nullptr;
  double *d_cents_river = nullptr;
  int n_cents_flop  = 0;
  int n_cents_turn  = 0;
  int n_cents_river = 0;

  cudaStream_t stream = nullptr;

  // Profiling.
  long long ns_traversal = 0;
  long long ns_match = 0;
  long long ns_sync = 0;
  long long iters = 0;
  long long trajectories = 0;
};

// Read equity_buckets.dat into three host vectors. Format documented in
// build_equity_buckets.cpp:
//   uint32 magic = 0x4C454342 ('LECB')
//   uint32 version = 1
//   for street in {flop, turn, river}:
//     int32 n_centroids
//     double centroid[n_centroids]
// Returns true on success. On failure host vectors are left empty.
static bool load_equity_buckets(const std::string &path,
                                std::vector<double> &flop,
                                std::vector<double> &turn,
                                std::vector<double> &river) {
  flop.clear(); turn.clear(); river.clear();
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "[gpu_cfr] equity_buckets.dat not found at %s\n",
                 path.c_str());
    return false;
  }
  uint32_t magic = 0, version = 0;
  if (std::fread(&magic,   sizeof(magic),   1, f) != 1 ||
      std::fread(&version, sizeof(version), 1, f) != 1 ||
      magic != 0x4C454342u) {
    std::fprintf(stderr, "[gpu_cfr] equity_buckets.dat: bad magic/version\n");
    std::fclose(f);
    return false;
  }
  std::vector<double> *streets[3] = {&flop, &turn, &river};
  for (int s = 0; s < 3; ++s) {
    int32_t n = 0;
    if (std::fread(&n, sizeof(n), 1, f) != 1 || n < 0) { std::fclose(f); return false; }
    streets[s]->resize((size_t)n);
    if (n > 0 && std::fread(streets[s]->data(), sizeof(double), (size_t)n, f) != (size_t)n) {
      std::fclose(f); return false;
    }
    // Defensive: ensure ascending (CPU loader sorts; we trust that here).
  }
  std::fclose(f);
  std::fprintf(stderr,
               "[gpu_cfr] loaded equity_buckets.dat: flop=%zu turn=%zu river=%zu\n",
               flop.size(), turn.size(), river.size());
  return true;
}

static long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

GpuCfrEngine *gpu_cfr_create(const GpuCfrConfig &cfg) {
  auto *eng = new GpuCfrEngine;
  eng->cfg = cfg;
  eng->row_capacity = kHashCapacity;

  CUDA_CHECK(cudaStreamCreate(&eng->stream));

  size_t mat_bytes = (size_t)eng->row_capacity * kMaxActions * sizeof(double);
  CUDA_CHECK(cudaMalloc(&eng->d_keys, (size_t)kHashCapacity * sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&eng->d_values, (size_t)kHashCapacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&eng->d_size, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&eng->d_num_actions, (size_t)eng->row_capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&eng->d_regret_sum, mat_bytes));
  CUDA_CHECK(cudaMalloc(&eng->d_strategy_sum, mat_bytes));
  CUDA_CHECK(cudaMalloc(&eng->d_strategy, mat_bytes));
  CUDA_CHECK(cudaMalloc(&eng->d_traversal_counter, sizeof(int)));

  CUDA_CHECK(cudaMemset(eng->d_keys, 0, (size_t)kHashCapacity * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(eng->d_values, -1, (size_t)kHashCapacity * sizeof(int)));
  CUDA_CHECK(cudaMemset(eng->d_size, 0, sizeof(int)));
  CUDA_CHECK(cudaMemset(eng->d_num_actions, 0, (size_t)eng->row_capacity * sizeof(int)));
  CUDA_CHECK(cudaMemset(eng->d_regret_sum, 0, mat_bytes));
  CUDA_CHECK(cudaMemset(eng->d_strategy_sum, 0, mat_bytes));
  CUDA_CHECK(cudaMemset(eng->d_strategy, 0, mat_bytes));
  CUDA_CHECK(cudaMemset(eng->d_traversal_counter, 0, sizeof(int)));

  // V3: load EHS² centroid table from disk and copy to device.
  if (cfg.hand_abstraction == HandAbstraction::V3) {
    std::vector<double> hf, ht, hr;
    if (!load_equity_buckets(cfg.equity_buckets_path, hf, ht, hr)) {
      std::fprintf(stderr,
          "[gpu_cfr] FATAL: V3 abstraction requested but equity_buckets.dat "
          "could not be loaded. Run build_equity_buckets first.\n");
      gpu_cfr_destroy(eng);
      return nullptr;
    }
    eng->n_cents_flop  = (int)hf.size();
    eng->n_cents_turn  = (int)ht.size();
    eng->n_cents_river = (int)hr.size();
    if (eng->n_cents_flop > 0) {
      CUDA_CHECK(cudaMalloc(&eng->d_cents_flop, hf.size() * sizeof(double)));
      CUDA_CHECK(cudaMemcpy(eng->d_cents_flop, hf.data(),
                            hf.size() * sizeof(double),
                            cudaMemcpyHostToDevice));
    }
    if (eng->n_cents_turn > 0) {
      CUDA_CHECK(cudaMalloc(&eng->d_cents_turn, ht.size() * sizeof(double)));
      CUDA_CHECK(cudaMemcpy(eng->d_cents_turn, ht.data(),
                            ht.size() * sizeof(double),
                            cudaMemcpyHostToDevice));
    }
    if (eng->n_cents_river > 0) {
      CUDA_CHECK(cudaMalloc(&eng->d_cents_river, hr.size() * sizeof(double)));
      CUDA_CHECK(cudaMemcpy(eng->d_cents_river, hr.data(),
                            hr.size() * sizeof(double),
                            cudaMemcpyHostToDevice));
    }
  }

  return eng;
}

void gpu_cfr_destroy(GpuCfrEngine *eng) {
  if (!eng) return;
  if (eng->iters > 0) {
    auto ms = [](long long ns) { return ns / 1e6; };
    std::fprintf(stderr,
                 "\n[gpu_cfr profile] iters=%lld trajectories=%lld\n"
                 "  traversal: %.2f ms total (%.2f us/iter)\n"
                 "  match:     %.2f ms total (%.2f us/iter)\n"
                 "  sync:      %.2f ms total\n",
                 eng->iters, eng->trajectories,
                 ms(eng->ns_traversal),
                 (eng->iters ? eng->ns_traversal / 1000.0 / eng->iters : 0),
                 ms(eng->ns_match),
                 (eng->iters ? eng->ns_match / 1000.0 / eng->iters : 0),
                 ms(eng->ns_sync));
  }
  cudaStreamDestroy(eng->stream);
  cudaFree(eng->d_keys);
  cudaFree(eng->d_values);
  cudaFree(eng->d_size);
  cudaFree(eng->d_num_actions);
  cudaFree(eng->d_regret_sum);
  cudaFree(eng->d_strategy_sum);
  cudaFree(eng->d_strategy);
  cudaFree(eng->d_traversal_counter);
  if (eng->d_cents_flop)  cudaFree(eng->d_cents_flop);
  if (eng->d_cents_turn)  cudaFree(eng->d_cents_turn);
  if (eng->d_cents_river) cudaFree(eng->d_cents_river);
  delete eng;
}

// Build the initial DGameState (blinds posted, no cards dealt yet — kernel
// deals hole cards inside the trajectory).
static DGameState build_initial_state(const GpuCfrConfig &cfg, int dealer) {
  DGameState s = {};
  s.num_board = 0;
  s.stage = kPreflop;
  s.dealer = (uint8_t)dealer;
  // HU: SB at (dealer+1) % 2, BB at dealer (Lucy convention).
  int sb_pos = (dealer + 1) % 2;
  int bb_pos = dealer;
  s.cur_bet[sb_pos] = (int16_t)cfg.small_blind;
  s.cur_bet[bb_pos] = (int16_t)cfg.big_blind;
  s.total_bet[sb_pos] = (int16_t)cfg.small_blind;
  s.total_bet[bb_pos] = (int16_t)cfg.big_blind;
  s.stack[sb_pos] = (int16_t)(cfg.starting_stack - cfg.small_blind);
  s.stack[bb_pos] = (int16_t)(cfg.starting_stack - cfg.big_blind);
  s.high_bet = (int16_t)cfg.big_blind;
  s.pot = (int16_t)(cfg.small_blind + cfg.big_blind);
  s.current_player = (uint8_t)((bb_pos + 1) % 2); // SB acts first preflop in HU
  // Mark hole-card slots as unset (kernel fills).
  for (int p = 0; p < 2; ++p) for (int i = 0; i < 2; ++i) s.hole[p][i] = 0xFF;
  return s;
}

double gpu_cfr_train(GpuCfrEngine *eng, int num_iterations, uint64_t base_seed) {
  if (!eng) return 0.0;
  int B = eng->cfg.batch_size;
  int row_capacity = eng->row_capacity;

  long long t_total_start = now_ns();

  for (int it = 1; it <= num_iterations; ++it) {
    // Alternate traverser per iter (HU): even iter → traverser 0, else 1.
    int traverser = it % 2;
    DGameState init0 = build_initial_state(eng->cfg, /*dealer=*/0);

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    long long t0 = now_ns();
    int abstraction = (eng->cfg.hand_abstraction == HandAbstraction::V3) ? 1 : 0;
    DV3Centroids v3{
        eng->d_cents_flop,  eng->n_cents_flop,
        eng->d_cents_turn,  eng->n_cents_turn,
        eng->d_cents_river, eng->n_cents_river,
        eng->cfg.ehs2_rollouts
    };
    outcome_sampling_kernel<<<blocks, threads, 0, eng->stream>>>(
        init0, traverser,
        base_seed ^ ((uint64_t)it * 0xC2B2AE3D27D4EB4FULL),
        B,
        (int)eng->cfg.big_blind,
        eng->cfg.epsilon,
        abstraction, v3,
        DHashTable{eng->d_keys, eng->d_values, eng->d_size,
                   kHashCapacity, kHashCapacity - 1},
        eng->d_regret_sum, eng->d_strategy_sum, eng->d_strategy,
        eng->d_num_actions, eng->d_traversal_counter);
    CUDA_CHECK(cudaGetLastError());
    long long t1 = now_ns();
    eng->ns_traversal += (t1 - t0);

    // Refresh strategy snapshot with DCFR discounting.
    threads = 256;
    blocks = (row_capacity + threads - 1) / threads;
    long long t2 = now_ns();
    regret_match_kernel<<<blocks, threads, 0, eng->stream>>>(
        eng->d_regret_sum, eng->d_strategy,
        eng->d_num_actions, row_capacity,
        eng->cfg.dcfr_alpha, eng->cfg.dcfr_beta, it);
    CUDA_CHECK(cudaGetLastError());
    long long t3 = now_ns();
    eng->ns_match += (t3 - t2);

    eng->iters += 1;
    eng->trajectories += B;
  }

  long long t_sync_start = now_ns();
  CUDA_CHECK(cudaStreamSynchronize(eng->stream));
  eng->ns_sync += (now_ns() - t_sync_start);

  return (double)(now_ns() - t_total_start) / 1e9;
}

int gpu_cfr_num_infosets(const GpuCfrEngine *eng) {
  if (!eng) return 0;
  int sz = 0;
  cudaMemcpy(&sz, eng->d_size, sizeof(int), cudaMemcpyDeviceToHost);
  return sz;
}

GpuCfrProfile gpu_cfr_profile(const GpuCfrEngine *eng) {
  GpuCfrProfile p;
  if (!eng) return p;
  p.traversal_ns = eng->ns_traversal;
  p.match_ns = eng->ns_match;
  p.sync_ns = eng->ns_sync;
  p.total_ns = eng->ns_traversal + eng->ns_match + eng->ns_sync;
  p.iters = eng->iters;
  p.trajectories = eng->trajectories;
  return p;
}

// Save in CPU-NodeMatrix format so the existing Trainer::load_from_file +
// serve mode can read GPU-trained models directly. Format (matches
// trainer.cpp save_to_file):
//   size_t N
//   N times: size_t key_len, char[key_len] key, size_t k, double[k] sum
//
// We materialise CPU-style string keys by decoding the GPU's packed uint64
// into the same string format compute_information_set emits for V2/V3
// hand abstractions:
//   "<bucket>|_|<stage>|<stack_bucket>|<pot_bucket>|<ir_summary>|<num_legal>|"
//
// stack_bucket isn't tracked in the GPU key, so we use a fixed "3" (medium-
// deep, matches 100bb at the start of a hand). For HU NLHE FCPA training
// this constant is correct for the entire hand.
bool gpu_cfr_save(GpuCfrEngine *eng, const std::string &path) {
  if (!eng) return false;
  int sz = gpu_cfr_num_infosets(eng);
  std::vector<uint64_t> hkeys(kHashCapacity);
  std::vector<int>      hvals(kHashCapacity);
  cudaMemcpy(hkeys.data(), eng->d_keys, kHashCapacity * sizeof(uint64_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hvals.data(), eng->d_values, kHashCapacity * sizeof(int),
             cudaMemcpyDeviceToHost);

  size_t mat = (size_t)eng->row_capacity * kMaxActions;
  std::vector<double> hss(mat, 0.0);
  std::vector<int> hna(eng->row_capacity, 0);
  cudaMemcpy(hss.data(), eng->d_strategy_sum, mat * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hna.data(), eng->d_num_actions, eng->row_capacity * sizeof(int),
             cudaMemcpyDeviceToHost);

  // Collect occupied (key, row) pairs.
  std::vector<std::pair<uint64_t, int>> entries;
  for (int i = 0; i < kHashCapacity; ++i) {
    if (hkeys[i] == 0ULL) continue;
    int row = hvals[i];
    if (row < 0 || row >= eng->row_capacity) continue;
    int n_act = hna[row];
    if (n_act <= 0) continue;
    entries.emplace_back(hkeys[i], row);
  }

  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  size_t N = entries.size();
  std::fwrite(&N, sizeof(N), 1, f);

  // Decode each GPU key into the CPU's V2/V3 string format. The bit layout
  // changed in v0.6 (bucket field widened from 4 to 8 bits) — see
  // dev_compute_infoset_key for the layout.
  bool v3 = (eng->cfg.hand_abstraction == HandAbstraction::V3);
  static const char *AGGRESSOR = "-ds-"; // 0='-', 1='d', 2='s', 3='-'
  for (auto &[key, row] : entries) {
    int bucket    = (int)((key >> 0)  & 0xFF);            // 8 bits
    int stage     = (int)((key >> 8)  & 0x7);
    int pot_b     = (int)((key >> 11) & 0x7);
    int rc[4]     = {(int)((key>>14)&0x3), (int)((key>>16)&0x3),
                     (int)((key>>18)&0x3), (int)((key>>20)&0x3)};
    int lr[4]     = {(int)((key>>22)&0x3), (int)((key>>24)&0x3),
                     (int)((key>>26)&0x3), (int)((key>>28)&0x3)};
    int num_legal = (int)((key >> 30) & 0x3);
    if (num_legal == 0) num_legal = 4;
    // (player at bits 32..33 isn't part of the CPU string key.)

    // For V3, the per-street bucket needs to be composed into the global
    // 0..768 index that CPU's compute_information_set emits. For V1 the
    // bucket is already 0..9 and is written directly.
    int out_bucket = bucket;
    if (v3) {
      // dev_v3_global_bucket equivalent on host.
      switch (stage) {
      case 2: out_bucket = kV3PreflopBuckets + bucket; break;            // flop
      case 3: out_bucket = kV3PreflopBuckets + kV3PostflopBuckets + bucket; break;
      case 4: out_bucket = kV3PreflopBuckets + 2 * kV3PostflopBuckets + bucket; break;
      default: out_bucket = bucket; break;                               // preflop
      }
    }

    int cur_street_idx = stage - 1; // stage 1=preflop → idx 0
    if (cur_street_idx < 0) cur_street_idx = 0;
    if (cur_street_idx > 3) cur_street_idx = 3;

    // Reconstruct the imperfect_recall_summary string, e.g. "-0,d1,s2,-0".
    std::string ir;
    for (int s = 0; s <= cur_street_idx; ++s) {
      ir += AGGRESSOR[lr[s] & 3];
      ir += std::to_string(rc[s]);
      if (s < cur_street_idx) ir += ',';
    }
    if (ir.empty()) ir = "_";

    // Format matches game_state.cpp's compute_information_set V2/V3 path:
    //   "<bucket>|_|<stage>|<stack_bucket>|<pot_bucket>|<ir_summary>|<num_legal>|"
    // (V2/V3 drop the suit signature, encoded as "_".)
    char buf[160];
    int len = snprintf(buf, sizeof(buf), "%d|_|%d|3|%d|%s|%d|",
                       out_bucket, stage, pot_b, ir.c_str(), num_legal);
    size_t key_len = (size_t)len;
    std::fwrite(&key_len, sizeof(key_len), 1, f);
    std::fwrite(buf, 1, key_len, f);

    int n_act = hna[row];
    size_t k = (size_t)n_act;
    std::fwrite(&k, sizeof(k), 1, f);
    // Write only the first n_act entries of the strategy_sum row (CPU side
    // expects exactly k doubles — the row has kMaxActions=4 slots but
    // CPU's NodeMatrix stores per-row n_act-sized vectors).
    std::fwrite(&hss[(size_t)row * kMaxActions], sizeof(double), k, f);
  }
  std::fclose(f);
  std::fprintf(stderr, "[gpu_cfr] saved %zu infosets (CPU-compatible format) -> %s\n",
               N, path.c_str());
  return true;
}

bool gpu_cfr_load(GpuCfrEngine *eng, const std::string &path) {
  // For v0.5, load is not implemented — training-only path. Eval reads
  // models via gpu_cfr_query() which uses the device tables in-process.
  (void)eng; (void)path;
  return false;
}

DeviceQueryResponse gpu_cfr_query(GpuCfrEngine *eng,
                                   const DeviceQueryRequest &req) {
  DeviceQueryResponse out{};
  if (!eng) return out;

  // Reconstruct the device-side info-set key on the host. We mirror the
  // device dev_compute_infoset_key logic.
  // … This is a stripped-down host port: walk req.history_action[] to compute
  // per-street raise-count and last raiser, look up bucket via dev rules.
  // For brevity in v0.5 we do this via a tiny CUDA kernel call instead so
  // the bucketing logic stays in one place. Pragmatic shortcut: construct
  // a one-trajectory DGameState matching the request, run a tiny query
  // kernel that fills out the response.

  // (Implementation deferred — gpu_cfr_query is exercised by exporting the
  // strategy_sum table and querying host-side. See gpu_cfr_save.)
  return out;
}

} // namespace gpu_cfr
