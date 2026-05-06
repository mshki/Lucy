// Lucy hand evaluation + bucketing.
//
// As of feat/fast-evaluator the 5/7-card evaluator is backed by OMPEval
// (zekyll/OMPEval, ISC) — a 200KB perfect-hash table that returns a 16-bit
// total ordering at ~270M evals/sec. The pre-existing brute-force
// `evaluate_5_cards` (sort+set+map+next_permutation, ~µs/eval) and the
// O(C(7,5)=21) wrapper around it have been deleted; their public API is
// preserved by thin OMP wrappers below.
//
// Return value semantics: evaluate_5_cards / evaluate_7_cards return OMP's
// raw rank value. Higher is strictly better. Category can be extracted by
// dividing by 4096 (1=high card, 2=pair, ..., 9=straight flush). The old
// 0x100000-stride encoding is gone — bucketize_hand has been migrated.
//
// CHANGELOG:
//   - DELETED: brute-force evaluate_5_cards (replaced by OMP perfect-hash)
//   - DELETED: 21-subset next_permutation enumerator (OMP evaluates 7-card
//              hands in a single 64-bit-add + LUT chain)
//   - DELETED: std::set<Rank> royals / std::map<Rank,int> counts machinery
//   - CHANGED: bucketize_hand thresholds migrated from 0xN00000 to N*4096
//              (lossless because OMP gives us strictly more information)
//   - CHANGED: calculate_display_equity now uses OMP evaluator at the inner
//              loop. Iteration count remains 1000 (Monte Carlo) — call
//              omp::EquityCalculator directly for true range-vs-range.

#include "../include/equity.h"
#include "../include/external/omp/HandEvaluator.h"
#include "../include/external/omp/Hand.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

namespace {

char rank_to_char(Rank rank) {
  static const char *ranks = "23456789TJQKA";
  return ranks[rank];
}

// OMPEval encodes a card as `4 * rank + suit` with rank in [0..12]
// (deuce..ace) and suit in [0..3]. Lucy's `Card{rank, suit}` uses identical
// integer ranges; only the symbolic *meaning* of suit differs (we call
// 0=clubs whereas OMP calls 0=spade), but evaluation depends only on
// equality of suit numbers, not their labels — so the conversion is direct.
inline unsigned to_omp_card_id(const Card &c) {
  return static_cast<unsigned>(c.rank) * 4u + static_cast<unsigned>(c.suit);
}

inline omp::Hand to_omp_hand(const std::vector<Card> &cards) {
  omp::Hand h = omp::Hand::empty();
  for (const auto &c : cards) {
    h += omp::Hand(to_omp_card_id(c));
  }
  return h;
}

// Singleton evaluator: the static LUTs init in HandEvaluator()'s ctor cost
// ~10 ms; do it once per process.
const omp::HandEvaluator &evaluator() {
  static const omp::HandEvaluator E;
  return E;
}

// OMP value -> hand category (1..9).
inline int omp_category(uint16_t v) { return v >> 12; }

} // namespace

bool compareCards(const Card &a, const Card &b) { return a.rank > b.rank; }

// ---------------------------------------------------------------------------
// EquityModule v2 abstraction state
//
// Default ctor tries to load `bucket_boundaries.dat` from the cwd at startup.
// Format: 4-byte magic 'LBKT', 4-byte version=1, then for each of
// {flop, turn, river}: 4-byte int n_cutoffs, then n_cutoffs 4-byte ints.
// ---------------------------------------------------------------------------

namespace {
constexpr uint32_t kBucketMagic = 0x4C424B54u; // 'LBKT'

// Map a 0..51 (rank, suit) Lucy card to OMP's same encoding.
inline unsigned to_omp_card_for_bucket(const Card &c) {
  return static_cast<unsigned>(c.rank) * 4u + static_cast<unsigned>(c.suit);
}
} // namespace

EquityModule::EquityModule() {
  try_load_bucket_boundaries("bucket_boundaries.dat");
  try_load_equity_buckets("equity_buckets.dat");
}

void EquityModule::try_load_bucket_boundaries(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return; // silently fall back to uniform binning
  uint32_t magic = 0, version = 0;
  in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  if (magic != kBucketMagic || version != 1) {
    std::cerr << "[equity] bucket_boundaries.dat magic/version mismatch; "
              << "ignoring.\n";
    return;
  }
  auto read_vec = [&](std::vector<int> &dst) {
    int32_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    dst.resize(n);
    in.read(reinterpret_cast<char *>(dst.data()), n * sizeof(int));
  };
  read_vec(flop_cutoffs_);
  read_vec(turn_cutoffs_);
  read_vec(river_cutoffs_);
  std::cerr << "[equity] loaded bucket_boundaries.dat: "
            << flop_cutoffs_.size() << "/" << turn_cutoffs_.size()
            << "/" << river_cutoffs_.size() << " cutoffs (flop/turn/river)\n";
}

void EquityModule::try_load_equity_buckets(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return;
  uint32_t magic = 0, version = 0;
  in.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  in.read(reinterpret_cast<char *>(&version), sizeof(version));
  // Magic for the EHS² centroids file: 'LECB' (Lucy Equity Cluster Buckets).
  if (magic != 0x4C454342u || version != 1) {
    std::cerr << "[equity] equity_buckets.dat magic/version mismatch; "
              << "ignoring.\n";
    return;
  }
  auto read_vec = [&](std::vector<double> &dst) {
    int32_t n = 0;
    in.read(reinterpret_cast<char *>(&n), sizeof(n));
    dst.resize(n);
    in.read(reinterpret_cast<char *>(dst.data()), n * sizeof(double));
    std::sort(dst.begin(), dst.end()); // ensure ascending so bucket id ↑ with strength
  };
  read_vec(flop_ehs_centroids_);
  read_vec(turn_ehs_centroids_);
  read_vec(river_ehs_centroids_);
  std::cerr << "[equity] loaded equity_buckets.dat: "
            << flop_ehs_centroids_.size() << "/" << turn_ehs_centroids_.size()
            << "/" << river_ehs_centroids_.size()
            << " EHS² centroids (flop/turn/river)\n";
}

int EquityModule::nearest_centroid(double v,
                                   const std::vector<double> &centroids) {
  if (centroids.empty()) return 0;
  auto it = std::lower_bound(centroids.begin(), centroids.end(), v);
  if (it == centroids.begin()) return 0;
  if (it == centroids.end()) return static_cast<int>(centroids.size()) - 1;
  int hi = static_cast<int>(it - centroids.begin());
  int lo = hi - 1;
  // Pick the closer of the two adjacent centroids.
  return (v - centroids[lo] < centroids[hi] - v) ? lo : hi;
}

double EquityModule::compute_ehs2_runtime(const std::vector<Card> &hole,
                                           const std::vector<Card> &board,
                                           int n_rollouts) const {
  if (hole.size() < 2) return 0.0;
  // Per-state PRNG seed: deterministic so the same (hole, board) gets the
  // same EHS² across CFR iterations. CFR's regret signal must be consistent
  // across visits to the same info-set; a non-deterministic feature would
  // make the bucket id wobble and pollute the regret table. The seed
  // hashes the dealt cards.
  std::uint64_t seed = 0xcbf29ce484222325ULL;
  auto fold_card = [&](const Card &c) {
    seed ^= static_cast<std::uint64_t>(c.rank) * 4u
            + static_cast<std::uint64_t>(c.suit);
    seed *= 0x100000001b3ULL;
  };
  for (const auto &c : hole) fold_card(c);
  for (const auto &c : board) fold_card(c);
  std::mt19937_64 gen(seed);

  // Build the per-card dealt mask.
  std::vector<bool> dealt(52, false);
  auto card_id = [](const Card &c) {
    return static_cast<int>(c.rank) * 4 + static_cast<int>(c.suit);
  };
  for (const auto &c : hole) dealt[card_id(c)] = true;
  for (const auto &c : board) dealt[card_id(c)] = true;
  std::vector<int> remaining;
  remaining.reserve(52 - hole.size() - board.size());
  for (int c = 0; c < 52; ++c) if (!dealt[c]) remaining.push_back(c);

  int board_needed = 5 - static_cast<int>(board.size());
  if (board_needed < 0) board_needed = 0;

  static const omp::HandEvaluator E;
  omp::Hand hero_partial = omp::Hand::empty();
  for (const auto &c : hole) hero_partial += omp::Hand(to_omp_card_for_bucket(c));
  omp::Hand board_base = omp::Hand::empty();
  for (const auto &c : board) board_base += omp::Hand(to_omp_card_for_bucket(c));

  double sum_sq = 0.0;
  int total = 0;
  for (int r = 0; r < n_rollouts; ++r) {
    int k = 2 + board_needed;
    if ((int)remaining.size() < k) break;
    for (int i = 0; i < k; ++i) {
      std::uniform_int_distribution<int> d(i, (int)remaining.size() - 1);
      int j = d(gen);
      std::swap(remaining[i], remaining[j]);
    }
    omp::Hand opp_partial = omp::Hand::empty();
    opp_partial += omp::Hand(static_cast<unsigned>(remaining[0]));
    opp_partial += omp::Hand(static_cast<unsigned>(remaining[1]));
    omp::Hand board_runout = board_base;
    for (int b = 0; b < board_needed; ++b) {
      board_runout += omp::Hand(static_cast<unsigned>(remaining[2 + b]));
    }
    uint16_t hv = E.evaluate(hero_partial + board_runout);
    uint16_t ov = E.evaluate(opp_partial + board_runout);
    double result;
    if (hv > ov)        result = 1.0;
    else if (hv == ov)  result = 0.5;
    else                result = 0.0;
    sum_sq += result * result;
    ++total;
  }
  return total > 0 ? sum_sq / total : 0.0;
}

int EquityModule::bucketize_hand_v3(const std::vector<Card> &hero_hand,
                                    const std::vector<Card> &board_cards,
                                    street st, int n_rollouts) const {
  if (hero_hand.size() < 2) return 0;
  if (st == PRE || board_cards.empty()) {
    return preflop_canonical_index(hero_hand[0], hero_hand[1]);
  }

  // Pick the right centroid table for the street. If the centroids weren't
  // loaded (no equity_buckets.dat), fall back to V2 quantile bucketing.
  const std::vector<double> *centroids = nullptr;
  switch (st) {
  case FLOP:  centroids = &flop_ehs_centroids_;  break;
  case TURN:  centroids = &turn_ehs_centroids_;  break;
  case RIVER: centroids = &river_ehs_centroids_; break;
  default:    return 0;
  }
  if (centroids->empty()) {
    return bucketize_hand_v2(hero_hand, board_cards, st);
  }
  double ehs2 = compute_ehs2_runtime(hero_hand, board_cards, n_rollouts);
  return nearest_centroid(ehs2, *centroids);
}

// ============================================================================
// GPU-compatible V3 bucketing.
//
// The GPU's dev_compute_ehs2 + dev_eval_7card_total uses xoroshiro128+ as its
// PRNG and a hand-rolled 32-bit total-ordering 7-card evaluator. The default
// `compute_ehs2_runtime` above uses std::mt19937_64 + OMPEval. For the same
// (hole, board) input both algorithms compute statistically equivalent EHS²
// values, but their per-rollout random draws differ → bit-identical
// reproduction is impossible without using the same algorithm on both sides.
//
// Without bit-identical reproduction, ~10–30% of borderline hands end up in
// different K-means clusters between CPU and GPU, which means CPU --serve
// loading a GPU-trained V3 model produces a per-query info-set string that
// often DOESN'T MATCH any key in the model → uniform fallback. The result is
// V3 GPU plays significantly weaker than V1 GPU when served from CPU, even
// though the GPU model itself is well-trained.
//
// The fix below ports the exact GPU device algorithm to host code:
//   * xoroshiro128+ PRNG seeded identically (FNV-1a from cards)
//   * Same partial Fisher-Yates shuffle (`u % (n_remaining - i)` index)
//   * Same 32-bit total-ordering evaluator (categorical 1..9 + kicker pack)
//
// Used by HandAbstraction::V3_IR mode in compute_information_set.
// ============================================================================

namespace {

// xoroshiro128+ — identical to dev_next_u64 in src/cuda/gpu_cfr.cu.
struct HRng { uint64_t s0, s1; };

inline uint64_t hr_rotl(uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}
inline uint64_t hr_next_u64(HRng &r) {
  uint64_t s0 = r.s0, s1 = r.s1;
  uint64_t result = s0 + s1;
  s1 ^= s0;
  r.s0 = hr_rotl(s0, 55) ^ s1 ^ (s1 << 14);
  r.s1 = hr_rotl(s1, 36);
  return result;
}

// FNV-1a 64-bit seed — matches dev_ehs2_seed exactly.
inline uint64_t hr_ehs2_seed(uint8_t hole0, uint8_t hole1,
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

// Total-ordering 7-card hand evaluator — identical to dev_eval_7card_total.
// Returns a 32-bit value where higher = stronger; bits 24..27 = category
// 1..9, bits 4..23 = up to 5 kicker ranks × 4 bits.
inline uint32_t hr_eval_7card_total(const uint8_t *cards, int num_cards) {
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
    int popcount = 0;
    for (int b = 0; b < 13; ++b) if (suit_mask[s] & (1u << b)) popcount++;
    if (popcount >= 5) { flush_suit = s; break; }
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

  int r[5] = {0, 0, 0, 0, 0};
  int kept = 0;
  for (int rr = 12; rr >= 0 && kept < 5; --rr) {
    if (rank_count[rr] >= 1) r[kept++] = rr;
  }
  return pack(1, r[0], r[1], r[2], r[3], r[4]);
}

// Host-side EHS² rollout — bit-for-bit identical to dev_compute_ehs2.
inline double hr_compute_ehs2(uint8_t hole0, uint8_t hole1,
                               const uint8_t *board, int num_board,
                               int n_rollouts) {
  uint8_t deck[52];
  bool used[52] = {false};
  used[hole0] = true;
  used[hole1] = true;
  for (int i = 0; i < num_board; ++i) used[board[i]] = true;
  int n_deck = 0;
  for (int c = 0; c < 52; ++c) if (!used[c]) deck[n_deck++] = (uint8_t)c;

  int board_needed = 5 - num_board;
  int k = 2 + board_needed;

  uint64_t seed = hr_ehs2_seed(hole0, hole1, board, num_board);
  HRng rng;
  rng.s0 = seed ^ 0x9E3779B97F4A7C15ULL;
  rng.s1 = seed * 0xBF58476D1CE4E5B9ULL ^ 0xC2B2AE3D27D4EB4FULL;

  uint8_t hero[7];
  uint8_t opp[7];
  hero[0] = hole0; hero[1] = hole1;
  for (int i = 0; i < num_board; ++i) hero[2 + i] = board[i];

  double sum_sq = 0.0;
  for (int r = 0; r < n_rollouts; ++r) {
    for (int i = 0; i < k; ++i) {
      uint64_t u = hr_next_u64(rng);
      int j = i + (int)(u % (uint64_t)(n_deck - i));
      uint8_t tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
    }
    opp[0] = deck[0]; opp[1] = deck[1];
    for (int i = 0; i < num_board; ++i) opp[2 + i] = board[i];
    for (int i = 0; i < board_needed; ++i) {
      hero[2 + num_board + i] = deck[2 + i];
      opp [2 + num_board + i] = deck[2 + i];
    }
    int n_full = 2 + num_board + board_needed;
    uint32_t hv = hr_eval_7card_total(hero, n_full);
    uint32_t ov = hr_eval_7card_total(opp,  n_full);
    double score = (hv > ov) ? 1.0 : ((hv == ov) ? 0.5 : 0.0);
    sum_sq += score * score;
  }
  return sum_sq / (double)n_rollouts;
}

} // anonymous namespace

int EquityModule::bucketize_hand_v3_gpu_compatible(
    const std::vector<Card> &hero_hand,
    const std::vector<Card> &board_cards,
    street st, int n_rollouts) const {
  if (hero_hand.size() < 2) return 0;
  if (st == PRE || board_cards.empty()) {
    return preflop_canonical_index(hero_hand[0], hero_hand[1]);
  }
  const std::vector<double> *centroids = nullptr;
  switch (st) {
  case FLOP:  centroids = &flop_ehs_centroids_;  break;
  case TURN:  centroids = &turn_ehs_centroids_;  break;
  case RIVER: centroids = &river_ehs_centroids_; break;
  default:    return 0;
  }
  if (centroids->empty()) {
    return bucketize_hand_v2(hero_hand, board_cards, st);
  }

  // Pack hole + board into uint8_t arrays (rank * 4 + suit, identical to
  // GPU's DGameState card encoding).
  uint8_t hole0 = (uint8_t)((int)hero_hand[0].rank * 4 + (int)hero_hand[0].suit);
  uint8_t hole1 = (uint8_t)((int)hero_hand[1].rank * 4 + (int)hero_hand[1].suit);
  uint8_t board[5] = {0};
  int num_board = 0;
  for (const auto &c : board_cards) {
    if (num_board < 5)
      board[num_board++] = (uint8_t)((int)c.rank * 4 + (int)c.suit);
  }
  double ehs2 = hr_compute_ehs2(hole0, hole1, board, num_board, n_rollouts);
  return nearest_centroid(ehs2, *centroids);
}

int EquityModule::bin_index(int v, const std::vector<int> &cutoffs,
                            int n_bins) {
  if (cutoffs.empty()) {
    // Fallback: divide OMP's [0, 9*4096] range uniformly into n_bins.
    constexpr int kOmpMax = 9 * 4096; // top of straight-flush category
    int b = (v * n_bins) / (kOmpMax + 1);
    if (b < 0) b = 0;
    if (b >= n_bins) b = n_bins - 1;
    return b;
  }
  auto it = std::upper_bound(cutoffs.begin(), cutoffs.end(), v);
  return static_cast<int>(it - cutoffs.begin());
}

int EquityModule::preflop_canonical_index(const Card &a, const Card &b) {
  // 169 canonical 2-card holdings. We index via:
  //   pairs (13 of them):    0..12   high-rank == 0..12
  //   suited (78 of them):  13..90   pair (lo, hi) lexicographic with
  //                                  hi > lo, high+low encoded as
  //                                  13 + (lo * 13 + hi) compressed
  //   offsuit (78 of them): 91..168  same encoding as suited but offset
  // We use a more direct scheme: the 169 = 13 (pairs) + C(13,2) (suited)
  //                                       + C(13,2) (offsuit).
  Rank hi = (a.rank > b.rank) ? a.rank : b.rank;
  Rank lo = (a.rank > b.rank) ? b.rank : a.rank;
  bool is_pair = a.rank == b.rank;
  bool is_suited = a.suit == b.suit;
  if (is_pair) {
    return static_cast<int>(hi); // 0..12
  }
  // hi > lo (strict). Convert (hi, lo) to a 0..77 index via lexicographic.
  // i = (hi * (hi - 1)) / 2 + lo  for hi in 1..12, lo in 0..hi-1.
  int idx = (static_cast<int>(hi) * (static_cast<int>(hi) - 1)) / 2
            + static_cast<int>(lo);
  return is_suited ? (13 + idx) : (13 + 78 + idx);
}

int EquityModule::v2_global_index(int st, int per_street) const {
  switch (st) {
  case PRE:   return per_street;                                   // 0..168
  case FLOP:  return counts_.preflop + per_street;                 // 169..368
  case TURN:  return counts_.preflop + counts_.flop + per_street;  // 369..568
  case RIVER: return counts_.preflop + counts_.flop + counts_.turn
                + per_street;                                       // 569..768
  default:    return 0;
  }
}

int EquityModule::bucketize_hand_v2(const std::vector<Card> &hero_hand,
                                    const std::vector<Card> &board_cards,
                                    street st) const {
  if (hero_hand.size() < 2) return 0;
  if (st == PRE || board_cards.empty()) {
    return preflop_canonical_index(hero_hand[0], hero_hand[1]);
  }

  // Build OMP hand with however many board cards we have.
  omp::Hand h = omp::Hand::empty();
  for (const auto &c : hero_hand) h += omp::Hand(to_omp_card_for_bucket(c));
  for (const auto &c : board_cards) h += omp::Hand(to_omp_card_for_bucket(c));

  uint16_t v = 0;
  static const omp::HandEvaluator E;
  v = E.evaluate(h);

  switch (st) {
  case FLOP:  return bin_index(v, flop_cutoffs_,  counts_.flop);
  case TURN:  return bin_index(v, turn_cutoffs_,  counts_.turn);
  case RIVER: return bin_index(v, river_cutoffs_, counts_.river);
  default:    return 0;
  }
}

int EquityModule::evaluate_5_cards(const std::vector<Card> &cards) {
  if (cards.size() != 5)
    return 0;
  return evaluator().evaluate(to_omp_hand(cards));
}

int EquityModule::evaluate_7_cards(const std::vector<Card> &cards) {
  if (cards.size() < 5)
    return 0;
  return evaluator().evaluate(to_omp_hand(cards));
}

int EquityModule::bucketize_hand(const std::vector<Card> &hero_hand,
                                 const std::vector<Card> &board_cards,
                                 street st) {
  if (hero_hand.empty() || hero_hand.size() < 2)
    return BucketID::AIR;

  if (st == PRE && board_cards.empty()) {
    Rank r1 = hero_hand[0].rank;
    Rank r2 = hero_hand[1].rank;
    Rank high = r1 > r2 ? r1 : r2;
    Rank low = r1 > r2 ? r2 : r1;
    bool suited = hero_hand[0].suit == hero_hand[1].suit;
    bool pair = r1 == r2;

    if (pair) {
      if (high >= Rank::QUEEN)
        return BucketID::NUTS;
      if (high >= Rank::NINE)
        return BucketID::OVER_PAIR;
      if (high >= Rank::SIX)
        return BucketID::TOP_PAIR;
      return BucketID::MIDDLE_PAIR;
    }

    if (high >= Rank::ACE && low >= Rank::TEN) {
      return suited ? BucketID::STRONG_MADE : BucketID::TOP_PAIR;
    }
    if (high >= Rank::KING && low >= Rank::TEN) {
      return suited ? BucketID::TOP_PAIR : BucketID::MIDDLE_PAIR;
    }
    if (high >= Rank::JACK && low >= Rank::NINE) {
      return suited ? BucketID::MIDDLE_PAIR : BucketID::WEAK_PAIR;
    }
    if (high >= Rank::TEN || suited) {
      return BucketID::WEAK_PAIR;
    }
    return BucketID::AIR;
  }

  std::vector<Card> all_cards = hero_hand;
  all_cards.insert(all_cards.end(), board_cards.begin(), board_cards.end());

  // OMP value: higher == better. Categories: 1=highcard, 2=pair, 3=two pair,
  // 4=trips, 5=straight, 6=flush, 7=fullhouse, 8=quads, 9=straight flush.
  int v = evaluate_7_cards(all_cards);
  int cat = omp_category(v);

  if (cat >= 5)            // straight or better
    return BucketID::STRONG_MADE;
  if (cat == 4)            // three of a kind
    return BucketID::STRONG_MADE;
  if (cat == 3)            // two pair
    return BucketID::TOP_PAIR;

  if (cat == 2) {          // single pair
    Rank board_high = Rank::TWO;
    for (const auto &c : board_cards)
      if (c.rank > board_high)
        board_high = c.rank;

    bool pocket_pair = (hero_hand[0].rank == hero_hand[1].rank);

    if (pocket_pair) {
      if (hero_hand[0].rank > board_high)
        return BucketID::OVER_PAIR;
      if (hero_hand[0].rank == board_high)
        return BucketID::TOP_PAIR;
      return BucketID::WEAK_PAIR;
    }

    for (const auto &c : hero_hand) {
      if (c.rank == board_high)
        return BucketID::TOP_PAIR;
    }
    return BucketID::MIDDLE_PAIR;
  }

  // High card — check for flush draw.
  bool flush_draw = false;
  if (board_cards.size() >= 2) {
    std::map<Suit, int> suit_counts;
    for (const auto &c : all_cards)
      suit_counts[c.suit]++;
    for (const auto &[suit, count] : suit_counts) {
      if (count >= 4)
        flush_draw = true;
    }
  }
  return flush_draw ? BucketID::STRONG_DRAW : BucketID::AIR;
}

std::string EquityModule::canonical_state_signature(
    const std::vector<Card> &hero_hand,
    const std::vector<Card> &board_cards) const {
  std::vector<Card> cards = hero_hand;
  cards.insert(cards.end(), board_cards.begin(), board_cards.end());

  std::sort(cards.begin(), cards.end(), [](const Card &lhs, const Card &rhs) {
    if (lhs.rank != rhs.rank)
      return lhs.rank > rhs.rank;
    return lhs.suit < rhs.suit;
  });

  std::map<Suit, char> suit_map;
  char next_label = 'a';
  std::string signature;
  signature.reserve(cards.size() * 3);

  for (const auto &card : cards) {
    auto it = suit_map.find(card.suit);
    if (it == suit_map.end()) {
      it = suit_map.emplace(card.suit, next_label).first;
      if (next_label < 'd')
        ++next_label;
    }
    signature.push_back(rank_to_char(card.rank));
    signature.push_back(it->second);
    signature.push_back('|');
  }
  return signature.empty() ? "_" : signature;
}

double
EquityModule::calculate_display_equity(const std::vector<Card> &hero_hand,
                                       const std::vector<Card> &board_cards) {
  if (hero_hand.size() != 2)
    return 0.0;

  // Inner loop now runs at OMP-perfect-hash speed; iter count kept at 1000
  // for the same precision the UI promises.
  int wins = 0;
  int ties = 0;
  int iterations = 1000;

  std::vector<Card> full_deck;
  for (int r = 0; r < 13; ++r)
    for (int s = 0; s < 4; ++s)
      full_deck.emplace_back(static_cast<Rank>(r), static_cast<Suit>(s));

  auto remove_card = [&](const Card &c) {
    full_deck.erase(std::remove_if(full_deck.begin(), full_deck.end(),
                                   [&](const Card &x) {
                                     return x.rank == c.rank &&
                                            x.suit == c.suit;
                                   }),
                    full_deck.end());
  };
  for (const auto &c : hero_hand) remove_card(c);
  for (const auto &c : board_cards) remove_card(c);

  // Pre-build a partial-Hand from hero + known board so the inner loop only
  // adds (5 - board_cards.size()) more cards per trial. This is the
  // OMPEval-recommended idiom and matches their EquityCalculator's hot loop.
  omp::Hand hero_partial = omp::Hand::empty();
  for (const auto &c : hero_hand) hero_partial += omp::Hand(to_omp_card_id(c));

  omp::Hand board_partial = omp::Hand::empty();
  for (const auto &c : board_cards) board_partial += omp::Hand(to_omp_card_id(c));

  std::mt19937 rng(std::random_device{}());

  for (int i = 0; i < iterations; ++i) {
    std::vector<Card> deck = full_deck;
    std::shuffle(deck.begin(), deck.end(), rng);

    omp::Hand opp_partial = omp::Hand::empty();
    opp_partial += omp::Hand(to_omp_card_id(deck[0]));
    opp_partial += omp::Hand(to_omp_card_id(deck[1]));

    omp::Hand board_runout = board_partial;
    int needed = 5 - static_cast<int>(board_cards.size());
    for (int k = 0; k < needed; ++k) {
      board_runout += omp::Hand(to_omp_card_id(deck[2 + k]));
    }

    uint16_t hero_v = evaluator().evaluate(hero_partial + board_runout);
    uint16_t opp_v  = evaluator().evaluate(opp_partial + board_runout);

    if (hero_v > opp_v)        wins++;
    else if (hero_v == opp_v)  ties++;
  }

  return (double)wins / iterations + ((double)ties / iterations) / 2.0;
}
