#ifndef EQUITY_MODULE_H
#define EQUITY_MODULE_H

#include <array>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

enum Rank {
  TWO,
  THREE,
  FOUR,
  FIVE,
  SIX,
  SEVEN,
  EIGHT,
  NINE,
  TEN,
  JACK,
  QUEEN,
  KING,
  ACE
};
enum Suit { CLUBS, DIAMONDS, HEARTS, SPADES };

struct Card {
  Rank rank;
  Suit suit;

  Card(Rank r, Suit s) : rank(r), suit(s) {}

  bool operator==(const Card &other) const {
    return rank == other.rank && suit == other.suit;
  }

  std::string to_string() {
    const char *ranks = "23456789TJQKA";
    const char *suits = "cdhs";
    return std::to_string(ranks[rank]) + suits[suit];
  }

  friend std::ostream &operator<<(std::ostream &os, const Card &c) {
    const char *ranks = "23456789TJQKA";
    const char *suits = "cdhs";
    os << ranks[c.rank] << suits[c.suit];
    return os;
  }
};

// Legacy 10-bucket BucketID (heuristic). DEPRECATED. New code should use
// the value-quantile buckets via bucketize_hand_v2 (compute_bucket_v2).
// Keeping this enum for back-compat with the v1 abstraction during the
// transition; it'll be deleted once v2 is the default in main.
enum BucketID {
  AIR = 0,
  WEAK_BACKDOOR = 1,
  WEAK_DRAW = 2,
  STRONG_DRAW = 3,
  WEAK_PAIR = 4,
  MIDDLE_PAIR = 5,
  TOP_PAIR = 6,
  OVER_PAIR = 7,
  STRONG_MADE = 8,
  NUTS = 9
};

// Buckets per street for the v2 (value-quantile) abstraction. 169 preflop
// because that's the exact count of suit-canonical 2-card holdings; 200 for
// the post-flop streets (matches Pluribus/Slumbot/Libratus standards).
struct BucketCounts {
  int preflop = 169;
  int flop    = 200;
  int turn    = 200;
  int river   = 200;
};

enum street { PRE, FLOP, TURN, RIVER };

class EquityModule {
public:
  EquityModule();

  // Core bucketization for MCCFR. Original heuristic 10-bucket abstraction.
  // Kept for back-compat. Prefer bucketize_hand_v2 for new training runs.
  int bucketize_hand(const std::vector<Card> &hero_hand,
                     const std::vector<Card> &board_cards, street street);

  // V2 abstraction: 169 / 200 / 200 / 200 buckets per street, scalar-OMP-
  // value-quantile feature. Misses draw potential (flush draws look like
  // high-card hands by current value). Use V3 for stronger play.
  int bucketize_hand_v2(const std::vector<Card> &hero_hand,
                        const std::vector<Card> &board_cards, street st) const;

  // V3 abstraction (Pluribus / Slumbot / Libratus standard): 169 preflop
  // canonical hands + 200 EHS²-cluster buckets per post-flop street.
  //
  // For each (hole, board) on a post-flop street, we compute Expected Hand
  // Strength squared by Monte Carlo: sample N opponent hole-card pairs and
  // sample one runout to the river per opponent. EHS² = mean(result²)
  // where result ∈ {0, 0.5, 1} for {loss, tie, win}. EHS² captures both
  // made-hand strength (high mean) AND draw potential (high variance →
  // higher EHS² for fixed mean).
  //
  // Buckets are 1-D KMeans cluster centroids over EHS² values precomputed
  // by `build_equity_buckets`, loaded from `equity_buckets.dat`. At query
  // time we compute EHS² for the current hand on the fly (~10 µs per
  // query at OMP speed) and snap to the nearest centroid.
  //
  // Falls back to V2 if `equity_buckets.dat` isn't present.
  int bucketize_hand_v3(const std::vector<Card> &hero_hand,
                        const std::vector<Card> &board_cards, street st,
                        int n_rollouts = 100) const;

  // V3 GPU-compatible variant. Uses the same algorithm as the GPU device
  // (xoroshiro128+ PRNG, integer-categorical 7-card eval, partial Fisher-
  // Yates over 52-card deck) so identical (hole, board) inputs produce
  // bit-identical EHS² values to the GPU's dev_compute_ehs2 — eliminating
  // the bucket drift that would otherwise break GPU-trained V3 model
  // serving from CPU. Used by HandAbstraction::V3_IR.
  int bucketize_hand_v3_gpu_compatible(const std::vector<Card> &hero_hand,
                                       const std::vector<Card> &board_cards,
                                       street st,
                                       int n_rollouts = 50) const;

  // Canonical suit-isomorphic signature for information sets
  std::string canonical_state_signature(const std::vector<Card> &hero_hand,
                                        const std::vector<Card> &board_cards) const;

  // Hand evaluation helper
  int evaluate_7_cards(const std::vector<Card> &cards);

  // Monte-Carlo equity for the UI ("Hero equity vs random").
  double calculate_display_equity(const std::vector<Card> &hero_hand,
                                  const std::vector<Card> &board_cards);

  // Number of buckets per street under v2 (for trainer's NodeMatrix sizing).
  BucketCounts v2_counts() const { return counts_; }

  // Total v2 bucket index spanning streets:
  //   preflop bucket_index in [0, 169)
  //   flop bucket_index in [169, 369)
  //   turn bucket_index in [369, 569)
  //   river bucket_index in [569, 769)
  int v2_global_index(int street, int per_street_bucket) const;

private:
  int evaluate_5_cards(const std::vector<Card> &cards);

  // V2 quantile cutoffs per street. Empty = fallback to uniform-OMP-value bins.
  std::vector<int> flop_cutoffs_;   // 199 entries → 200 bins
  std::vector<int> turn_cutoffs_;
  std::vector<int> river_cutoffs_;
  // V3 EHS² centroids per street (sorted ascending → bucket index = position).
  std::vector<double> flop_ehs_centroids_;
  std::vector<double> turn_ehs_centroids_;
  std::vector<double> river_ehs_centroids_;
  BucketCounts counts_;

  void try_load_bucket_boundaries(const std::string &path);
  void try_load_equity_buckets(const std::string &path);
  // Map a raw OMP value to a bin index using cutoffs.
  static int bin_index(int v, const std::vector<int> &cutoffs, int n_bins);
  // Map an EHS² value to the nearest centroid index.
  static int nearest_centroid(double v, const std::vector<double> &centroids);
  // Compute EHS² on the fly via N Monte Carlo rollouts. Uses a per-thread
  // RNG seeded from the hand state so repeated queries on the same state
  // give the same answer (otherwise CFR's regret signal becomes inconsistent
  // across iterations for the same info-set).
  double compute_ehs2_runtime(const std::vector<Card> &hole,
                              const std::vector<Card> &board,
                              int n_rollouts) const;

public:
  // Helper exposed for the build_bucket_boundaries tool. Computes the
  // canonical-preflop-hand index (0..168) for two hole cards.
  static int preflop_canonical_index(const Card &a, const Card &b);
};

#endif
