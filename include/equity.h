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

  // V2 abstraction: 169 / 200 / 200 / 200 buckets per street.
  //   preflop:  one of 169 canonical 2-card holdings (suit isomorphism)
  //   flop:     OMP 5-card value, binned into 200 quantiles
  //   turn:     OMP 6-card value, binned into 200 quantiles
  //   river:    OMP 7-card value, binned into 200 quantiles
  // Quantile cutoffs are loaded from `bucket_boundaries.dat` if present in
  // the cwd; otherwise fall back to uniform-OMP-value bins (less accurate
  // distribution but still 200 distinct buckets).
  int bucketize_hand_v2(const std::vector<Card> &hero_hand,
                        const std::vector<Card> &board_cards, street st) const;

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

  // Quantile cutoffs per street. NULL = fallback to uniform-OMP-value bins.
  std::vector<int> flop_cutoffs_;   // 199 entries → 200 bins
  std::vector<int> turn_cutoffs_;
  std::vector<int> river_cutoffs_;
  BucketCounts counts_;

  void try_load_bucket_boundaries(const std::string &path);
  // Map a raw OMP value to a bin index using cutoffs.
  static int bin_index(int v, const std::vector<int> &cutoffs, int n_bins);

public:
  // Helper exposed for the build_bucket_boundaries tool. Computes the
  // canonical-preflop-hand index (0..168) for two hole cards.
  static int preflop_canonical_index(const Card &a, const Card &b);
};

#endif
