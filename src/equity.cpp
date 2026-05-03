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
