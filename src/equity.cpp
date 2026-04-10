#include "../include/equity.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <set>

namespace {

char rank_to_char(Rank rank) {
  static const char *ranks = "23456789TJQKA";
  return ranks[rank];
}

} // namespace

// --- Internal Hand Evaluation Logic ---

// Helper to sort cards by rank descending
bool compareCards(const Card &a, const Card &b) { return a.rank > b.rank; }

int EquityModule::evaluate_5_cards(const std::vector<Card> &cards) {
  if (cards.size() != 5)
    return 0;

  std::vector<Card> sorted = cards;
  std::sort(sorted.begin(), sorted.end(), compareCards);

  bool flush = true;
  for (size_t i = 1; i < 5; ++i) {
    if (sorted[i].suit != sorted[0].suit) {
      flush = false;
      break;
    }
  }

  bool straight = true;
  bool royal = true;
  std::set<Rank> royals = {Rank::TEN, 
                            Rank::JACK,
                            Rank::QUEEN,
                            Rank::KING,
                            Rank::ACE};
  for (size_t i = 0; i < 4; ++i) {
    if (sorted[i].rank != sorted[i + 1].rank + 1) {
      straight = false;
      break;
    }
    if (royals.find(sorted[i].rank) == royals.end()) {
      royal = false;
    }
  }
  // Wheel case: A 5 4 3 2
  if (!straight && sorted[0].rank == Rank::ACE &&
      sorted[1].rank == Rank::FIVE && sorted[2].rank == Rank::FOUR &&
      sorted[3].rank == Rank::THREE && sorted[4].rank == Rank::TWO) {
    straight = true;
  }

  if (flush && straight) {
    if (sorted[0].rank == Rank::ACE && sorted[1].rank == Rank::KING)
      return 0x900000;                // Royal
    return 0x800000 + sorted[0].rank; // Straight Flush
  }

  std::map<Rank, int> counts;
  for (const auto &c : sorted)
    counts[c.rank]++;

  bool four_kind = false;
  bool three_kind = false;
  int pairs = 0;

  for (auto const &[rank, count] : counts) {
    if (count == 4)
      four_kind = true;
    if (count == 3)
      three_kind = true;
    if (count == 2)
      pairs++;
  }

  if (straight && flush && royal)
    return 0x900000;
  if (straight && flush) 
    return 0x800000;
  if (four_kind)
    return 0x700000; // Simplified score
  if (three_kind && pairs >= 1)
    return 0x600000;
  if (flush)
    return 0x500000;
  if (straight)
    return 0x400000;
  if (three_kind)
    return 0x300000;
  if (pairs == 2)
    return 0x200000;
  if (pairs == 1)
    return 0x100000;

  return sorted[0].rank;
}

int EquityModule::evaluate_7_cards(const std::vector<Card> &cards) {
  if (cards.size() < 5)
    return 0;
  if (cards.size() == 5)
    return evaluate_5_cards(cards);

  int max_score = 0;
  std::vector<int> p(cards.size());
  std::fill(p.begin(), p.begin() + 5, 1);
  std::fill(p.begin() + 5, p.end(), 0);

  do {
    std::vector<Card> combo;
    for (size_t i = 0; i < cards.size(); ++i) {
      if (p[i])
        combo.push_back(cards[i]);
    }
    int score = evaluate_5_cards(combo);
    if (score > max_score)
      max_score = score;
  } while (std::prev_permutation(p.begin(), p.end()));

  return max_score;
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

  int score = evaluate_7_cards(all_cards);

  if (score >= 0x400000)
    return BucketID::STRONG_MADE;
  if (score >= 0x300000)
    return BucketID::STRONG_MADE;
  if (score >= 0x200000)
    return BucketID::TOP_PAIR;

  if (score >= 0x100000) {
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

  if (flush_draw)
    return BucketID::STRONG_DRAW;

  return BucketID::AIR;
}

std::string EquityModule::canonical_state_signature(
    const std::vector<Card> &hero_hand, const std::vector<Card> &board_cards) const {
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

// Fast Monte Carlo simulation for display equity
double
EquityModule::calculate_display_equity(const std::vector<Card> &hero_hand,
                                       const std::vector<Card> &board_cards) {
  if (hero_hand.size() != 2)
    return 0.0;

  int wins = 0;
  int ties = 0;
  int iterations = 1000; // Enough for display precision

  // Create a full deck
  std::vector<Card> full_deck;
  for (int r = 0; r < 13; ++r) {
    for (int s = 0; s < 4; ++s) {
      full_deck.emplace_back(static_cast<Rank>(r), static_cast<Suit>(s));
    }
  }

  // Remove known cards
  auto remove_card = [&](const Card &c) {
    full_deck.erase(std::remove_if(full_deck.begin(), full_deck.end(),
                                   [&](const Card &x) {
                                     return x.rank == c.rank &&
                                            x.suit == c.suit;
                                   }),
                    full_deck.end());
  };

  for (const auto &c : hero_hand)
    remove_card(c);
  for (const auto &c : board_cards)
    remove_card(c);

  std::mt19937 rng(std::random_device{}());

  for (int i = 0; i < iterations; ++i) {
    std::vector<Card> deck = full_deck;
    std::shuffle(deck.begin(), deck.end(), rng);

    // Deal opponent hand
    std::vector<Card> opp_hand = {deck[0], deck[1]};

    // Deal remaining board
    std::vector<Card> current_board = board_cards;
    int board_idx = 2;
    while (current_board.size() < 5) {
      current_board.push_back(deck[board_idx++]);
    }

    // Evaluate
    std::vector<Card> hero_full = hero_hand;
    hero_full.insert(hero_full.end(), current_board.begin(),
                     current_board.end());

    std::vector<Card> opp_full = opp_hand;
    opp_full.insert(opp_full.end(), current_board.begin(), current_board.end());

    int hero_score = evaluate_7_cards(hero_full);
    int opp_score = evaluate_7_cards(opp_full);

    if (hero_score > opp_score)
      wins++;
    else if (hero_score == opp_score)
      ties++;
  }

  return (double)wins / iterations + ((double)ties / iterations) / 2.0;
}
