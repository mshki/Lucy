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

enum street { PRE, FLOP, TURN, RIVER };

class EquityModule {
public:
  // Core bucketization for MCCFR
  int bucketize_hand(const std::vector<Card> &hero_hand,
                     const std::vector<Card> &board_cards, street street);

  // Canonical suit-isomorphic signature for information sets
  std::string canonical_state_signature(const std::vector<Card> &hero_hand,
                                        const std::vector<Card> &board_cards) const;

  // Hand evaluation helper
  int evaluate_7_cards(const std::vector<Card> &cards);

  // New: Display-focused equity calculation (Monte Carlo)
  double calculate_display_equity(const std::vector<Card> &hero_hand,
                                  const std::vector<Card> &board_cards);

private:
  int evaluate_5_cards(const std::vector<Card> &cards);
};

#endif
