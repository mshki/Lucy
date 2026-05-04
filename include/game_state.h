#ifndef GAME_STATE_MODULE_H
#define GAME_STATE_MODULE_H

#include "equity.h"
#include "risk_profiler.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

enum class Stage {
  START,
  PREFLOP,
  FLOP,
  TURN,
  RIVER,
  SHOWDOWN,
};

enum class ActionType { FOLD, CHECK, CALL, BET, RAISE, ALLIN };

// Betting tree abstraction.
//   LEGACY: Lucy's original 5-bet-size abstraction
//           {fold, check/call, 0.33p, 0.66p, 1p, 2p, allin}
//   FCPA:   research-standard 4-action abstraction matching OpenSpiel's
//           universal_poker bettingAbstraction=fcpa
//           {fold, check/call, pot-sized bet/raise, all-in}
// Action ordering in FCPA mode preserves OpenSpiel's stable IDs:
//   0 = FOLD, 1 = CHECK/CALL, 2 = POT, 3 = ALLIN
// Action abstraction for the betting tree.
//   LEGACY      — original 5-bet sizes {0.33p, 0.66p, p, 2p, allin}, same per street
//   FCPA        — research-standard 4-action {fold, call, pot, allin}, same per street
//   STREET_RICH — Slumbot-style street-specific sizing (recommended for play):
//                   preflop: {fold, call, 2.5×BB, 3×BB, 4×BB, allin}        (6)
//                   flop:    {fold, check/call, 0.33p, 0.66p, p, allin}     (5-6)
//                   turn:    {fold, check/call, 0.5p, p, 1.5p, allin}       (5-6)
//                   river:   {fold, check/call, 0.5p, p, 1.5p, 2p, allin}   (6-7)
//                 Captures human-poker tactics (pre-flop blind-multiplier raises,
//                 river overbets, polarized turn sizing) that FCPA can't express.
enum class BettingAbstraction { LEGACY, FCPA, STREET_RICH };

// Hand-strength abstraction:
//   V1 — legacy 10-bucket heuristic (BucketID enum). Coarse, ignores
//        draw potential, ignores most board texture.
//   V2 — 169/200/200/200 buckets per street using OMP-value quantile cuts.
//        Captures current made-hand strength but NOT draw potential.
//   V3 — 169/200/200/200 buckets per street using EHS² (Expected Hand
//        Strength squared) cluster centroids. Captures both made-hand
//        strength AND draw potential. Pluribus / Slumbot / Libratus standard.
enum class HandAbstraction {
  V1_HEURISTIC_10,
  V2_VALUE_QUANTILES,
  V3_EHS_CLUSTERS,
};

// Minimal player struct for MCCFR
struct Player {
  int id;
  vector<Card> hole_cards;
  double stack;
  double total_bet_size;
  double current_bet; // amount to bet in the current street
  bool is_folded;
  bool is_all_in;
  bool is_human;

  bool has_acted_this_street;

  Player(int _id, double _stack, bool _is_human);
};

struct Action {
  int player_id;
  ActionType type;
  double amount;
  double previous_bet;

  Action(int pid, ActionType t, double amt = 0)
      : player_id(pid), type(t), amount(amt), previous_bet(0) {}
};

enum class StateType { CHANCE, PLAY, TERMINAL };

struct GameState {
  vector<Player> players;
  vector<Card> community_cards; // Manual input

  std::vector<Action> history;

  // Modules
  RiskProfiler *risk_profiler;
  EquityModule *equity_module;

  double pot_size;
  double current_street_highest_bet;

  int num_players;
  int dealer_index;
  int current_player_index;

  double small_blind_amount;
  double big_blind_amount;

  Stage stage;
  StateType type;
  BettingAbstraction betting_abstraction = BettingAbstraction::LEGACY;
  HandAbstraction hand_abstraction = HandAbstraction::V1_HEURISTIC_10;

  GameState(RiskProfiler *rp, EquityModule *em);

  // Init
  void init_game_setup(int n_players, double stack_size, double sb, double bb);
  void start_hand(int input_dealer = -1);

  // Manual Input Methods
  void set_community_cards(const std::vector<Card> &cards);
  void set_player_cards(int player_id, const std::vector<Card> &cards);

  // Flow
  void next_street();
  void resolve_winner(); // Manual winner resolution or simple equity calc
  bool is_hand_over();
  bool is_betting_round_over();

  // Actions
  bool record_action(int player_idx, Action action, bool is_train);
  std::vector<Action> get_legal_actions();
  void apply_action(Action action, bool is_train);

  // MCCFR Support
  bool is_terminal();
  string compute_information_set(int player_id);
  Player *get_player(int player_id);

  // Abstraction Helpers
  int abstract_stack_size(double stack_bb) const;
  int abstract_pot_size(double pot_bb) const;
  std::string abstract_bet_size(double bet_amount) const;
  std::string abstract_action_history() const;
  // Imperfect-recall summary (Pluribus standard):
  //   - aggressor relative-position per prior street (preflop, flop, turn),
  //     where each is the player_id of the last raiser or 'X' if no raise
  //   - number of bets/raises *this street*, capped at 4
  // Drops the full action sequence — collapses strategically equivalent
  // histories that differ only in non-strategic permutations of
  // call/check/raise sequencing.
  std::string imperfect_recall_summary() const;

  // Helpers
  int get_active_player_count();
  Player *get_current_player();
  void next_player();
  void determine_next_state();
};

#endif