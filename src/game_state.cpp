#include "../include/game_state.h"
#include <algorithm>
#include <iomanip>
#include <iostream>

// Player Constructor
Player::Player(int _id, double _stack, bool _is_human)
    : id(_id), stack(_stack), current_bet(0), is_folded(false),
      is_all_in(false), is_human(_is_human), has_acted_this_street(false) {}

// GameState Constructor
GameState::GameState(RiskProfiler *rp, EquityModule *em)
    : risk_profiler(rp), equity_module(em), pot_size(0),
      current_street_highest_bet(0), num_players(0), dealer_index(0),
      current_player_index(0), small_blind_amount(0), big_blind_amount(0),
      stage(Stage::START), type(StateType::CHANCE) {}

void GameState::init_game_setup(int n_players, double stack_size, double sb,
                                double bb) {
  num_players = n_players;
  small_blind_amount = sb;
  big_blind_amount = bb;

  players.clear();
  // Player 0 is Human
  players.emplace_back(0, stack_size, true);
  players[0].total_bet_size = 0;
  if (risk_profiler)
    risk_profiler->add_player(0, stack_size);

  // Others are Bots
  for (int i = 1; i < num_players; i++) {
    players.emplace_back(i, stack_size, false);
    players[i].total_bet_size = 0;
    if (risk_profiler)
      risk_profiler->add_player(i, stack_size);
  }

  dealer_index = 0;
}

void GameState::start_hand(int input_dealer) {
  pot_size = 0;
  current_street_highest_bet = 0;
  stage = Stage::START;
  type = StateType::CHANCE; // Waiting for cards
  community_cards.clear();
  history.clear();

  // manual selection of the dealer
  if (input_dealer != -1) {
    // check input
    dealer_index = input_dealer % num_players;
  } else {
    // default
    dealer_index = (dealer_index + 1) % num_players;
  }
  //dealer_index = (dealer_index + 1) % num_players;

  if (risk_profiler)
    risk_profiler->reset_hand();

  for (auto &p : players) {
    p.hole_cards.clear();
    p.is_folded = false;
    p.is_all_in = false;
    p.current_bet = 0;
    p.has_acted_this_street = false;
  }

  // Post Blinds
  int sb_pos = (dealer_index + 1) % num_players;
  int bb_pos = (dealer_index + 2) % num_players;

  // SB
  int sb_paid = min(small_blind_amount, players[sb_pos].stack);
  players[sb_pos].stack -= sb_paid;
  players[sb_pos].current_bet = sb_paid;
  pot_size += sb_paid;

  // BB
  int bb_paid = min(big_blind_amount, players[bb_pos].stack);
  players[bb_pos].stack -= bb_paid;
  players[bb_pos].current_bet = bb_paid;
  pot_size += bb_paid;
  current_street_highest_bet = bb_paid;

  // Action starts left of BB
  current_player_index = (bb_pos + 1) % num_players;

  // Advance to PREFLOP immediately after setup
  stage = Stage::PREFLOP;
  type = StateType::PLAY;
}

void GameState::set_community_cards(const std::vector<Card> &cards) {
  community_cards = cards;
}

void GameState::set_player_cards(int player_id,
                                 const std::vector<Card> &cards) {
  if (player_id >= 0 && player_id < (int)players.size()) {
    players[player_id].hole_cards = cards;
  }
}

void GameState::next_street() {
  for (auto &p : players) {
    p.current_bet = 0;
    p.has_acted_this_street = false;
  }
  current_street_highest_bet = 0;

  // Advance stage
  if (stage == Stage::PREFLOP)
    stage = Stage::FLOP;
  else if (stage == Stage::FLOP)
    stage = Stage::TURN;
  else if (stage == Stage::TURN)
    stage = Stage::RIVER;
  else if (stage == Stage::RIVER)
    stage = Stage::SHOWDOWN;

  // Reset action: in HU NLHE the BB acts first postflop; in 3+ player games
  // the SB (= player to the left of the dealer) acts first.
  if (num_players == 2) {
    // BB sits at (dealer+2) % 2 == dealer position in Lucy's blinds layout.
    current_player_index = dealer_index;
  } else {
    current_player_index = (dealer_index + 1) % num_players;
  }

  // Skip folded/all-in players
  int attempts = 0;
  while ((players[current_player_index].is_folded ||
          players[current_player_index].is_all_in) &&
         attempts < num_players) {
    current_player_index = (current_player_index + 1) % num_players;
    attempts++;
  }
}

bool GameState::record_action(int player_idx, Action action, bool is_train) {
  Player &p = players[player_idx];
  action.previous_bet = p.current_bet;
  history.push_back(action);
  p.has_acted_this_street = true;

  if (action.type == ActionType::FOLD) {
    p.is_folded = true;
    if (risk_profiler)
      risk_profiler->update_player_profile(p.id, "fold", 0, pot_size);
    return true;
  }

  int amount_added = 0;
  if (action.type == ActionType::CALL) {
    amount_added = min(p.stack, current_street_highest_bet - p.current_bet);
    if (risk_profiler)
      risk_profiler->update_player_profile(p.id, "call", amount_added,
                                           pot_size);
  } else if (action.type == ActionType::BET ||
             action.type == ActionType::RAISE) {
    amount_added = min(p.stack, action.amount - p.current_bet);
    current_street_highest_bet = action.amount;
    if (risk_profiler)
      risk_profiler->update_player_profile(p.id, "raise", amount_added,
                                           pot_size);
  } else if (action.type == ActionType::ALLIN) {
    amount_added = p.stack;
    if (p.current_bet + amount_added > current_street_highest_bet) {
      current_street_highest_bet = p.current_bet + amount_added;
    }
    if (risk_profiler)
      risk_profiler->update_player_profile(p.id, "allin", amount_added,
                                           pot_size);
  } else if (action.type == ActionType::CHECK) {
    return true;
  }

  // if (!is_train) {
    p.stack -= amount_added;
    pot_size += amount_added;
  // }
  p.current_bet += amount_added;
  p.total_bet_size += amount_added;
  
  if (p.stack <= 0) {
    p.is_all_in = true;
    p.stack = 0;
  }
    
  return true;
}

void GameState::apply_action(Action action, bool is_train) {
  record_action(action.player_id, action, is_train);
  determine_next_state();
}

void GameState::determine_next_state() {
  // If the hand ended by folding
  if (get_active_player_count() <= 1) {
    stage = Stage::SHOWDOWN;
    type = StateType::TERMINAL;
    return;
  }

  // If betting is not over, just rotate action
  if (!is_betting_round_over()) {
    next_player();
    return;
  }

  // If betting ended and we are on RIVER, hand is terminal.
  if (stage == Stage::RIVER) {
    stage = Stage::SHOWDOWN;
    type = StateType::TERMINAL;
    return;
  }

  // Otherwise:
  // *** IMPORTANT ***
  // DO NOT MOVE STREETS HERE.
  // CFR will call next_street() after dealing flop/turn/river.
  return;
}

bool GameState::is_betting_round_over() {
  for (const auto &p : players) {
    if (!p.is_folded && !p.is_all_in) {
      if (!p.has_acted_this_street)
        return false;
      if (p.current_bet != current_street_highest_bet)
        return false;
    }
  }
  return true;
}

void GameState::next_player() {
  int attempts = 0;
  do {
    current_player_index = (current_player_index + 1) % num_players;
    attempts++;
  } while ((players[current_player_index].is_folded ||
            players[current_player_index].is_all_in) &&
           attempts <= num_players);
}

int GameState::get_active_player_count() {
  int count = 0;
  for (const auto &p : players)
    if (!p.is_folded)
      count++;
  return count;
}

Player *GameState::get_current_player() {
  return &players[current_player_index];
}

Player *GameState::get_player(int player_id) {
  for (auto &p : players)
    if (p.id == player_id)
      return &p;
  return nullptr;
}

bool GameState::is_terminal() { return type == StateType::TERMINAL || stage == Stage::SHOWDOWN; }

// MCCFR Information Set
string GameState::compute_information_set(int player_id) {
  std::string info;
  Player *p = get_player(player_id);
  if (!p)
    return "INVALID";

  street st = PRE;
  if (stage == Stage::FLOP)
    st = FLOP;
  else if (stage == Stage::TURN)
    st = TURN;
  else if (stage == Stage::RIVER)
    st = RIVER;

  int bucket = 0;
  if (equity_module && p->hole_cards.size() >= 2) {
    if (hand_abstraction == HandAbstraction::V3_EHS_CLUSTERS) {
      int per_street = equity_module->bucketize_hand_v3(p->hole_cards,
                                                        community_cards, st);
      bucket = equity_module->v2_global_index(static_cast<int>(st), per_street);
    } else if (hand_abstraction == HandAbstraction::V2_VALUE_QUANTILES) {
      int per_street = equity_module->bucketize_hand_v2(p->hole_cards,
                                                        community_cards, st);
      bucket = equity_module->v2_global_index(static_cast<int>(st), per_street);
    } else {
      bucket = equity_module->bucketize_hand(p->hole_cards,
                                             community_cards, st);
    }
  }

  // V1 includes the suit-canonical signature in the info-set key (huge for
  // distinguishing strategically equivalent boards). V2 / V3 already encode
  // hand strength via OMP / EHS² clustering, so we drop the signature to
  // keep infoset count manageable.
  std::string suit_signature = "_";
  if (hand_abstraction == HandAbstraction::V1_HEURISTIC_10
      && equity_module && p->hole_cards.size() >= 2) {
    suit_signature =
        equity_module->canonical_state_signature(p->hole_cards, community_cards);
  }

  // for (auto c : p->hole_cards) {
  //   info += c.to_string();
  // }
  // if (!p->hole_cards.empty()) {
  //   info += "|";
  // }
  // for (auto c : community_cards) {
  //   info += c.to_string();
  // }
  // if (!community_cards.empty()) {
  //   info += "|";
  // }
  
  info += std::to_string(bucket) + "|";
  info += suit_signature + "|";
  info += std::to_string((int)stage) + "|";

  // ABSTRACTION: Normalize to big blinds
  double bb = big_blind_amount;
  if (bb <= 0)
    bb = 1.0; // Safety check

  // Abstract stack size to buckets
  double stack_bb = p->stack / bb;
  int stack_bucket = abstract_stack_size(stack_bb);
  info += std::to_string(stack_bucket) + "|";
  // info += std::to_string(p->stack);

  // Abstract pot size to buckets
  double pot_bb = pot_size / bb;
  int pot_bucket = abstract_pot_size(pot_bb);
  info += std::to_string(pot_bucket) + "|";
  // info += std::to_string(pot_size);

  // Abstract action history with bet sizing
  info += abstract_action_history() + "|";

  // Make info-set unique per distinct action count
  auto legal = get_legal_actions();
  info += std::to_string(legal.size()) + "|";

  return info;
}
// Abstract stack into buckets based on big blinds
int GameState::abstract_stack_size(double stack_bb) const {
  if (stack_bb < 10)
    return 0; // Short stack
  if (stack_bb < 25)
    return 1; // Medium-short
  if (stack_bb < 50)
    return 2; // Medium
  if (stack_bb < 100)
    return 3; // Medium-deep
  return 4;   // Deep stack
}

// Abstract pot into buckets
int GameState::abstract_pot_size(double pot_bb) const {
  if (pot_bb < 5)
    return 0; // Small pot
  if (pot_bb < 15)
    return 1; // Medium pot
  if (pot_bb < 30)
    return 2; // Large pot
  if (pot_bb < 60)
    return 3; // Very large pot
  return 4;   // Huge pot
}

// Abstract bet size relative to pot
std::string GameState::abstract_bet_size(double bet_amount) const {
  double pot = std::max(pot_size, big_blind_amount);
  double pot_fraction = bet_amount / pot;

  if (pot_fraction < 0.4)
    return "S"; // Small (< 1/3 pot)
  if (pot_fraction < 0.75)
    return "M"; // Medium (1/2 pot)
  if (pot_fraction < 1.5)
    return "P"; // Pot-sized
  if (pot_fraction < 2.5)
    return "L"; // Large (2x pot)
  return "A";   // All-in / Overbet
}

// Simplify action history with abstracted bet sizes and RELATIVE positions
std::string GameState::abstract_action_history() const {
  std::string result;
  for (const auto &action : history) {
    // Calculate relative position: (action_player - dealer + num_players) %
    // num_players This makes 0=Dealer, 1=SB, 2=BB, etc. regardless of absolute
    // ID
    int relative_pos =
        (action.player_id - dealer_index + num_players) % num_players;

    result += std::to_string(relative_pos);
    double added = action.amount - action.previous_bet;

    switch (action.type) {
    case ActionType::FOLD:
      result += "F";
      break;
    case ActionType::CHECK:
      result += "X";
      break;
    case ActionType::CALL:
      result += "C";
      break;
    case ActionType::BET:
    case ActionType::RAISE:
      result += "R" + abstract_bet_size(added);
      break;
    case ActionType::ALLIN:
      result += "A";
      break;
    }
  }
  return result.empty() ? "_" : result;
}

std::vector<Action> GameState::get_legal_actions() {
  std::vector<Action> actions;
  if (is_terminal())
    return actions;

  Player *p = get_current_player();
  int call_amt = current_street_highest_bet - p->current_bet;

  // ---- FCPA abstraction (matches OpenSpiel universal_poker fcpa) ----
  // OpenSpiel action IDs are stable across states:
  //   0 = FOLD, 1 = CHECK/CALL, 2 = POT-sized bet/raise, 3 = ALLIN
  // Fold is illegal when there is no bet to call (matches OpenSpiel).
  if (betting_abstraction == BettingAbstraction::FCPA) {
    if (call_amt > 0) {
      actions.emplace_back(p->id, ActionType::FOLD, 0);
    }

    if (call_amt == 0) {
      actions.emplace_back(p->id, ActionType::CHECK, 0);
    } else {
      double call = std::min((double)p->stack, (double)call_amt);
      actions.emplace_back(p->id, ActionType::CALL, call);
    }

    if (call_amt == 0) {
      double pot = pot_size > 0 ? pot_size : big_blind_amount;
      double add = pot;
      if (add > 0 && add < p->stack) {
        actions.emplace_back(p->id, ActionType::BET, p->current_bet + add);
      }
    } else {
      double pot = std::max((double)pot_size, big_blind_amount);
      double base = pot + call_amt;
      double raise_to = current_street_highest_bet + base;
      if (raise_to > current_street_highest_bet &&
          (raise_to - p->current_bet) < p->stack) {
        actions.emplace_back(p->id, ActionType::RAISE, raise_to);
      }
    }

    if (p->stack > 0) {
      actions.emplace_back(p->id, ActionType::ALLIN, p->stack);
    }
    return actions;
  }

  // ---- LEGACY abstraction (original behavior) ----
  actions.emplace_back(p->id, ActionType::FOLD, 0);

  if (call_amt == 0) {
    actions.emplace_back(p->id, ActionType::CHECK, 0);

    double pot;

    if (pot_size == 0) {
      pot = big_blind_amount;
    } else {
      pot = pot_size;
    }

    double contribs[] = {0.33 * pot, 0.66 * pot, 1.00 * pot, 2.00 * pot};

    for (double add_amount : contribs) {
      if (add_amount <= p->stack) {
        double raise_to = p->current_bet + add_amount;
        actions.emplace_back(p->id, ActionType::BET, raise_to);
      }
    }

    actions.emplace_back(p->id, ActionType::ALLIN, p->stack);

    // if (p->stack >= pot / 2.0)
    //  actions.emplace_back(p->id, ActionType::BET, ((double)(1/3) * pot));
    // if (p->stack >= pot)
    //   actions.emplace_back(p->id, ActionType::BET, pot);
    // actions.emplace_back(p->id, ActionType::ALLIN, p->stack);
  } else {
    // Facing a bet
    if (p->stack < call_amt) {
      actions.emplace_back(p->id, ActionType::CALL, p->stack);
    } else {
      actions.emplace_back(p->id, ActionType::CALL, call_amt);
    }

    if (p->stack > call_amt) {
      double pot = std::max(pot_size, big_blind_amount);
      // Use (pot + call_amt) as base for raise sizing
      double base = pot + call_amt;
      double contribs[] = {0.33 * base, 0.66 * base, 1.00 * base, 2.00 * base};

      for (double add_amount : contribs) {
        double raise_to = current_street_highest_bet + add_amount;

        if (raise_to > current_street_highest_bet &&
            (raise_to - p->current_bet) <= p->stack) {
          actions.emplace_back(p->id, ActionType::RAISE, raise_to);
        }
      }
      actions.emplace_back(p->id, ActionType::ALLIN, p->stack);
    }
  }
  return actions;
}

void GameState::resolve_winner() {
  // Manual resolution or simple equity check
  // For now, simple equity check if we have cards
  int winner_id = -1;
  int best_score = -1;

  for (auto &p : players) {
    if (p.is_folded)
      continue;

    // If we don't have opponent cards (manual mode), we can't determine winner
    // automatically unless we ask user. For now, let's just print "Showdown"
    if (p.hole_cards.empty())
      continue;

    std::vector<Card> hand = p.hole_cards;
    hand.insert(hand.end(), community_cards.begin(), community_cards.end());
    int score = equity_module->evaluate_7_cards(hand);
    if (score > best_score) {
      best_score = score;
      winner_id = p.id;
    }
  }

  if (winner_id != -1) {
    cout << "Winner determined by cards: Player " << winner_id << "\n";
  } else {
    cout << "Showdown! (Enter winner manually in next version)\n";
  }
}
