#include "../../include/mccfr/trainer.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

Trainer::Trainer(GameState *g)
    : game(g), em(*(g->equity_module)), node_matrix_(kMaxActions) {}

Trainer::~Trainer() = default;

int Trainer::get_or_create_node_id(const std::string &info, int n_actions) {
  auto it = info_to_id_.find(info);
  if (it != info_to_id_.end())
    return it->second;
  int id = node_matrix_.add_node(n_actions);
  info_to_id_.emplace(info, id);
  return id;
}

// Helper function to deal random hole cards
void Trainer::deal_random_hole_cards(GameState &state, std::mt19937 &gen) {
  // Create a full deck
  std::vector<Card> deck;
  for (int r = 0; r < 13; ++r) {
    for (int s = 0; s < 4; ++s) {
      deck.emplace_back(static_cast<Rank>(r), static_cast<Suit>(s));
    }
  }

  // Shuffle the deck
  std::shuffle(deck.begin(), deck.end(), gen);

  // Deal 2 cards to each player
  int card_idx = 0;
  for (auto &player : state.players) {
    player.hole_cards.clear();
    player.hole_cards.push_back(deck[card_idx++]);
    player.hole_cards.push_back(deck[card_idx++]);
  }
}

// Helper function to deal random community cards
void Trainer::deal_random_community_cards(GameState &state, int num_cards,
                                          std::mt19937 &gen) {
  // Create a full deck
  std::vector<Card> deck;
  for (int r = 0; r < 13; ++r) {
    for (int s = 0; s < 4; ++s) {
      deck.emplace_back(static_cast<Rank>(r), static_cast<Suit>(s));
    }
  }

  // Remove already dealt cards (player hole cards + existing community cards)
  auto is_dealt = [&](const Card &c) {
    // Check player cards
    for (const auto &p : state.players) {
      for (const auto &pc : p.hole_cards) {
        if (pc.rank == c.rank && pc.suit == c.suit)
          return true;
      }
    }
    // Check existing community cards
    for (const auto &cc : state.community_cards) {
      if (cc.rank == c.rank && cc.suit == c.suit)
        return true;
    }
    return false;
  };

  deck.erase(std::remove_if(deck.begin(), deck.end(), is_dealt), deck.end());

  // Shuffle remaining deck
  std::shuffle(deck.begin(), deck.end(), gen);

  // Deal the specified number of cards
  for (int i = 0; i < num_cards && i < (int)deck.size(); ++i) {
    state.community_cards.push_back(deck[i]);
  }
}

void Trainer::train(int iterations, int num_players, unsigned seed,
                    BettingAbstraction abs, bool randomize_config,
                    double stack_bb_arg, double sb_arg, double bb_arg) {
  if (!game) {
    return;
  }

  std::mt19937 gen;
  if (seed != 0) {
    gen.seed(seed);
  } else {
    std::random_device rd;
    gen.seed(rd());
  }

  // Configuration options for legacy randomized sampling.
  std::vector<int> player_counts = {2, 3, 4, 5, 6};
  std::vector<double> stack_bb_options = {10, 25, 50, 100, 200};

  int batched_traversals = 0;
  int flush_count = 0;
  int log_every = std::max(1, iterations / 20);

  for (int i = 0; i < iterations; ++i) {

    if (i == 0 || i % log_every == 0) {
      std::cout << "[lucy] iter " << i << "/" << iterations
                << " nodes=" << node_matrix_.num_nodes() << std::endl;
    }

    int sampled_players;
    double stack_bb, sb, bb;
    if (randomize_config) {
      // Legacy behavior: sample players and stack each iteration.
      sampled_players = num_players > 0 ? num_players : 5;
      stack_bb = stack_bb_options[gen() % stack_bb_options.size()];
      sb = 1.0;
      bb = 2.0;
    } else {
      // Benchmark mode: fixed config every iteration.
      sampled_players = num_players;
      stack_bb = stack_bb_arg;
      sb = sb_arg;
      bb = bb_arg;
    }
    double stack = stack_bb * bb;

    // External sampling: traverse from each player's perspective
    for (int traverser = 0; traverser < sampled_players; ++traverser) {
      GameState s(nullptr, game->equity_module);
      s.betting_abstraction = abs;
      s.init_game_setup(sampled_players, stack, sb, bb);
      s.start_hand();
      deal_random_hole_cards(s, gen);

      std::vector<double> reach(sampled_players, 1.0);
      cfr(s, traverser, 1.0, reach, 1.0, gen, 0);

      if (++batched_traversals >= batch_size_) {
        ++flush_count;
        node_matrix_.flush(flush_count, dcfr_);
        batched_traversals = 0;
      }
    }
  }

  // Final flush so any leftover deltas land in the strategy_sum.
  if (batched_traversals > 0) {
    ++flush_count;
    node_matrix_.flush(flush_count, dcfr_);
  }

  std::cout << "[lucy] training complete: " << iterations << " iterations, "
            << node_matrix_.num_nodes() << " infosets\n";
}

std::vector<double> Trainer::calculate_payoffs(GameState &state) {
  int pot = state.pot_size;
  std::vector<double> payoff(state.num_players, 0.0);

  int best_rank = -1;
  std::vector<int> rank(state.num_players, -9999);

  for (int i = 0; i < state.num_players; ++i) {
    Player *p = state.get_player(i);

    if (p->is_folded) {
      payoff[i] = -p->total_bet_size;
      continue;
    }

    std::vector<Card> hand;
    hand.insert(hand.end(), p->hole_cards.begin(), p->hole_cards.end());
    hand.insert(hand.end(), state.community_cards.begin(),
                state.community_cards.end());

    rank[i] = em.evaluate_7_cards(hand);
    best_rank = std::max(best_rank, rank[i]);
  }

  for (int i = 0; i < state.num_players; ++i) {
    Player *p = state.get_player(i);

    if (p->is_folded) {
      // already set above
    } else if (rank[i] == best_rank) {
      payoff[i] = pot - p->total_bet_size;
    } else {
      payoff[i] = -p->total_bet_size;
    }
  }

  return payoff;
}

double Trainer::get_terminal_payoff(GameState &state, int player_id) {
  return calculate_payoffs(state)[player_id];
}

double Trainer::cfr(GameState &state, int traverser, double prob_traverser,
                    std::vector<double> &reach, double prob_chance,
                    std::mt19937 &gen, int depth) {
  if (state.is_terminal() || depth > 200)
    return get_terminal_payoff(state, traverser);

  if (state.is_betting_round_over() && state.stage != Stage::SHOWDOWN) {
    if (state.stage == Stage::PREFLOP && state.community_cards.empty()) {
      deal_random_community_cards(state, 3, gen);
    } else if (state.stage == Stage::FLOP &&
               state.community_cards.size() == 3) {
      deal_random_community_cards(state, 1, gen);
    } else if (state.stage == Stage::TURN &&
               state.community_cards.size() == 4) {
      deal_random_community_cards(state, 1, gen);
    }
    state.next_street();

    if (state.is_terminal())
      return get_terminal_payoff(state, traverser);
  }

  Player *curr = state.get_current_player();
  if (!curr) {
    return 0.0;
  }
  int acting = curr->id;

  std::string info = state.compute_information_set(acting);

  auto legal = state.get_legal_actions();
  if (legal.empty())
    return get_terminal_payoff(state, traverser);

  if (static_cast<int>(legal.size()) > kMaxActions) {
    throw std::runtime_error("Trainer::cfr: legal action count exceeds "
                             "kMaxActions; bump kMaxActions in trainer.h");
  }

  int node_id = get_or_create_node_id(info, static_cast<int>(legal.size()));

  // Strategy snapshot from cache (uniform on the very first visit).
  const double *strat_row = node_matrix_.strategy_row(node_id);
  std::vector<double> strategy(strat_row, strat_row + legal.size());
  node_matrix_.accumulate_realization_weight(node_id, reach[curr->id]);

  // -----------------------------------------------------
  //     TRAVERSER — EXPLORE ALL ACTIONS
  // -----------------------------------------------------
  if (acting == traverser) {
    double node_util = 0.0;
    std::vector<double> utils(legal.size());

    for (size_t i = 0; i < legal.size(); ++i) {
      GameState next = state;
      next.apply_action(legal[i], true);

      std::vector<double> next_reach = reach;
      next_reach[traverser] *= strategy[i];

      utils[i] = cfr(next, traverser, prob_traverser * strategy[i], next_reach,
                     prob_chance, gen, depth + 1);

      node_util += strategy[i] * utils[i];
    }

    double scale = prob_chance;
    for (size_t p = 0; p < reach.size(); ++p) {
      if ((int)p != traverser) {
        scale *= reach[p];
      }
    }

    for (size_t i = 0; i < legal.size(); ++i) {
      double regret = (utils[i] - node_util) * scale;
      node_matrix_.accumulate_regret(node_id, static_cast<int>(i), regret);
    }

    return node_util;
  }

  // -----------------------------------------------------
  //     OPPONENT — SAMPLE ONE ACTION
  // -----------------------------------------------------
  std::discrete_distribution<> dist(strategy.begin(), strategy.end());
  int a = dist(gen);

  GameState next = state;
  next.apply_action(legal[a], true);

  std::vector<double> next_reach = reach;
  next_reach[acting] *= strategy[a];

  return cfr(next, traverser, prob_traverser, next_reach, prob_chance, gen,
             depth + 1);
}

std::vector<double> Trainer::get_strategy(const std::string &info) {
  auto it = info_to_id_.find(info);
  if (it == info_to_id_.end())
    return {};
  return node_matrix_.average_strategy(it->second);
}

Action Trainer::get_action_recommendation(GameState &state, int player_id,
                                          std::vector<double> &probs) {
  std::string info = state.compute_information_set(player_id);
  auto legal = state.get_legal_actions();

  if (legal.empty()) {
    probs.clear();
    return Action(-1, ActionType::FOLD, 0);
  }

  probs = get_strategy(info);
  if (probs.empty())
    probs.assign(legal.size(), 1.0 / legal.size());

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  int idx = dist(gen) % legal.size();
  return legal[idx];
}

void Trainer::save_to_file(const std::string &fn) {
  std::ofstream out(fn, std::ios::binary);
  if (!out) {
    std::cerr << "Cannot write file " << fn << "\n";
    return;
  }

  size_t N = info_to_id_.size();
  out.write((char *)&N, sizeof(N));

  for (auto &[key, id] : info_to_id_) {
    size_t len = key.size();
    out.write((char *)&len, sizeof(len));
    out.write(key.c_str(), len);

    auto sum = node_matrix_.strategy_sum_row(id);
    size_t k = sum.size();
    out.write((char *)&k, sizeof(k));
    out.write((char *)sum.data(), sizeof(double) * k);
  }
}

void Trainer::load_from_file(const std::string &fn) {
  std::ifstream in(fn, std::ios::binary);
  if (!in) {
    std::cerr << "Cannot open file " << fn << "\n";
    return;
  }

  // Reset state.
  info_to_id_.clear();
  node_matrix_ = NodeMatrix(kMaxActions);

  size_t N;
  in.read((char *)&N, sizeof(N));

  for (size_t i = 0; i < N; ++i) {
    size_t len;
    in.read((char *)&len, sizeof(len));

    std::string key(len, '\0');
    in.read(&key[0], len);

    size_t k;
    in.read((char *)&k, sizeof(k));

    std::vector<double> sum(k);
    in.read((char *)sum.data(), sizeof(double) * k);

    int id = node_matrix_.add_node(static_cast<int>(k));
    info_to_id_.emplace(std::move(key), id);
    node_matrix_.load_strategy_sum_row(id, sum);
  }
}
