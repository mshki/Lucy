#ifndef TRAINER_H
#define TRAINER_H

#include "equity.h"
#include "game_state.h"
#include "mccfr/node_matrix.h"
#include <random>
#include <unordered_map>
#include <vector>

using InfoSetKey = std::string;

class Trainer {
private:
  GameState *game;
  EquityModule em;

  // K is the maximum number of legal actions any info set will ever expose.
  // The current action abstraction in game_state.cpp emits at most ~5 actions
  // (FOLD, CHECK/CALL, BET/RAISE, ALLIN). 8 leaves headroom for tweaks without
  // forcing a rebuild.
  static constexpr int kMaxActions = 8;

  // Default flush cadence: how many top-level CFR traversals (one per
  // traverser-iteration) are accumulated before flushing to GPU.
  static constexpr int kDefaultBatchSize = 64;

  NodeMatrix node_matrix_;
  std::unordered_map<InfoSetKey, int> info_to_id_;
  int batch_size_ = kDefaultBatchSize;
  // DCFR (alpha, beta, gamma). Default = vanilla CFR (no discount). Trainer's
  // train() loop passes `flush_count` as the iteration counter to flush() so
  // the discount is applied per-flush, not per-CFR-traversal — keeping the
  // schedule independent of batch_size.
  DcfrParams dcfr_ = {1e30, 1e30, 0.0};

  double cfr(GameState &state, int player_id, double prob_traverser,
             std::vector<double> &reach, double prob_chance, std::mt19937 &gen,
             int depth = 0);

  std::vector<double> calculate_payoffs(GameState &state);

  double get_terminal_payoff(GameState &state, int player_id);

  // Helper functions for card dealing
  void deal_random_hole_cards(GameState &state, std::mt19937 &gen);
  void deal_random_community_cards(GameState &state, int num_cards,
                                   std::mt19937 &gen);

  // Look up an info-set id, creating a new NodeMatrix row if needed.
  int get_or_create_node_id(const std::string &info, int n_actions);

public:
  explicit Trainer(GameState *game);

  ~Trainer();

  void set_batch_size(int n) { batch_size_ = n > 0 ? n : 1; }
  void set_dcfr(DcfrParams p) { dcfr_ = p; }
  DcfrParams dcfr() const { return dcfr_; }

  // Train with a fixed configuration suitable for benchmarking against an
  // external solver. Set `randomize_config=false` for reproducible benchmarks
  // where every iteration uses the same (num_players, stack, blinds, abs).
  // A non-zero `seed` makes the run deterministic.
  void train(int iterations, int num_players = 2, unsigned seed = 0,
             BettingAbstraction abs = BettingAbstraction::LEGACY,
             bool randomize_config = true,
             double stack_bb = 100.0,
             double sb = 1.0,
             double bb = 2.0);

  std::vector<double> get_strategy(const std::string &info_set);

  Action get_action_recommendation(GameState &state, int player_id,
                                   std::vector<double> &probabilities);

  void save_to_file(const std::string &filename);
  void load_from_file(const std::string &filename);
};

#endif
