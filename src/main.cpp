#include "../include/game_state.h"
#include "../include/mccfr/trainer.h"
#include "../include/cuda/gpu_cfr.h"
#include "../include/external/json.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using nlohmann::json;
using namespace std;

// ============================================================================
// Card parsing (used by both interactive solver mode and the JSON serve mode)
// ============================================================================

static Card parse_card(const string &s) {
  if (s.length() != 2)
    return Card(Rank::TWO, Suit::CLUBS);

  Rank r;
  switch (s[0]) {
  case '2': r = Rank::TWO; break;
  case '3': r = Rank::THREE; break;
  case '4': r = Rank::FOUR; break;
  case '5': r = Rank::FIVE; break;
  case '6': r = Rank::SIX; break;
  case '7': r = Rank::SEVEN; break;
  case '8': r = Rank::EIGHT; break;
  case '9': r = Rank::NINE; break;
  case 'T': case 't': r = Rank::TEN; break;
  case 'J': case 'j': r = Rank::JACK; break;
  case 'Q': case 'q': r = Rank::QUEEN; break;
  case 'K': case 'k': r = Rank::KING; break;
  case 'A': case 'a': r = Rank::ACE; break;
  default:  r = Rank::TWO;
  }

  Suit suit;
  switch (s[1]) {
  case 'c': suit = Suit::CLUBS; break;
  case 'd': suit = Suit::DIAMONDS; break;
  case 'h': suit = Suit::HEARTS; break;
  case 's': suit = Suit::SPADES; break;
  default:  suit = Suit::CLUBS;
  }
  return Card(r, suit);
}

static std::vector<Card> parse_cards_line(const string &line) {
  std::vector<Card> cards;
  std::stringstream ss(line);
  std::string item;
  while (ss >> item) {
    cards.push_back(parse_card(item));
  }
  return cards;
}

static std::vector<Card> parse_cards_json(const json &arr) {
  std::vector<Card> out;
  for (const auto &v : arr) {
    out.push_back(parse_card(v.get<std::string>()));
  }
  return out;
}

// ============================================================================
// Interactive solver mode (preserved from original main.cpp, minus verbose UI)
// ============================================================================

static void solver_mode(Trainer &trainer) {
  RiskProfiler rp;
  EquityModule em;
  GameState game(&rp, &em);

  cout << "\n=== MCCFR Poker Solver (Manual Mode) ===\n";

  int num_players;
  double stack, sb, bb;
  cout << "Number of players: "; cin >> num_players;
  cout << "Stack size: ";        cin >> stack;
  cout << "Small Blind: ";       cin >> sb;
  cout << "Big Blind: ";         cin >> bb;
  cin.ignore(10000, '\n');

  game.init_game_setup(num_players, stack, sb, bb);

  while (true) {
    int d_pos = -1;
    cout << "\nEnter Dealer Position (0 to " << num_players - 1
         << ") [Enter for Rotation]: ";
    string input_pos;
    getline(cin, input_pos);
    if (!input_pos.empty()) {
      try { d_pos = stoi(input_pos); } catch (...) { d_pos = -1; }
    }

    cout << "\n--- New Hand ---\n";
    game.start_hand(d_pos);

    cout << "Enter Hero Cards (e.g. Ah Kd): ";
    string line; getline(cin, line);
    std::vector<Card> hero_cards = parse_cards_line(line);
    game.set_player_cards(0, hero_cards);

    while (!game.is_terminal()) {
      Player *p = game.get_current_player();
      double to_call = game.current_street_highest_bet - p->current_bet;

      cout << "\n--- Stage " << (int)game.stage << " | Pot " << game.pot_size
           << " | To call " << to_call << " ---\n";
      cout << "Action on player " << p->id
           << (p->is_human ? " (HERO)" : "") << "\n";

      if (p->is_human) {
        std::vector<double> probs;
        Action best = trainer.get_action_recommendation(game, p->id, probs);
        cout << "Solver suggests: type=" << (int)best.type
             << " amount=" << best.amount << "\n";
      }

      cout << "Enter action (f / c / b <amt> / a): ";
      string action_str; getline(cin, action_str);

      Action selected(p->id, ActionType::FOLD, 0);
      if (action_str == "f") {
        selected = Action(p->id, ActionType::FOLD);
      } else if (action_str == "c") {
        if (to_call == 0) selected = Action(p->id, ActionType::CHECK);
        else              selected = Action(p->id, ActionType::CALL, to_call);
      } else if (action_str == "a") {
        selected = Action(p->id, ActionType::ALLIN, p->stack);
      } else if (!action_str.empty() && action_str[0] == 'b') {
        double amt = stod(action_str.substr(2));
        if (game.current_street_highest_bet == 0)
          selected = Action(p->id, ActionType::BET, amt);
        else
          selected = Action(p->id, ActionType::RAISE, amt);
      }
      game.apply_action(selected, false);

      if (game.is_betting_round_over() && game.stage != Stage::SHOWDOWN) {
        game.next_street();
        cout << "Enter board cards: ";
        getline(cin, line);
        auto extra = parse_cards_line(line);
        auto board = game.community_cards;
        board.insert(board.end(), extra.begin(), extra.end());
        game.set_community_cards(board);
      }
    }

    cout << "Play another hand? (y/n): ";
    string ans; getline(cin, ans);
    if (ans != "y") break;
  }
}

// ============================================================================
// JSON serve mode  (used by the benchmark harness)
//
// Protocol: line-delimited JSON over stdin/stdout. One request -> one response.
//
// Request shape (all fields required):
//   {
//     "player_id": 0,
//     "num_players": 2,
//     "dealer": 0,                  # SB sits at (dealer+1)%n in Lucy.
//     "sb": 1.0, "bb": 2.0,
//     "starting_stack": 200.0,
//     "stage": "preflop|flop|turn|river",
//     "hole":  ["Ah","Kd"],         # 2 cards; querying player's hand
//     "board": ["..." , ...],       # 0/3/4/5 cards depending on stage
//     "history": [                  # OpenSpiel-style action sequence in order
//       {"player": 1, "action": 1},   # action ids match Lucy's FCPA ordering
//       ...                            # 0=fold, 1=check/call, 2=pot, 3=allin
//     ],
//     "abstraction": "fcpa"
//   }
//
// Response:
//   {
//     "ok": true,
//     "infoset": "...",
//     "legal_actions": [0,1,2,3],
//     "probabilities": [p0,p1,p2,p3], # aligned with legal_actions
//     "action": 2                      # sampled / argmax
//   }
//
// Or on error: { "ok": false, "error": "<msg>" }
// ============================================================================

static const char *stage_name(Stage s) {
  switch (s) {
  case Stage::PREFLOP:  return "preflop";
  case Stage::FLOP:     return "flop";
  case Stage::TURN:     return "turn";
  case Stage::RIVER:    return "river";
  case Stage::SHOWDOWN: return "showdown";
  default:              return "start";
  }
}

// Translate a single action ID to a concrete Action object given the
// current Lucy state. Encoding depends on the betting abstraction:
//
//   FCPA: stable {0=fold, 1=check/call, 2=pot, 3=allin}, matches
//         OpenSpiel's universal_poker fcpa action IDs.
//   STREET_RICH: ID is the index into get_legal_actions() at the current
//         state. Variable-length per state (agreed on by the harness via
//         the serve reply's `legal_actions` list).
//   LEGACY: use the index-into-legal-actions convention as well.
static Action action_from_id(GameState &state, int player_id, int action_id) {
  Player *p = state.get_player(player_id);
  double call_amt = state.current_street_highest_bet - p->current_bet;

  // FCPA stable mapping (matches OpenSpiel universal_poker fcpa).
  if (state.betting_abstraction == BettingAbstraction::FCPA) {
    switch (action_id) {
    case 0: return Action(player_id, ActionType::FOLD, 0);
    case 1:
      if (call_amt == 0) return Action(player_id, ActionType::CHECK, 0);
      return Action(player_id, ActionType::CALL,
                    std::min((double)p->stack, (double)call_amt));
    case 2: {
      if (call_amt == 0) {
        double pot = state.pot_size > 0 ? state.pot_size
                                         : state.big_blind_amount;
        return Action(player_id, ActionType::BET, p->current_bet + pot);
      }
      double pot = std::max((double)state.pot_size, state.big_blind_amount);
      double base = pot + call_amt;
      double raise_to = state.current_street_highest_bet + base;
      return Action(player_id, ActionType::RAISE, raise_to);
    }
    case 3:
      return Action(player_id, ActionType::ALLIN, p->stack);
    default:
      return Action(player_id, ActionType::FOLD, 0);
    }
  }

  // STREET_RICH / LEGACY: index-into-legal-actions semantics.
  int saved = state.current_player_index;
  state.current_player_index = player_id;
  auto legal = state.get_legal_actions();
  state.current_player_index = saved;

  if (legal.empty() || action_id < 0 || action_id >= (int)legal.size()) {
    return Action(player_id, ActionType::FOLD, 0);
  }
  return legal[action_id];
}

static void deal_board_for_stage(GameState &state, Stage target,
                                 const std::vector<Card> &board) {
  size_t needed = 0;
  switch (target) {
  case Stage::PREFLOP: needed = 0; break;
  case Stage::FLOP:    needed = 3; break;
  case Stage::TURN:    needed = 4; break;
  case Stage::RIVER:   needed = 5; break;
  default:             needed = 0;
  }
  std::vector<Card> chosen(board.begin(),
                           board.begin() + std::min(needed, board.size()));
  state.set_community_cards(chosen);
}

static int serve_mode(Trainer &trainer, BettingAbstraction abs,
                      HandAbstraction hand_abs) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr << "[lucy-serve] ready abstraction="
            << (abs == BettingAbstraction::FCPA ? "fcpa" : "legacy")
            << " hand="
            << (hand_abs == HandAbstraction::V3_EHS_CLUSTERS ? "v3"
                : hand_abs == HandAbstraction::V2_VALUE_QUANTILES ? "v2"
                : "v1")
            << "\n";

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    json req;
    try {
      req = json::parse(line);
    } catch (const std::exception &e) {
      json resp = {{"ok", false}, {"error", std::string("parse: ") + e.what()}};
      std::cout << resp.dump() << "\n";
      continue;
    }

    try {
      int player_id      = req.at("player_id").get<int>();
      int num_players    = req.at("num_players").get<int>();
      int dealer         = req.at("dealer").get<int>();
      double sb          = req.at("sb").get<double>();
      double bb          = req.at("bb").get<double>();
      double starting    = req.at("starting_stack").get<double>();
      std::string stage  = req.at("stage").get<std::string>();
      auto hole_cards    = parse_cards_json(req.at("hole"));
      auto board_cards   = parse_cards_json(req.at("board"));
      const auto &hist   = req.at("history");

      // Build a fresh GameState that mirrors the request.
      EquityModule em;
      GameState s(nullptr, &em);
      s.betting_abstraction = abs;
      s.hand_abstraction = hand_abs;
      s.init_game_setup(num_players, starting, sb, bb);
      s.start_hand(dealer);

      // Set hole cards for the *querying* player only — Lucy's info-set
      // bucketing is hero-only.
      s.set_player_cards(player_id, hole_cards);

      Stage target_stage = Stage::PREFLOP;
      if      (stage == "flop")  target_stage = Stage::FLOP;
      else if (stage == "turn")  target_stage = Stage::TURN;
      else if (stage == "river") target_stage = Stage::RIVER;

      // Replay history. After each action that closes a betting round we
      // advance the street and reveal the appropriate board prefix.
      for (const auto &h : hist) {
        int p_act = h.at("player").get<int>();
        int a_id  = h.at("action").get<int>();
        while (s.is_betting_round_over() && s.stage != Stage::SHOWDOWN
               && s.stage != target_stage) {
          s.next_street();
          deal_board_for_stage(s, s.stage, board_cards);
        }
        Action act = action_from_id(s, p_act, a_id);
        s.apply_action(act, true);
      }

      while (s.is_betting_round_over() && s.stage != Stage::SHOWDOWN
             && s.stage != target_stage) {
        s.next_street();
        deal_board_for_stage(s, s.stage, board_cards);
      }
      // Defensive: pad board to match request stage even if no transition
      // was triggered above.
      deal_board_for_stage(s, target_stage, board_cards);

      // Look up strategy.
      std::vector<double> probs;
      Action best = trainer.get_action_recommendation(s, player_id, probs);

      // Map Lucy's legal actions back to OpenSpiel-style FCPA IDs.
      auto legal = s.get_legal_actions();
      std::vector<int> legal_ids;
      legal_ids.reserve(legal.size());
      for (const auto &a : legal) {
        switch (a.type) {
        case ActionType::FOLD:  legal_ids.push_back(0); break;
        case ActionType::CHECK:
        case ActionType::CALL:  legal_ids.push_back(1); break;
        case ActionType::BET:
        case ActionType::RAISE: legal_ids.push_back(2); break;
        case ActionType::ALLIN: legal_ids.push_back(3); break;
        }
      }

      int chosen_id = 0;
      switch (best.type) {
      case ActionType::FOLD:  chosen_id = 0; break;
      case ActionType::CHECK:
      case ActionType::CALL:  chosen_id = 1; break;
      case ActionType::BET:
      case ActionType::RAISE: chosen_id = 2; break;
      case ActionType::ALLIN: chosen_id = 3; break;
      }

      json resp = {
        {"ok", true},
        {"infoset", s.compute_information_set(player_id)},
        {"legal_actions", legal_ids},
        {"probabilities", probs},
        {"action", chosen_id},
        {"stage", stage_name(s.stage)},
        {"pot", s.pot_size},
      };
      std::cout << resp.dump() << "\n";
    } catch (const std::exception &e) {
      json resp = {{"ok", false}, {"error", std::string("eval: ") + e.what()}};
      std::cout << resp.dump() << "\n";
    }
  }
  return 0;
}

// ============================================================================
// CLI parsing
// ============================================================================

struct Args {
  enum class Cmd { NONE, TRAIN, SERVE, INTERACTIVE } cmd = Cmd::NONE;
  int iterations = 0;
  int players = 2;
  unsigned seed = 0;
  std::string out_path = "poker_model.dat";
  std::string in_path  = "poker_model.dat";
  BettingAbstraction abstraction = BettingAbstraction::LEGACY;
  HandAbstraction hand_abstraction = HandAbstraction::V1_HEURISTIC_10;
  bool randomize_config = false;
  double stack_bb = 100.0;
  double sb = 1.0;
  double bb = 2.0;
  int batch_size = 64;

  // CFR variant — drives DcfrParams.
  // "vanilla" (CFR), "linear" (Linear CFR), "plus" (CFR+), "dcfr" (default).
  std::string cfr_variant = "vanilla";

  // MCCFR sampler — "external" or "outcome".
  std::string sampler = "external";
  double outcome_epsilon = 0.6;
  // Compute device — "cpu" (the default) or "gpu" (uses gpu_cfr engine,
  // GPU-resident outcome-sampling MCCFR with V1 hand abstraction).
  std::string device = "cpu";
  int gpu_traj_per_iter = 4096;
};

static void print_usage() {
  std::cerr <<
    "PokerBotMAIF — Lucy MCCFR poker solver\n"
    "Usage:\n"
    "  PokerBotMAIF --train <iters> [options]    Train a model\n"
    "  PokerBotMAIF --serve <model_path> [opts]  JSON IPC strategy server\n"
    "  PokerBotMAIF --interactive                Original interactive mode\n"
    "  PokerBotMAIF                              (deprecated) menu mode\n"
    "\n"
    "Options:\n"
    "  --players N            Number of players (default 2)\n"
    "  --seed N               PRNG seed (0 = nondeterministic)\n"
    "  --out PATH             Save model to PATH (train mode)\n"
    "  --abstraction MODE     legacy|fcpa|street-rich  (default legacy)\n"
    "                         legacy      = 5-bet-size old default (0.33p..2p,allin)\n"
    "                         fcpa        = OpenSpiel-compat 4-action {fold,call,pot,allin}\n"
    "                         street-rich = Slumbot-style street-specific 5-7 actions\n"
    "  --hand-abstraction H   v1|v2|v3  (default v1)\n"
    "                         v1 = legacy 10-bucket heuristic\n"
    "                         v2 = OMP-value quantile, 169/200/200/200 per street\n"
    "                         v3 = EHS² cluster centroids, 169/200/200/200 per street\n"
    "  --randomize-config     Use legacy randomized stack/players sampling\n"
    "  --stack-bb N           Fixed stack size in BB (default 100)\n"
    "  --sb N                 Small blind chips (default 1)\n"
    "  --bb N                 Big blind chips (default 2)\n"
    "  --batch-size N         Trainer flush cadence (default 64)\n"

    "  --cfr-variant V        vanilla|linear|plus|dcfr  (default vanilla)\n"

    "  --sampler S            MCCFR sampler: external|outcome  (default external)\n"
    "  --outcome-epsilon E    Exploration mixing for outcome-sampling (default 0.6)\n"
    "  --device D             cpu|gpu  (default cpu; gpu = GPU-resident outcome-sampling)\n"
    "  --gpu-traj N           Trajectories per GPU iteration (default 4096)\n"

    "\n";
}

static Args parse_args(int argc, char **argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto next = [&](const char *flag) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << flag << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (s == "--train") {
      a.cmd = Args::Cmd::TRAIN;
      a.iterations = std::stoi(next("--train"));
    } else if (s == "--serve") {
      a.cmd = Args::Cmd::SERVE;
      a.in_path = next("--serve");
    } else if (s == "--interactive") {
      a.cmd = Args::Cmd::INTERACTIVE;
    } else if (s == "--players")    a.players = std::stoi(next("--players"));
    else if (s == "--seed")         a.seed = (unsigned)std::stoul(next("--seed"));
    else if (s == "--out")          a.out_path = next("--out");
    else if (s == "--abstraction") {
      std::string v = next("--abstraction");
      if (v == "fcpa")             a.abstraction = BettingAbstraction::FCPA;
      else if (v == "legacy")      a.abstraction = BettingAbstraction::LEGACY;
      else if (v == "street-rich") a.abstraction = BettingAbstraction::STREET_RICH;
      else { std::cerr << "unknown abstraction: " << v << "\n"; std::exit(2); }
    }
    else if (s == "--hand-abstraction") {
      std::string v = next("--hand-abstraction");
      if (v == "v1")      a.hand_abstraction = HandAbstraction::V1_HEURISTIC_10;
      else if (v == "v2") a.hand_abstraction = HandAbstraction::V2_VALUE_QUANTILES;
      else if (v == "v3") a.hand_abstraction = HandAbstraction::V3_EHS_CLUSTERS;
      else { std::cerr << "unknown hand-abstraction: " << v << "\n"; std::exit(2); }
    }
    else if (s == "--randomize-config") a.randomize_config = true;
    else if (s == "--stack-bb")     a.stack_bb = std::stod(next("--stack-bb"));
    else if (s == "--sb")           a.sb = std::stod(next("--sb"));
    else if (s == "--bb")           a.bb = std::stod(next("--bb"));
    else if (s == "--batch-size")   a.batch_size = std::stoi(next("--batch-size"));

    else if (s == "--cfr-variant")  a.cfr_variant = next("--cfr-variant");

    else if (s == "--sampler")      a.sampler = next("--sampler");
    else if (s == "--outcome-epsilon") a.outcome_epsilon = std::stod(next("--outcome-epsilon"));
    else if (s == "--device")       a.device = next("--device");
    else if (s == "--gpu-traj")     a.gpu_traj_per_iter = std::stoi(next("--gpu-traj"));

    else if (s == "-h" || s == "--help") { print_usage(); std::exit(0); }
    else {
      std::cerr << "unknown arg: " << s << "\n"; print_usage(); std::exit(2);
    }
  }
  return a;
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char *argv[]) {
  Args a = parse_args(argc, argv);

  RiskProfiler rp;
  EquityModule em;
  GameState game(&rp, &em);
  game.betting_abstraction = a.abstraction;
  game.hand_abstraction = a.hand_abstraction;
  Trainer trainer(&game);
  trainer.set_batch_size(a.batch_size);

  // Map the --cfr-variant string to DcfrParams.
  // (alpha, beta, gamma):
  //   vanilla → (∞, ∞, 0)   no discount
  //   linear  → (∞, ∞, 1)   linear strategy averaging
  //   plus    → (∞, -∞, 1)  CFR+ (clamp negative regrets, linear avg)
  //   dcfr    → (1.5, 0, 2) Brown & Sandholm 2019 default
  if (a.cfr_variant == "vanilla") trainer.set_dcfr({1e30, 1e30, 0.0});
  else if (a.cfr_variant == "linear") trainer.set_dcfr({1e30, 1e30, 1.0});
  else if (a.cfr_variant == "plus") trainer.set_dcfr({1e30, -1e30, 1.0});
  else if (a.cfr_variant == "dcfr") trainer.set_dcfr({1.5, 0.0, 2.0});
  else {
    std::cerr << "unknown --cfr-variant: " << a.cfr_variant << "\n";
    return 2;
  }

  if (a.sampler == "external") trainer.set_sampler(SamplerType::ExternalSampling);
  else if (a.sampler == "outcome") trainer.set_sampler(SamplerType::OutcomeSampling);
  else {
    std::cerr << "unknown --sampler: " << a.sampler << "\n";
    return 2;
  }
  trainer.set_outcome_epsilon(a.outcome_epsilon);


  if (a.cmd == Args::Cmd::TRAIN) {
    auto t0 = std::chrono::steady_clock::now();

    if (a.device == "gpu") {
      // GPU-resident outcome-sampling MCCFR. V1 hand abstraction + FCPA
      // betting only in v0.5; richer abstractions are CPU-only for now.
      gpu_cfr::GpuCfrConfig gcfg;
      gcfg.num_players      = a.players;
      gcfg.small_blind      = a.sb;
      gcfg.big_blind        = a.bb;
      gcfg.starting_stack   = a.stack_bb * a.bb;
      gcfg.epsilon          = a.outcome_epsilon;
      gcfg.batch_size       = a.gpu_traj_per_iter;
      // Map --cfr-variant to DCFR knobs (match CPU side).
      if (a.cfr_variant == "vanilla") {
        gcfg.dcfr_alpha = 1e30; gcfg.dcfr_beta = 1e30; gcfg.dcfr_gamma = 0.0;
      } else if (a.cfr_variant == "linear") {
        gcfg.dcfr_alpha = 1e30; gcfg.dcfr_beta = 1e30; gcfg.dcfr_gamma = 1.0;
      } else if (a.cfr_variant == "plus") {
        gcfg.dcfr_alpha = 1e30; gcfg.dcfr_beta = -1e30; gcfg.dcfr_gamma = 1.0;
      } else /* dcfr */ {
        gcfg.dcfr_alpha = 1.5; gcfg.dcfr_beta = 0.0; gcfg.dcfr_gamma = 2.0;
      }

      auto *eng = gpu_cfr::gpu_cfr_create(gcfg);
      if (!eng) {
        std::cerr << "[lucy] --device gpu requested but engine creation "
                  << "failed (no CUDA?). Aborting.\n";
        return 1;
      }
      uint64_t seed64 = a.seed != 0 ? (uint64_t)a.seed : 0xCAFEBABEDEADBEEFULL;
      double secs_dev = gpu_cfr::gpu_cfr_train(eng, a.iterations, seed64);
      int n_info = gpu_cfr::gpu_cfr_num_infosets(eng);
      std::cerr << "[lucy-gpu] " << a.iterations << " iters x "
                << gcfg.batch_size << " traj/iter = "
                << ((long long)a.iterations * gcfg.batch_size)
                << " trajectories in " << secs_dev << "s ("
                << ((long long)a.iterations * gcfg.batch_size / secs_dev)
                << " traj/s); " << n_info << " infosets\n";
      gpu_cfr::gpu_cfr_save(eng, a.out_path);
      gpu_cfr::gpu_cfr_destroy(eng);
    } else {
      trainer.train(a.iterations, a.players, a.seed, a.abstraction,
                    a.randomize_config, a.stack_bb, a.sb, a.bb);
      trainer.save_to_file(a.out_path);
    }

    auto t1 = std::chrono::steady_clock::now();
    auto secs = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[lucy] elapsed " << std::fixed << std::setprecision(2)
              << secs << "s\n";
    std::cerr << "[lucy] saved -> " << a.out_path << "\n";
    return 0;
  }

  if (a.cmd == Args::Cmd::SERVE) {
    trainer.load_from_file(a.in_path);
    return serve_mode(trainer, a.abstraction, a.hand_abstraction);
  }

  if (a.cmd == Args::Cmd::INTERACTIVE) {
    trainer.load_from_file(a.in_path);
    solver_mode(trainer);
    return 0;
  }

  // Legacy menu (back-compat with original main).
  cout << "1. Train MCCFR\n2. Solver Mode\nSelect: ";
  int choice; cin >> choice; cin.ignore(10000, '\n');
  if (choice == 1) {
    int iter; cout << "Iterations: "; cin >> iter;
    trainer.train(iter, a.players, a.seed, a.abstraction);
    trainer.save_to_file(a.out_path);
  } else {
    trainer.load_from_file(a.in_path);
    solver_mode(trainer);
  }
  return 0;
}
