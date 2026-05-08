#ifndef GPU_CFR_H
#define GPU_CFR_H

#include <cstdint>
#include <string>
#include <vector>

namespace gpu_cfr {

constexpr int kMaxActions = 4;
constexpr int kMaxDepth = 32;
constexpr int kHashCapacityLog2 = 20;
constexpr int kHashCapacity = 1 << kHashCapacityLog2;

enum class HandAbstraction : uint8_t {
  V1 = 0,
  V3 = 1,
};

constexpr int kV3PreflopBuckets = 169;
constexpr int kV3PostflopBuckets = 200;

struct GpuCfrConfig {
  int num_players       = 2;
  double small_blind    = 1.0;
  double big_blind      = 2.0;
  double starting_stack = 200.0;
  double epsilon        = 0.6;
  double dcfr_alpha     = 1.5;
  double dcfr_beta      = 0.0;
  double dcfr_gamma     = 2.0;
  int batch_size        = 4096;
  HandAbstraction hand_abstraction = HandAbstraction::V1;
  std::string equity_buckets_path  = "equity_buckets.dat";
  int ehs2_rollouts                = 50;
};

struct GpuCfrEngine;

GpuCfrEngine *gpu_cfr_create(const GpuCfrConfig &cfg);
void gpu_cfr_destroy(GpuCfrEngine *eng);

double gpu_cfr_train(GpuCfrEngine *eng, int num_iterations,
                     uint64_t base_seed);

int gpu_cfr_num_infosets(const GpuCfrEngine *eng);

bool gpu_cfr_save(GpuCfrEngine *eng, const std::string &path);
bool gpu_cfr_load(GpuCfrEngine *eng, const std::string &path);

struct DeviceQueryRequest {
  int player_id;
  int dealer;
  uint8_t hole_cards[2];
  uint8_t board_cards[5];
  uint8_t num_board;
  uint8_t stage;
  int num_history;
  uint8_t history_action[kMaxDepth];
  uint8_t history_player[kMaxDepth];
};
struct DeviceQueryResponse {
  bool found;
  uint64_t infoset_key;
  double probabilities[kMaxActions];
  int num_legal;
  int legal_actions[kMaxActions];
};
DeviceQueryResponse gpu_cfr_query(GpuCfrEngine *eng,
                                  const DeviceQueryRequest &req);

struct GpuCfrProfile {
  long long total_ns        = 0;
  long long traversal_ns    = 0;
  long long match_ns        = 0;
  long long strategy_dl_ns  = 0;
  long long sync_ns         = 0;
  long long iters           = 0;
  long long trajectories    = 0;
};
GpuCfrProfile gpu_cfr_profile(const GpuCfrEngine *eng);

} // namespace gpu_cfr

#endif
