// CPU-only fallback for cuda_kernels.h. Same math as the kernels, sequential.
// Used when nvcc is unavailable so the trainer/main binary still builds.
#include "cuda/cuda_kernels.h"

#include <stdexcept>

void launch_apply_regret_deltas(std::vector<double> &host_regret_sum,
                                const std::vector<double> &host_deltas,
                                int n_rows, int max_actions) {
  size_t total = static_cast<size_t>(n_rows) * max_actions;
  if (host_regret_sum.size() != total || host_deltas.size() != total) {
    throw std::invalid_argument("launch_apply_regret_deltas: size mismatch");
  }
  for (size_t i = 0; i < total; ++i) {
    host_regret_sum[i] += host_deltas[i];
  }
}

void launch_regret_match(const std::vector<double> &host_regret_sum,
                        std::vector<double> &host_strategy,
                        std::vector<double> &host_strategy_sum,
                        const std::vector<int> &host_num_actions,
                        const std::vector<double> &host_realization_weights,
                        int n_rows, int max_actions) {
  size_t mat = static_cast<size_t>(n_rows) * max_actions;
  if (host_regret_sum.size() != mat || host_strategy.size() != mat ||
      host_strategy_sum.size() != mat ||
      host_num_actions.size() != static_cast<size_t>(n_rows) ||
      host_realization_weights.size() != static_cast<size_t>(n_rows)) {
    throw std::invalid_argument("launch_regret_match: size mismatch");
  }

  for (int i = 0; i < n_rows; ++i) {
    int row_offset = i * max_actions;
    int actions = host_num_actions[i];
    if (actions <= 0)
      continue;

    double normalizing_sum = 0.0;
    for (int a = 0; a < actions; ++a) {
      double pos = host_regret_sum[row_offset + a] > 0.0
                      ? host_regret_sum[row_offset + a]
                      : 0.0;
      host_strategy[row_offset + a] = pos;
      normalizing_sum += pos;
    }
    for (int a = 0; a < actions; ++a) {
      int idx = row_offset + a;
      if (normalizing_sum > 0.0) {
        host_strategy[idx] /= normalizing_sum;
      } else {
        host_strategy[idx] = 1.0 / static_cast<double>(actions);
      }
      host_strategy_sum[idx] +=
          host_realization_weights[i] * host_strategy[idx];
    }
    for (int a = actions; a < max_actions; ++a) {
      host_strategy[row_offset + a] = 0.0;
    }
  }
}

void launch_regret_update(std::vector<double> &host_regret_sum,
                          const std::vector<double> &host_action_utils,
                          const std::vector<double> &host_node_utils,
                          const std::vector<double> &host_scales,
                          const std::vector<int> &host_num_actions, int n_rows,
                          int max_actions) {
  size_t mat = static_cast<size_t>(n_rows) * max_actions;
  if (host_regret_sum.size() != mat || host_action_utils.size() != mat ||
      host_node_utils.size() != static_cast<size_t>(n_rows) ||
      host_scales.size() != static_cast<size_t>(n_rows) ||
      host_num_actions.size() != static_cast<size_t>(n_rows)) {
    throw std::invalid_argument("launch_regret_update: size mismatch");
  }
  for (int i = 0; i < n_rows; ++i) {
    int actions = host_num_actions[i];
    if (actions <= 0)
      continue;
    int row_offset = i * max_actions;
    double node_util = host_node_utils[i];
    double scale = host_scales[i];
    for (int a = 0; a < actions; ++a) {
      int idx = row_offset + a;
      host_regret_sum[idx] +=
          (host_action_utils[idx] - node_util) * scale;
    }
  }
}
