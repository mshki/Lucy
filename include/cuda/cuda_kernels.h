#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <vector>

// Adds host_deltas[i] into host_regret_sum[i] elementwise.
// Both vectors must be of length n_rows * max_actions.
void launch_apply_regret_deltas(std::vector<double> &host_regret_sum,
                                const std::vector<double> &host_deltas,
                                int n_rows, int max_actions);

// Regret matching: derives strategy from regret_sum, then accumulates
// strategy_sum[i, a] += realization_weights[i] * strategy[i, a].
// Pads padded action slots (a >= num_actions[i]) with 0 in `strategy`.
void launch_regret_match(const std::vector<double> &host_regret_sum,
                         std::vector<double> &host_strategy,
                         std::vector<double> &host_strategy_sum,
                         const std::vector<int> &host_num_actions,
                         const std::vector<double> &host_realization_weights,
                         int n_rows, int max_actions);

// Update used by the demo path: applies one (action_utils, node_util, scale)
// tuple per row in a single kernel call. Kept for the demo target; the trainer
// uses the deltas path instead since a node may be visited multiple times in a
// batch.
void launch_regret_update(std::vector<double> &host_regret_sum,
                          const std::vector<double> &host_action_utils,
                          const std::vector<double> &host_node_utils,
                          const std::vector<double> &host_scales,
                          const std::vector<int> &host_num_actions, int n_rows,
                          int max_actions);

#endif
