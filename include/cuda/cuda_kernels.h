#ifndef CUDA_KERNELS_H
#define CUDA_KERNELS_H

#include <vector>

// Discounted CFR parameters. Three knobs: (alpha, beta, gamma).
//   alpha  — discounts positive cumulative regrets
//   beta   — discounts negative cumulative regrets
//   gamma  — discounts old strategy_sum contributions
//
// Reference: Brown & Sandholm, AAAI 2019 "Solving Imperfect-Information
// Games via Discounted Regret Minimization."
//
// Common variants:
//   {INF, INF, 0}    — vanilla CFR  (no discount)
//   {INF, INF, 1}    — Linear CFR   (linear strategy averaging)
//   {INF, -INF, 1}   — CFR+         (RM+ clamp + linear avg)
//   {1.5, 0, 2}      — DCFR default (Brown & Sandholm 2019, also Pluribus)
//
// `iteration` is the current CFR iteration count t (>=1) used to compute the
// per-flush discount factors. With batched flushing, callers may pass either
// the running iteration count when the flush happens or t = (flushes_so_far).
// The latter makes batch_size irrelevant to the discount schedule.
struct DcfrParams {
  double alpha = 1.5;
  double beta = 0.0;
  double gamma = 2.0;
};

// Applies the DCFR regret update:
//   regret_sum[i,a] = regret_sum[i,a] * D⁺(t,α)   if regret_sum > 0
//                   = regret_sum[i,a] * D⁻(t,β)   if regret_sum ≤ 0
//   regret_sum[i,a] += deltas[i,a]
// where D⁺(t,α) = t^α/(t^α+1), D⁻(t,β) = t^β/(t^β+1). With α=∞, β=∞ the
// discounts are ≈1 and the call reduces to the original "add deltas".
void launch_apply_regret_deltas(std::vector<double> &host_regret_sum,
                                const std::vector<double> &host_deltas,
                                int n_rows, int max_actions,
                                int iteration = 1,
                                DcfrParams dcfr = {1e30, 1e30, 0.0});

// Regret matching: derives strategy from regret_sum, then accumulates
//   strategy_sum[i,a] = strategy_sum[i,a] * (t/(t+1))^γ
//                     + realization_weights[i] * strategy[i,a]
// Pads slots a >= num_actions[i] with 0 in `strategy`.
void launch_regret_match(const std::vector<double> &host_regret_sum,
                         std::vector<double> &host_strategy,
                         std::vector<double> &host_strategy_sum,
                         const std::vector<int> &host_num_actions,
                         const std::vector<double> &host_realization_weights,
                         int n_rows, int max_actions,
                         int iteration = 1,
                         DcfrParams dcfr = {1e30, 1e30, 0.0});

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
