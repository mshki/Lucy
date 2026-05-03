// CPU-only fallback for cuda_kernels.h. Same math as the kernels, sequential.
// Used when nvcc is unavailable so the trainer/main binary still builds.
//
// As of feat/dcfr these kernels also implement Discounted CFR (Brown &
// Sandholm AAAI 2019). The discount factors are computed once per flush
// (host-side) and applied to every regret/strategy_sum entry. With α=∞ and
// β=∞ the per-flush multipliers degenerate to ≈1.0 and the kernel reduces to
// vanilla CFR — preserving bit-exact backward compatibility for callers
// that don't opt into DCFR.
#include "cuda/cuda_kernels.h"

#include <cmath>
#include <stdexcept>

namespace {

// D(t, p) = t^p / (t^p + 1). Saturates to 1.0 as p → ∞ for any finite t,
// to 0.5 as p → 0, and depends nontrivially on t in between.
inline double dcfr_discount(double t, double p) {
  if (p >= 1e29) return 1.0;        // vanilla CFR: no discount
  if (p <= -1e29) return 0.0;       // CFR+ on the negative branch (clamp)
  double tp = std::pow(t, p);
  if (!std::isfinite(tp)) return 1.0; // numerical infinity
  return tp / (tp + 1.0);
}

} // namespace

void launch_apply_regret_deltas(std::vector<double> &host_regret_sum,
                                const std::vector<double> &host_deltas,
                                int n_rows, int max_actions,
                                int iteration, DcfrParams dcfr) {
  size_t total = static_cast<size_t>(n_rows) * max_actions;
  if (host_regret_sum.size() != total || host_deltas.size() != total) {
    throw std::invalid_argument("launch_apply_regret_deltas: size mismatch");
  }
  // Compute discount factors once per call — they're constant across all rows.
  double t = std::max(1.0, static_cast<double>(iteration));
  double dpos = dcfr_discount(t, dcfr.alpha);
  double dneg = dcfr_discount(t, dcfr.beta);
  bool needs_discount = (dpos != 1.0) || (dneg != 1.0);

  if (needs_discount) {
    for (size_t i = 0; i < total; ++i) {
      double r = host_regret_sum[i];
      double scaled = (r > 0.0) ? r * dpos : r * dneg;
      host_regret_sum[i] = scaled + host_deltas[i];
    }
  } else {
    // Hot path for vanilla CFR — same as before, no extra mul/branch.
    for (size_t i = 0; i < total; ++i) {
      host_regret_sum[i] += host_deltas[i];
    }
  }
}

void launch_regret_match(const std::vector<double> &host_regret_sum,
                        std::vector<double> &host_strategy,
                        std::vector<double> &host_strategy_sum,
                        const std::vector<int> &host_num_actions,
                        const std::vector<double> &host_realization_weights,
                        int n_rows, int max_actions,
                        int iteration, DcfrParams dcfr) {
  size_t mat = static_cast<size_t>(n_rows) * max_actions;
  if (host_regret_sum.size() != mat || host_strategy.size() != mat ||
      host_strategy_sum.size() != mat ||
      host_num_actions.size() != static_cast<size_t>(n_rows) ||
      host_realization_weights.size() != static_cast<size_t>(n_rows)) {
    throw std::invalid_argument("launch_regret_match: size mismatch");
  }

  // Strategy_sum discount factor: (t/(t+1))^γ. Constant per flush.
  double t = std::max(1.0, static_cast<double>(iteration));
  double sw = (dcfr.gamma == 0.0) ? 1.0 : std::pow(t / (t + 1.0), dcfr.gamma);
  bool needs_strategy_discount = (sw != 1.0);

  for (int i = 0; i < n_rows; ++i) {
    int row_offset = i * max_actions;
    int actions = host_num_actions[i];
    if (actions <= 0)
      continue;

    // Regret matching from the (now-discounted) regret_sum.
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
      // DCFR strategy_sum update: discount old contributions, add new.
      double old_sum = host_strategy_sum[idx];
      host_strategy_sum[idx] = (needs_strategy_discount ? old_sum * sw : old_sum)
                             + host_realization_weights[i] * host_strategy[idx];
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
