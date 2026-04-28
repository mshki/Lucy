#ifndef NODE_MATRIX_H
#define NODE_MATRIX_H

#include <vector>

// Dense N x K host-side buffer for MCCFR info-set state. During a batch of CFR
// traversals the trainer pushes regret deltas and realization weights into
// per-batch accumulators; flush() applies them in bulk and refreshes the
// strategy cache via the GPU (or CPU fallback).
class NodeMatrix {
public:
  explicit NodeMatrix(int max_actions);

  // Allocate a new info-set row, return its integer id.
  int add_node(int n_actions);

  int num_nodes() const { return n_; }
  int max_actions() const { return k_; }
  int num_actions(int id) const { return num_actions_[id]; }

  // Read-only pointer to the cached strategy for `id` (length = max_actions).
  // For never-flushed nodes returns the uniform strategy installed at add_node.
  const double *strategy_row(int id) const { return &strategy_[id * k_]; }

  // Push (action_idx, delta) into the pending regret accumulator for `id`.
  void accumulate_regret(int id, int action, double delta);

  // Push realization weight into the pending strategy_sum accumulator for `id`.
  void accumulate_realization_weight(int id, double weight);

  // Apply all pending deltas and weights, refresh the strategy cache.
  void flush();

  // Per-node average strategy from strategy_sum (for inference / save).
  std::vector<double> average_strategy(int id) const;

  // Raw strategy_sum row (for save_to_file compatibility).
  std::vector<double> strategy_sum_row(int id) const;

  // Restore a strategy_sum row from saved data.
  void load_strategy_sum_row(int id, const std::vector<double> &row);

private:
  int n_ = 0; // current number of rows
  int k_;     // max actions across all rows (fixed)

  std::vector<double> regret_sum_;   // n_ * k_
  std::vector<double> strategy_;     // n_ * k_, cache used during traversal
  std::vector<double> strategy_sum_; // n_ * k_
  std::vector<int> num_actions_;     // n_

  // Per-batch accumulators, zeroed after each flush.
  std::vector<double> pending_deltas_;  // n_ * k_
  std::vector<double> pending_weights_; // n_

  void grow_to(int new_n);
};

#endif
