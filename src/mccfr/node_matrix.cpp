#include "mccfr/node_matrix.h"

#include "cuda/cuda_kernels.h"

#include <algorithm>
#include <stdexcept>

NodeMatrix::NodeMatrix(int max_actions) : k_(max_actions) {
  if (k_ <= 0) {
    throw std::invalid_argument("NodeMatrix: max_actions must be > 0");
  }
}

void NodeMatrix::grow_to(int new_n) {
  if (new_n <= n_)
    return;
  size_t new_mat = static_cast<size_t>(new_n) * k_;
  regret_sum_.resize(new_mat, 0.0);
  strategy_.resize(new_mat, 0.0);
  strategy_sum_.resize(new_mat, 0.0);
  num_actions_.resize(new_n, 0);
  pending_deltas_.resize(new_mat, 0.0);
  pending_weights_.resize(new_n, 0.0);
  n_ = new_n;
}

int NodeMatrix::add_node(int n_actions) {
  if (n_actions <= 0 || n_actions > k_) {
    throw std::invalid_argument("NodeMatrix::add_node: n_actions out of range");
  }
  int id = n_;
  grow_to(n_ + 1);
  num_actions_[id] = n_actions;

  // Seed strategy cache with uniform over valid actions so traversals before
  // the first flush pick reasonable opponent actions.
  double uniform = 1.0 / static_cast<double>(n_actions);
  int row_offset = id * k_;
  for (int a = 0; a < n_actions; ++a) {
    strategy_[row_offset + a] = uniform;
  }
  return id;
}

void NodeMatrix::accumulate_regret(int id, int action, double delta) {
  pending_deltas_[id * k_ + action] += delta;
}

void NodeMatrix::accumulate_realization_weight(int id, double weight) {
  pending_weights_[id] += weight;
}

void NodeMatrix::flush() {
  if (n_ == 0)
    return;

  launch_apply_regret_deltas(regret_sum_, pending_deltas_, n_, k_);
  launch_regret_match(regret_sum_, strategy_, strategy_sum_, num_actions_,
                      pending_weights_, n_, k_);

  std::fill(pending_deltas_.begin(), pending_deltas_.end(), 0.0);
  std::fill(pending_weights_.begin(), pending_weights_.end(), 0.0);
}

std::vector<double> NodeMatrix::average_strategy(int id) const {
  int actions = num_actions_[id];
  std::vector<double> avg(actions, 0.0);
  if (actions <= 0)
    return avg;
  int row_offset = id * k_;
  double sum = 0.0;
  for (int a = 0; a < actions; ++a) {
    sum += strategy_sum_[row_offset + a];
  }
  for (int a = 0; a < actions; ++a) {
    if (sum > 0.0) {
      avg[a] = strategy_sum_[row_offset + a] / sum;
    } else {
      avg[a] = 1.0 / static_cast<double>(actions);
    }
  }
  return avg;
}

std::vector<double> NodeMatrix::strategy_sum_row(int id) const {
  int actions = num_actions_[id];
  std::vector<double> row(actions, 0.0);
  int row_offset = id * k_;
  for (int a = 0; a < actions; ++a) {
    row[a] = strategy_sum_[row_offset + a];
  }
  return row;
}

void NodeMatrix::load_strategy_sum_row(int id, const std::vector<double> &row) {
  int actions = num_actions_[id];
  if (static_cast<int>(row.size()) != actions) {
    throw std::invalid_argument(
        "NodeMatrix::load_strategy_sum_row: size mismatch");
  }
  int row_offset = id * k_;
  for (int a = 0; a < actions; ++a) {
    strategy_sum_[row_offset + a] = row[a];
  }
}
