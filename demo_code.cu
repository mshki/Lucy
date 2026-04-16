#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#define N 4 // info sets (rows)
#define K 3 // max actions per info set (cols)

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

// One thread handles one info set row.
// strategy(i, a) = max(regret_sum(i, a), 0) / sum_pos_regret(i)
// fallback when no positive regret: uniform over valid actions in row i.
__global__ void regret_match_kernel(const double *regret_sum, double *strategy,
                                    double *strategy_sum,
                                    const int *num_actions,
                                    const double *realization_weights,
                                    int n_rows, int max_actions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_rows) {
    return;
  }

  int row_offset = i * max_actions;
  int actions = num_actions[i];
  if (actions <= 0) {
    return;
  }

  double normalizing_sum = 0.0;
  for (int a = 0; a < actions; ++a) {
    int idx = row_offset + a;
    double positive_regret = regret_sum[idx] > 0.0 ? regret_sum[idx] : 0.0;
    strategy[idx] = positive_regret;
    normalizing_sum += positive_regret;
  }

  for (int a = 0; a < actions; ++a) {
    int idx = row_offset + a;
    if (normalizing_sum > 0.0) {
      strategy[idx] /= normalizing_sum;
    } else {
      strategy[idx] = 1.0 / static_cast<double>(actions);
    }
    strategy_sum[idx] += realization_weights[i] * strategy[idx];
  }

  // Keep padded actions deterministic and unused.
  for (int a = actions; a < max_actions; ++a) {
    int idx = row_offset + a;
    strategy[idx] = 0.0;
  }
}

// Regret update for traverser nodes after utilities are computed on CPU/GPU.
// regret_sum(i, a) += (action_utils(i, a) - node_util(i)) * scale(i)
__global__ void update_regret_sum_kernel(double *regret_sum,
                                         const double *action_utils,
                                         const double *node_utils,
                                         const double *scales,
                                         const int *num_actions, int n_rows,
                                         int max_actions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_rows) {
    return;
  }

  int row_offset = i * max_actions;
  int actions = num_actions[i];
  if (actions <= 0) {
    return;
  }

  double node_util = node_utils[i];
  double scale = scales[i];

  for (int a = 0; a < actions; ++a) {
    int idx = row_offset + a;
    double regret = (action_utils[idx] - node_util) * scale;
    regret_sum[idx] += regret;
  }
}

void launch_regret_match(const std::vector<double> &host_regret_sum,
                         std::vector<double> &host_strategy,
                         std::vector<double> &host_strategy_sum,
                         const std::vector<int> &host_num_actions,
                         const std::vector<double> &host_realization_weights,
                         int n_rows, int max_actions) {
  if (host_regret_sum.size() != static_cast<size_t>(n_rows * max_actions) ||
      host_strategy.size() != static_cast<size_t>(n_rows * max_actions) ||
      host_strategy_sum.size() != static_cast<size_t>(n_rows * max_actions) ||
      host_num_actions.size() != static_cast<size_t>(n_rows) ||
      host_realization_weights.size() != static_cast<size_t>(n_rows)) {
    throw std::invalid_argument("launch_regret_match: invalid input sizes");
  }

  double *d_regret_sum = nullptr;
  double *d_strategy = nullptr;
  double *d_strategy_sum = nullptr;
  int *d_num_actions = nullptr;
  double *d_realization_weights = nullptr;

  size_t mat_bytes = static_cast<size_t>(n_rows * max_actions) * sizeof(double);
  size_t row_bytes = static_cast<size_t>(n_rows) * sizeof(double);
  size_t row_int_bytes = static_cast<size_t>(n_rows) * sizeof(int);

  CUDA_CHECK(cudaMalloc(&d_regret_sum, mat_bytes));
  CUDA_CHECK(cudaMalloc(&d_strategy, mat_bytes));
  CUDA_CHECK(cudaMalloc(&d_strategy_sum, mat_bytes));
  CUDA_CHECK(cudaMalloc(&d_num_actions, row_int_bytes));
  CUDA_CHECK(cudaMalloc(&d_realization_weights, row_bytes));

  CUDA_CHECK(cudaMemcpy(d_regret_sum, host_regret_sum.data(), mat_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_strategy, host_strategy.data(), mat_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_strategy_sum, host_strategy_sum.data(), mat_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_num_actions, host_num_actions.data(), row_int_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_realization_weights, host_realization_weights.data(),
                        row_bytes, cudaMemcpyHostToDevice));

  const int threads = 128;
  const int blocks = (n_rows + threads - 1) / threads;
  regret_match_kernel<<<blocks, threads>>>(
      d_regret_sum, d_strategy, d_strategy_sum, d_num_actions,
      d_realization_weights, n_rows, max_actions);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_strategy.data(), d_strategy, mat_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_strategy_sum.data(), d_strategy_sum, mat_bytes,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_regret_sum));
  CUDA_CHECK(cudaFree(d_strategy));
  CUDA_CHECK(cudaFree(d_strategy_sum));
  CUDA_CHECK(cudaFree(d_num_actions));
  CUDA_CHECK(cudaFree(d_realization_weights));
}

void launch_regret_update(std::vector<double> &host_regret_sum,
                          const std::vector<double> &host_action_utils,
                          const std::vector<double> &host_node_utils,
                          const std::vector<double> &host_scales,
                          const std::vector<int> &host_num_actions,
                          int n_rows, int max_actions) {
  if (host_regret_sum.size() != static_cast<size_t>(n_rows * max_actions) ||
      host_action_utils.size() != static_cast<size_t>(n_rows * max_actions) ||
      host_node_utils.size() != static_cast<size_t>(n_rows) ||
      host_scales.size() != static_cast<size_t>(n_rows) ||
      host_num_actions.size() != static_cast<size_t>(n_rows)) {
    throw std::invalid_argument("launch_regret_update: invalid input sizes");
  }

  double *d_regret_sum = nullptr;
  double *d_action_utils = nullptr;
  double *d_node_utils = nullptr;
  double *d_scales = nullptr;
  int *d_num_actions = nullptr;

  size_t mat_bytes = static_cast<size_t>(n_rows * max_actions) * sizeof(double);
  size_t row_bytes = static_cast<size_t>(n_rows) * sizeof(double);
  size_t row_int_bytes = static_cast<size_t>(n_rows) * sizeof(int);

  CUDA_CHECK(cudaMalloc(&d_regret_sum, mat_bytes));
  CUDA_CHECK(cudaMalloc(&d_action_utils, mat_bytes));
  CUDA_CHECK(cudaMalloc(&d_node_utils, row_bytes));
  CUDA_CHECK(cudaMalloc(&d_scales, row_bytes));
  CUDA_CHECK(cudaMalloc(&d_num_actions, row_int_bytes));

  CUDA_CHECK(cudaMemcpy(d_regret_sum, host_regret_sum.data(), mat_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_action_utils, host_action_utils.data(), mat_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_node_utils, host_node_utils.data(), row_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scales, host_scales.data(), row_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_num_actions, host_num_actions.data(), row_int_bytes,
                        cudaMemcpyHostToDevice));

  const int threads = 128;
  const int blocks = (n_rows + threads - 1) / threads;
  update_regret_sum_kernel<<<blocks, threads>>>(
      d_regret_sum, d_action_utils, d_node_utils, d_scales, d_num_actions,
      n_rows, max_actions);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_regret_sum.data(), d_regret_sum, mat_bytes,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_regret_sum));
  CUDA_CHECK(cudaFree(d_action_utils));
  CUDA_CHECK(cudaFree(d_node_utils));
  CUDA_CHECK(cudaFree(d_scales));
  CUDA_CHECK(cudaFree(d_num_actions));
}

int main() {
  try {
    const int n_rows = N;
    const int max_actions = K;

    std::vector<double> regret_sum = {
        0.5, -0.3, 0.1,  // row 0
        0.2, 0.8, -0.1,  // row 1
        -0.4, 0.6, 0.3,  // row 2
        -0.1, -0.2, -0.5 // row 3
    };
    std::vector<double> strategy(n_rows * max_actions, 0.0);
    std::vector<double> strategy_sum(n_rows * max_actions, 0.0);

    std::vector<int> num_actions = {3, 3, 3, 3};
    std::vector<double> realization_weights = {1.0, 1.0, 1.0, 1.0};

    launch_regret_match(regret_sum, strategy, strategy_sum, num_actions,
                        realization_weights, n_rows, max_actions);

    std::cout << "Strategy after regret matching:\n";
    for (int i = 0; i < n_rows; ++i) {
      std::cout << "row " << i << ": ";
      for (int a = 0; a < max_actions; ++a) {
        std::cout << strategy[i * max_actions + a] << " ";
      }
      std::cout << "\n";
    }

    // Example regret update inputs (typically computed by CFR traversal).
    std::vector<double> action_utils = {
        0.30, 0.10, 0.20, // row 0
        0.40, 0.55, 0.05, // row 1
        0.10, 0.80, 0.60, // row 2
        0.25, 0.15, 0.05  // row 3
    };
    std::vector<double> node_utils = {0.22, 0.46, 0.50, 0.16};
    std::vector<double> scales = {1.0, 1.0, 1.0, 1.0};

    launch_regret_update(regret_sum, action_utils, node_utils, scales,
                         num_actions, n_rows, max_actions);

    std::cout << "\nRegret sum after update:\n";
    for (int i = 0; i < n_rows; ++i) {
      std::cout << "row " << i << ": ";
      for (int a = 0; a < max_actions; ++a) {
        std::cout << regret_sum[i * max_actions + a] << " ";
      }
      std::cout << "\n";
    }
  } catch (const std::exception &ex) {
    std::cerr << "fatal: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
