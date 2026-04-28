#include "cuda/cuda_kernels.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void apply_regret_deltas_kernel(double *regret_sum,
                                           const double *deltas,
                                            int total_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_size)
    return;
  regret_sum[idx] += deltas[idx];
}

__global__ void regret_match_kernel(const double *regret_sum, double *strategy,
                                    double *strategy_sum,
                                    const int *num_actions,
                                    const double *realization_weights,
                                    int n_rows, int max_actions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_rows)
    return;

  int row_offset = i * max_actions;
  int actions = num_actions[i];
  if (actions <= 0)
    return;

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

  for (int a = actions; a < max_actions; ++a) {
    strategy[row_offset + a] = 0.0;
  }
}

__global__ void update_regret_sum_kernel(double *regret_sum,
                                         const double *action_utils,
                                         const double *node_utils,
                                         const double *scales,
                                         const int *num_actions, int n_rows,
                                          int max_actions) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_rows)
    return;

  int row_offset = i * max_actions;
  int actions = num_actions[i];
  if (actions <= 0)
    return;

  double node_util = node_utils[i];
  double scale = scales[i];

  for (int a = 0; a < actions; ++a) {
    int idx = row_offset + a;
    regret_sum[idx] += (action_utils[idx] - node_util) * scale;
  }
}

void launch_apply_regret_deltas(std::vector<double> &host_regret_sum,
                                const std::vector<double> &host_deltas,
                                int n_rows, int max_actions) {
  size_t total = static_cast<size_t>(n_rows) * max_actions;
  if (host_regret_sum.size() != total || host_deltas.size() != total) {
    throw std::invalid_argument("launch_apply_regret_deltas: size mismatch");
  }
  if (total == 0)
    return;

  size_t bytes = total * sizeof(double);

  double *d_regret_sum = nullptr;
  double *d_deltas = nullptr;
  CUDA_CHECK(cudaMalloc(&d_regret_sum, bytes));
  CUDA_CHECK(cudaMalloc(&d_deltas, bytes));

  CUDA_CHECK(cudaMemcpy(d_regret_sum, host_regret_sum.data(), bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_deltas, host_deltas.data(), bytes,
                        cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (static_cast<int>(total) + threads - 1) / threads;
  apply_regret_deltas_kernel<<<blocks, threads>>>(d_regret_sum, d_deltas,
                                                  static_cast<int>(total));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(host_regret_sum.data(), d_regret_sum, bytes,
                        cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_regret_sum));
  CUDA_CHECK(cudaFree(d_deltas));
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
  if (n_rows == 0)
    return;

  size_t mat_bytes = mat * sizeof(double);
  size_t row_bytes = static_cast<size_t>(n_rows) * sizeof(double);
  size_t row_int_bytes = static_cast<size_t>(n_rows) * sizeof(int);

  double *d_regret_sum = nullptr;
  double *d_strategy = nullptr;
  double *d_strategy_sum = nullptr;
  int *d_num_actions = nullptr;
  double *d_realization_weights = nullptr;

  CUDA_CHECK(cudaMalloc(&d_regret_sum, mat_bytes));
  CUDA_CHECK(cudaMalloc(&d_strategy, mat_bytes));
  CUDA_CHECK(cudaMalloc(&d_strategy_sum, mat_bytes));
  CUDA_CHECK(cudaMalloc(&d_num_actions, row_int_bytes));
  CUDA_CHECK(cudaMalloc(&d_realization_weights, row_bytes));

  CUDA_CHECK(cudaMemcpy(d_regret_sum, host_regret_sum.data(), mat_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_strategy_sum, host_strategy_sum.data(), mat_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_num_actions, host_num_actions.data(), row_int_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_realization_weights, host_realization_weights.data(),
                        row_bytes, cudaMemcpyHostToDevice));

  const int threads = 128;
  const int blocks = (n_rows + threads - 1) / threads;
  regret_match_kernel<<<blocks, threads>>>(d_regret_sum, d_strategy,
                                           d_strategy_sum, d_num_actions,
                                           d_realization_weights, n_rows,
                                           max_actions);
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
                          const std::vector<int> &host_num_actions, int n_rows,
                          int max_actions) {
  size_t mat = static_cast<size_t>(n_rows) * max_actions;
  if (host_regret_sum.size() != mat || host_action_utils.size() != mat ||
      host_node_utils.size() != static_cast<size_t>(n_rows) ||
      host_scales.size() != static_cast<size_t>(n_rows) ||
      host_num_actions.size() != static_cast<size_t>(n_rows)) {
    throw std::invalid_argument("launch_regret_update: size mismatch");
  }
  if (n_rows == 0)
    return;

  size_t mat_bytes = mat * sizeof(double);
  size_t row_bytes = static_cast<size_t>(n_rows) * sizeof(double);
  size_t row_int_bytes = static_cast<size_t>(n_rows) * sizeof(int);

  double *d_regret_sum = nullptr;
  double *d_action_utils = nullptr;
  double *d_node_utils = nullptr;
  double *d_scales = nullptr;
  int *d_num_actions = nullptr;

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
  update_regret_sum_kernel<<<blocks, threads>>>(d_regret_sum, d_action_utils,
                                                d_node_utils, d_scales,
                                                d_num_actions, n_rows,
                                                max_actions);
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
