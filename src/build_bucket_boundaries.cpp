// build_bucket_boundaries: precompute per-street quantile cutoffs over
// OMP hand-rank values for use by EquityModule's V2 bucketing.
//
// Output format (binary, little-endian):
//   uint32_t magic = 'LBKT' (0x4C424B54)
//   uint32_t version = 1
//   for each of {flop, turn, river}:
//     int32_t  n_cutoffs           (=199 for 200 bins)
//     int32_t  cutoffs[n_cutoffs]  (sorted ascending; bin i = (cutoffs[i-1], cutoffs[i]])
//
// Strategy: sample N (hole_cards, board) tuples uniformly at random, run
// the OMP evaluator for each, sort the resulting values, take the (i/N_BINS)-
// quantile boundary for i in 1..N_BINS-1. With N=200k samples per street and
// OMP at ~270M evals/sec, the whole build is dominated by the random draw
// and runs in under a second.
//
// Usage:
//   ./build_bucket_boundaries [--samples 200000] [--out bucket_boundaries.dat]

#include "../include/external/omp/HandEvaluator.h"
#include "../include/external/omp/Hand.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kBucketMagic = 0x4C424B54u; // 'LBKT'
constexpr int kNumBins = 200;

void write_cutoffs(std::ofstream &out, const std::vector<int> &cutoffs) {
  int32_t n = static_cast<int32_t>(cutoffs.size());
  out.write(reinterpret_cast<const char *>(&n), sizeof(n));
  out.write(reinterpret_cast<const char *>(cutoffs.data()),
            n * sizeof(int));
}

std::vector<int> compute_cutoffs(const std::vector<int> &samples,
                                 int n_bins) {
  std::vector<int> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  std::vector<int> cutoffs;
  cutoffs.reserve(n_bins - 1);
  for (int b = 1; b < n_bins; ++b) {
    size_t idx = (sorted.size() * b) / n_bins;
    if (idx >= sorted.size()) idx = sorted.size() - 1;
    cutoffs.push_back(sorted[idx]);
  }
  return cutoffs;
}

// Sample N (hole_cards, n_board_cards) tuples, evaluate, return values.
std::vector<int> sample_values(const omp::HandEvaluator &E,
                               int n_samples, int n_board_cards,
                               std::mt19937 &gen) {
  std::vector<int> out;
  out.reserve(n_samples);

  // 52 cards 0..51. Draw 2 + n_board_cards distinct cards each iteration.
  // Reservoir-style swap to amortize.
  std::vector<int> deck(52);
  for (int i = 0; i < 52; ++i) deck[i] = i;

  for (int it = 0; it < n_samples; ++it) {
    int total_needed = 2 + n_board_cards;
    // Partial Fisher–Yates of the first `total_needed` slots.
    for (int i = 0; i < total_needed; ++i) {
      std::uniform_int_distribution<int> d(i, 51);
      int j = d(gen);
      std::swap(deck[i], deck[j]);
    }
    omp::Hand h = omp::Hand::empty();
    for (int i = 0; i < total_needed; ++i) {
      h += omp::Hand(static_cast<unsigned>(deck[i]));
    }
    out.push_back(E.evaluate(h));
  }
  return out;
}
} // namespace

int main(int argc, char **argv) {
  int n_samples = 200000;
  std::string out_path = "bucket_boundaries.dat";
  uint32_t seed = 12345;
  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--samples" && i + 1 < argc) n_samples = std::atoi(argv[++i]);
    else if (s == "--out" && i + 1 < argc) out_path = argv[++i];
    else if (s == "--seed" && i + 1 < argc) seed = std::atoi(argv[++i]);
    else {
      std::cerr << "Usage: " << argv[0]
                << " [--samples N=200000] [--out PATH] [--seed N]\n";
      return 2;
    }
  }

  std::cerr << "[build_buckets] samples=" << n_samples
            << " out=" << out_path << " seed=" << seed << "\n";

  omp::HandEvaluator E;
  std::mt19937 gen(seed);

  std::cerr << "[build_buckets] sampling flop (5-card) ...\n";
  auto flop_vals  = sample_values(E, n_samples, 3, gen);
  std::cerr << "[build_buckets] sampling turn (6-card) ...\n";
  auto turn_vals  = sample_values(E, n_samples, 4, gen);
  std::cerr << "[build_buckets] sampling river (7-card) ...\n";
  auto river_vals = sample_values(E, n_samples, 5, gen);

  auto flop_cuts  = compute_cutoffs(flop_vals,  kNumBins);
  auto turn_cuts  = compute_cutoffs(turn_vals,  kNumBins);
  auto river_cuts = compute_cutoffs(river_vals, kNumBins);

  std::ofstream out(out_path, std::ios::binary);
  if (!out) {
    std::cerr << "[build_buckets] cannot write " << out_path << "\n";
    return 1;
  }
  uint32_t magic = kBucketMagic;
  uint32_t version = 1;
  out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char *>(&version), sizeof(version));
  write_cutoffs(out, flop_cuts);
  write_cutoffs(out, turn_cuts);
  write_cutoffs(out, river_cuts);

  std::cerr << "[build_buckets] wrote " << flop_cuts.size() << "/"
            << turn_cuts.size() << "/" << river_cuts.size()
            << " cutoffs (flop/turn/river) -> " << out_path << "\n";

  // Print sample distribution summary so the user can sanity check.
  auto print_summary = [](const char *name, const std::vector<int> &cuts) {
    if (cuts.empty()) return;
    std::cerr << "[build_buckets] " << name
              << " min=" << cuts.front()
              << " p25=" << cuts[cuts.size() / 4]
              << " p50=" << cuts[cuts.size() / 2]
              << " p75=" << cuts[(cuts.size() * 3) / 4]
              << " max=" << cuts.back() << "\n";
  };
  print_summary("flop ", flop_cuts);
  print_summary("turn ", turn_cuts);
  print_summary("river", river_cuts);
  return 0;
}
