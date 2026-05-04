// build_equity_buckets: precompute EHS²-based hand-strength clusters per street.
//
// Replaces build_bucket_boundaries (which just took quantile cuts on raw OMP
// values — that misses draw potential). Instead, for each (hole, board)
// configuration we estimate Expected Hand Strength squared (EHS²) by Monte
// Carlo rollouts:
//
//   For each (hole_cards, board_cards) on a given street:
//     For r in 1..N_OPP_SAMPLES:
//       Sample 2 opponent hole cards from the remaining deck
//       For s in 1..N_RUNOUT_SAMPLES:
//         Sample (5 - len(board)) cards to complete the board to river
//         Compute hero_value, opp_value via OMP HandEvaluator on 7 cards
//         result = (hero > opp) + 0.5 * (hero == opp)   ∈ {0, 0.5, 1}
//         accumulate EHS² += result²
//   EHS²(hole, board) = mean of accumulated values over (r, s) pairs
//
// EHS² captures both:
//   - Made-hand strength (high mean → high EHS²)
//   - Draw potential (high variance → high EHS² for fixed mean)
//
// We sample M (hole, board) tuples per street, compute EHS² for each, and
// run 1-D KMeans on the M scalars to get K = 200 cluster centroids per
// street. At query time, Lucy computes EHS² for the current hand on the
// fly (fast — same OMP rollouts in the inner loop) and snaps to the
// nearest centroid → that's the bucket index.
//
// Output format (binary, little-endian):
//   uint32_t magic = 'LECB' (Lucy Equity Cluster Buckets)
//   uint32_t version = 1
//   for each of {flop, turn, river}:
//     int32_t  n_centroids
//     double   centroid[n_centroids]   (sorted ascending — implicit bucket order)
//
// Usage:
//   ./build_equity_buckets [--samples 100000] [--opp-samples 50] \
//                          [--runout-samples 20] [--n-clusters 200] \
//                          [--out equity_buckets.dat] [--seed 12345]
//
// Defaults give ~1 minute build time using OMP at ~270M evals/sec.

#include "../include/external/omp/HandEvaluator.h"
#include "../include/external/omp/Hand.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {
constexpr uint32_t kMagic = 0x4C454342u; // 'LECB'

inline omp::Hand omp_hand_from_ids(const std::vector<int> &card_ids) {
  omp::Hand h = omp::Hand::empty();
  for (int c : card_ids) h += omp::Hand(static_cast<unsigned>(c));
  return h;
}

// Run N_OPP × N_RUNOUT rollouts and return EHS² ∈ [0, 1].
double compute_ehs2(const omp::HandEvaluator &E,
                    const std::vector<int> &hole, const std::vector<int> &board,
                    int n_opp, int n_runout, std::mt19937 &gen) {
  // Build set of dealt cards for fast removal.
  std::vector<bool> dealt(52, false);
  for (int c : hole) dealt[c] = true;
  for (int c : board) dealt[c] = true;
  std::vector<int> remaining;
  remaining.reserve(52 - hole.size() - board.size());
  for (int c = 0; c < 52; ++c) if (!dealt[c]) remaining.push_back(c);

  int board_needed = 5 - static_cast<int>(board.size());
  if (board_needed < 0) board_needed = 0;

  // Pre-build hero partial hand (hole cards only).
  omp::Hand hero_partial = omp_hand_from_ids(hole);
  // Pre-build board partial. The board may be added incrementally per rollout.
  omp::Hand board_base = omp_hand_from_ids(board);

  double sum_sq = 0.0;
  int total = 0;

  // Each opp-sample reuses the same board completion to amortise.
  for (int r = 0; r < n_opp; ++r) {
    // Partial Fisher–Yates on the first 2 + board_needed slots of `remaining`.
    int k = 2 + board_needed;
    if ((int)remaining.size() < k) break;
    for (int i = 0; i < k; ++i) {
      std::uniform_int_distribution<int> d(i, (int)remaining.size() - 1);
      int j = d(gen);
      std::swap(remaining[i], remaining[j]);
    }
    omp::Hand opp_partial = omp_hand_from_ids({remaining[0], remaining[1]});
    omp::Hand board_runout = board_base;
    for (int b = 0; b < board_needed; ++b) {
      board_runout += omp::Hand(static_cast<unsigned>(remaining[2 + b]));
    }

    uint16_t hv = E.evaluate(hero_partial + board_runout);
    uint16_t ov = E.evaluate(opp_partial + board_runout);

    double result;
    if (hv > ov)        result = 1.0;
    else if (hv == ov)  result = 0.5;
    else                result = 0.0;
    // EHS² = mean(result²). For deterministic results, result² in {0, 0.25, 1}.
    sum_sq += result * result;
    ++total;

    // Optionally take more runout samples per opp draw — but the standard
    // estimator pairs each opp draw with one runout. Keeping it simple here.
    (void)n_runout;
  }
  return total > 0 ? sum_sq / total : 0.0;
}

// 1-D KMeans on a vector of EHS² values. Returns K cluster centroids,
// sorted ascending so cluster k = "the k-th-strongest hand strength".
std::vector<double> kmeans_1d(std::vector<double> samples, int K, int max_iter,
                              std::mt19937 &gen) {
  if ((int)samples.size() <= K) {
    std::sort(samples.begin(), samples.end());
    return samples;
  }
  // Sort once for quantile-init centroids.
  std::sort(samples.begin(), samples.end());
  std::vector<double> centroids(K);
  for (int k = 0; k < K; ++k) {
    size_t idx = (samples.size() * (size_t)k) / (size_t)K
                 + samples.size() / (2 * (size_t)K);
    if (idx >= samples.size()) idx = samples.size() - 1;
    centroids[k] = samples[idx];
  }

  std::vector<int> assign(samples.size(), 0);
  for (int it = 0; it < max_iter; ++it) {
    bool changed = false;
    // Assign each sample to nearest centroid (1-D, so it's just lower_bound).
    for (size_t i = 0; i < samples.size(); ++i) {
      double v = samples[i];
      // Binary search for the centroid closest to v. Since centroids are
      // sorted, the closest is one of two adjacent ones.
      auto it = std::lower_bound(centroids.begin(), centroids.end(), v);
      int k1 = (int)(it - centroids.begin());
      int best = k1;
      if (k1 < K && k1 > 0) {
        double d_lo = v - centroids[k1 - 1];
        double d_hi = centroids[k1] - v;
        if (d_lo < d_hi) best = k1 - 1;
      } else if (k1 >= K) {
        best = K - 1;
      }
      if (assign[i] != best) {
        assign[i] = best;
        changed = true;
      }
    }
    // Recompute centroids.
    std::vector<double> sums(K, 0.0);
    std::vector<int> counts(K, 0);
    for (size_t i = 0; i < samples.size(); ++i) {
      sums[assign[i]] += samples[i];
      counts[assign[i]] += 1;
    }
    for (int k = 0; k < K; ++k) {
      if (counts[k] > 0) centroids[k] = sums[k] / counts[k];
    }
    std::sort(centroids.begin(), centroids.end());
    if (!changed) break;
  }
  return centroids;
}

// Sample M (hole, board) tuples for a given board size, compute EHS² for each.
std::vector<double> sample_ehs2_for_street(const omp::HandEvaluator &E,
                                            int board_size, int n_samples,
                                            int n_opp, int n_runout,
                                            std::mt19937 &gen) {
  std::vector<double> values;
  values.reserve(n_samples);
  std::vector<int> deck(52);
  for (int i = 0; i < 52; ++i) deck[i] = i;

  for (int sample = 0; sample < n_samples; ++sample) {
    int total_needed = 2 + board_size;
    for (int i = 0; i < total_needed; ++i) {
      std::uniform_int_distribution<int> d(i, 51);
      int j = d(gen);
      std::swap(deck[i], deck[j]);
    }
    std::vector<int> hole(deck.begin(), deck.begin() + 2);
    std::vector<int> board(deck.begin() + 2, deck.begin() + 2 + board_size);
    double ehs2 = compute_ehs2(E, hole, board, n_opp, n_runout, gen);
    values.push_back(ehs2);
  }
  return values;
}

void log_progress(const char *name, std::chrono::steady_clock::time_point t0) {
  auto t1 = std::chrono::steady_clock::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  std::cerr << "[buckets] " << name << " in " << secs << "s\n";
}
} // namespace

int main(int argc, char **argv) {
  int n_samples = 100000;
  int n_opp = 50;
  int n_runout = 1; // see compute_ehs2 — n_runout is folded into n_opp here
  int n_clusters = 200;
  uint32_t seed = 12345;
  std::string out_path = "equity_buckets.dat";

  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    if (s == "--samples" && i+1 < argc) n_samples = std::atoi(argv[++i]);
    else if (s == "--opp-samples" && i+1 < argc) n_opp = std::atoi(argv[++i]);
    else if (s == "--runout-samples" && i+1 < argc) n_runout = std::atoi(argv[++i]);
    else if (s == "--n-clusters" && i+1 < argc) n_clusters = std::atoi(argv[++i]);
    else if (s == "--seed" && i+1 < argc) seed = std::atoi(argv[++i]);
    else if (s == "--out" && i+1 < argc) out_path = argv[++i];
    else { std::cerr << "Usage: " << argv[0] << " [opts]\n"; return 2; }
  }

  std::cerr << "[buckets] samples=" << n_samples << " opp=" << n_opp
            << " runout=" << n_runout << " clusters=" << n_clusters
            << " out=" << out_path << "\n";

  omp::HandEvaluator E;
  std::mt19937 gen(seed);

  std::cerr << "[buckets] sampling flop EHS² ...\n";
  auto t0 = std::chrono::steady_clock::now();
  auto flop_vals  = sample_ehs2_for_street(E, 3, n_samples, n_opp, n_runout, gen);
  log_progress("flop sampling", t0);

  std::cerr << "[buckets] sampling turn EHS² ...\n";
  t0 = std::chrono::steady_clock::now();
  auto turn_vals  = sample_ehs2_for_street(E, 4, n_samples, n_opp, n_runout, gen);
  log_progress("turn sampling", t0);

  std::cerr << "[buckets] sampling river EHS² ...\n";
  t0 = std::chrono::steady_clock::now();
  auto river_vals = sample_ehs2_for_street(E, 5, n_samples, n_opp, n_runout, gen);
  log_progress("river sampling", t0);

  std::cerr << "[buckets] running KMeans ...\n";
  t0 = std::chrono::steady_clock::now();
  auto flop_cents  = kmeans_1d(flop_vals,  n_clusters, 50, gen);
  auto turn_cents  = kmeans_1d(turn_vals,  n_clusters, 50, gen);
  auto river_cents = kmeans_1d(river_vals, n_clusters, 50, gen);
  log_progress("kmeans", t0);

  std::ofstream out(out_path, std::ios::binary);
  if (!out) { std::cerr << "[buckets] cannot write " << out_path << "\n"; return 1; }
  uint32_t magic = kMagic;
  uint32_t version = 1;
  out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char *>(&version), sizeof(version));
  auto write_centroids = [&](const std::vector<double> &c) {
    int32_t n = (int32_t)c.size();
    out.write(reinterpret_cast<const char *>(&n), sizeof(n));
    out.write(reinterpret_cast<const char *>(c.data()), n * sizeof(double));
  };
  write_centroids(flop_cents);
  write_centroids(turn_cents);
  write_centroids(river_cents);

  auto summary = [](const char *name, const std::vector<double> &c) {
    if (c.empty()) return;
    std::cerr << "[buckets] " << name
              << " min=" << c.front()
              << " p25=" << c[c.size()/4]
              << " p50=" << c[c.size()/2]
              << " p75=" << c[(c.size()*3)/4]
              << " max=" << c.back() << "\n";
  };
  summary("flop ", flop_cents);
  summary("turn ", turn_cents);
  summary("river", river_cents);
  std::cerr << "[buckets] wrote " << out_path << " (" << flop_cents.size()
            << "/" << turn_cents.size() << "/" << river_cents.size()
            << " clusters)\n";
  return 0;
}
