# Lucy

GPU-resident outcome-sampling MCCFR solver for heads-up no-limit Texas Hold'em.

## Build

Requires CUDA 12+, CMake 3.18+, a C++20 compiler.

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="60;70;75;80;86;89;90"
cmake --build build -j
```

CPU-only build (no CUDA): omit `-DCMAKE_CUDA_ARCHITECTURES`. Falls back to CPU stubs.

## Train

```
# Build the EHS² centroid table once
./build/bin/build_equity_buckets --samples 100000 --opp-samples 50

# V1 (10-bucket heuristic, fast)
./build/bin/PokerBotMAIF --train 50000 --gpu-traj 8192 \
  --abstraction fcpa --hand-abstraction v1 --cfr-variant dcfr \
  --device gpu --seed 42 --out lucy_v1.dat

# V3 (EHS² K-means clusters, stronger)
./build/bin/PokerBotMAIF --train 200000 --gpu-traj 8192 \
  --abstraction fcpa --hand-abstraction v3 --cfr-variant dcfr \
  --device gpu --seed 42 --out lucy_v3.dat
```

## Play

Interactive terminal:

```
./build/bin/PokerBotMAIF --interactive
```

JSON IPC (one-shot query per line on stdin):

```
./build/bin/PokerBotMAIF --serve lucy_v3.dat \
  --abstraction fcpa --hand-abstraction v3ir
```

Use `--hand-abstraction v1ir` for V1-trained GPU models, `v3ir` for V3-trained
GPU models. The `*ir` variants apply the GPU-side imperfect-recall key format.

## Browser coach demo

A multi-seat NLHE table where humans play and Lucy provides real-time advice
in the spectator view.

```
pip install flask
python -m demo.browser.server \
  --lucy-binary build/bin/PokerBotMAIF \
  --lucy-model lucy_v3.dat \
  --hand-abstraction v3ir \
  --port 5050
```

Open `http://localhost:5050`.

## Slumbot benchmark

Plays HU NLHE vs the public Slumbot 2017 bot at slumbot.com.

```
pip install requests
python -m bench.slumbot.slumbot_client \
  --lucy-binary build/bin/PokerBotMAIF \
  --lucy-model lucy_v3.dat \
  --num-hands 1000 \
  --out slumbot_result.json
```

## Layout

```
src/                 C++ / CUDA sources
include/             headers + bundled OMPEval / nlohmann/json / libdivide
bench/               Python benchmark harness (vs OpenSpiel and Slumbot)
demo/browser/        Flask + HTML coach UI
equity_buckets.dat   precomputed V3 EHS² K-means centroids
bucket_boundaries.dat precomputed V2 OMP-value quantile cutoffs
```
