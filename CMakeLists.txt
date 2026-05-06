cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(PokerBotMAIF VERSION 1.0 LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/mccfr)
# OMPEval headers expect flat #include "HandEvaluator.h" style
include_directories(${PROJECT_SOURCE_DIR}/include/external/omp)

# Core C++ sources (game engine + trainer + node matrix host code)
set(SOURCES
    src/main.cpp
    src/equity.cpp
    src/risk_profiler.cpp
    src/game_state.cpp
    src/mccfr/trainer.cpp
    src/mccfr/node.cpp
    src/mccfr/node_matrix.cpp
    # OMPEval (zekyll/OMPEval, ISC) — fast 7-card hand evaluator + equity
    # calculator. Replaces the old brute-force evaluate_5/7_cards.
    src/external/omp/HandEvaluator.cpp
    src/external/omp/CardRange.cpp
    src/external/omp/CombinedRange.cpp
    src/external/omp/EquityCalculator.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
)

# Pick the kernel backend: real CUDA when available, CPU fallback otherwise.
if(CMAKE_CUDA_COMPILER)
    target_sources(${PROJECT_NAME} PRIVATE
        src/cuda/cuda_kernels.cu
        src/cuda/gpu_cfr.cu                # GPU-resident CFR traversal kernel
    )
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
else()
    target_sources(${PROJECT_NAME} PRIVATE
        src/cuda/cuda_kernels_cpu.cpp
        src/cuda/gpu_cfr_cpu_stub.cpp     # No-op stub when nvcc unavailable
    )
endif()

# Bucket-boundaries builder — V2 abstraction (OMP-value quantile cuts).
add_executable(build_bucket_boundaries
    src/build_bucket_boundaries.cpp
    src/external/omp/HandEvaluator.cpp
)
set_target_properties(build_bucket_boundaries PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
)

# Equity-bucket builder — V3 abstraction (EHS² + KMeans cluster centroids).
# Run once before training:
#   ./build/bin/build_equity_buckets --samples 100000 --opp-samples 50
add_executable(build_equity_buckets
    src/build_equity_buckets.cpp
    src/external/omp/HandEvaluator.cpp
)
set_target_properties(build_equity_buckets PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
)

# Demo executable (kernel sandbox) — only when CUDA is available.
if(CMAKE_CUDA_COMPILER)
    add_executable(demo_code demo_code.cu)
    target_compile_features(demo_code PRIVATE cxx_std_20)
    set_target_properties(demo_code PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
    )
endif()
