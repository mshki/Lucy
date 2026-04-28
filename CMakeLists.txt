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

# Core C++ sources (game engine + trainer + node matrix host code)
set(SOURCES
    src/main.cpp
    src/equity.cpp
    src/risk_profiler.cpp
    src/game_state.cpp
    src/mccfr/trainer.cpp
    src/mccfr/node.cpp
    src/mccfr/node_matrix.cpp
)

add_executable(${PROJECT_NAME} ${SOURCES})

target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -pedantic)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/bin
)

# Pick the kernel backend: real CUDA when available, CPU fallback otherwise.
if(CMAKE_CUDA_COMPILER)
    target_sources(${PROJECT_NAME} PRIVATE src/cuda/cuda_kernels.cu)
    set_target_properties(${PROJECT_NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
    )
else()
    target_sources(${PROJECT_NAME} PRIVATE src/cuda/cuda_kernels_cpu.cpp)
endif()

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
