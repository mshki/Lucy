#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 4 // info sets (rows)
#define K 3 // actions (max) per info set (cols)

// #define CUDA_CHECK(call) do {
//   cudaError_t err = (call);
//   if (err != cudaSuccess) {
//     printf("error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));
//     exit(1);
//   }
// } while(0)


/*
Currently

Node {
  vector<double> regret_sum; length = num actions
  vector<double> strategy; length = num actions
  vector<double> strat sum; length = num actions
}

all across N heap allocs, indexed by string key

we can represent instead 
regret_sum [N rows x K cols] one row per info set and one col per action
strategy [N rows x K cols] computed from regret sum each iter
strat sum [N rows x K cols] accumulated over iters

stored in mem as double[N*K], row major 
elem(i, a) is at index [i * K + a]

with 4 info sets and k = 3:

  regret_sum[] = {
  0.5, -0.3, 0.1,   // row 0: info set 0, actions 0 - 2
  0.2, 0.8, -0.1,   // row 1: info set 1, actions 0 - 2
  -0.4, 0.6, 0.3,   // row 2: info set 2, acitons 0 - 2
  -0.1, -0.2, -0.5  // row 3: info set 3, actions 0 - 2
  }

  we then have one meta data vector that tellsu s like num actions 

  num_actions[N] = { 3, 3, 3, 3} -- valid cols per row

  if an info set only had 2 actions then its 3rd col would be 0 for padding
  the kernel then uses num actions[i] for real data

  the kernel is kinda like one thread per info set, all N threads run simultaneously
  we could use this to replace Node::get_strategy and the updating regret sum Node::update_regret_sum()
  for all info sets at once
*/

__global__ void regret_match(
    double* regret_sum,
    double* strategy,
    )
