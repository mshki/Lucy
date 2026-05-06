// CPU fallback for gpu_cfr.h. Compiles when nvcc isn't available so the
// rest of Lucy can still build. The stubs return failure / no-op — the
// CLI in main.cpp checks if the engine is available before dispatching.
#include "cuda/gpu_cfr.h"

#include <cstdio>

namespace gpu_cfr {

GpuCfrEngine *gpu_cfr_create(const GpuCfrConfig &) {
  std::fprintf(stderr,
               "[gpu_cfr] this build was compiled without CUDA — GPU-CFR "
               "not available. Rebuild with -DLUCY_USE_CUDA=ON.\n");
  return nullptr;
}
void gpu_cfr_destroy(GpuCfrEngine *) {}
double gpu_cfr_train(GpuCfrEngine *, int, uint64_t) { return 0.0; }
int gpu_cfr_num_infosets(const GpuCfrEngine *) { return 0; }
bool gpu_cfr_save(GpuCfrEngine *, const std::string &) { return false; }
bool gpu_cfr_load(GpuCfrEngine *, const std::string &) { return false; }
DeviceQueryResponse gpu_cfr_query(GpuCfrEngine *, const DeviceQueryRequest &) {
  DeviceQueryResponse r{}; return r;
}
GpuCfrProfile gpu_cfr_profile(const GpuCfrEngine *) {
  GpuCfrProfile p; return p;
}

} // namespace gpu_cfr
