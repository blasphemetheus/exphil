// Host-side CUDA fault injection for the selective-scan backward launcher.
// No GPU kernel is executed. Link with --wrap for the CUDA APIs below.
//
// History: before the 2026-09-25 fix the launcher ignored a failed
// cudaMallocAsync and launched anyway. Since 2026-09-26 the workspace is a
// cached, grow-only cudaMalloc buffer (one per device, freed stream-ordered
// only when it must grow) — the per-step mallocAsync/freeAsync churn was the
// reproducible Xid 31 crash under concurrent eager XLA work. This test pins
// that contract: allocation failure propagates without a launch, launch
// errors propagate, a same-size call reuses the buffer, a larger call frees
// the old one (stream-ordered) and allocates once, and a failed free on
// growth propagates without a launch.
#include <cstdio>
#include <cuda_runtime.h>
extern "C" int fused_selective_scan_backward_launch(cudaStream_t,
  const float*, const float*, const float*, const float*, const float*,
  const float*, float*, int, int, int, int);
static int launches = 0, frees = 0, mallocs = 0;
static cudaError_t allocation_error = cudaErrorMemoryAllocation;
static cudaError_t launch_error = cudaSuccess, free_error = cudaSuccess;
extern "C" cudaError_t __wrap_cudaGetDevice(int* device) { *device = 0; return cudaSuccess; }
extern "C" cudaError_t __wrap_cudaMalloc(void **ptr, size_t) {
  ++mallocs;
  *ptr = allocation_error == cudaSuccess ? (void*)0x1000 : nullptr;
  return allocation_error;
}
extern "C" cudaError_t __wrap_cudaMemsetAsync(void*, int, size_t, cudaStream_t) {
  return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaLaunchKernel(const void*, dim3, dim3, void**, size_t, cudaStream_t) {
  ++launches;
  return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaFreeAsync(void*, cudaStream_t) {
  ++frees;
  return free_error;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return launch_error; }
static int call(int batch) {
  static float dummy[1024] = {};
  return fused_selective_scan_backward_launch(nullptr, dummy, dummy, dummy,
    dummy, dummy, dummy, dummy, batch, 2, 4, 16);
}
int main() {
  int result = call(1);
  printf("allocation failure: result=%d expected=%d launches=%d frees=%d mallocs=%d\n",
    result, (int)cudaErrorMemoryAllocation, launches, frees, mallocs);
  if (!(result == cudaErrorMemoryAllocation && launches == 0 && frees == 0 && mallocs == 1)) return 1;

  allocation_error = cudaSuccess;
  launch_error = cudaErrorInvalidConfiguration;
  result = call(1);
  if (!(result == launch_error && launches == 1 && frees == 0 && mallocs == 2)) return 2;

  launch_error = cudaSuccess;
  result = call(1);  // same size: reuse, no malloc, no free
  if (!(result == cudaSuccess && launches == 2 && frees == 0 && mallocs == 2)) return 3;

  result = call(2);  // larger: free old (stream-ordered) + one malloc
  if (!(result == cudaSuccess && launches == 3 && frees == 1 && mallocs == 3)) return 4;

  free_error = cudaErrorInvalidValue;
  result = call(4);  // growth whose free fails: propagate, no launch, no malloc
  if (!(result == free_error && launches == 3 && frees == 2 && mallocs == 3)) return 5;

  puts("PASS allocation, launch, reuse, growth and cleanup errors behave");
  return 0;
}
