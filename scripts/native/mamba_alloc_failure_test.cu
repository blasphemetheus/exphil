// Host-side CUDA fault injection. No GPU kernel is executed.
// Link with --wrap for the five CUDA APIs below. Before the fix, the wrapper
// launches a kernel after allocation fails and returns success: both are bugs.
#include <cstdio>
#include <cuda_runtime.h>
extern "C" int fused_selective_scan_backward_launch(cudaStream_t,
  const float*, const float*, const float*, const float*, const float*,
  const float*, float*, int, int, int, int);
static int launches = 0, frees = 0;
static cudaError_t allocation_error = cudaErrorMemoryAllocation;
static cudaError_t launch_error = cudaSuccess, free_error = cudaSuccess;
extern "C" cudaError_t __wrap_cudaMallocAsync(void **ptr, size_t, cudaStream_t) {
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
int main() {
  float dummy[1024] = {};
  int result = fused_selective_scan_backward_launch(nullptr, dummy, dummy, dummy,
    dummy, dummy, dummy, dummy, 1, 2, 4, 16);
  printf("allocation failure: result=%d expected=%d launches=%d frees=%d\n",
    result, (int)cudaErrorMemoryAllocation, launches, frees);
  if (!(result == cudaErrorMemoryAllocation && launches == 0 && frees == 0)) return 1;
  allocation_error = cudaSuccess;
  launch_error = cudaErrorInvalidConfiguration;
  result = fused_selective_scan_backward_launch(nullptr, dummy, dummy, dummy,
    dummy, dummy, dummy, dummy, 1, 2, 4, 16);
  if (!(result == launch_error && launches == 1 && frees == 1)) return 2;
  launch_error = cudaSuccess;
  free_error = cudaErrorInvalidValue;
  result = fused_selective_scan_backward_launch(nullptr, dummy, dummy, dummy,
    dummy, dummy, dummy, dummy, 1, 2, 4, 16);
  if (!(result == free_error && launches == 2 && frees == 2)) return 3;
  puts("PASS allocation, launch, and cleanup errors propagate");
  return 0;
}
