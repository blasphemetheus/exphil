/* LD_PRELOAD guard: turn unchecked CUDA async allocation failure into a
 * precise host-side report before a kernel receives an invalid workspace. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

int cudaMallocAsync(void **ptr, size_t size, void *stream) {
  typedef int (*alloc_fn)(void **, size_t, void *);
  alloc_fn real = (alloc_fn)dlsym(RTLD_NEXT, "cudaMallocAsync");
  if (!real) {
    void *handle = dlopen("libcudart.so.12", RTLD_NOW | RTLD_LOCAL);
    if (handle) real = (alloc_fn)dlsym(handle, "cudaMallocAsync");
  }
  if (!real) { fprintf(stderr, "MAMBA_AUDIT: missing cudaMallocAsync\n"); abort(); }
  int result = real(ptr, size, stream);
  if (result) {
    fprintf(stderr, "MAMBA_AUDIT: cudaMallocAsync failed code=%d bytes=%zu stream=%p\n",
            result, size, stream);
    fflush(stderr);
    abort();
  }
  return result;
}
