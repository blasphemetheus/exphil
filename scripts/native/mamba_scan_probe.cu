// Standalone invocation of the SAME backward kernel linked into EXLA.
// Build with -I../edifice/native/cuda; suitable for compute-sanitizer without BEAM.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "fused_selective_scan_backward.cu"

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 2; } } while (0)

int main(int argc, char **argv) {
  int batch = argc > 1 ? atoi(argv[1]) : 2;
  int time = argc > 2 ? atoi(argv[2]) : 8;
  int hidden = argc > 3 ? atoi(argv[3]) : 7;
  int state = 16;
  size_t bth = (size_t)batch * time * hidden, bts = (size_t)batch * time * state;
  size_t hs = (size_t)hidden * state;
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  float *x, *dt, *a, *b, *c, *dy, *out;
  CHECK(cudaMalloc(&x, bth * 4)); CHECK(cudaMalloc(&dt, bth * 4));
  CHECK(cudaMalloc(&a, hs * 4)); CHECK(cudaMalloc(&b, bts * 4));
  CHECK(cudaMalloc(&c, bts * 4)); CHECK(cudaMalloc(&dy, bth * 4));
  CHECK(cudaMalloc(&out, (2*bth + 2*bts) * 4));
  std::vector<float> vx(bth, 0.1f), vd(bth, 0.05f), va(hs, -0.5f);
  std::vector<float> vb(bts, 0.2f), vc(bts, 0.3f), vy(bth, 1.0f);
  CHECK(cudaMemcpy(x, vx.data(), bth*4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dt, vd.data(), bth*4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(a, va.data(), hs*4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b, vb.data(), bts*4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(c, vc.data(), bts*4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dy, vy.data(), bth*4, cudaMemcpyHostToDevice));
  int code = fused_selective_scan_backward_launch(stream, x, dt, a, b, c, dy,
    out, batch, time, hidden, state);
  CHECK((cudaError_t)code);
  CHECK(cudaStreamSynchronize(stream));
  std::vector<float> grads(2*bth+2*bts);
  CHECK(cudaMemcpy(grads.data(), out, grads.size()*4, cudaMemcpyDeviceToHost));
  for (float g : grads) if (!std::isfinite(g)) return 3;
  // Independent scalar derivative of sum(y) with respect to x at t=0.
  double expected = 0, decay = 1;
  for (int t = 0; t < time; ++t) {
    expected += state * 0.3 * 0.05 * 0.2 * decay;
    decay *= std::exp(-0.5 * 0.05);
  }
  if (std::abs(grads[0]-expected) > 2e-5) {
    fprintf(stderr, "gradient mismatch got=%g expected=%g\n", grads[0], expected);
    return 4;
  }
  printf("PASS B=%d T=%d H=%d grad_x0=%g\n", batch, time, hidden, grads[0]);
  CHECK(cudaFree(x)); CHECK(cudaFree(dt)); CHECK(cudaFree(a));
  CHECK(cudaFree(b)); CHECK(cudaFree(c)); CHECK(cudaFree(dy)); CHECK(cudaFree(out));
  CHECK(cudaStreamDestroy(stream));
  return 0;
}
