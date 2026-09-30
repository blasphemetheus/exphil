// Standalone reproducer, v2: PJRT's stream topology (2026-09-30).
//
// v1 (pool_unmap_race.cu) put the syncing thread on its own stream and stayed
// clean in six variants. In the XLA process the eager embedding's kernels run
// on the SAME compute stream as training, interleaved with the backward
// kernels, and its host transfers wait on compute-stream events before the
// host syncs. v2 reproduces that shape:
//
//   compute stream C (shared):
//     thread A: loop { mallocAsync(ws, C); big_kernel<<<C>>>(ws); freeAsync(ws, C) }   x2 (two layers)
//     thread B: loop { small_kernel<<<C>>>(buf); eventRecord(e, C);
//                      streamWaitEvent(D2H, e); memcpyAsync(host<-buf, D2H); streamSynchronize(D2H);
//                      memcpyAsync(buf2<-host, H2D); eventRecord(e2, H2D); streamWaitEvent(C, e2) }
//   plus a large held cudaMalloc reservation (EXLA's BFC preallocation).
//
// A fault here (Xid 31 in dmesg / cudaErrorIllegalAddress) = the driver
// released the workspace pages under a running kernel with textbook usage
// in the topology XLA uses → driver bug, portable reproducer. Clean = the
// remaining factor is inside XLA's allocator interaction.
//
//   nvcc -O2 -arch=sm_120 pool_unmap_race_v2.cu -o pool_unmap_race_v2
//   ./pool_unmap_race_v2 [seconds=180] [workspace_mib=640] [release_threshold_max=0] [opportunistic=1] [pressure_pct=45]
//
// Exit 0 = clean in the window; 3 = CUDA error.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thread>

#define CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
    failed.store(true); return; } } while (0)

static std::atomic<bool> stop{false};
static std::atomic<bool> failed{false};
static std::atomic<long> a_iters{0}, b_iters{0};
static cudaStream_t compute, h2d, d2h;

__global__ void big_write(float* ws, size_t n, int passes) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int p = 0; p < passes; ++p)
        for (size_t k = i; k < n; k += stride) ws[k] = ws[k] * 0.5f + (float)p;
}

__global__ void small_op(float* buf, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) buf[i] = buf[i] * 1.0001f + 1.0f;
}

// Training: two "layers" per step, each mallocAsync → kernel → freeAsync on C.
static void thread_a(size_t bytes, int passes) {
    size_t n = bytes / sizeof(float);
    while (!stop.load() && !failed.load()) {
        for (int layer = 0; layer < 2; ++layer) {
            float* ws = nullptr;
            CHECK(cudaMallocAsync((void**)&ws, bytes, compute));
            big_write<<<1024, 256, 0, compute>>>(ws, n, passes);
            CHECK(cudaGetLastError());
            CHECK(cudaFreeAsync(ws, compute));
        }
        a_iters.fetch_add(1);
    }
    CHECK(cudaStreamSynchronize(compute));
}

// Eager embedding: small kernels on C, D2H gated by a C event then a host
// sync, then an H2D whose completion C waits on (PJRT's transfer pattern).
static void thread_b() {
    const size_t n = 1 << 18;  // 1 MiB of floats
    float *buf = nullptr, *buf2 = nullptr, *host = nullptr;
    cudaEvent_t e, e2;
    CHECK(cudaMalloc((void**)&buf, n * sizeof(float)));
    CHECK(cudaMalloc((void**)&buf2, n * sizeof(float)));
    CHECK(cudaMallocHost((void**)&host, n * sizeof(float)));
    CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&e2, cudaEventDisableTiming));
    while (!stop.load() && !failed.load()) {
        for (int k = 0; k < 8; ++k) {  // a handful of eager ops
            small_op<<<(unsigned)((n + 255) / 256), 256, 0, compute>>>(buf, n);
            CHECK(cudaGetLastError());
        }
        CHECK(cudaEventRecord(e, compute));
        CHECK(cudaStreamWaitEvent(d2h, e, 0));
        CHECK(cudaMemcpyAsync(host, buf, n * sizeof(float), cudaMemcpyDeviceToHost, d2h));
        CHECK(cudaStreamSynchronize(d2h));           // Nx.to_binary
        CHECK(cudaMemcpyAsync(buf2, host, n * sizeof(float), cudaMemcpyHostToDevice, h2d));
        CHECK(cudaEventRecord(e2, h2d));
        CHECK(cudaStreamWaitEvent(compute, e2, 0));  // next compute op consumes the upload
        b_iters.fetch_add(1);
    }
}

int main(int argc, char** argv) {
    int seconds = argc > 1 ? atoi(argv[1]) : 180;
    size_t mib = argc > 2 ? (size_t)atoll(argv[2]) : 640;
    int threshold_max = argc > 3 ? atoi(argv[3]) : 0;
    int opportunistic = argc > 4 ? atoi(argv[4]) : 1;
    int pressure_pct = argc > 5 ? atoi(argv[5]) : 45;
    int passes = 8;

    void* reservation = nullptr;
    if (pressure_pct > 0) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        size_t want = total_b / 100 * pressure_pct;
        if (cudaMalloc(&reservation, want) != cudaSuccess) { puts("reservation failed"); return 2; }
        printf("reserved %zu MiB (BFC stand-in)\n", want >> 20);
    }

    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) != cudaSuccess) { puts("no default pool"); return 2; }
    if (threshold_max) {
        unsigned long long max = ~0ULL;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &max);
    }
    int opp = opportunistic ? 1 : 0;
    cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowOpportunistic, &opp);

    if (cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&h2d, cudaStreamNonBlocking) != cudaSuccess ||
        cudaStreamCreateWithFlags(&d2h, cudaStreamNonBlocking) != cudaSuccess) {
        puts("stream creation failed");
        return 2;
    }

    printf("pool_unmap_race_v2: %d s, workspace %zu MiB, release_threshold_max=%d, opportunistic=%d, pressure_pct=%d\n",
           seconds, mib, threshold_max, opportunistic, pressure_pct);

    std::thread ta(thread_a, mib << 20, passes);
    std::thread tb(thread_b);
    auto t0 = std::chrono::steady_clock::now();
    while (!failed.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("  %.0fs  A(steps) %ld  B(eager rounds) %ld\n", el, a_iters.load(), b_iters.load());
        fflush(stdout);
        if (el >= seconds) break;
    }
    stop.store(true);
    ta.join();
    tb.join();
    cudaError_t last = cudaDeviceSynchronize();
    if (failed.load() || last != cudaSuccess) {
        printf("FAULT: %s\n", cudaGetErrorString(last));
        return 3;
    }
    puts("CLEAN: no fault in the window");
    return 0;
}
