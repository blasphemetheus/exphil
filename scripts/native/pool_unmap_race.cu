// Standalone two-thread reproducer for the selective-scan workspace fault
// (2026-09-26). No XLA, no BEAM. Decides driver-bug vs ordering-rule:
//
//   thread A (training):  loop { cudaMallocAsync(ws, N, sA);
//                                 slow_write_kernel<<<sA>>>(ws);   // writes all of ws
//                                 cudaFreeAsync(ws, sA); }
//   thread B (embedding): loop { small D2H memcpy on sB; cudaStreamSynchronize(sB); }
//
// This is exactly the pre-fix kernel's allocation pattern with the eager
// embedding's synchronization pattern beside it. A fault (Xid 31 in dmesg,
// or cudaErrorIllegalAddress here) with textbook usage means the driver's
// pool released pages under a running kernel. A clean run over the full
// duration means the fault needs something XLA-side.
//
//   nvcc -O2 -arch=sm_120 pool_unmap_race.cu -o pool_unmap_race
//   ./pool_unmap_race [seconds=180] [workspace_mib=640] [release_threshold_max=0] [opportunistic=1] [gap_ms=0] [pressure_pct=0] [pool_peer=0]
//
// Exit 0 = no fault in the window; exit 3 = CUDA error (details printed).
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

// Touch every element several times so the kernel runs long enough for a
// sync on the other stream to land while it is in flight (~ms per pass).
__global__ void slow_write(float* ws, size_t n, int passes) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int p = 0; p < passes; ++p)
        for (size_t k = i; k < n; k += stride) ws[k] = ws[k] * 0.5f + (float)p;
}

static int gap_ms = 0;
static void thread_a(size_t bytes, int passes) {
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    size_t n = bytes / sizeof(float);
    while (!stop.load() && !failed.load()) {
        float* ws = nullptr;
        CHECK(cudaMallocAsync((void**)&ws, bytes, s));
        slow_write<<<1024, 256, 0, s>>>(ws, n, passes);
        CHECK(cudaGetLastError());
        CHECK(cudaFreeAsync(ws, s));
        a_iters.fetch_add(1);
        // XLA-like gap: the rest of the training step runs before the next call,
        // so the freed block sits idle in the pool long enough to be trimmed.
        if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
        // A second, smaller allocation on the same stream, as the second Mamba layer would do.
        float* ws2 = nullptr;
        CHECK(cudaMallocAsync((void**)&ws2, bytes / 2, s));
        slow_write<<<1024, 256, 0, s>>>(ws2, n / 2, passes);
        CHECK(cudaGetLastError());
        CHECK(cudaFreeAsync(ws2, s));
    }
    CHECK(cudaStreamSynchronize(s));
}

static int pool_peer = 0;
// A second pool user on its own stream (XLA and its libraries also take
// stream-ordered allocations): small allocate/touch/free cycles.
static void thread_c() {
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    while (!stop.load() && !failed.load()) {
        void* p = nullptr;
        CHECK(cudaMallocAsync(&p, 1 << 20, s));
        CHECK(cudaMemsetAsync(p, 1, 1 << 20, s));
        CHECK(cudaFreeAsync(p, s));
        CHECK(cudaStreamSynchronize(s));
    }
}

static void thread_b() {
    cudaStream_t s;
    CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    float* dev = nullptr;
    float* host = nullptr;
    CHECK(cudaMalloc((void**)&dev, 1 << 20));
    CHECK(cudaMallocHost((void**)&host, 1 << 20));
    while (!stop.load() && !failed.load()) {
        // The eager embedding: small device work, a host transfer, a sync.
        CHECK(cudaMemsetAsync(dev, 0, 1 << 20, s));
        CHECK(cudaMemcpyAsync(host, dev, 1 << 20, cudaMemcpyDeviceToHost, s));
        CHECK(cudaStreamSynchronize(s));
        b_iters.fetch_add(1);
    }
}

int main(int argc, char** argv) {
    int seconds = argc > 1 ? atoi(argv[1]) : 180;
    size_t mib = argc > 2 ? (size_t)atoll(argv[2]) : 640;
    int threshold_max = argc > 3 ? atoi(argv[3]) : 0;
    int opportunistic = argc > 4 ? atoi(argv[4]) : 1;
    gap_ms = argc > 5 ? atoi(argv[5]) : 0;
    int pressure_pct = argc > 6 ? atoi(argv[6]) : 0;
    pool_peer = argc > 7 ? atoi(argv[7]) : 0;
    int passes = 8;
    // XLA-like pressure: hold a large plain cudaMalloc reservation (EXLA keeps 45%).
    void* reservation = nullptr;
    if (pressure_pct > 0) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        size_t want = total_b / 100 * pressure_pct;
        if (cudaMalloc(&reservation, want) != cudaSuccess) { puts("reservation failed"); return 2; }
        printf("reserved %zu MiB\n", want >> 20);
    }

    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) != cudaSuccess) { puts("no default pool"); return 2; }
    if (threshold_max) {
        unsigned long long max = ~0ULL;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &max);
    }
    int opp = opportunistic ? 1 : 0;
    cudaMemPoolSetAttribute(pool, cudaMemPoolReuseAllowOpportunistic, &opp);
    printf("pool_unmap_race: %d s, workspace %zu MiB, release_threshold_max=%d, opportunistic=%d, gap_ms=%d, pressure_pct=%d, pool_peer=%d\n",
           seconds, mib, threshold_max, opportunistic, gap_ms, pressure_pct, pool_peer);

    std::thread ta(thread_a, mib << 20, passes);
    std::thread tb(thread_b);
    std::thread tc;
    if (pool_peer) tc = std::thread(thread_c);
    auto t0 = std::chrono::steady_clock::now();
    while (!failed.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("  %.0fs  A(alloc/kernel/free) %ld  B(sync) %ld\n", el, a_iters.load(), b_iters.load());
        fflush(stdout);
        if (el >= seconds) break;
    }
    stop.store(true);
    ta.join();
    tb.join();
    if (tc.joinable()) tc.join();
    cudaError_t last = cudaDeviceSynchronize();
    if (failed.load() || last != cudaSuccess) {
        printf("FAULT: %s\n", cudaGetErrorString(last));
        return 3;
    }
    puts("CLEAN: no fault in the window");
    return 0;
}
