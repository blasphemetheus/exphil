/*
 * Triton AOT Linear Scan NIF
 *
 * Loads a pre-compiled cubin (from Triton AOT compilation) and launches
 * the linear scan kernel via CUDA driver API. Zero Python at runtime.
 *
 * Build:
 *   python3 compile_aot.py   # generates .cubin + metadata header
 *   gcc -shared -fPIC -o triton_scan_nif.so triton_scan_nif.c \
 *       -I${ERL_INCLUDE} -lcuda
 */

#include <erl_nif.h>
#include <cuda.h>
#include <string.h>
#include <stdio.h>

#include "linear_scan_kernel_meta.h"

/* CUDA module + function — loaded once from cubin on first call */
static CUmodule cuda_module = NULL;
static CUfunction cuda_function = NULL;
static CUcontext cuda_context = NULL;
static int cuda_initialized = 0;

/* Embedded cubin (loaded from file at NIF load time) */
static unsigned char *cubin_data = NULL;
static size_t cubin_size = 0;

/* ========================================================================== */
/* CUDA Initialization                                                         */
/* ========================================================================== */

static int init_cuda(void) {
    CUresult res;
    CUdevice device;

    if (cuda_initialized) return 0;

    res = cuInit(0);
    if (res != CUDA_SUCCESS) return -1;

    res = cuDeviceGet(&device, 0);
    if (res != CUDA_SUCCESS) return -1;

    res = cuCtxCreate(&cuda_context, 0, device);
    if (res != CUDA_SUCCESS) return -1;

    if (!cubin_data || cubin_size == 0) return -1;

    res = cuModuleLoadData(&cuda_module, cubin_data);
    if (res != CUDA_SUCCESS) {
        cuCtxDestroy(cuda_context);
        cuda_context = NULL;
        return -1;
    }

    res = cuModuleGetFunction(&cuda_function, cuda_module, TRITON_KERNEL_NAME);
    if (res != CUDA_SUCCESS) {
        cuModuleUnload(cuda_module);
        cuCtxDestroy(cuda_context);
        cuda_module = NULL;
        cuda_context = NULL;
        return -1;
    }

    cuda_initialized = 1;
    return 0;
}

/* ========================================================================== */
/* NIF Functions                                                               */
/* ========================================================================== */

/**
 * ping() -> "pong from triton_scan_nif"
 */
static ERL_NIF_TERM nif_ping(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
    return enif_make_string(env, "pong from triton_scan_nif", ERL_NIF_LATIN1);
}

/**
 * linear_scan_nif(a_bin, b_bin, h0_bin, {batch, seq_len, hidden}) -> result_bin
 *
 * a_bin:  f32 binary [batch * seq_len * hidden]
 * b_bin:  f32 binary [batch * seq_len * hidden]
 * h0_bin: f32 binary [batch * hidden]
 *
 * Returns: f32 binary [batch * seq_len * hidden]
 */
static ERL_NIF_TERM nif_linear_scan(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    ErlNifBinary a_bin, b_bin, h0_bin;
    int batch, seq_len, hidden;
    const ERL_NIF_TERM *shape_tuple;
    int shape_arity;

    /* Parse arguments */
    if (!enif_inspect_binary(env, argv[0], &a_bin) ||
        !enif_inspect_binary(env, argv[1], &b_bin) ||
        !enif_inspect_binary(env, argv[2], &h0_bin) ||
        !enif_get_tuple(env, argv[3], &shape_arity, &shape_tuple) ||
        shape_arity != 3 ||
        !enif_get_int(env, shape_tuple[0], &batch) ||
        !enif_get_int(env, shape_tuple[1], &seq_len) ||
        !enif_get_int(env, shape_tuple[2], &hidden)) {
        return enif_make_badarg(env);
    }

    int64_t total_3d = (int64_t)batch * seq_len * hidden;
    int64_t total_2d = (int64_t)batch * hidden;

    /* Validate binary sizes */
    if ((int64_t)a_bin.size != total_3d * (int64_t)sizeof(float) ||
        (int64_t)b_bin.size != total_3d * (int64_t)sizeof(float) ||
        (int64_t)h0_bin.size != total_2d * (int64_t)sizeof(float)) {
        return enif_make_tuple2(
            env, enif_make_atom(env, "error"),
            enif_make_string(env, "binary size mismatch", ERL_NIF_LATIN1));
    }

    /* Initialize CUDA on first call */
    if (!cuda_initialized) {
        if (init_cuda() != 0) {
            return enif_make_tuple2(
                env, enif_make_atom(env, "error"),
                enif_make_string(env, "CUDA initialization failed", ERL_NIF_LATIN1));
        }
    }

    CUresult res;

    /* Allocate GPU memory */
    CUdeviceptr a_dev, b_dev, h0_dev, out_dev;

    res = cuMemAlloc(&a_dev, total_3d * sizeof(float));
    if (res != CUDA_SUCCESS) goto cuda_error;

    res = cuMemAlloc(&b_dev, total_3d * sizeof(float));
    if (res != CUDA_SUCCESS) { cuMemFree(a_dev); goto cuda_error; }

    res = cuMemAlloc(&h0_dev, total_2d * sizeof(float));
    if (res != CUDA_SUCCESS) { cuMemFree(a_dev); cuMemFree(b_dev); goto cuda_error; }

    res = cuMemAlloc(&out_dev, total_3d * sizeof(float));
    if (res != CUDA_SUCCESS) { cuMemFree(a_dev); cuMemFree(b_dev); cuMemFree(h0_dev); goto cuda_error; }

    /* Copy data to GPU */
    cuMemcpyHtoD(a_dev, a_bin.data, total_3d * sizeof(float));
    cuMemcpyHtoD(b_dev, b_bin.data, total_3d * sizeof(float));
    cuMemcpyHtoD(h0_dev, h0_bin.data, total_2d * sizeof(float));
    cuMemsetD8(out_dev, 0, total_3d * sizeof(float));

    /* Launch kernel */
    /* Triton: each program = one CUDA block, tl.program_id(0) = blockIdx.x */
    /* One program per (batch, hidden) pair */
    int num_programs = batch * hidden;
    /* Block dim = num_warps * warp_size (Triton manages thread-level parallelism) */
    int threads_per_block = TRITON_NUM_WARPS * 32;

    void *params[] = {
        &a_dev, &b_dev, &h0_dev, &out_dev,
        &batch, &seq_len, &hidden
    };

    res = cuLaunchKernel(
        cuda_function,
        num_programs, 1, 1,         /* grid: one block per program */
        threads_per_block, 1, 1,    /* block: num_warps * warp_size */
        TRITON_SHARED_MEM,          /* shared memory */
        0,                          /* stream (default) */
        params,                     /* kernel params */
        NULL                        /* extra */
    );

    if (res != CUDA_SUCCESS) {
        cuMemFree(a_dev);
        cuMemFree(b_dev);
        cuMemFree(h0_dev);
        cuMemFree(out_dev);
        goto cuda_error;
    }

    /* Synchronize */
    cuCtxSynchronize();

    /* Copy result back */
    ERL_NIF_TERM result_term;
    unsigned char *result_data = enif_make_new_binary(
        env, total_3d * sizeof(float), &result_term);

    cuMemcpyDtoH(result_data, out_dev, total_3d * sizeof(float));

    /* Cleanup GPU memory */
    cuMemFree(a_dev);
    cuMemFree(b_dev);
    cuMemFree(h0_dev);
    cuMemFree(out_dev);

    return result_term;

cuda_error:
    return enif_make_tuple2(
        env, enif_make_atom(env, "error"),
        enif_make_string(env, "CUDA operation failed", ERL_NIF_LATIN1));
}

/* ========================================================================== */
/* NIF Lifecycle                                                               */
/* ========================================================================== */

static int nif_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    /* Load cubin from file — try several paths */
    FILE *fp = NULL;

    /* Try several paths */
    const char *paths[] = {
        "priv/native/linear_scan_kernel.cubin",
        "native/triton_scan/linear_scan_kernel.cubin",
        "../../native/triton_scan/linear_scan_kernel.cubin",
        NULL
    };

    for (int i = 0; paths[i] != NULL; i++) {
        fp = fopen(paths[i], "rb");
        if (fp) break;
    }

    if (!fp) {
        /* Last resort: try app dir */
        /* This is set during load by Elixir */
        return 0;  /* Don't fail load, just mark as unavailable */
    }

    fseek(fp, 0, SEEK_END);
    cubin_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    cubin_data = (unsigned char *)enif_alloc(cubin_size);
    if (!cubin_data) {
        fclose(fp);
        return 0;
    }

    if (fread(cubin_data, 1, cubin_size, fp) != cubin_size) {
        enif_free(cubin_data);
        cubin_data = NULL;
        cubin_size = 0;
        fclose(fp);
        return 0;
    }

    fclose(fp);
    return 0;
}

static void nif_unload(ErlNifEnv *env, void *priv_data) {
    if (cuda_function) cuda_function = NULL;

    if (cuda_module) {
        cuModuleUnload(cuda_module);
        cuda_module = NULL;
    }

    if (cuda_context) {
        cuCtxDestroy(cuda_context);
        cuda_context = NULL;
    }

    if (cubin_data) {
        enif_free(cubin_data);
        cubin_data = NULL;
        cubin_size = 0;
    }

    cuda_initialized = 0;
}

/* ========================================================================== */
/* NIF Table                                                                   */
/* ========================================================================== */

static ErlNifFunc nif_funcs[] = {
    {"ping", 0, nif_ping, 0},
    {"linear_scan_nif", 4, nif_linear_scan, ERL_NIF_DIRTY_JOB_CPU_BOUND}
};

ERL_NIF_INIT(Elixir.ExPhil.Native.TritonScan, nif_funcs, nif_load, NULL, NULL, nif_unload)
