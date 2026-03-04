/*
 * ThunderKittens Linear Scan NIF
 *
 * Loads the compiled CUDA kernel (.so) via dlopen and launches
 * the linear scan kernel. Same NIF pattern as triton_scan_nif.c
 * but using CUDA runtime API instead of driver API.
 *
 * Build:
 *   See Makefile — builds kernel.cu → .so, then links this NIF against it.
 */

#include <erl_nif.h>
#include <dlfcn.h>
#include <string.h>
#include <stdio.h>
#include <cuda_runtime.h>

/* Function pointers loaded from the kernel .so */
typedef void (*tk_linear_scan_fn)(
    const float*, const float*, const float*,
    float*, int, int, int, cudaStream_t);
typedef void (*tk_synchronize_fn)(void);
typedef int (*tk_device_info_fn)(char*, int, int*, int*);

static void *kernel_lib = NULL;
static tk_linear_scan_fn kernel_scan = NULL;
static tk_synchronize_fn kernel_sync = NULL;
static tk_device_info_fn kernel_info = NULL;
static int initialized = 0;

/* ========================================================================== */
/* NIF Functions                                                               */
/* ========================================================================== */

static ERL_NIF_TERM nif_ping(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
    return enif_make_string(env, "pong from thunderkittens_scan_nif", ERL_NIF_LATIN1);
}

static ERL_NIF_TERM nif_device_info(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    if (!initialized || !kernel_info) {
        return enif_make_tuple2(
            env, enif_make_atom(env, "error"),
            enif_make_string(env, "kernel library not loaded", ERL_NIF_LATIN1));
    }

    char name[256];
    int sm_major, sm_minor;
    if (kernel_info(name, sizeof(name), &sm_major, &sm_minor) != 0) {
        return enif_make_tuple2(
            env, enif_make_atom(env, "error"),
            enif_make_string(env, "no CUDA device", ERL_NIF_LATIN1));
    }

    ERL_NIF_TERM name_term = enif_make_string(env, name, ERL_NIF_LATIN1);
    ERL_NIF_TERM sm_term = enif_make_tuple2(
        env, enif_make_int(env, sm_major), enif_make_int(env, sm_minor));
    ERL_NIF_TERM tk_capable = enif_make_atom(env, sm_major >= 8 ? "true" : "false");

    return enif_make_tuple4(
        env, enif_make_atom(env, "ok"),
        name_term, sm_term, tk_capable);
}

static ERL_NIF_TERM nif_linear_scan(ErlNifEnv *env, int argc,
                                     const ERL_NIF_TERM argv[]) {
    ErlNifBinary a_bin, b_bin, h0_bin;
    int batch, seq_len, hidden;
    const ERL_NIF_TERM *shape_tuple;
    int shape_arity;

    if (!initialized || !kernel_scan) {
        return enif_make_tuple2(
            env, enif_make_atom(env, "error"),
            enif_make_string(env, "kernel library not loaded", ERL_NIF_LATIN1));
    }

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

    /* Allocate GPU memory */
    void *a_dev, *b_dev, *h0_dev, *out_dev;
    cudaError_t err;

    err = cudaMalloc(&a_dev, total_3d * sizeof(float));
    if (err != cudaSuccess) goto cuda_error;

    err = cudaMalloc(&b_dev, total_3d * sizeof(float));
    if (err != cudaSuccess) { cudaFree(a_dev); goto cuda_error; }

    err = cudaMalloc(&h0_dev, total_2d * sizeof(float));
    if (err != cudaSuccess) { cudaFree(a_dev); cudaFree(b_dev); goto cuda_error; }

    err = cudaMalloc(&out_dev, total_3d * sizeof(float));
    if (err != cudaSuccess) { cudaFree(a_dev); cudaFree(b_dev); cudaFree(h0_dev); goto cuda_error; }

    /* Copy data to GPU */
    cudaMemcpy(a_dev, a_bin.data, total_3d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_dev, b_bin.data, total_3d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(h0_dev, h0_bin.data, total_2d * sizeof(float), cudaMemcpyHostToDevice);

    /* Launch kernel */
    kernel_scan((const float*)a_dev, (const float*)b_dev, (const float*)h0_dev,
                (float*)out_dev, batch, seq_len, hidden, 0);

    /* Synchronize */
    kernel_sync();

    /* Copy result back */
    ERL_NIF_TERM result_term;
    unsigned char *result_data = enif_make_new_binary(
        env, total_3d * sizeof(float), &result_term);

    cudaMemcpy(result_data, out_dev, total_3d * sizeof(float), cudaMemcpyDeviceToHost);

    /* Cleanup */
    cudaFree(a_dev);
    cudaFree(b_dev);
    cudaFree(h0_dev);
    cudaFree(out_dev);

    return result_term;

cuda_error:
    return enif_make_tuple2(
        env, enif_make_atom(env, "error"),
        enif_make_string(env, "CUDA memory allocation failed", ERL_NIF_LATIN1));
}

/* ========================================================================== */
/* NIF Lifecycle                                                               */
/* ========================================================================== */

static int nif_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    /* Try to load the kernel shared library */
    const char *paths[] = {
        "priv/native/libthunderkittens_kernel.so",
        "native/thunderkittens_scan/libthunderkittens_kernel.so",
        "../../native/thunderkittens_scan/libthunderkittens_kernel.so",
        NULL
    };

    for (int i = 0; paths[i] != NULL; i++) {
        kernel_lib = dlopen(paths[i], RTLD_NOW);
        if (kernel_lib) break;
    }

    if (!kernel_lib) {
        fprintf(stderr, "thunderkittens_scan_nif: kernel library not found\n");
        return 0;  /* Don't fail load, just mark as unavailable */
    }

    kernel_scan = (tk_linear_scan_fn)dlsym(kernel_lib, "tk_linear_scan");
    kernel_sync = (tk_synchronize_fn)dlsym(kernel_lib, "tk_synchronize");
    kernel_info = (tk_device_info_fn)dlsym(kernel_lib, "tk_device_info");

    if (!kernel_scan || !kernel_sync || !kernel_info) {
        fprintf(stderr, "thunderkittens_scan_nif: missing symbols in kernel library\n");
        dlclose(kernel_lib);
        kernel_lib = NULL;
        return 0;
    }

    initialized = 1;
    return 0;
}

static void nif_unload(ErlNifEnv *env, void *priv_data) {
    if (kernel_lib) {
        dlclose(kernel_lib);
        kernel_lib = NULL;
    }
    kernel_scan = NULL;
    kernel_sync = NULL;
    kernel_info = NULL;
    initialized = 0;
}

/* ========================================================================== */
/* NIF Table                                                                   */
/* ========================================================================== */

static ErlNifFunc nif_funcs[] = {
    {"ping", 0, nif_ping, 0},
    {"device_info_nif", 0, nif_device_info, 0},
    {"linear_scan_nif", 4, nif_linear_scan, ERL_NIF_DIRTY_JOB_CPU_BOUND}
};

ERL_NIF_INIT(Elixir.ExPhil.Native.ThunderKittensScan, nif_funcs, nif_load, NULL, NULL, nif_unload)
