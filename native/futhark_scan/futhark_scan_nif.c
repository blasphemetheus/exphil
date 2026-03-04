/*
 * Futhark Linear Scan NIF Wrapper
 *
 * Thin C wrapper that bridges Erlang NIF API to Futhark's generated C library.
 * Futhark compiles linear_scan.fut → linear_scan.{c,h}, which provides a
 * CUDA-backed parallel prefix scan.
 *
 * Build:
 *   futhark cuda --library linear_scan.fut   # generates linear_scan.{c,h}
 *   gcc -shared -fPIC -o futhark_scan_nif.so \
 *       futhark_scan_nif.c linear_scan.c      \
 *       -I${ERL_INCLUDE} -lcuda -lnvrtc -lcudart
 */

#include <erl_nif.h>
#include <string.h>
#include "linear_scan.h"

/* Futhark context — initialized once on NIF load */
static struct futhark_context_config *cfg = NULL;
static struct futhark_context *ctx = NULL;

/* ========================================================================== */
/* NIF Functions                                                               */
/* ========================================================================== */

/**
 * ping() -> "pong from futhark_scan_nif"
 */
static ERL_NIF_TERM nif_ping(ErlNifEnv *env, int argc,
                              const ERL_NIF_TERM argv[]) {
    return enif_make_string(env, "pong from futhark_scan_nif", ERL_NIF_LATIN1);
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

    /* Create Futhark 1D arrays from flat data */
    struct futhark_f32_1d *a_fut =
        futhark_new_f32_1d(ctx, (const float *)a_bin.data, total_3d);
    struct futhark_f32_1d *b_fut =
        futhark_new_f32_1d(ctx, (const float *)b_bin.data, total_3d);
    struct futhark_f32_1d *h0_fut =
        futhark_new_f32_1d(ctx, (const float *)h0_bin.data, total_2d);

    if (!a_fut || !b_fut || !h0_fut) {
        const char *msg = futhark_context_get_error(ctx);
        ERL_NIF_TERM error = enif_make_tuple2(
            env, enif_make_atom(env, "error"),
            enif_make_string(env, msg ? msg : "failed to create arrays", ERL_NIF_LATIN1));
        if (a_fut) futhark_free_f32_1d(ctx, a_fut);
        if (b_fut) futhark_free_f32_1d(ctx, b_fut);
        if (h0_fut) futhark_free_f32_1d(ctx, h0_fut);
        return error;
    }

    /* Call Futhark entry point */
    struct futhark_f32_1d *result_fut = NULL;
    int err = futhark_entry_linear_scan_flat(
        ctx, &result_fut,
        a_fut, b_fut, h0_fut,
        (int64_t)batch, (int64_t)seq_len, (int64_t)hidden);

    if (err != 0) {
        const char *msg = futhark_context_get_error(ctx);
        ERL_NIF_TERM error = enif_make_tuple2(
            env, enif_make_atom(env, "error"),
            enif_make_string(env, msg ? msg : "scan kernel failed", ERL_NIF_LATIN1));
        futhark_free_f32_1d(ctx, a_fut);
        futhark_free_f32_1d(ctx, b_fut);
        futhark_free_f32_1d(ctx, h0_fut);
        if (result_fut) futhark_free_f32_1d(ctx, result_fut);
        return error;
    }

    /* Synchronize (ensures GPU work is complete) */
    futhark_context_sync(ctx);

    /* Copy result to Erlang binary */
    ERL_NIF_TERM result_term;
    unsigned char *result_data = enif_make_new_binary(
        env, total_3d * sizeof(float), &result_term);

    err = futhark_values_f32_1d(ctx, result_fut, (float *)result_data);

    /* Need to sync again after values copy */
    futhark_context_sync(ctx);

    /* Cleanup */
    futhark_free_f32_1d(ctx, a_fut);
    futhark_free_f32_1d(ctx, b_fut);
    futhark_free_f32_1d(ctx, h0_fut);
    futhark_free_f32_1d(ctx, result_fut);

    if (err != 0) {
        return enif_make_tuple2(
            env, enif_make_atom(env, "error"),
            enif_make_string(env, "failed to read result", ERL_NIF_LATIN1));
    }

    return result_term;
}

/* ========================================================================== */
/* NIF Lifecycle                                                               */
/* ========================================================================== */

static int nif_load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {
    cfg = futhark_context_config_new();
    if (!cfg) return 1;

    ctx = futhark_context_new(cfg);
    if (!ctx) {
        futhark_context_config_free(cfg);
        cfg = NULL;
        return 1;
    }

    return 0;
}

static void nif_unload(ErlNifEnv *env, void *priv_data) {
    if (ctx) {
        futhark_context_free(ctx);
        ctx = NULL;
    }
    if (cfg) {
        futhark_context_config_free(cfg);
        cfg = NULL;
    }
}

/* ========================================================================== */
/* NIF Table                                                                   */
/* ========================================================================== */

static ErlNifFunc nif_funcs[] = {
    {"ping", 0, nif_ping, 0},
    {"linear_scan_nif", 4, nif_linear_scan, 0}
};

ERL_NIF_INIT(Elixir.ExPhil.Native.FutharkScan, nif_funcs, nif_load, NULL, NULL, nif_unload)
