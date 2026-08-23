# JIT warmup elimination — living plan

Started 2026-08-24 (Bradley: "it is now becoming the weakest link in
our chain to have to wait 20 seconds each time we start a game").
UPDATE THIS DOC as items land.

## The problem, plainly

The policy is described as math (Nx/Axon); the GPU runs compiled CUDA
kernels. On every session start, XLA compiles the model for our exact
shapes: graph optimization, kernel fusion, ptxas, and AUTOTUNING
(benchmarking candidate algorithms per matmul and keeping the
fastest). Measured: **19.8s on the 5090** (ms_g19_ep4, GRU-60,
constant across sessions). The answer never changes — same checkpoint,
same shapes, same GPU — but each new OS process starts with an empty
in-memory cache and throws the work away on exit.

Current mitigation: warmup is BACKGROUNDED and overlapped with menu
navigation (CSS interlock keeps games from starting on a half-JIT'd
policy). Menu overhead beyond JIT is ~3-4s local / ~5s netplay (the
2026-08-23 profiling+fixes arc), so JIT is now >80% of launch-to-game.

## Options (ordered by leverage/cost)

1. **EXLA persistent compilation cache** — CONFIRMED SUPPORTED by our
   exla fork (0.13.1): the `:cache` compiler option accepts a
   FILESYSTEM PATH ("set it to a binary, representing a filesystem
   path to store the cache"); `EXLA.NIF.deserialize_executable` exists.
   Wire `cache: <dir keyed by checkpoint hash + shapes>` into the
   Agent's jit options; expect second-boot warmup ~1-3s.
   STATUS: [ ] not started. First step: find where Agent/warmup calls
   EXLA.jit / Nx.Defn.jit and thread the option; cold/warm A-B measure
   via the "JIT warmup complete (NNNNms)" log line.
2. **XLA autotune cache dir** — env-only fallback if (1) stalls:
   XLA_FLAGS autotune-cache flags persist the benchmarking results
   (usually the bulk of compile time). STATUS: [ ] untried.
3. **Resident policy server** — one long-lived beam JITs once and
   serves inference to every session (games and dolphins come and go).
   Also what unattended rematch and eval fleets want (gate sweeps run
   hundreds of session starts). STATUS: [ ] design only. Structural;
   do after (1) since (1) may make per-session compile cheap enough.
4. **ONNX runtime deploy path** — we already export ONNX INT8 with
   0.55ms inference; ORT loads in ~1s with zero JIT. Needs parity
   validation for GRU + autoregressive heads + sampling before any
   deploy trust (behavioral gate, not loss). STATUS: [ ] parked as the
   parallel runway.
5. **Compile less** (autotune level down, graph splits) — last resort;
   (1) should make it moot.

## Measurement protocol

The log line `JIT warmup complete (NNNNms)` is the metric; A-B =
consecutive session starts with the same checkpoint (first = cold
cache, second = warm). Guard runtime-regression with a qtrace lag
check on the first warm-cache netplay session (a cache should be
bit-identical, but verify once).

## Log

- 2026-08-24: doc created; exla `:cache`-as-path support confirmed in
  the local fork's source (exla/lib/exla.ex). Baseline: 19.8s
  (measured repeatedly 2026-08-23, ms_g19_ep4 GRU-60 on the 5090).
