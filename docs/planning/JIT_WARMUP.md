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
   STATUS: [~] ATTEMPTED 2026-08-24, DEFAULT-OFF after a live
   conviction. Wired into all four compile sites (predict, trunk_step,
   heads, fused samplers; Utils.xla_exec_cache/2 +
   EXPHIL_XLA_EXEC_CACHE env). Measured: cold 26.5s (compile+write),
   warm 13.4s — the caches hit (no key-mismatch warnings) but only
   ~6s came back; the residual 13s is NOT the fused samplers (their
   cache moved nothing) — unattributed (candidate: driver PTX->SASS
   JIT for sm_120, cuDNN handle init). THEN the live conviction: the
   first real netplay game with cached executables HUNG the inference
   process mid-game (counter frozen at 2612, inputs latched on down-B
   = crouch + held shine); the identical session with the cache
   disabled played normally. Deserialized executables are unsafe on
   this stack (xla 0.10 / exla 0.13.1 / RTX 5090) pending a bisect
   (enable per-function to find the poisoned one; suspect the big
   9.8MB predict executable or device-state assumptions in
   EXLA.Executable.load).
2. **XLA autotune cache dir** — env-only fallback if (1) stalls:
   XLA_FLAGS autotune-cache flags persist the benchmarking results
   (usually the bulk of compile time). STATUS: [x] DEAD 2026-08-24:
   `--xla_gpu_autotune_level=0` changed nothing (19.8s → 20.0s), so
   autotuning is ~0% of our compile; a persistent autotune cache can
   save nothing.
2b. **Stateful step path (`--stateful-step`)** — compile a
   single-timestep trunk graph instead of the 60-frame unroll.
   STATUS: [x] MEASURED 2026-08-24: **warmup 19.9s → 1.5s** (embed
   0.2s + trunk_step 0.55s + fused heads sampler 0.75s). No cache, no
   deserialization safety question — the graph is just 60x smaller.
   Same-day fix: the stateful warmup branch previously warmed only the
   bare heads predict, NOT the fused sampler the live loop actually
   calls (`Policy.sample(heads_predict_fn, features)`) — the 08-07
   netplay-freeze class waiting to happen; warmup now routes through
   `warmup_sample` with both prev_buttons variants. Equivalence is
   pinned (stateful_step_equivalence_test, max logit delta 3.6e-7) and
   the path is already mandatory for headless probes (GOTCHA #69).
   LIVE-VALIDATED same night (eval_runs/0824_stateful_live): stateful
   vs windowed control, identical settings (ms_g19_ep4, local, CPU-3,
   d3) — warmup 1,482ms vs 19,973ms; BOTH arms qtrace-sharp at the
   nominal lag peak 5 (d3+2), stateful marginally sharper (99.9% vs
   99.6%); staleness 1/9192 vs 7/9153. Use `--stateful-step` in local
   deploy recipes now. REMAINING: netplay-vs-human rung before the
   script default flips (deploy-rung rule).
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
- 2026-08-24 (attribution session): the residual is ATTRIBUTED. Stage
  lines (standalone boot, ms_g19_ep4): embed 209ms, **sample1 19.6s**,
  sample2 5ms, confidence 1ms — the entire cost is the ONE fused
  predict+sampler compile of the 60-frame unrolled GRU graph. Process
  watch during compile: beam.smp ~200% CPU for ~16s (XLA HLO passes,
  in-process) then a single `ptxas -arch sm_120a` for ~4s. Driver
  PTX->SASS JIT hypothesis DEAD: ptxas targets sm_120a natively (CUDA
  12.9) and ~/.nv/ComputeCache stayed untouched. Autotune hypothesis
  DEAD: `--xla_gpu_autotune_level=0` → 20.0s (no change). The winning
  lever is 2b: **`--stateful-step` warmup = 1.5s** (single-step graph;
  19.6s of HLO work simply never exists). Also fixed the stateful
  warmup gap (fused sampler now warmed — was the 08-07 freeze class).
  Attribution harness: scratchpad warmup_attribution.exs (boots Agent
  standalone, no Dolphin, WARMUP_STATEFUL=1 toggles the path).
- 2026-08-24 (later): step 1 wired + measured (26.5 cold / 13.4 warm)
  then CONVICTED — cached executables hang inference mid-game
  (details at option 1) — default-off. Also: ~13s of warmup is
  unattributed by the executable caches at all (warmup stage
  instrumentation added to Agent). Next levers: attribute the
  residual 13s (likely driver-level), then option 2 (autotune cache
  flags) or option 3 (resident policy server, which sidesteps the
  deserialize question entirely).
