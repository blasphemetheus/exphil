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
   STATUS: [x] BISECTED + CLOSED 2026-08-24 (eval_runs/
   0824_cache_bisect): 3-arm warm-boot matrix — all-cached FATAL,
   predict-only FATAL, sampling-only CLEAN — **the 9.8MB `predict`
   executable's cache is the poison, and the mechanism is TIMING, not
   corruption**: deserialization defers expensive finalization to
   first use in the calling process, so the Inference process's first
   live call re-loads MID-GAME, blocking the frame loop for seconds →
   local spectator disconnect / the netplay both-peers freeze that
   originally convicted the cache ("counter frozen, latched down-B" =
   frame-loop starvation seen from outside). A cold cache-writing
   boot also once hard-wedged every scheduler post-warmup (do_wait) —
   same load path, worse day. Sampling caches are innocent AND
   worthless (~0 warmup saved). VERDICT: stays DEFAULT-OFF
   permanently; per-function knob EXPHIL_XLA_EXEC_CACHE_ONLY + pins
   in test/exphil/training/xla_exec_cache_test.exs. A future revival
   would need first-use finalization forced at WARMUP TIME in the
   inference process — but the stateful path (2b) already makes the
   whole question moot for probes, and the resident policy server
   (option 3) for deploys.
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
   LIVE-VALIDATED for LATENCY same night (eval_runs/0824_stateful_live):
   warmup 1,482ms vs 19,973ms; both arms qtrace-sharp at nominal peak 5.
   BUT the deploy rung FAILED on BEHAVIOR (eval_runs/0824_stateful_netplay
   + canonical rescore of the local A/B): ShineChain-over-replays shows
   the stateful arm chains ~4x less than windowed even locally at the
   trained id (sustained 2 vs 8, 38.9 vs 61.4 shines/min), and netplay
   games capped at chain 1-2 (those also confounded by an untrained
   delay-id — bare --frame-delay 4 sets id4; record knobs are d4/id3).
   Mechanism: carried GRU state diverges from trained sliding-window
   semantics after frame 60 (the equivalence test only pins the first
   window). VERDICT: default stays OFF for play; the g6 lesson holds —
   latency rungs don't crown, chain strength at the deploy rung does.
   Still correct for headless probes (GOTCHA #69). UPDATE (same day,
   eval_runs/0824_resync_local): **`--stateful-resync 60` at a trained
   id RESTORES windowed-grade chains locally** (sustained 9 vs
   windowed's 8 vs plain-stateful's 2, 57.2 shines/min) — the
   as-if-windowed rebuild works. Remaining rung before the deploy
   default flips: netplay TAIL evidence (a 20+ chain on the resync arm
   during organic play; per-game netplay chains are heavy-tailed, so
   n=1 medians prove nothing — the 0824 crown-decider distribution
   {32,12} vs {3,3,13} shows the tail is where arms separate).
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
- 2026-08-24 (evening close-out): options 1 and 2 both reached final
  verdicts (1 = bisected, predict-cache load timing, permanently off;
  2 = autotune is ~0% of compile). Scoreboard: probes/headless =
  SOLVED (stateful, 1.5s); local play = 20s windowed (resync arm
  pending netplay tails could bring 1.5s); deploys = 20s windowed.
  **The only structural lever left is option 3, the resident policy
  server** — now also the eval-fleet multiplier (gate sweeps and
  deciders pay 20s per session start today). Option 4 (ONNX) stays
  the parallel runway behind a behavioral gate.
