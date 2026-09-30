# Fox Mamba crash investigation and regression gates

Updated 2026-09-25, Codex. Full campaign remains stopped after its CUDA crash.

## Confirmed defect

Edifice's `native/cuda/fused_selective_scan_backward.cu` ignored the return
value of `cudaMallocAsync`. At the training shape (batch 128, expanded hidden
1024, window 80, state 16), each backward scan requests 640 MiB outside EXLA's
reserved pool. A failed allocation still launched a kernel with an invalid
workspace. Both the standalone wrapper and EXLA FFI handler now check it.
Launch and cleanup errors are also returned explicitly.

The host fault-injection regression failed before the fix: allocation returned
error 2, but one kernel launched and the wrapper returned success. After the
fix, the allocation error is returned without launching or freeing anything.

This establishes a real bug and a plausible mechanism for the original Xid31
invalid write. It does **not yet prove** that allocation failure caused the
original failure around update 11001. The later original-chunks replay caught
a 640 MiB allocation failure, but Ollama independently loaded a 6.2 GiB model
during that reproduction. Leave that unrelated service alone.

An attempted XLA FFI ScratchAllocator replacement failed at both 70% and 45%
GPU reservation and was removed. Its failure is not evidence of GPU OOM; the
underlying API error was hidden by that experiment. The retained implementation
uses checked CUDA allocation. The lower reservation is a diagnostic setting,
not an established optimal production setting.

## Recovery defects and protections

The original first periodic save at 25000 updates lost all progress on the
11001-update crash. The Fox driver now saves before training, after update 1,
and every 500 updates into two rotating recovery slots. These preserve weights
and optimizer state; they do **not** restore the shuffled data cursor.

Saved config tensors were also left on EXLA even though parameters/optimizer
were converted to BinaryBackend. Configuration is now converted recursively;
the GPU regression checks direct and nested tensors. This avoids persisting
process-local GPU handles in recovery artifacts.

## Repeatable checks

Only run Mix with no GPU training/probe active (shared EXLA library).

```bash
devenv shell -- mix test \
  test/exphil/training/callbacks/rolling_checkpoint_test.exs \
  test/exphil/training/callbacks/checkpoint_callback_test.exs \
  test/exphil/training/checkpoint_roundtrip_test.exs \
  test/exphil/training/checkpoint_config_backend_test.exs --include gpu

devenv shell -- python3 scripts/native/test_mamba.py
# Optional: MAMBA_MEMCHECK=/path/to/compute-sanitizer for full-size memcheck.
```

Native tests build standalone binaries in a temporary directory, without
replacing the shared EXLA library. They inject allocation/launch/free errors,
check analytic gradient agreement and finite outputs at small, uneven-thread,
and full training shapes, and optionally run Compute Sanitizer memcheck.

Saved-batch probes avoid reparsing: `scripts/mamba_crash_probe.exs`.
Real original-chunk transition probes: `scripts/mamba_crash_stream.exs`.
Both save pre-step state and input artifacts for offline diagnosis.
The chunk probe rotates paired captures every 100 updates and records exact
file lists, original chunk index, batch index, and shuffle seed.

## Evidence so far

- Original kernel standalone full-shape memcheck: zero errors, analytic
  gradient check passed (`logs/mamba_native_memcheck.log`).
- Thirty saved-batch shapes (128 down to 99): passed before kernel fix
  (`logs/mamba_crash_shapes_audit.log`). This does not reproduce long-run pressure.
- Recovery callback plus existing callback tests: 12 passed initially;
  expanded checkpoint suite including GPU configuration serialization: **18 passed**
  (`regression/unit.log`).
- Native fault injection, four shape/gradient cases, and full-size memcheck:
  **3 tests passed, zero sanitizer errors** (`regression/native.log`).
- Original-chunks 12–13 with checked allocation and 45% reservation:
  `logs/mamba_crash_chunks12_13_guard.log`: **1879 updates passed**.
- Full-size unfused reference at 45% reservation needs a 14.95 GiB allocation
  and fails cleanly with OOM. Numerical parity uses batch 64 for both paths;
  native memcheck and streaming endurance retain production batch 128.
- Numerical parity at batch64/window80/hidden512: sampled features after six
  training steps agree exactly (max absolute error 0). Fused median update
  14.897 ms vs unfused 28.915 ms. See `regression_v2/parity.json` and profiles.
- Still required: more than 11001 updates across repeated chunk transitions before
  treating the full-corpus run as ready.

Current fail-fast supervisor: `scripts/mamba_regression_campaign.py`, systemd
user unit `exphil-mamba-regressions-v2`. The live markdown and
`eval_runs/0925_fox_mamba/regression_v2/status.json` record its current stage,
child PID, log, and exit status. Unit/native passes are in the preceding
`regression/` directory; v2 resumes at numerical parity. No full-corpus training
starts automatically. GPU usage is sampled every two seconds in
`regression/gpu_memory.csv` (30-minute bounded monitor).

## Second transition run failed — allocation guard is not the complete fix

At 23:12:43 the repeated parallel original-chunks 12–13 test failed at update
639 with Xid31 invalid WRITE, despite the allocation guard and about 14 GiB
free GPU memory. `regression_v2/transition_failed` is the final supervisor
state; the endurance stage never started. This rejects the simple explanation
that all observed failures were unchecked OOM. The native guard remains a
valid independently tested fix.

Pre-update 600 weights/optimizer and batch were preserved and copied to
`crash/step600/`; repeating that saved batch for 1000 further updates passed
(`logs/mamba_step600_repeat.log`). This does not rule out a specific later
batch, but strengthens the concurrent-preparation hypothesis.

At 23:15, full-pipeline Compute Sanitizer memcheck is running with
`--target-processes application-only --track-stream-ordered-races all`;
log `logs/mamba_parallel_memcheck.log`, artifacts `crash/parallel_memcheck/`.
The application-only setting avoids the earlier `erl_child_setup` failure
caused by tracking Erlang's child processes. No other training run is active.

Recovery callback now also rejects non-`.axon` paths before training, avoiding
an unsupported suffix silently overwriting the primary checkpoint. Its five
targeted tests passed (four prior tests plus this new regression).

23:17 correction: the full-pipeline sanitizer **disabled itself** with
`CUDA initialized before the Sanitizer`; that run is NOT a valid memory-check
pass. It reproduced the illegal write at update 81. Testing
`CUDA_LAUNCH_BLOCKING=1` on the same parallel chunks next
(`logs/mamba_parallel_blocking.log`). Saved-batch replay remained stable.

23:20: synchronous launch test passed all 1879 updates. Directly launching
`beam.smp` under Compute Sanitizer (with BINDIR/ROOTDIR set), instead of the
Mix shell wrapper, successfully attaches the checker. Helper:
`scripts/mamba_sanitizer_direct.py`. It uses captured BEAM arguments in
`crash/beam_argv.json`; recapture after changing the devenv Erlang installation.
Two saved-batch updates completed under instrumentation; the reported errors
were XLA `cuModuleGetGlobal_v2` missing-symbol API returns, not invalid memory
accesses. The full parallel replay is now instrumented with API-return
reporting disabled and stream-ordered allocation race tracking enabled:
`logs/mamba_direct_parallel_memcheck.log`. Memory errors still fail the run.

Original campaign log/split remain intact under
`eval_runs/0925_fox_mamba/campaign/` and
`checkpoints/fox_mamba_v1_20260925/split.json`.

## Root cause found and fixed — workspace pool churn under concurrent eager XLA work (2026-09-26)

**Reproducer.** `scripts/mamba_race_probe.exs CAPTURE OUT MODE STEPS [CHUNK]`
trains from the step-600 capture on one saved batch while a background Task
runs the real 64-game chunk preparation in a loop (continuous overlap instead
of the transition's ~35 s window). Mode bisect, 6000 steps each, fraction 0.45:

| background work | result |
| --- | --- |
| none | pass |
| parse_only (CPU parse, no Nx) | pass, 231 loops |
| cpu (parse + `Streaming.create_dataset` on BinaryBackend) | pass, 8 loops |
| real (parse + eager GPU embedding, the pipeline) | **crash 2/3** — steps 4098, 172; pass |
| real + `--xla_gpu_enable_command_buffer=` | **crash 2/2** — steps 4498, 172 (command buffers ruled out) |
| real, workspace fix (below) | **pass 3/3** |

Every crash: Xid 31, `MMU Fault: ENGINE GRAPHICS GPC7 GPCCLIENT_T1_3 …
FAULT_PDE ACCESS_TYPE_VIRT_WRITE`, at `0x3_26018000` (the 01:45 transition
crash: `0x3_26000000`) — a write to an unmapped page at a stable address,
from the BEAM's own scheduler thread. The second run of a page-cached batch
died at step 172 in both the baseline and the command-buffer control, so the
race is near-deterministic once the corpus is in page cache.

**Cause.** `fused_selective_scan_backward` allocated its `[B,H,T,S]` f32
workspace (640 MiB at production shape) with `cudaMallocAsync` and released
it with `cudaFreeAsync` on the compute stream on EVERY call — two layers, so
1.3 GB mapped and unmapped through the driver's stream-ordered pool per
training step. That pool unmaps freed pages at the next synchronization
point. Training alone rarely synchronizes; the pipeline's background chunk
embedding on another host thread synchronizes constantly (host transfers),
and a kernel still writing the workspace faulted on a page the pool had
released. Whether the driver's in-use tracking is wrong or an ordering rule
is being violated is NOT yet established (experiments below); the mechanism
— unmap-on-sync of the per-call workspace — is.

**Fix** (Edifice `native/cuda/fused_selective_scan_backward.cu`):
`selective_scan_backward_workspace/3` — one grow-only `cudaMalloc` buffer per
device behind a mutex, reused in stream order for the life of the process;
growth frees the old buffer with `cudaFreeAsync` (stream-ordered, so an
in-flight kernel is safe). Both the NIF launcher and the FFI handler use it.
No per-step pool traffic remains. Native suite: allocation-failure harness
rewritten for the new contract (failure → no launch; launch error
propagates; same size reuses; larger frees once + allocates once; failed
growth free propagates), gradient shapes 1×1×1 … 128×80×1024 pass.

**Still to establish (fast, ~1 h, old code path):** (1) pool release
threshold unlimited alone; (2) opportunistic cross-stream reuse off alone;
(3) a standalone two-thread C++ reproducer (fault ⇒ driver, clean ⇒ an
XLA-side ordering interaction); (4) driver A/B if (3) faults.

Evidence: `logs/mamba_race_{real,none,parse_only,cpu,h2_cmdbuf,h1_fix}_*.log`,
`eval_runs/0925_fox_mamba/crash/race_*/`, `journalctl -k | grep Xid`
(11:50:42, 11:51:04, 12:11:xx, 12:12:xx), `logs/mamba_native_tests_0926.log`.

**Gate passed on the fix (2026-09-26, `eval_runs/0925_fox_mamba/regression_v4`,
unit `exphil-mamba-gate-v4`, launched 12:26):** parity max_abs_error within
1e-4; transition (chunks 12–13, parallel) 1879/1879; **endurance (chunks
1–16, parallel) 15170 updates, 0 Xid** — past the original 11001-update
failure with the pipeline's concurrent preparation on. `gates_passed`.

### Mechanism experiments (2026-09-26 18:07–21:50) — what is and is not established

In-process, on the race reproducer (6000 steps each, fixed lib with the
`EDIFICE_SSB_WORKSPACE` diagnostic switch, `logs/mamba_race_pool*_*.log`):

| `EDIFICE_SSB_WORKSPACE` | pool behaviour | runs |
| --- | --- | --- |
| `pool` (pre-fix path restored) | per-call mallocAsync/freeAsync | **crash**, pass, pass |
| `pool_keep` | release threshold unlimited: freed pages never unmapped | pass, pass, pass |
| `pool_noopp` | opportunistic cross-stream reuse off (pool trims more) | **crash, crash, crash** |
| `cached` (the fix) | no pool traffic | pass ×3 + gate |

All four crashes: Xid 31 VIRT_WRITE at `0x3_26018000`. **Established:** the
fault is the driver unmapping (releasing) the freed workspace pages while the
backward kernel is still writing them — never releasing removes it, releasing
more often makes it deterministic.

Standalone (`scripts/native/pool_unmap_race.cu`, no XLA): thread A does the
exact pre-fix pattern (mallocAsync → 640 MiB writing kernel → freeAsync, one
stream) while thread B syncs another stream continuously. **Clean** in every
variant: default (13,459 cycles vs 725k syncs in 180 s), 25 ms idle gap
before reuse, 60 % of the card reserved, opportunistic reuse off, and a third
thread as a second pool user on its own stream. So the textbook pattern does
not fault by itself; **the XLA process contributes a factor that these
variants do not model** (stream/event topology of PJRT, its own stream-
ordered allocations, or a driver bug that needs that topology). Driver-bug
vs ordering-rule is therefore **NOT decided**; the mechanism and the fix
are. Next step if anyone wants the last word: an nsys trace of one `pool`
crash (which stream the kernel and the free actually land on, and what the
other threads enqueue in between), then a driver A/B.

Decision: the cached workspace is the fix (no pool traffic, gate passed
15170 updates with concurrent prep). The `pool*` modes stay for diagnosis.

### Tools ready for the last question (2026-09-30, run when the GPU is next free)

1. **`scripts/native/pool_unmap_race_v2.cu`** — standalone, PJRT topology: the
   workspace kernels AND a thread of small "eager" kernels share one compute
   stream; host transfers wait on compute-stream events, sync on a D2H
   stream, and the compute stream waits on the H2D event (PJRT's transfer
   pattern); 45 % held reservation as the BFC stand-in. Fault ⇒ driver bug
   with a portable reproducer; clean ⇒ the factor is inside XLA's allocator
   interaction. Build: `nvcc -O2 -arch=sm_120 … -o pool_unmap_race_v2`;
   run 180 s each: default; `180 640 0 0` (reuse off, the in-process
   deterministic-crash setting); `180 640 1` (release threshold unlimited,
   must be clean).
2. **`scripts/mamba_nsys_direct.py`** — the race reproducer under Nsight
   Systems with `EDIFICE_SSB_WORKSPACE=pool` (crashes in minutes). Needs
   `nix shell nixpkgs#cudaPackages.nsight_systems`; direct-BEAM launch as the
   sanitizer used. Read the report for the stream of each
   `fused_selective_scan_backward_kernel` and `cudaFreeAsync`, and what the
   embedding thread enqueued between them.
3. **Driver A/B**: `~/dotfiles/configuration.nix` pins
   `hardware.nvidia.package = …nvidiaPackages.beta` (595.45.04 today). Swap
   to `.production`, `sudo nixos-rebuild switch --flake ~/dotfiles#nixos_slanka`,
   reboot, rerun the in-process `pool_noopp` triple (3/3 crash on beta). A
   pass on production = driver regression; report upstream with (1).

Order: (1) three runs, ~10 min; (2) only if (1) is clean; (3) only if (1)
faults or Bradley wants the upstream report airtight.
