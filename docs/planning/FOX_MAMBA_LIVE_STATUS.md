# Fox Mamba live status

Updated: 2026-09-25 23:27 CDT

Phase: **instrumented crash diagnosis**. Active BEAM PID **2753971**, under
Compute Sanitizer; latest marker step217 in original chunk12. Earlier
regression supervisor PID 2511669 has exited.

Current log: `logs/mamba_direct_parallel_memcheck.log`.
Current artifacts: `eval_runs/0925_fox_mamba/crash/direct_parallel_memcheck/`.
GPU telemetry: `eval_runs/0925_fox_mamba/regression/gpu_memory.csv`.

Full-corpus campaign remains stopped. Checkpoint tests, native fault injection,
native memcheck, and fused/fallback parity passed. The repeated asynchronous
two-chunk gate crashed with ample free memory; the 16-chunk gate never started.
The allocation guard fixes a real bug but does not resolve this remaining race.
Saved-batch 1000-step replay passed. Synchronous CUDA two-chunk replay passed
1879 updates. The current run instruments the asynchronous pipeline directly.

Probes use 45% EXLA GPU reservation. Unrelated Ollama is left alone. Do not run
Mix/rebuild shared EXLA while this probe is active. No automatic next stage.

Evidence and limitations: [crash regressions](FOX_MAMBA_CRASH_REGRESSIONS.md). No full training restart or automatic promotion occurs here.

Resume from [the latest handoff](HANDOFF_2026-09-25a.md). Code committed on user
request: ExPhil a1420270; Edifice c90990a. Nothing pushed.

23:39 CDT: user requested `CUDA.md` as Claude's completion signal. Durable
watcher `exphil-cuda-report.service`, PID2969609, observes both diagnostic
process identities and atomically creates `/home/blewf/git/exphil/CUDA.md`
after they exit. The file intentionally does not exist yet. It will include
final sanitizer evidence, completion status, prior findings and remaining
uncertainty. Watcher regression tests: 4 passed. No follow-up training starts.
