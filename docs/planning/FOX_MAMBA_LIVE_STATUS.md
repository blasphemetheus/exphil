# Fox Mamba live status

Updated: 2026-09-27T14:11:04.015901+00:00

Phase: **complete**. Unit: `exphil-fox-mamba-v1`. Supervisor PID: 1093124; active child: None.

Current log: `None`. Last exit: `0`.

One epoch, 512-wide two-layer Mamba, window 80, batch 128, F32, fused scan, stride 5. Sixteen disjoint validation games. No style conditioning. Large streaming caches disabled.

Candidate: `checkpoints/fox_mamba_v1_20260925/model_best_policy.bin`. GRU opponent: `eval_runs/0923_ppo/eval_candidate/candidate_policy.bin`.

After fit: reload/Agent latency, then four FD games with swapped ports. This is a development comparison, not a promotion gate or an isolated architecture comparison (GRU also received PPO).

Do not run Mix or edit code this campaign calls while it is active. Existing viewers and unrelated Phoenix are untouched. No commits/pushes.

Performance evidence: `docs/planning/FOX_MAMBA_PROFILE_2026-09-25.md`.
