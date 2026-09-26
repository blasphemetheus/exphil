# Fox Mamba profiling and first training pass

User commissioned Fox Mamba imitation and comparison with the strongest
human-tested GRU, plus training/inference profiling. No promotion is implied.
Live process/phase/log record: [FOX_MAMBA_LIVE_STATUS.md](FOX_MAMBA_LIVE_STATUS.md).

## Measurements and changes

RTX 5090, F32 tensors, `EXPHIL_EXLA_PRECISION=highest`, window 80,
two Mamba layers, state size 16, expansion 2, convolution 4. Final candidate:
hidden 512, 3,627,793 parameters, 264 source-supported input channels,
autoregressive controller head. Profiles use two actual Fox replay games,
seed 905; 20 measured training steps after warmup, batch 128.

| Measurement | Before | After |
| --- | ---: | ---: |
| Training step median, fallback vs fused scan | 59.394 ms | 28.644 ms |
| Full Agent decision median, GPU vs host embedding | 9.678 ms | 5.562 ms |
| Full Agent decision p95 | 12.217 ms | 7.200 ms |
| Full Agent decision maximum, 400 samples | 17.188 ms | 9.326 ms |
| Embedding alone median, including transfer/readback | 4.514 ms | 0.150 ms |

Enable `EDIFICE_FUSED_CUSTOM_CALL=1` for the linked CUDA kernels; profiler
confirms selective_scan custom calls. Identical-seed fused/fallback final
trunk features differ by at most 4.77e-7 after 23 updates; final losses differ
by 7.63e-6. This is a development parity check, not a general kernel proof.

Agent now constructs the small scalar/one-hot embedding on BinaryBackend,
then transfers the completed tensor to its original backend once. One hundred
recorded frames had exactly equal host/GPU embeddings. Existing embedding,
observe/history, stateful-step and live-queue tests: **24 passed**. Export,
reload, canary and real Agent inference also passed. No full test suite run.

Full Agent results include mailbox, embedding, window assembly, stochastic
AR sampling and controller conversion. They exclude Dolphin transport and
rendering. No >16.67ms decisions after the change in this 400-frame sample;
this is not a universal latency bound. Sampler alone is ~3.6ms.

Batch materialization averaged ~5.1ms per batch. The real streaming smoke fit
ran ~32–33ms/step including trainer overhead. Batch 256 (~58ms) gave no useful
throughput advantage over 128 (~29ms), so use 128. EXLA reserves ~70% VRAM;
observed process-time GPU totals were ~24.4 GiB including desktop/allocator
reservation, **not measured peak live tensor memory**. Cached datasets are
disabled to protect the 48GiB remaining root space.

This improves measured bottlenecks; it is not a claim of globally optimal
Mamba performance. True Mamba still recomputes an 80-frame window each decision.
Its GatedSSM-named incremental helper is not a true-Mamba step implementation.
Carried-state BPTT currently supports GRU only. BF16 and true incremental Mamba
are not enabled without separate numerical and train/inference parity checks.

## Training and evaluation

The generic streaming trainer has no windowed validation holdout and reports
training loss as val_loss in that case. `scripts/train_fox_mamba.exs` reuses
Pipeline/Trainer/callbacks but reserves 16 whole games before training, selected
by SHA256 path order; smoke runs reserve 10% capped at 16. Split manifests are
saved alongside checkpoints. Validation samples nonoverlapping 80-frame windows.
The initial candidate has no style conditioning. No held-out frames enter
training chunks. This is a development validation set, not a final test set.

Validation smoke: 21 training games, 2 separate validation games, 337 steps,
train loss 6.3543, held-out loss 4.4444, successful best/final policy exports.
Artifacts: `checkpoints/fox_mamba_20260925_holdout_smoke/`.
The earlier 23-game two-epoch smoke had **no genuine holdout**; its reported
val_loss 4.0877 is training loss. Do not compare it with the held-out result.

`scripts/sim_policy_match.exs` supports distinct policies and each checkpoint's
execution contract. Two 240-frame smoke matches against
`eval_runs/0923_ppo/eval_candidate/candidate_policy.bin` completed, swapping ports,
with both agents producing controls. Those short games are plumbing checks,
not strength evidence. The campaign schedules four longer FD matches with
matched seeds per swapped-port pair. Native sim has previously crashed in long
jobs; failure is recorded and stops the campaign, never counted as a win.

`scripts/fox_mamba_campaign.py` runs one full corpus epoch, registers the
best validation candidate, profiles the reloaded Agent, then plays those four
matches. No automatic promotion or indefinite repeat. Mamba imitation versus
a GRU that also received PPO is not an isolated architecture comparison.
Candidate path: `checkpoints/fox_mamba_v1_20260925/model_best_policy.bin`.
Checkpoint every 25,000 batches plus best/final/epoch exports; no overwrite of
an existing completed campaign. Each phase command is saved as JSON.

## Reproduction and evidence

Inside devenv, with no training/loop active:

```bash
EDIFICE_FUSED_CUSTOM_CALL=1 EXPHIL_EXLA_PRECISION=highest mix run scripts/profile_fox_mamba.exs --hidden 512 --batch 128 --window 80 --steps 20 --out eval_runs/mamba_profile_new
```

Use `EDIFICE_DISABLE_FUSED=1` instead for the fallback comparison. Every profile
exports a smoke policy; these short profiler artifacts are not trained bots.
Early `h256_*` profiles also used a different embedding/sampling configuration;
use `final_h512_b128_{fused,fallback}` for the final comparison.

Evidence under `eval_runs/0925_fox_mamba/`: final profile JSONs include losses,
features, compile times, input provenance and timing distributions;
`source_h512_b128_fused/agent_{before,after}.json` plus embedding parity samples;
`match_smoke/` contains the two successful control-loop checks.

Training's inhibitor child prints `sh failed with exit status 1` when stdin
closes at normal VM shutdown. Both smoke Mix processes actually exited **0**;
the message alone is not a training failure. Check the process exit and
`completed.json`, not that line.
