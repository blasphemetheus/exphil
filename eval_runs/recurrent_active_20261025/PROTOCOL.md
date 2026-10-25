# Active-opponent recurrent resync screen

Predeclared before completion: four seeds (1003–1006), 1,800 frames per
environment, Final Destination, stochastic sampling with an explicit per-frame
key, highest CUDA precision and portable kernels. The opponent is always
`coh_prev_q_ck4` MinGRU in its original windowed mode. Both sides use synchronous
zero-delay inference. These are NIF simulator rollouts, not Dolphin or human
netplay matches.

Window-trained MinGRU, Mamba and Mamba-2 each run windowed, indefinitely carried,
resync30 and resync60. BPTT GRU runs only carried: resetting its history periodically
would violate training semantics. MinGRU/Mamba/GRU use eight environments per
run. Archived independent-head Mamba-2 uses four sequential single-Agent
environments, matching the supported fallback without converting heads.

Total planned: 52 runs, 176 environment-minutes. Each arm has four seed-level
replicates; environment rows are not counted as independent statistical samples.
The seeds and per-frame random keys are shared across modes, but resulting game
trajectories diverge. Reset timing can therefore diverge too. Fixed mode order and
a shared host mean rollout wall times are not isolated latency benchmarks.

Resync is triggered by a threshold of 30/60 completed inference steps and only
when a full history is available. The counter resets to zero on the resync step;
steady uninterrupted schedules therefore replay every 31/61 calls. It is not a
wall-clock timer. Partial resets keep other rows' carry intact.

Read self-destructs, deaths, kills, damage dealt/taken and recovery together.
Lower damage taken alone can reflect disengagement; fewer self-destructs alone
do not prove stronger play. Report mean metrics and per-seed mode-minus-windowed
differences, without declaring significance from four seeds. Do not rank
architectures against each other: these checkpoints have different training,
characters and heads. In particular, the Mamba-2 export is an archived Mewtwo
model with legacy leaked labels, while the other policies are Fox.

No training, checkpoint promotion or deployment-default change is included.
Delay-matched evaluation and other stages remain necessary before any deployment
recommendation. Existing CPU/two-seed screens are historical context, not pooled
into this CUDA experiment.

Run inside the ExPhil devenv with compiled dependencies and a free GPU:

```sh
bash scripts/compare_recurrent_active.sh eval_runs/recurrent_active_20261025 1003 1004 1005 1006
python3 scripts/summarize_recurrent_active.py eval_runs/recurrent_active_20261025
```

The launcher stops on failure and skips completed JSON files on resume. It uses
the bare-VM source-loading runner, avoiding shared native rebuilds. Do not resume
into the same output directory with changed model/source/backend settings.
`manifest.json` records source/checkpoint hashes and the exact runner source.
The runner was formatted after the completed measurements; its exact measured
source remains embedded in the manifest. Model/Agent/replay computation did not
change during the experiment.

The first `mamba_windowed_s1005` attempt failed while writing XLA's temporary
PTX file (`No space left on device`) and emitted no result JSON. Free disk space
had returned at inspection. The queue resumed from that arm with compiler
temporary files under `/dev/shm/exphil-recurrent-validation` (`TMPDIR`); no other
job's files were deleted. Completed arms were retained and no failed timing or
partial rollout was included. This changes compiler scratch location, not model
execution settings.
