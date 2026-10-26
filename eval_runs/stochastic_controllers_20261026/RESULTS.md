# Stochastic controller inference

The deterministic benchmarks understated deployed stochastic CUDA latency.
With fresh RNG keys and all controller work included, steady inference is
about **3.0–3.8 ms per request** on the RTX 5090, whether requesting one policy
or eight batched policies. CPU steady carry is **0.5–0.8 ms for one policy** and
**0.7–1.4 ms for eight** in this screen. These are different trained models,
not a parameter-matched architecture comparison.

See [all latency tables](TABLES.md). Raw JSON retains all 100 timings per scenario.

## Implications for ExPhil

- Stochastic sampling is the next CUDA optimization target. Deterministic
  CUDA carry is roughly 0.9–1.5 ms in the matched runs. Standalone fresh-key
  construction plus synchronization costs only 0.068 ms median; constructing
  the same key on the host reduces that to 0.033 ms, too little to explain
  the full gap. Production sampling was not changed.
- CPU deserves an end-to-end deployment trial for steady single-policy carry,
  especially BPTT GRU. This is not sufficient evidence to switch defaults:
  replay, training/evaluation concurrency and emulator integration matter too.
- CUDA remains useful for substantial Mamba replay/window workloads. Eight-row
  stochastic Mamba all-row resync is 9.258 ms versus 30.996 ms on CPU; full
  window recomputation is 4.966 versus 153.856 ms. Reset/resync spikes remain
  distinct from ordinary-step latency.
- Resync remains opt-in. These latency measurements do not override the
  context-dependent gameplay results from the delayed evaluation screen.

## Protocol and validity

32 primary runs: MinGRU, BPTT GRU, Mamba and Mamba-2; one/eight rows;
deterministic/stochastic; CPU/CUDA. Each scenario uses five warmups and 100
measurements after compilation. Timers cover the complete Agent request:
embedding, trunk, heads, fresh cryptographic RNG-key construction when sampling,
six controller components and device-to-host transfer. They exclude fixture
restoration, row-reset API calls, compilation and simulator/emulator IPC.

The fixture restores a saturated history and fixed carry before every timed
call, then samples the next recorded observation. Thus these are controlled
per-scenario latency distributions, not sustained closed-loop rollout averages.
Timed stochastic draws use fresh keys; priming and rollback use fixed explicit
keys. Temperature is 1, buttons are stochastic, and mode-of-N is off. All
stochastic ordinary cases produced multiple controller groups; every deterministic
case produced one. Public single-Agent snapshots reproduce a recorded suffix
exactly with explicit keys. Batched rollback is an internal complete-state
fixture check, not a new public batch snapshot API. No equality is claimed
between stochastic single and batched trajectories.

Hardware: Ryzen 9 9950X3D, RTX 5090, driver 595.104.02. EXLA highest precision;
portable kernels (`EDIFICE_FUSED_CUSTOM_CALL=0`). CPU runs used EXLA's host
client. Source, checkpoint and replay SHA-256 hashes were verified unchanged
through each complete 16-run suite. The manifests preserve exact commands.

CPU primary runs were 12:55:12–12:59:06 America/Chicago while another agent's
test workload was active. CUDA primary runs were 13:00:19–13:03:17; the harness
checked that the GPU was free before every run, and the next competing test
process started at 13:04:25. CPU/GPU measurements are not simultaneous isolated
hardware trials. Small differences, including stochastic occasionally measuring
faster on CPU, should be treated as run variability.

The archived Mamba-2 export has independent heads and its known legacy leaked-label
training warning. Its latency is useful implementation evidence, not evidence of
policy quality. The other three exports have autoregressive heads. No policy,
training contract, live inference default or production sampling code changed.

## Attribution diagnostics

`rng_host.json` and `rng_cuda.json` compare the existing fresh-key constructor
with host construction of the identical two-word key. Each has 200 timed samples
after ten warmups, including a return synchronization. Seed parity covers
0, 1, 42 and the maximum unsigned 32-bit value. The idle CUDA diagnostic ran
at 13:03:41, before the competing test restarted.

`heads_host.json` and `heads_cuda.json` isolate trained heads/sampling and six
output transfers on fixed features obtained from recorded observations. They
compare deterministic, fresh-key stochastic and fixed-key stochastic paths for
MinGRU's AR heads and Mamba-2's independent heads at one/eight rows. **The later
CUDA head-only diagnostic overlapped the competing test process**; its roughly
2.4–3.3 ms stochastic results are attribution hints, not uncontended latency
baselines. Supplying a precomputed key did not remove the observed cost. Profile
the fused random-draw/categorical kernels under exclusive GPU use before changing
their implementation or claiming a speedup.

## Reproduce

From ExPhil, use a new output directory (the harness refuses overwrites):

```sh
python3 scripts/profile_stochastic_controllers.py --client host --out eval_runs/new_profile
python3 scripts/profile_stochastic_controllers.py --client cuda --out eval_runs/new_profile
python3 scripts/summarize_stochastic_controllers.py eval_runs/new_profile
```

The summarizer requires the recorded source/checkpoint/replay hashes to still
match, validates all timing statistics, and regenerates `TABLES.md`. The diagnostic
scripts run through `scripts/run_recurrent_validation.exs` with `RECURRENT_CLIENT`
set to `host` or `cuda` and a single output-JSON path argument.
