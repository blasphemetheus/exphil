# Recurrent resync under fixed deployment latency

Predeclared before examining delayed outcomes: 96 trials, 384 environment-minutes.
Actors are the same causal-label, delay-0-trained MinGRU and Mamba checkpoints
used in the preceding active-opponent screen. Compare windowed, indefinite
carry and resync-30 with paired seeds 1101–1104, eight environments and 1,800
frames per trial. Each seed is one replicate; rows are not independent replicates.
The resync threshold is 30; because replay resets the decision counter to zero,
continuous full-history inference replays every 31 decision calls.

| Context | Stage | Action delay, both players | Opponent |
| --- | --- | ---: | --- |
| fd3_min_gru | Final Destination | 3 | Windowed MinGRU |
| fd4_min_gru | Final Destination | 4 | Windowed MinGRU |
| bf3_min_gru | Battlefield | 3 | Windowed MinGRU |
| fd3_gru | Final Destination | 3 | Carried BPTT GRU |

The last three contexts each change one factor from the first. The GRU
opponent always retains its trained carry contract, with no periodic resync.
Delay 3/4 represents a deployment-latency stress test outside these particular
checkpoints' training distribution. Do not interpret it as an aligned-delay
quality result, and do not change deployment defaults from this screen.

`ExPhil.Sim.ActionDelay` holds emitted controllers for d simulator calls:
the action chosen from state t controls state t+d+1. Pending inputs start
neutral; finished environments clear only their own pending controllers and
Agent state. Both players have the same physical delay. Policies receive
current observations and retain their own previously emitted action history.
This is a synchronous fixed-actuation FIFO, not observation lag, network
jitter, asynchronous inference or a Dolphin/netplay trial. The existing
Agent queue-depth restriction remains; no new queue-conditioned policy support.

Unit tests verify frame-tagged action timing for delays 0/1/3/4 and partial
reset isolation. A full zero-delay seeded rerun exactly matches every prior
reported gameplay metric in `../recurrent_active_20261025/min_gru_windowed_s1003.json`.

Highest-precision EXLA CUDA on RTX 5090, portable kernels, seeded stochastic
sampling, unchanged checkpoints and native NIF simulator. `manifest.json`
captures source/checkpoint/simulator SHA256 before the grid; the summarizer
refuses changed sources or incomplete protocols. After four completed arms,
the remaining grid was split across four isolated VMs (one context per shard,
`ERL_FLAGS='+S 4:4'`) to reduce CPU-bound rollout wall time. The manifest retains
the original runner and this scheduling amendment; arm settings are unchanged.
Each VM runs arms sequentially to reuse compiled programs. Raw JSON is saved after each arm.
The launcher can resume missing arms; do not mix artifacts from different manifests.

Report deaths, self-destructs, kills, damage, recovery, repetition and neutral
input share separately. No composite winner, win-rate inference, statistical
superiority claim, policy promotion, training restart or inference-default change.

Run inside ExPhil devenv with `EDIFICE_FUSED_CUSTOM_CALL=0`:

```sh
python3 scripts/summarize_recurrent_delays.py eval_runs/recurrent_delays_20261025 --allow-partial
elixir -pa '_build/dev/lib/*/ebin' scripts/run_recurrent_validation.exs \
  scripts/compare_recurrent_delays.exs eval_runs/recurrent_delays_20261025
python3 scripts/summarize_recurrent_delays.py eval_runs/recurrent_delays_20261025
```
