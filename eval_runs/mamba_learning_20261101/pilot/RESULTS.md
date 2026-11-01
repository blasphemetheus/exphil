# Mamba learning pilot results

**AI-generated.** Read `../README.md` for the method and limits.

Status: complete; 6 completed runs.
All models start fresh on the same 29 training games and three held-out games.
Two seeds are a pilot, not a reliable estimate of uncertainty.

| Model and seed | Budget | Updates | Training seconds | Held-out loss | Warm ms/update |
| --- | --- | ---: | ---: | ---: | ---: |
| mamba_n16_s905 | equal_steps | 300 | 39.405 | 5.830 | 47.702 |
| mamba_n16_s905 | equal_time | 1799 | 120.097 | 4.708 | 52.811 |
| mamba2_full_n16_s905 | equal_steps | 300 | 33.757 | 4.872 | 30.836 |
| mamba2_full_n16_s905 | equal_time | 2588 | 120.023 | 3.972 | 36.910 |
| mamba2_full_n128_s905 | equal_steps | 300 | 37.617 | 4.866 | 32.809 |
| mamba2_full_n128_s905 | equal_time | 2301 | 120.219 | 4.133 | 40.179 |
| mamba2_full_n128_s906 | equal_steps | 300 | 37.748 | 4.827 | 33.033 |
| mamba2_full_n128_s906 | equal_time | 2301 | 120.978 | 4.312 | 40.481 |
| mamba2_full_n16_s906 | equal_steps | 300 | 33.917 | 4.863 | 29.973 |
| mamba2_full_n16_s906 | equal_time | 2609 | 120.012 | 3.930 | 36.448 |
| mamba_n16_s906 | equal_steps | 300 | 39.357 | 5.469 | 47.251 |
| mamba_n16_s906 | equal_time | 1795 | 120.020 | 4.777 | 52.838 |

Lower held-out loss is better on this objective. Warm ms/update subtracts the
first training step (which includes compilation). It still includes batch
loading and epoch boundaries; it is not a GPU kernel benchmark.

Equal-time budgets include first-step compilation but exclude dataset setup,
validation and policy export. GPU memory snapshots are in the per-arm JSON.
Replay/source hashes and contention observations are in `manifest.json`.

These results do not establish gameplay strength or convergence on the full
corpus. Policy files are local and are not promoted automatically.

**AI-generated poem**

We give each learner time and ground;
then write the limits of what we found.

## Short simulator readout

Each snapshot faces the same frozen master-ep3 Fox, both ports, four environments,
30 seconds per session, Final Destination, zero delay. Policy sampling remains
stochastic. Simulator seeds are 71/72. This is a small compatibility/gameplay
pilot. A lead at the time limit is not a completed-match win.

| Snapshot | Completed wins/losses/draws | Time-limit leads/trails/ties |
| --- | ---: | ---: |
| mamba2_full_n128_s905_equal_steps | 0/0/0 | 0/8/0 |
| mamba2_full_n128_s905_equal_time | 0/0/0 | 1/7/0 |
| mamba2_full_n128_s906_equal_steps | 0/0/0 | 0/8/0 |
| mamba2_full_n128_s906_equal_time | 0/0/0 | 1/6/1 |
| mamba2_full_n16_s905_equal_steps | 0/0/0 | 0/7/1 |
| mamba2_full_n16_s905_equal_time | 0/0/0 | 1/7/0 |
| mamba2_full_n16_s906_equal_steps | 0/1/0 | 0/7/1 |
| mamba2_full_n16_s906_equal_time | 0/0/0 | 0/7/1 |
| mamba_n16_s905_equal_steps | 0/0/0 | 1/7/0 |
| mamba_n16_s905_equal_time | 0/0/0 | 2/6/0 |
| mamba_n16_s906_equal_steps | 0/0/0 | 0/6/2 |
| mamba_n16_s906_equal_time | 0/0/0 | 2/5/1 |
