# Mamba learning pilot results

**AI-generated.** Read `../README.md` for the method and limits.

Status: complete; 4 completed runs.
All models start fresh on the same 232 training games and 24 held-out games.
Two seeds are a pilot, not a reliable estimate of uncertainty.

| Model and seed | Budget | Updates | Training seconds | Held-out loss | Warm ms/update |
| --- | --- | ---: | ---: | ---: | ---: |
| mamba_n16_s905 | equal_steps | 3000 | 182.653 | 4.642 | 52.553 |
| mamba_n16_s905 | equal_time | 10755 | 600.006 | 3.791 | 53.465 |
| mamba2_full_n16_s905 | equal_steps | 3000 | 130.592 | 4.138 | 35.196 |
| mamba2_full_n16_s905 | equal_time | 15951 | 600.020 | 3.600 | 36.049 |
| mamba2_full_n16_s906 | equal_steps | 3000 | 127.867 | 4.229 | 34.212 |
| mamba2_full_n16_s906 | equal_time | 16294 | 600.101 | 3.581 | 35.281 |
| mamba_n16_s906 | equal_steps | 3000 | 177.359 | 4.621 | 50.961 |
| mamba_n16_s906 | equal_time | 11050 | 600.089 | 3.856 | 52.092 |

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

## Simulator readout

Both player ports are used on Final Destination with zero delay. Policy
sampling remains stochastic. Per-pair settings and simulator seeds are saved
in the league JSON. Direct comparisons are separate from the frozen-opponent
comparisons. A lead at the time limit is not a completed-match win.

| Snapshot | Opponent | Session length | Completed wins/losses/draws | Time-limit leads/trails/ties |
| --- | --- | ---: | ---: | ---: |
| mamba_n16_s905_equal_time | mamba2_full_n16_s905_equal_time | 180.0s | 6/11/0 | 1/5/2 |
| mamba_n16_s906_equal_time | mamba2_full_n16_s906_equal_time | 180.0s | 2/5/0 | 6/2/0 |
| mamba2_full_n16_s905_equal_time | master_ep3 | 180.0s | 0/10/0 | 1/6/1 |
| mamba2_full_n16_s906_equal_time | master_ep3 | 180.0s | 0/7/0 | 1/7/0 |
| mamba_n16_s905_equal_time | master_ep3 | 180.0s | 0/16/0 | 0/8/0 |
| mamba_n16_s906_equal_time | master_ep3 | 180.0s | 0/11/0 | 0/5/3 |
