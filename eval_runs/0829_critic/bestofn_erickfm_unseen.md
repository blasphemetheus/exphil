# Offline Best-of-N

Policy `fox_gen_v1_20260825_210355_ep10.bin`, critic `critic_fox_gen_v1_ep10.bin`, 20 replays (eval_runs/0829_critic/erickfm_unseen41_60/*.slp).

| decode on 48191 decision frames, N=16 | match rate |
|---|---|
| sampling pass@1 (today's decode) | 14.9% |
| mode-of-N (critic-free re-ranker) | 22.9% |
| **selector Best-of-N** | **19.0%** |
| oracle pass@16 (ceiling) | 43.3% |

Gap recovered: 14.6%.
