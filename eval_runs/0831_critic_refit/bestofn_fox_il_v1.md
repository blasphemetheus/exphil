# Offline Best-of-N

Policy `fox_gen_v1.2_ARrefit_policy.bin`, critic `critic.nx`, 20 replays (replays/fox_il_v1/*.slp).

| decode on 19146 decision frames, N=16 | match rate |
|---|---|
| sampling pass@1 (today's decode) | 0.5% |
| mode-of-N (critic-free re-ranker) | 3.0% |
| **selector Best-of-N** | **4.0%** |
| oracle pass@16 (ceiling) | 6.2% |

Gap recovered: 61.6%.
