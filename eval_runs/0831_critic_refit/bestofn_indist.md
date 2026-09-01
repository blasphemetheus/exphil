# Offline Best-of-N

Policy `fox_gen_v1.2_ARrefit_policy.bin`, critic `critic.nx`, 20 replays (replays/erickfm_ranked/FOX/extracted/*.slp).

| decode on 37502 decision frames, N=16 | match rate |
|---|---|
| sampling pass@1 (today's decode) | 1.3% |
| mode-of-N (critic-free re-ranker) | 7.2% |
| **selector Best-of-N** | **10.8%** |
| oracle pass@16 (ceiling) | 13.9% |

Gap recovered: 75.5%.
