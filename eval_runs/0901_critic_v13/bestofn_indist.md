# Offline Best-of-N

Policy `fox_gen_v1.3_ARrefit_policy.bin`, critic `critic.nx`, 20 replays (replays/erickfm_ranked/FOX/extracted/*.slp).

| decode on 37502 decision frames, N=16 | match rate |
|---|---|
| sampling pass@1 (today's decode) | 5.2% |
| mode-of-N (critic-free re-ranker) | 18.3% |
| **selector Best-of-N** | **24.5%** |
| oracle pass@16 (ceiling) | 32.8% |

Gap recovered: 69.9%.
