# Offline Best-of-N

Policy `fox_gen_v1.1_AR_20260831_080100_policy.bin`, critic `critic.nx`, 20 replays (replays/erickfm_ranked/FOX/extracted/*.slp).

| decode on 37502 decision frames, N=16 | match rate |
|---|---|
| sampling pass@1 (today's decode) | 3.4% |
| mode-of-N (critic-free re-ranker) | 11.0% |
| **selector Best-of-N** | **13.7%** |
| oracle pass@16 (ceiling) | 21.4% |

Gap recovered: 57.4%.
