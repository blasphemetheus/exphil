# Offline Best-of-N

Policy `fox_gen_v1.3_ARrefit_policy.bin`, critic `critic.nx`, 20 replays (replays/fox_il_v1/*.slp).

| decode on 19146 decision frames, N=16 | match rate |
|---|---|
| sampling pass@1 (today's decode) | 2.1% |
| mode-of-N (critic-free re-ranker) | 8.9% |
| **selector Best-of-N** | **11.0%** |
| oracle pass@16 (ceiling) | 16.0% |

Gap recovered: 64.1%.
