# Offline Best-of-N

Policy `fox_gen_v1.1_AR_20260831_080100_policy.bin`, critic `critic.nx`, 20 replays (replays/fox_il_v1/*.slp).

| decode on 19146 decision frames, N=16 | match rate |
|---|---|
| sampling pass@1 (today's decode) | 1.3% |
| mode-of-N (critic-free re-ranker) | 5.1% |
| **selector Best-of-N** | **6.5%** |
| oracle pass@16 (ceiling) | 11.1% |

Gap recovered: 52.5%.
