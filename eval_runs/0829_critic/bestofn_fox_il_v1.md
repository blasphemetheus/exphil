# Offline Best-of-N

Policy `fox_gen_v1_20260825_210355_ep10.bin`, critic `critic_fox_gen_v1_ep10.bin`, 20 replays (replays/fox_il_v1/*.slp).

| decode on 19146 decision frames, N=16 | match rate |
|---|---|
| sampling pass@1 (today's decode) | 3.6% |
| mode-of-N (critic-free re-ranker) | 6.8% |
| **selector Best-of-N** | **4.9%** |
| oracle pass@16 (ceiling) | 15.0% |

Gap recovered: 12.2%.
