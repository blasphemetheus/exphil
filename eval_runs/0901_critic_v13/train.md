# D2 critic training report

Data: cache/critic/v13arrefit_erickfm40_k16_r2.nx
Rows train/eval 55607/20331, replays 25/8, decision rows, K=16, L2 0.001.

| metric | value |
|---|---|
| V held-out R^2 | 0.092 |
| V pair-rank acc (chance 0.5) | 0.607 |
| V pair-rank acc, shuffled-target control | 0.512 |
| S top-1 among K+1 incl. master (chance 5.9%) | 58.7% |
| **sampling pass@1** (random sample == master) | 5.3% |
| **selector pass@1** (argmax-scored sample == master) | **23.8%** |
| oracle pass@16 (any sample == master) | 33.8% |
| selector pass@1, shuffled-label control | 15.1% |


**Verdict:** SUSPECT: the shuffled-label control also beats sampling — the gain is not from the (state, action) pairing.

Gap recovered: 65.0% of (oracle - sampling).
