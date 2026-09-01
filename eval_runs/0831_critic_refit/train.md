# D2 critic training report

Data: cache/critic/v12arrefit_erickfm40_k16.nx
Rows train/eval 55607/20331, replays 25/8, decision rows, K=16, L2 0.001.

| metric | value |
|---|---|
| V held-out R^2 | 0.059 |
| V pair-rank acc (chance 0.5) | 0.593 |
| V pair-rank acc, shuffled-target control | 0.51 |
| S top-1 among K+1 incl. master (chance 5.9%) | 61.4% |
| **sampling pass@1** (random sample == master) | 1.6% |
| **selector pass@1** (argmax-scored sample == master) | **12.7%** |
| oracle pass@16 (any sample == master) | 17.0% |
| selector pass@1, shuffled-label control | 9.3% |


**Verdict:** SUSPECT: the shuffled-label control also beats sampling — the gain is not from the (state, action) pairing.

Gap recovered: 72.1% of (oracle - sampling).
