# D2 critic training report

Data: cache/critic/fox_gen_v1_ep10_erickfm40.nx
Rows train/eval 66797/23706, replays 30/10, decision rows, K=8, L2 0.001.

| metric | value |
|---|---|
| V held-out R^2 | 0.046 |
| V pair-rank acc (chance 0.5) | 0.599 |
| V pair-rank acc, shuffled-target control | 0.514 |
| S top-1 among K+1 incl. master (chance 11.1%) | 49.8% |
| **sampling pass@1** (random sample == master) | 16.0% |
| **selector pass@1** (argmax-scored sample == master) | **22.2%** |
| oracle pass@8 (any sample == master) | 38.8% |
| selector pass@1, shuffled-label control | 12.1% |


**Verdict:** PARTIAL: the selector recovers 27.0% of the gap. Real but linear-limited — try a small MLP head before wiring live.

Gap recovered: 27.0% of (oracle - sampling).
