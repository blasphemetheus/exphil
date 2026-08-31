# C3 — reaction latency (frames to next action-state change; cap 45)

| set | after opp lands | after opp grabs ledge | after self lands |
|---|---|---|---|
| expert | med 8 · mean 13.1 · p90 35 · ≥cap 6% (n=26584) | med 9 · mean 13.0 · p90 30 · ≥cap 3% (n=2413) | med 9 · mean 10.1 · p90 20 · ≥cap 1% (n=32035) |
| AR | med 32 · mean 28.5 · p90 45 · ≥cap 47% (n=408) | med 6 · mean 13.8 · p90 45 · ≥cap 25% (n=4) | med 7 · mean 7.9 · p90 11 · ≥cap 0% (n=225) |
| IND | med 45 · mean 34.0 · p90 45 · ≥cap 64% (n=401) | – (0) | med 5 · mean 7.3 · p90 10 · ≥cap 0% (n=246) |
| ep10_cpu | med 12 · mean 18.5 · p90 45 · ≥cap 20% (n=352) | med 19 · mean 17.7 · p90 32 · ≥cap 8% (n=12) | med 7 · mean 8.4 · p90 17 · ≥cap 0% (n=393) |
| B1_human | med 17 · mean 22.1 · p90 45 · ≥cap 22% (n=246) | med 16 · mean 23.4 · p90 45 · ≥cap 24% (n=17) | med 10 · mean 10.8 · p90 26 · ≥cap 0% (n=219) |

Caveats: action-state change is a proxy for "reacted" (landing lag ends
count as changes; identical for every set, so comparisons stand). The cap
share is the dithering signal.
