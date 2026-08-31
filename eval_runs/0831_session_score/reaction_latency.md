# C3 — reaction latency (frames to next action-state change; cap 45)

| set | after opp lands | after opp grabs ledge | after self lands |
|---|---|---|---|
| expert | med 8 · mean 13.1 · p90 35 · ≥cap 6% (n=26584) | med 9 · mean 13.0 · p90 30 · ≥cap 3% (n=2413) | med 9 · mean 10.1 · p90 20 · ≥cap 1% (n=32035) |
| AR_human | med 18 · mean 21.3 · p90 45 · ≥cap 17% (n=650) | med 20 · mean 23.4 · p90 45 · ≥cap 18% (n=40) | med 10 · mean 11.2 · p90 26 · ≥cap 0% (n=568) |
| IND_human | med 21 · mean 22.5 · p90 45 · ≥cap 20% (n=217) | med 32 · mean 26.6 · p90 45 · ≥cap 18% (n=11) | med 10 · mean 11.6 · p90 26 · ≥cap 0% (n=215) |
| B1_human | med 17 · mean 22.1 · p90 45 · ≥cap 22% (n=246) | med 16 · mean 23.4 · p90 45 · ≥cap 24% (n=17) | med 10 · mean 10.8 · p90 26 · ≥cap 0% (n=219) |

Caveats: action-state change is a proxy for "reacted" (landing lag ends
count as changes; identical for every set, so comparisons stand). The cap
share is the dithering signal.
