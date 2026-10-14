# C3 — reaction latency (frames to next action-state change; cap 45)

| set | after opp lands | after opp grabs ledge | after self lands |
|---|---|---|---|
| expert | med 8 · mean 12.4 · p90 32 · ≥cap 5% (n=25841) | med 9 · mean 12.5 · p90 29 · ≥cap 2% (n=2377) | med 8 · mean 9.6 · p90 20 · ≥cap 0% (n=31206) |
| bot | med 10 · mean 14.1 · p90 37 · ≥cap 7% (n=2302) | med 9 · mean 14.0 · p90 36 · ≥cap 8% (n=167) | med 9 · mean 10.4 · p90 26 · ≥cap 0% (n=2472) |

Caveats: action-state change is a proxy for "reacted" (landing lag ends
count as changes; identical for every set, so comparisons stand). The cap
share is the dithering signal.
