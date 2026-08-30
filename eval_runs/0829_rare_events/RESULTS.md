# Rare-event coverage (E2) — replays/erickfm_ranked/FOX/extracted/*.slp

7905 games, 23725.4 minutes of play, subject port 1. Temporal recipe
stride 5 → a single-frame event is a window label ≈ 1 in 5 times.

| behaviour | count | per game | per minute | ≈ targets / epoch (÷5) |
|---|---:|---:|---:|---:|
| dash initiation (stand/walk/turn → DASH) | 787335 | 99.6 | 33.2 | 157467 |
| dash-dance bursts (≥3 flips / 24 f) | 84010 | 10.6 | 3.5 | 16802 |
| grab out of dash/run | 4830 | 0.6 | 0.2 | 966 |
| grab entries (any) | 61670 | 7.8 | 2.6 | 12334 |
| throw from a held grab | 29711 | 3.8 | 1.3 | 5942 |
| up-B (firefox) start while offstage | 33798 | 4.3 | 1.4 | 6759 |
| side-B (illusion) start while offstage | 16084 | 2.0 | 0.7 | 3216 |
| airdodge while offstage | 4949 | 0.6 | 0.2 | 989 |

Grab-hold frames (GRAB_WAIT + PUMMEL): 512235 → mean hold 8.3 f per grab entry.

## Read

- "Dozens" would mean a data problem; anything in the thousands per epoch
  means the recipe sees the behaviour many times and still does not
  reproduce it — the loss is downstream (label weighting, window/stride
  alignment, discretization, or the model). Compare to the bot's rates in
  `eval_runs/0829_situation_hist/README.md`.
- Throw-from-grab vs airdodge-offstage ratio is the direct denominator for
  A1 §2/§3.
