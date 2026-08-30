# Coach Report — 20260829_224718

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.5 |
| Approaches (total) | 12 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 0 |
| Death sequences | 3 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T221716.slp | 0.5 | 12 | 3 | 8 | 0 | 3 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T221716.slp:6604 — passive_run=435f mid_dist=34.6
- **neutral_loss** Game_20260829T221716.slp:1465 — hit@1555 opener=p2_action68 dist_90f_prior=46.6
- **neutral_loss** Game_20260829T221716.slp:2227 — hit@2317 opener=p2_action66 dist_90f_prior=3.9
- **neutral_loss** Game_20260829T221716.slp:2780 — hit@2870 opener=p2_action65 dist_90f_prior=101.7
- **neutral_loss** Game_20260829T221716.slp:3277 — hit@3367 opener=p2_action68 dist_90f_prior=21.8
- **neutral_loss** Game_20260829T221716.slp:4343 — hit@4433 opener=p2_action67 dist_90f_prior=9.8
- **neutral_loss** Game_20260829T221716.slp:4995 — hit@5085 opener=p2_action56 dist_90f_prior=52.6
- **neutral_loss** Game_20260829T221716.slp:5918 — hit@6008 opener=p2_action67 dist_90f_prior=60.4
- **neutral_loss** Game_20260829T221716.slp:6288 — hit@6378 opener=p2_action68 dist_90f_prior=16.3
- **death_sequence** Game_20260829T221716.slp:3082 — death@3083 elapsed=1f opener=p2_action27

Appended 10 new gap(s) to `scenarios/gaps.json` (1830 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
