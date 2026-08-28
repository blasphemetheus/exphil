# Coach Report — 20260828_123230

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 6 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 6 |
| Death sequences | 2 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 6 | 2 | 10 | 6 | 2 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:3634 — passive_run=372f mid_dist=8.3
- **passivity_window** r3.slp:6499 — passive_run=674f mid_dist=8.3
- **dropped_punish** r3.slp:419 — opening@419 start_pct=9.7 window=120f
- **dropped_punish** r3.slp:1721 — opening@1721 start_pct=69.9 window=120f
- **dropped_punish** r3.slp:2206 — opening@2206 start_pct=92.1 window=120f
- **dropped_punish** r3.slp:2643 — opening@2643 start_pct=112.3 window=120f
- **dropped_punish** r3.slp:3384 — opening@3384 start_pct=140.8 window=120f
- **dropped_punish** r3.slp:4816 — opening@4816 start_pct=244.8 window=120f
- **neutral_loss** r3.slp:432 — hit@522 opener=p2_action356 dist_90f_prior=19.8
- **neutral_loss** r3.slp:1408 — hit@1498 opener=p2_action48 dist_90f_prior=6.7

Appended 10 new gap(s) to `scenarios/gaps.json` (751 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
