# Coach Report — 20260829_120256

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 6 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.49 | 6 | 0 | 7 | 2 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r4.slp:2151 — passive_run=309f mid_dist=6.2
- **passivity_window** r4.slp:5239 — passive_run=325f mid_dist=8.3
- **dropped_punish** r4.slp:1531 — opening@1531 start_pct=83.4 window=120f
- **dropped_punish** r4.slp:4111 — opening@4111 start_pct=155.7 window=120f
- **neutral_loss** r4.slp:1133 — hit@1223 opener=p2_action44 dist_90f_prior=19.9
- **neutral_loss** r4.slp:1822 — hit@1912 opener=p2_action44 dist_90f_prior=0.5
- **neutral_loss** r4.slp:2359 — hit@2449 opener=p2_action60 dist_90f_prior=6.6
- **neutral_loss** r4.slp:3250 — hit@3340 opener=p2_action60 dist_90f_prior=39.8
- **neutral_loss** r4.slp:3609 — hit@3699 opener=p2_action57 dist_90f_prior=52.4
- **neutral_loss** r4.slp:3885 — hit@3975 opener=p2_action57 dist_90f_prior=0.2

Appended 10 new gap(s) to `scenarios/gaps.json` (1459 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
