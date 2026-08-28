# Coach Report — 20260828_171159

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.75 |
| Approaches (total) | 9 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 2 |
| Death sequences | 4 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T054605.slp | 0.75 | 9 | 1 | 11 | 2 | 4 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T054605.slp:794 — passive_run=315f mid_dist=9.6
- **passivity_window** Game_20260828T054605.slp:7568 — passive_run=320f mid_dist=8.3
- **dropped_punish** Game_20260828T054605.slp:564 — opening@564 start_pct=53.0 window=120f
- **dropped_punish** Game_20260828T054605.slp:5569 — opening@5569 start_pct=154.3 window=120f
- **neutral_loss** Game_20260828T054605.slp:887 — hit@977 opener=p2_action44 dist_90f_prior=24.4
- **neutral_loss** Game_20260828T054605.slp:1129 — hit@1219 opener=p2_action63 dist_90f_prior=75.5
- **neutral_loss** Game_20260828T054605.slp:1956 — hit@2046 opener=p2_action69 dist_90f_prior=1.3
- **neutral_loss** Game_20260828T054605.slp:2479 — hit@2569 opener=p2_action65 dist_90f_prior=32.6
- **neutral_loss** Game_20260828T054605.slp:3475 — hit@3565 opener=p2_action63 dist_90f_prior=56.5
- **neutral_loss** Game_20260828T054605.slp:4728 — hit@4818 opener=p2_action213 dist_90f_prior=29.7

Appended 10 new gap(s) to `scenarios/gaps.json` (1239 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
