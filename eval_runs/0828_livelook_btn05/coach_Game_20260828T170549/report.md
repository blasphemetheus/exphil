# Coach Report — 20260828_171136

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 7 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 10 |
| Dropped punishes | 3 |
| Death sequences | 4 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T170549.slp | 0.0 | 7 | 0 | 10 | 3 | 4 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T170549.slp:2309 — passive_run=433f mid_dist=15.4
- **dropped_punish** Game_20260828T170549.slp:1521 — opening@1521 start_pct=50.9 window=120f
- **dropped_punish** Game_20260828T170549.slp:1872 — opening@1872 start_pct=83.8 window=120f
- **dropped_punish** Game_20260828T170549.slp:6583 — opening@6583 start_pct=145.8 window=120f
- **neutral_loss** Game_20260828T170549.slp:1967 — hit@2057 opener=p2_action215 dist_90f_prior=27.2
- **neutral_loss** Game_20260828T170549.slp:3361 — hit@3451 opener=p2_action57 dist_90f_prior=71.2
- **neutral_loss** Game_20260828T170549.slp:4063 — hit@4153 opener=p2_action56 dist_90f_prior=22.1
- **neutral_loss** Game_20260828T170549.slp:5094 — hit@5184 opener=p2_action20 dist_90f_prior=99.3
- **neutral_loss** Game_20260828T170549.slp:5404 — hit@5494 opener=p2_action65 dist_90f_prior=29.5
- **neutral_loss** Game_20260828T170549.slp:5847 — hit@5937 opener=p2_action65 dist_90f_prior=72.4

Appended 10 new gap(s) to `scenarios/gaps.json` (1151 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
