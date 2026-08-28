# Coach Report — 20260828_162821

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 6 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r8.slp | 0.49 | 2 | 1 | 7 | 6 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r8.slp:3323 — passive_run=334f mid_dist=8.3
- **passivity_window** r8.slp:4237 — passive_run=347f mid_dist=5.2
- **dropped_punish** r8.slp:706 — opening@706 start_pct=14.0 window=120f
- **dropped_punish** r8.slp:1773 — opening@1773 start_pct=52.7 window=120f
- **dropped_punish** r8.slp:3083 — opening@3083 start_pct=159.8 window=120f
- **dropped_punish** r8.slp:4562 — opening@4562 start_pct=220.6 window=120f
- **dropped_punish** r8.slp:4802 — opening@4802 start_pct=230.6 window=120f
- **dropped_punish** r8.slp:7062 — opening@7062 start_pct=99.2 window=120f
- **neutral_loss** r8.slp:778 — hit@868 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r8.slp:1251 — hit@1341 opener=p2_action44 dist_90f_prior=30.2

Appended 10 new gap(s) to `scenarios/gaps.json` (1071 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
