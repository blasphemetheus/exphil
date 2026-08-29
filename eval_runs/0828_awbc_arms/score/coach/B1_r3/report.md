# Coach Report — 20260829_120242

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.96 |
| Approaches (total) | 8 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 7 |
| Death sequences | 3 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 1.96 | 8 | 2 | 6 | 7 | 3 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:4957 — passive_run=347f mid_dist=8.3
- **passivity_window** r3.slp:6118 — passive_run=304f mid_dist=8.3
- **dropped_punish** r3.slp:385 — opening@385 start_pct=51.2 window=120f
- **dropped_punish** r3.slp:1043 — opening@1043 start_pct=83.4 window=120f
- **dropped_punish** r3.slp:1514 — opening@1514 start_pct=99.4 window=120f
- **dropped_punish** r3.slp:4430 — opening@4430 start_pct=158.4 window=120f
- **dropped_punish** r3.slp:4685 — opening@4685 start_pct=165.4 window=120f
- **dropped_punish** r3.slp:5521 — opening@5521 start_pct=194.5 window=120f
- **dropped_punish** r3.slp:6614 — opening@6614 start_pct=243.5 window=120f
- **neutral_loss** r3.slp:1258 — hit@1348 opener=p2_action56 dist_90f_prior=21.4

Appended 10 new gap(s) to `scenarios/gaps.json` (1289 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
