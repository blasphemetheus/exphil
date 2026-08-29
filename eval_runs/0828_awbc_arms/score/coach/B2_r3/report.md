# Coach Report — 20260829_120248

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 3 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 5 |
| Death sequences | 3 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 3 | 0 | 6 | 5 | 3 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:3049 — passive_run=308f mid_dist=8.3
- **passivity_window** r3.slp:3558 — passive_run=326f mid_dist=8.3
- **passivity_window** r3.slp:6133 — passive_run=422f mid_dist=10.5
- **dropped_punish** r3.slp:1354 — opening@1354 start_pct=45.4 window=120f
- **dropped_punish** r3.slp:4020 — opening@4020 start_pct=168.2 window=120f
- **dropped_punish** r3.slp:4917 — opening@4917 start_pct=23.1 window=120f
- **dropped_punish** r3.slp:5976 — opening@5976 start_pct=58.5 window=120f
- **dropped_punish** r3.slp:7219 — opening@7219 start_pct=95.1 window=120f
- **neutral_loss** r3.slp:317 — hit@407 opener=p2_action44 dist_90f_prior=13.5
- **neutral_loss** r3.slp:849 — hit@939 opener=p2_action44 dist_90f_prior=19.4

Appended 10 new gap(s) to `scenarios/gaps.json` (1369 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
