# Coach Report — 20260829_224715

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.51 |
| Approaches (total) | 12 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 4 |
| Death sequences | 4 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260829T221405.slp | 0.51 | 12 | 2 | 9 | 4 | 4 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260829T221405.slp:1688 — passive_run=302f mid_dist=12.7
- **passivity_window** Game_20260829T221405.slp:3518 — passive_run=322f mid_dist=19.3
- **passivity_window** Game_20260829T221405.slp:4132 — passive_run=350f mid_dist=5.4
- **dropped_punish** Game_20260829T221405.slp:824 — opening@824 start_pct=23.4 window=120f
- **dropped_punish** Game_20260829T221405.slp:1996 — opening@1996 start_pct=76.6 window=120f
- **dropped_punish** Game_20260829T221405.slp:2483 — opening@2483 start_pct=91.3 window=120f
- **dropped_punish** Game_20260829T221405.slp:6460 — opening@6460 start_pct=3.0 window=120f
- **neutral_loss** Game_20260829T221405.slp:976 — hit@1066 opener=p2_action65 dist_90f_prior=45.0
- **neutral_loss** Game_20260829T221405.slp:1446 — hit@1536 opener=p2_action67 dist_90f_prior=9.0
- **neutral_loss** Game_20260829T221405.slp:2866 — hit@2956 opener=p2_action67 dist_90f_prior=15.2

Appended 10 new gap(s) to `scenarios/gaps.json` (1812 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
