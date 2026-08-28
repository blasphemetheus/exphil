# Coach Report — 20260828_171147

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.04 |
| Approaches (total) | 10 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T053254.slp | 1.04 | 10 | 1 | 11 | 1 | 4 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T053254.slp:2011 — passive_run=482f mid_dist=6.1
- **passivity_window** Game_20260828T053254.slp:2568 — passive_run=323f mid_dist=35.5
- **dropped_punish** Game_20260828T053254.slp:3477 — opening@3477 start_pct=90.9 window=120f
- **neutral_loss** Game_20260828T053254.slp:813 — hit@903 opener=p2_action63 dist_90f_prior=34.5
- **neutral_loss** Game_20260828T053254.slp:1191 — hit@1281 opener=p2_action215 dist_90f_prior=35.0
- **neutral_loss** Game_20260828T053254.slp:1500 — hit@1590 opener=p2_action67 dist_90f_prior=0.5
- **neutral_loss** Game_20260828T053254.slp:2121 — hit@2211 opener=p2_action68 dist_90f_prior=31.5
- **neutral_loss** Game_20260828T053254.slp:2495 — hit@2585 opener=p2_action215 dist_90f_prior=43.9
- **neutral_loss** Game_20260828T053254.slp:2993 — hit@3083 opener=p2_action215 dist_90f_prior=55.2
- **neutral_loss** Game_20260828T053254.slp:3611 — hit@3701 opener=p2_action57 dist_90f_prior=14.5

Appended 10 new gap(s) to `scenarios/gaps.json` (1161 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
