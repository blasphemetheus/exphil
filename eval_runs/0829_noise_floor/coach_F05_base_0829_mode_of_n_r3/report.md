# Coach Report — 20260829_231237

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 9 |
| Conversions (total) | 4 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 4 |
| Death sequences | 3 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 9 | 4 | 5 | 4 | 3 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:4208 — passive_run=338f mid_dist=8.3
- **passivity_window** r3.slp:4889 — passive_run=398f mid_dist=8.3
- **dropped_punish** r3.slp:736 — opening@736 start_pct=43.4 window=120f
- **dropped_punish** r3.slp:1927 — opening@1927 start_pct=82.7 window=120f
- **dropped_punish** r3.slp:2735 — opening@2735 start_pct=113.3 window=120f
- **dropped_punish** r3.slp:3055 — opening@3055 start_pct=123.3 window=120f
- **neutral_loss** r3.slp:340 — hit@430 opener=p2_action44 dist_90f_prior=6.5
- **neutral_loss** r3.slp:3212 — hit@3302 opener=p2_action60 dist_90f_prior=26.8
- **neutral_loss** r3.slp:5495 — hit@5585 opener=p2_action60 dist_90f_prior=6.8
- **neutral_loss** r3.slp:6001 — hit@6091 opener=p2_action60 dist_90f_prior=15.8

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
