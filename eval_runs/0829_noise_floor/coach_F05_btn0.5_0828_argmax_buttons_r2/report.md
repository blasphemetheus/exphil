# Coach Report — 20260829_231226

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 5 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.49 | 2 | 1 | 8 | 5 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r2.slp:1108 — opening@1108 start_pct=63.5 window=120f
- **dropped_punish** r2.slp:3265 — opening@3265 start_pct=128.4 window=120f
- **dropped_punish** r2.slp:4196 — opening@4196 start_pct=155.7 window=120f
- **dropped_punish** r2.slp:5064 — opening@5064 start_pct=0.0 window=120f
- **dropped_punish** r2.slp:6616 — opening@6616 start_pct=70.0 window=120f
- **neutral_loss** r2.slp:536 — hit@626 opener=p2_action53 dist_90f_prior=58.1
- **neutral_loss** r2.slp:1148 — hit@1238 opener=p2_action44 dist_90f_prior=39.7
- **neutral_loss** r2.slp:1850 — hit@1940 opener=p2_action60 dist_90f_prior=32.2
- **neutral_loss** r2.slp:2132 — hit@2222 opener=p2_action44 dist_90f_prior=50.5
- **neutral_loss** r2.slp:3087 — hit@3177 opener=p2_action44 dist_90f_prior=6.9

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
