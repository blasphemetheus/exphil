# Coach Report — 20260829_231251

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 4 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.49 | 2 | 0 | 11 | 4 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r2.slp:958 — opening@958 start_pct=47.6 window=120f
- **dropped_punish** r2.slp:1258 — opening@1258 start_pct=52.1 window=120f
- **dropped_punish** r2.slp:2216 — opening@2216 start_pct=81.1 window=120f
- **dropped_punish** r2.slp:6257 — opening@6257 start_pct=35.7 window=120f
- **neutral_loss** r2.slp:1041 — hit@1131 opener=p2_action53 dist_90f_prior=50.3
- **neutral_loss** r2.slp:1295 — hit@1385 opener=p2_action53 dist_90f_prior=56.2
- **neutral_loss** r2.slp:1952 — hit@2042 opener=p2_action45 dist_90f_prior=11.4
- **neutral_loss** r2.slp:2570 — hit@2660 opener=p2_action60 dist_90f_prior=6.9
- **neutral_loss** r2.slp:3448 — hit@3538 opener=p2_action60 dist_90f_prior=33.5
- **neutral_loss** r2.slp:3834 — hit@3924 opener=p2_action60 dist_90f_prior=51.6

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
