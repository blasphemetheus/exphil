# Coach Report — 20260826_123831

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.49 | 2 | 0 | 9 | 3 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r1.slp:1123 — opening@1123 start_pct=44.6 window=120f
- **dropped_punish** r1.slp:3091 — opening@3091 start_pct=172.8 window=120f
- **dropped_punish** r1.slp:3629 — opening@3629 start_pct=177.8 window=120f
- **neutral_loss** r1.slp:779 — hit@869 opener=p2_action57 dist_90f_prior=23.4
- **neutral_loss** r1.slp:1282 — hit@1372 opener=p2_action60 dist_90f_prior=7.3
- **neutral_loss** r1.slp:1704 — hit@1794 opener=p2_action60 dist_90f_prior=40.8
- **neutral_loss** r1.slp:2293 — hit@2383 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r1.slp:2686 — hit@2776 opener=p2_action44 dist_90f_prior=64.3
- **neutral_loss** r1.slp:3978 — hit@4068 opener=p2_action44 dist_90f_prior=15.2
- **neutral_loss** r1.slp:4681 — hit@4771 opener=p2_action44 dist_90f_prior=15.0

Appended 10 new gap(s) to `scenarios/gaps.json` (266 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
