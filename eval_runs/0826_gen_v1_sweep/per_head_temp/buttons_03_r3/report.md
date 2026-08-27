# Coach Report — 20260827_181712

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 7 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 7 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.49 | 7 | 3 | 7 | 7 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r3.slp:1899 — opening@1899 start_pct=14.8 window=120f
- **dropped_punish** r3.slp:2271 — opening@2271 start_pct=19.2 window=120f
- **dropped_punish** r3.slp:2639 — opening@2639 start_pct=26.2 window=120f
- **dropped_punish** r3.slp:3254 — opening@3254 start_pct=48.8 window=120f
- **dropped_punish** r3.slp:4301 — opening@4301 start_pct=55.8 window=120f
- **dropped_punish** r3.slp:5198 — opening@5198 start_pct=80.0 window=120f
- **dropped_punish** r3.slp:6385 — opening@6385 start_pct=129.7 window=120f
- **neutral_loss** r3.slp:394 — hit@484 opener=p2_action44 dist_90f_prior=6.7
- **neutral_loss** r3.slp:922 — hit@1012 opener=p2_action44 dist_90f_prior=7.2
- **neutral_loss** r3.slp:1630 — hit@1720 opener=p2_action44 dist_90f_prior=2.2

Appended 10 new gap(s) to `scenarios/gaps.json` (604 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
