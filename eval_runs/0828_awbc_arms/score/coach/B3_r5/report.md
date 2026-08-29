# Coach Report — 20260829_120257

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.47 |
| Approaches (total) | 7 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 6 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r5.slp | 1.47 | 7 | 1 | 7 | 6 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r5.slp:1578 — opening@1578 start_pct=83.3 window=120f
- **dropped_punish** r5.slp:2309 — opening@2309 start_pct=7.0 window=120f
- **dropped_punish** r5.slp:3602 — opening@3602 start_pct=46.2 window=120f
- **dropped_punish** r5.slp:5848 — opening@5848 start_pct=114.1 window=120f
- **dropped_punish** r5.slp:6173 — opening@6173 start_pct=114.1 window=120f
- **dropped_punish** r5.slp:6795 — opening@6795 start_pct=138.8 window=120f
- **neutral_loss** r5.slp:1613 — hit@1703 opener=p2_action56 dist_90f_prior=17.0
- **neutral_loss** r5.slp:2685 — hit@2775 opener=p2_action44 dist_90f_prior=30.8
- **neutral_loss** r5.slp:3135 — hit@3225 opener=p2_action57 dist_90f_prior=14.8
- **neutral_loss** r5.slp:3427 — hit@3517 opener=p2_action44 dist_90f_prior=14.3

Appended 10 new gap(s) to `scenarios/gaps.json` (1469 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
