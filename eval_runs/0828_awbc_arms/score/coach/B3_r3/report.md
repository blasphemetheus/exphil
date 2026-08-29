# Coach Report — 20260829_120256

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.47 |
| Approaches (total) | 9 |
| Conversions (total) | 3 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 7 |
| Death sequences | 1 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 1.47 | 9 | 3 | 7 | 7 | 1 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r3.slp:691 — opening@691 start_pct=63.1 window=120f
- **dropped_punish** r3.slp:1661 — opening@1661 start_pct=119.7 window=120f
- **dropped_punish** r3.slp:3303 — opening@3303 start_pct=143.3 window=120f
- **dropped_punish** r3.slp:3702 — opening@3702 start_pct=152.3 window=120f
- **dropped_punish** r3.slp:5701 — opening@5701 start_pct=42.1 window=120f
- **dropped_punish** r3.slp:6108 — opening@6108 start_pct=70.1 window=120f
- **dropped_punish** r3.slp:6366 — opening@6366 start_pct=71.1 window=120f
- **neutral_loss** r3.slp:388 — hit@478 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r3.slp:1095 — hit@1185 opener=p2_action44 dist_90f_prior=18.1
- **neutral_loss** r3.slp:2414 — hit@2504 opener=p2_action44 dist_90f_prior=10.3

Appended 10 new gap(s) to `scenarios/gaps.json` (1449 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
