# Coach Report — 20260828_162818

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 5 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.49 | 5 | 1 | 7 | 5 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r4.slp:1974 — opening@1974 start_pct=137.4 window=120f
- **dropped_punish** r4.slp:2385 — opening@2385 start_pct=166.9 window=120f
- **dropped_punish** r4.slp:3064 — opening@3064 start_pct=29.2 window=120f
- **dropped_punish** r4.slp:3809 — opening@3809 start_pct=29.2 window=120f
- **dropped_punish** r4.slp:4140 — opening@4140 start_pct=47.2 window=120f
- **neutral_loss** r4.slp:476 — hit@566 opener=p2_action44 dist_90f_prior=70.8
- **neutral_loss** r4.slp:1193 — hit@1283 opener=p2_action44 dist_90f_prior=33.3
- **neutral_loss** r4.slp:1632 — hit@1722 opener=p2_action44 dist_90f_prior=6.6
- **neutral_loss** r4.slp:3272 — hit@3362 opener=p2_action60 dist_90f_prior=61.2
- **neutral_loss** r4.slp:4431 — hit@4521 opener=p2_action44 dist_90f_prior=6.5

Appended 10 new gap(s) to `scenarios/gaps.json` (1031 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
