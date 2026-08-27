# Coach Report — 20260827_182849

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 6 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 4 |
| Dropped punishes | 5 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 6 | 1 | 4 | 5 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r3.slp:965 — opening@965 start_pct=54.3 window=120f
- **dropped_punish** r3.slp:3259 — opening@3259 start_pct=143.0 window=120f
- **dropped_punish** r3.slp:3691 — opening@3691 start_pct=145.0 window=120f
- **dropped_punish** r3.slp:3998 — opening@3998 start_pct=160.0 window=120f
- **dropped_punish** r3.slp:5480 — opening@5480 start_pct=52.4 window=120f
- **neutral_loss** r3.slp:361 — hit@451 opener=p2_action44 dist_90f_prior=19.7
- **neutral_loss** r3.slp:2853 — hit@2943 opener=p2_action44 dist_90f_prior=10.2
- **neutral_loss** r3.slp:4439 — hit@4529 opener=p2_action44 dist_90f_prior=63.8
- **neutral_loss** r3.slp:5173 — hit@5263 opener=p2_action44 dist_90f_prior=32.8
- **death_sequence** r3.slp:2574 — death@2575 elapsed=1f opener=p2_action358

Appended 10 new gap(s) to `scenarios/gaps.json` (654 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
