# Coach Report — 20260829_120253

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 1 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 8 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r8.slp | 0.0 | 1 | 0 | 7 | 8 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r8.slp:6647 — passive_run=419f mid_dist=8.3
- **dropped_punish** r8.slp:628 — opening@628 start_pct=59.9 window=120f
- **dropped_punish** r8.slp:1713 — opening@1713 start_pct=111.1 window=120f
- **dropped_punish** r8.slp:2208 — opening@2208 start_pct=121.1 window=120f
- **dropped_punish** r8.slp:2994 — opening@2994 start_pct=145.9 window=120f
- **dropped_punish** r8.slp:3295 — opening@3295 start_pct=147.9 window=120f
- **dropped_punish** r8.slp:3665 — opening@3665 start_pct=156.9 window=120f
- **dropped_punish** r8.slp:4901 — opening@4901 start_pct=195.8 window=120f
- **dropped_punish** r8.slp:5355 — opening@5355 start_pct=203.8 window=120f
- **neutral_loss** r8.slp:864 — hit@954 opener=p2_action44 dist_90f_prior=6.5

Appended 10 new gap(s) to `scenarios/gaps.json` (1419 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
