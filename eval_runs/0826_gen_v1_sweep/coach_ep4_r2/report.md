# Coach Report — 20260826_125354

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 2 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.98 | 4 | 1 | 11 | 2 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r2.slp:2182 — opening@2182 start_pct=154.7 window=120f
- **dropped_punish** r2.slp:3269 — opening@3269 start_pct=164.7 window=120f
- **neutral_loss** r2.slp:421 — hit@511 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r2.slp:1161 — hit@1251 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r2.slp:1530 — hit@1620 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r2.slp:1855 — hit@1945 opener=p2_action44 dist_90f_prior=7.0
- **neutral_loss** r2.slp:3389 — hit@3479 opener=p2_action44 dist_90f_prior=50.6
- **neutral_loss** r2.slp:3923 — hit@4013 opener=p2_action44 dist_90f_prior=7.7
- **neutral_loss** r2.slp:4330 — hit@4420 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r2.slp:5173 — hit@5263 opener=p2_action44 dist_90f_prior=24.1

Appended 10 new gap(s) to `scenarios/gaps.json` (335 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
