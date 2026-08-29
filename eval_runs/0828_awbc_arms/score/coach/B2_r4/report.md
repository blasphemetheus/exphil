# Coach Report — 20260829_120249

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 4 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 11 |
| Dropped punishes | 6 |
| Death sequences | 1 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.49 | 4 | 0 | 11 | 6 | 1 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r4.slp:727 — opening@727 start_pct=17.3 window=120f
- **dropped_punish** r4.slp:976 — opening@976 start_pct=21.4 window=120f
- **dropped_punish** r4.slp:2299 — opening@2299 start_pct=92.5 window=120f
- **dropped_punish** r4.slp:4797 — opening@4797 start_pct=160.7 window=120f
- **dropped_punish** r4.slp:5166 — opening@5166 start_pct=178.5 window=120f
- **dropped_punish** r4.slp:6505 — opening@6505 start_pct=49.6 window=120f
- **neutral_loss** r4.slp:390 — hit@480 opener=p2_action44 dist_90f_prior=9.2
- **neutral_loss** r4.slp:796 — hit@886 opener=p2_action57 dist_90f_prior=23.5
- **neutral_loss** r4.slp:1462 — hit@1552 opener=p2_action60 dist_90f_prior=26.6
- **neutral_loss** r4.slp:1855 — hit@1945 opener=p2_action44 dist_90f_prior=27.0

Appended 10 new gap(s) to `scenarios/gaps.json` (1379 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
