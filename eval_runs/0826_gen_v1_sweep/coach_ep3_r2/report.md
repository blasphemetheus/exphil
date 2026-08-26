# Coach Report — 20260826_124537

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.49 |
| Approaches (total) | 4 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 3 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r2.slp | 0.49 | 4 | 1 | 8 | 3 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r2.slp:1565 — passive_run=341f mid_dist=8.3
- **passivity_window** r2.slp:2376 — passive_run=382f mid_dist=7.1
- **dropped_punish** r2.slp:803 — opening@803 start_pct=82.7 window=120f
- **dropped_punish** r2.slp:3460 — opening@3460 start_pct=47.2 window=120f
- **dropped_punish** r2.slp:7177 — opening@7177 start_pct=305.3 window=120f
- **neutral_loss** r2.slp:442 — hit@532 opener=p2_action44 dist_90f_prior=26.2
- **neutral_loss** r2.slp:846 — hit@936 opener=p2_action57 dist_90f_prior=57.1
- **neutral_loss** r2.slp:1161 — hit@1251 opener=p2_action57 dist_90f_prior=58.5
- **neutral_loss** r2.slp:1443 — hit@1533 opener=p2_action44 dist_90f_prior=78.0
- **neutral_loss** r2.slp:2643 — hit@2733 opener=p2_action44 dist_90f_prior=24.9

Appended 10 new gap(s) to `scenarios/gaps.json` (305 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
