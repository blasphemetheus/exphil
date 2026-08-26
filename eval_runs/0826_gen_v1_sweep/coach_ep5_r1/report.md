# Coach Report — 20260826_130210

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 3 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 3 |
| Dropped punishes | 2 |
| Death sequences | 2 |
| Passivity windows | 4 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.98 | 3 | 2 | 3 | 2 | 2 | 4 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:1405 — passive_run=344f mid_dist=8.3
- **passivity_window** r1.slp:2118 — passive_run=317f mid_dist=8.3
- **passivity_window** r1.slp:4294 — passive_run=432f mid_dist=8.3
- **passivity_window** r1.slp:4733 — passive_run=442f mid_dist=8.3
- **dropped_punish** r1.slp:722 — opening@722 start_pct=39.8 window=120f
- **dropped_punish** r1.slp:6982 — opening@6982 start_pct=306.8 window=120f
- **neutral_loss** r1.slp:3031 — hit@3121 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r1.slp:4176 — hit@4266 opener=p2_action44 dist_90f_prior=48.4
- **neutral_loss** r1.slp:5642 — hit@5732 opener=p2_action57 dist_90f_prior=57.1
- **death_sequence** r1.slp:1893 — death@1894 elapsed=1f opener=p2_action14

Appended 10 new gap(s) to `scenarios/gaps.json` (355 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
