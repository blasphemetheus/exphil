# Coach Report — 20260828_124553

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 1 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 1 |
| Death sequences | 0 |
| Passivity windows | 5 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 1 | 0 | 6 | 1 | 0 | 5 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:2462 — passive_run=339f mid_dist=8.3
- **passivity_window** r1.slp:3752 — passive_run=436f mid_dist=8.3
- **passivity_window** r1.slp:4194 — passive_run=620f mid_dist=8.3
- **passivity_window** r1.slp:5265 — passive_run=569f mid_dist=8.3
- **passivity_window** r1.slp:5840 — passive_run=572f mid_dist=8.3
- **dropped_punish** r1.slp:665 — opening@665 start_pct=56.5 window=120f
- **neutral_loss** r1.slp:1232 — hit@1322 opener=p2_action60 dist_90f_prior=67.6
- **neutral_loss** r1.slp:2922 — hit@3012 opener=p2_action57 dist_90f_prior=16.6
- **neutral_loss** r1.slp:3286 — hit@3376 opener=p2_action44 dist_90f_prior=12.4
- **neutral_loss** r1.slp:5049 — hit@5139 opener=p2_action44 dist_90f_prior=6.9

Appended 10 new gap(s) to `scenarios/gaps.json` (781 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, passivity.
