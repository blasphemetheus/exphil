# Coach Report — 20260829_120254

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 7 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 9 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.98 | 7 | 2 | 9 | 2 | 1 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:825 — passive_run=361f mid_dist=15.7
- **dropped_punish** r1.slp:1598 — opening@1598 start_pct=100.6 window=120f
- **dropped_punish** r1.slp:5166 — opening@5166 start_pct=207.6 window=120f
- **neutral_loss** r1.slp:430 — hit@520 opener=p2_action60 dist_90f_prior=23.4
- **neutral_loss** r1.slp:758 — hit@848 opener=p2_action44 dist_90f_prior=55.3
- **neutral_loss** r1.slp:1642 — hit@1732 opener=p2_action56 dist_90f_prior=57.1
- **neutral_loss** r1.slp:2014 — hit@2104 opener=p2_action53 dist_90f_prior=58.3
- **neutral_loss** r1.slp:2625 — hit@2715 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r1.slp:3006 — hit@3096 opener=p2_action356 dist_90f_prior=142.8
- **neutral_loss** r1.slp:3449 — hit@3539 opener=p2_action44 dist_90f_prior=6.9

Appended 10 new gap(s) to `scenarios/gaps.json` (1429 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
