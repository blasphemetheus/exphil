# Coach Report — 20260829_231248

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 1 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 0 |
| Death sequences | 1 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r3.slp | 0.0 | 1 | 0 | 5 | 0 | 1 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r3.slp:4302 — passive_run=559f mid_dist=8.3
- **passivity_window** r3.slp:5422 — passive_run=425f mid_dist=8.3
- **passivity_window** r3.slp:6741 — passive_run=502f mid_dist=8.3
- **neutral_loss** r3.slp:2089 — hit@2179 opener=p2_action44 dist_90f_prior=11.9
- **neutral_loss** r3.slp:3225 — hit@3315 opener=p2_action356 dist_90f_prior=25.2
- **neutral_loss** r3.slp:3622 — hit@3712 opener=p2_action44 dist_90f_prior=69.3
- **neutral_loss** r3.slp:5284 — hit@5374 opener=p2_action57 dist_90f_prior=41.9
- **neutral_loss** r3.slp:6535 — hit@6625 opener=p2_action44 dist_90f_prior=22.7
- **death_sequence** r3.slp:4740 — death@4741 elapsed=1f opener=p2_action15

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
