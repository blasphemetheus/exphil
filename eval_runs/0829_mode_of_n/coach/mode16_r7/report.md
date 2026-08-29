# Coach Report — 20260829_150203

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 0 |
| Death sequences | 4 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r7.slp | 0.0 | 2 | 0 | 6 | 0 | 4 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r7.slp:1015 — passive_run=1313f mid_dist=8.9
- **passivity_window** r7.slp:2440 — passive_run=327f mid_dist=11.0
- **neutral_loss** r7.slp:321 — hit@411 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r7.slp:681 — hit@771 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r7.slp:1058 — hit@1148 opener=p2_action44 dist_90f_prior=5.6
- **neutral_loss** r7.slp:1548 — hit@1638 opener=p2_action44 dist_90f_prior=7.1
- **neutral_loss** r7.slp:2026 — hit@2116 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r7.slp:2497 — hit@2587 opener=p2_action44 dist_90f_prior=6.9
- **death_sequence** r7.slp:2706 — death@2707 elapsed=1f opener=p2_action14
- **death_sequence** r7.slp:2968 — death@2969 elapsed=1f opener=p2_action16

Appended 10 new gap(s) to `scenarios/gaps.json` (1629 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, deaths, passivity.
