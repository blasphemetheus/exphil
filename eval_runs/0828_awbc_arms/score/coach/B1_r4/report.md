# Coach Report — 20260829_120242

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 3 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.0 | 2 | 0 | 7 | 3 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** r4.slp:1447 — opening@1447 start_pct=72.4 window=120f
- **dropped_punish** r4.slp:2744 — opening@2744 start_pct=85.8 window=120f
- **dropped_punish** r4.slp:5013 — opening@5013 start_pct=204.9 window=120f
- **neutral_loss** r4.slp:1619 — hit@1709 opener=p2_action64 dist_90f_prior=31.5
- **neutral_loss** r4.slp:2247 — hit@2337 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r4.slp:3453 — hit@3543 opener=p2_action57 dist_90f_prior=28.8
- **neutral_loss** r4.slp:3770 — hit@3860 opener=p2_action44 dist_90f_prior=28.6
- **neutral_loss** r4.slp:4648 — hit@4738 opener=p2_action60 dist_90f_prior=103.9
- **neutral_loss** r4.slp:5683 — hit@5773 opener=p2_action44 dist_90f_prior=35.2
- **neutral_loss** r4.slp:6114 — hit@6204 opener=p2_action44 dist_90f_prior=7.1

Appended 10 new gap(s) to `scenarios/gaps.json` (1299 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
