# Coach Report — 20260826_133514

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.98 |
| Approaches (total) | 4 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 4 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.98 | 4 | 2 | 8 | 4 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:4898 — passive_run=376f mid_dist=8.3
- **dropped_punish** r1.slp:1181 — opening@1181 start_pct=56.8 window=120f
- **dropped_punish** r1.slp:1443 — opening@1443 start_pct=64.5 window=120f
- **dropped_punish** r1.slp:5831 — opening@5831 start_pct=264.8 window=120f
- **dropped_punish** r1.slp:7294 — opening@7294 start_pct=46.3 window=120f
- **neutral_loss** r1.slp:767 — hit@857 opener=p2_action44 dist_90f_prior=19.8
- **neutral_loss** r1.slp:1902 — hit@1992 opener=p2_action48 dist_90f_prior=17.1
- **neutral_loss** r1.slp:2521 — hit@2611 opener=p2_action60 dist_90f_prior=12.1
- **neutral_loss** r1.slp:4360 — hit@4450 opener=p2_action44 dist_90f_prior=28.8
- **neutral_loss** r1.slp:4794 — hit@4884 opener=p2_action44 dist_90f_prior=84.1

Appended 10 new gap(s) to `scenarios/gaps.json` (475 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
