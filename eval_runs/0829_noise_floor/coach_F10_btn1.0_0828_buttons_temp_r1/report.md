# Coach Report — 20260829_231257

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 2 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 1 |
| Death sequences | 2 |
| Passivity windows | 1 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r1.slp | 0.0 | 2 | 1 | 7 | 1 | 2 | 1 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r1.slp:5308 — passive_run=541f mid_dist=8.3
- **dropped_punish** r1.slp:6724 — opening@6724 start_pct=300.8 window=120f
- **neutral_loss** r1.slp:1319 — hit@1409 opener=p2_action60 dist_90f_prior=57.1
- **neutral_loss** r1.slp:1672 — hit@1762 opener=p2_action44 dist_90f_prior=18.9
- **neutral_loss** r1.slp:3738 — hit@3828 opener=p2_action57 dist_90f_prior=65.0
- **neutral_loss** r1.slp:4402 — hit@4492 opener=p2_action63 dist_90f_prior=5.9
- **neutral_loss** r1.slp:5742 — hit@5832 opener=p2_action44 dist_90f_prior=19.8
- **neutral_loss** r1.slp:6407 — hit@6497 opener=p2_action44 dist_90f_prior=10.3
- **neutral_loss** r1.slp:6786 — hit@6876 opener=p2_action60 dist_90f_prior=61.2
- **death_sequence** r1.slp:390 — death@391 elapsed=1f opener=p2_action14

Appended 0 new gap(s) to `scenarios/gaps.json` (1832 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
