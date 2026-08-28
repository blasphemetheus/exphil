# Coach Report — 20260828_121906

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 0 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 2 |
| Death sequences | 1 |
| Passivity windows | 2 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| r4.slp | 0.0 | 0 | 0 | 6 | 2 | 1 | 2 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** r4.slp:6141 — passive_run=603f mid_dist=8.3
- **passivity_window** r4.slp:6750 — passive_run=467f mid_dist=8.3
- **dropped_punish** r4.slp:718 — opening@718 start_pct=110.1 window=120f
- **dropped_punish** r4.slp:2089 — opening@2089 start_pct=149.6 window=120f
- **neutral_loss** r4.slp:360 — hit@450 opener=p2_action44 dist_90f_prior=6.9
- **neutral_loss** r4.slp:775 — hit@865 opener=p2_action60 dist_90f_prior=57.3
- **neutral_loss** r4.slp:2138 — hit@2228 opener=p2_action57 dist_90f_prior=57.2
- **neutral_loss** r4.slp:3828 — hit@3918 opener=p2_action44 dist_90f_prior=23.0
- **neutral_loss** r4.slp:4314 — hit@4404 opener=p2_action44 dist_90f_prior=6.8
- **neutral_loss** r4.slp:5688 — hit@5778 opener=p2_action44 dist_90f_prior=6.7

Appended 10 new gap(s) to `scenarios/gaps.json` (711 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
