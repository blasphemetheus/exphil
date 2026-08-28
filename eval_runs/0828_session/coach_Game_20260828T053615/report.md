# Coach Report — 20260828_171150

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.35 |
| Approaches (total) | 9 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 6 |
| Dropped punishes | 3 |
| Death sequences | 4 |
| Passivity windows | 3 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T053615.slp | 1.35 | 9 | 0 | 6 | 3 | 4 | 3 |

## Top drill candidates (appended to gap ledger)

- **passivity_window** Game_20260828T053615.slp:1900 — passive_run=323f mid_dist=6.5
- **passivity_window** Game_20260828T053615.slp:6628 — passive_run=430f mid_dist=8.3
- **passivity_window** Game_20260828T053615.slp:7468 — passive_run=475f mid_dist=8.3
- **dropped_punish** Game_20260828T053615.slp:470 — opening@470 start_pct=40.2 window=120f
- **dropped_punish** Game_20260828T053615.slp:1857 — opening@1857 start_pct=45.1 window=120f
- **dropped_punish** Game_20260828T053615.slp:2841 — opening@2841 start_pct=65.3 window=120f
- **neutral_loss** Game_20260828T053615.slp:520 — hit@610 opener=p2_action213 dist_90f_prior=72.5
- **neutral_loss** Game_20260828T053615.slp:1135 — hit@1225 opener=p2_action50 dist_90f_prior=34.6
- **neutral_loss** Game_20260828T053615.slp:3228 — hit@3318 opener=p2_action53 dist_90f_prior=18.2
- **neutral_loss** Game_20260828T053615.slp:4465 — hit@4555 opener=p2_action356 dist_90f_prior=59.0

Appended 10 new gap(s) to `scenarios/gaps.json` (1180 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths, passivity.
