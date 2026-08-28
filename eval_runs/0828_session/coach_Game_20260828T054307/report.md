# Coach Report — 20260828_171156

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.55 |
| Approaches (total) | 7 |
| Conversions (total) | 0 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T054307.slp | 0.55 | 7 | 0 | 7 | 1 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260828T054307.slp:626 — opening@626 start_pct=3.7 window=120f
- **neutral_loss** Game_20260828T054307.slp:1269 — hit@1359 opener=p2_action215 dist_90f_prior=27.8
- **neutral_loss** Game_20260828T054307.slp:1663 — hit@1753 opener=p2_action67 dist_90f_prior=77.6
- **neutral_loss** Game_20260828T054307.slp:2554 — hit@2644 opener=p2_action67 dist_90f_prior=19.9
- **neutral_loss** Game_20260828T054307.slp:3359 — hit@3449 opener=p2_action215 dist_90f_prior=8.3
- **neutral_loss** Game_20260828T054307.slp:4193 — hit@4283 opener=p2_action67 dist_90f_prior=5.2
- **neutral_loss** Game_20260828T054307.slp:5285 — hit@5375 opener=p2_action215 dist_90f_prior=44.1
- **neutral_loss** Game_20260828T054307.slp:5742 — hit@5832 opener=p2_action213 dist_90f_prior=66.0
- **death_sequence** Game_20260828T054307.slp:405 — death@406 elapsed=1f opener=p2_action20
- **death_sequence** Game_20260828T054307.slp:2921 — death@2922 elapsed=1f opener=p2_action26

Appended 10 new gap(s) to `scenarios/gaps.json` (1219 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
