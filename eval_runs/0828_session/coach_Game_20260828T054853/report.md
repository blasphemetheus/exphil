# Coach Report — 20260828_171201

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.28 |
| Approaches (total) | 6 |
| Conversions (total) | 2 |
| Neutral losses (opened up) | 8 |
| Dropped punishes | 1 |
| Death sequences | 4 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T054853.slp | 1.28 | 6 | 2 | 8 | 1 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260828T054853.slp:4228 — opening@4228 start_pct=108.2 window=120f
- **neutral_loss** Game_20260828T054853.slp:508 — hit@598 opener=p2_action63 dist_90f_prior=30.3
- **neutral_loss** Game_20260828T054853.slp:1357 — hit@1447 opener=p2_action63 dist_90f_prior=5.4
- **neutral_loss** Game_20260828T054853.slp:1597 — hit@1687 opener=p2_action63 dist_90f_prior=26.6
- **neutral_loss** Game_20260828T054853.slp:2611 — hit@2701 opener=p2_action69 dist_90f_prior=35.4
- **neutral_loss** Game_20260828T054853.slp:3462 — hit@3552 opener=p2_action44 dist_90f_prior=34.5
- **neutral_loss** Game_20260828T054853.slp:3761 — hit@3851 opener=p2_action67 dist_90f_prior=50.6
- **neutral_loss** Game_20260828T054853.slp:4578 — hit@4668 opener=p2_action63 dist_90f_prior=17.0
- **neutral_loss** Game_20260828T054853.slp:4895 — hit@4985 opener=p2_action50 dist_90f_prior=25.4
- **death_sequence** Game_20260828T054853.slp:861 — death@862 elapsed=1f opener=p2_action42

Appended 10 new gap(s) to `scenarios/gaps.json` (1249 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
