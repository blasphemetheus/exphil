# Coach Report — 20260828_171148

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.0 |
| Approaches (total) | 10 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 5 |
| Dropped punishes | 1 |
| Death sequences | 3 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T053456.slp | 0.0 | 10 | 1 | 5 | 1 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260828T053456.slp:1009 — opening@1009 start_pct=35.5 window=120f
- **neutral_loss** Game_20260828T053456.slp:525 — hit@615 opener=p2_action213 dist_90f_prior=22.5
- **neutral_loss** Game_20260828T053456.slp:1311 — hit@1401 opener=p2_action68 dist_90f_prior=17.4
- **neutral_loss** Game_20260828T053456.slp:2247 — hit@2337 opener=p2_action66 dist_90f_prior=77.4
- **neutral_loss** Game_20260828T053456.slp:2917 — hit@3007 opener=p2_action213 dist_90f_prior=5.1
- **neutral_loss** Game_20260828T053456.slp:3646 — hit@3736 opener=p2_action50 dist_90f_prior=44.1
- **death_sequence** Game_20260828T053456.slp:2440 — death@2441 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260828T053456.slp:2716 — death@2717 elapsed=1f opener=p2_action42
- **death_sequence** Game_20260828T053456.slp:4382 — death@4383 elapsed=1f opener=p2_action14

Appended 9 new gap(s) to `scenarios/gaps.json` (1170 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
