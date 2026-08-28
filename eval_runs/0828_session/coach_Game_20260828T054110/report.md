# Coach Report — 20260828_171155

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 1.09 |
| Approaches (total) | 8 |
| Conversions (total) | 1 |
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
| Game_20260828T054110.slp | 1.09 | 8 | 1 | 7 | 1 | 4 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260828T054110.slp:1242 — opening@1242 start_pct=122.2 window=120f
- **neutral_loss** Game_20260828T054110.slp:448 — hit@538 opener=p2_action213 dist_90f_prior=55.3
- **neutral_loss** Game_20260828T054110.slp:833 — hit@923 opener=p2_action213 dist_90f_prior=20.2
- **neutral_loss** Game_20260828T054110.slp:1633 — hit@1723 opener=p2_action213 dist_90f_prior=19.6
- **neutral_loss** Game_20260828T054110.slp:2042 — hit@2132 opener=p2_action213 dist_90f_prior=63.1
- **neutral_loss** Game_20260828T054110.slp:3668 — hit@3758 opener=p2_action63 dist_90f_prior=28.4
- **neutral_loss** Game_20260828T054110.slp:4508 — hit@4598 opener=p2_action67 dist_90f_prior=46.7
- **neutral_loss** Game_20260828T054110.slp:6047 — hit@6137 opener=p2_action50 dist_90f_prior=42.7
- **death_sequence** Game_20260828T054110.slp:2315 — death@2316 elapsed=1f opener=p2_action42
- **death_sequence** Game_20260828T054110.slp:4000 — death@4001 elapsed=1f opener=p2_action345

Appended 10 new gap(s) to `scenarios/gaps.json` (1209 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
