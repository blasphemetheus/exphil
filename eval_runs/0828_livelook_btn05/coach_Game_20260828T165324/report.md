# Coach Report — 20260828_171129

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.63 |
| Approaches (total) | 5 |
| Conversions (total) | 1 |
| Neutral losses (opened up) | 7 |
| Dropped punishes | 1 |
| Death sequences | 2 |
| Passivity windows | 0 |

Armed-approaches/min ≈ 0 with conversions present is the gate-10 story:
the bot converts once in, but does not initiate. Passivity windows and
dropped punishes localize that to drillable moments.

## Per game

| Game | armed/min | appr | conv | neutral_loss | dropped | deaths | passive |
|------|-----------|------|------|--------------|---------|--------|---------|
| Game_20260828T165324.slp | 0.63 | 5 | 1 | 7 | 1 | 2 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260828T165324.slp:1855 — opening@1855 start_pct=66.5 window=120f
- **neutral_loss** Game_20260828T165324.slp:793 — hit@883 opener=p2_action66 dist_90f_prior=46.0
- **neutral_loss** Game_20260828T165324.slp:2319 — hit@2409 opener=p2_action57 dist_90f_prior=69.7
- **neutral_loss** Game_20260828T165324.slp:2800 — hit@2890 opener=p2_action57 dist_90f_prior=36.6
- **neutral_loss** Game_20260828T165324.slp:3273 — hit@3363 opener=p2_action215 dist_90f_prior=27.2
- **neutral_loss** Game_20260828T165324.slp:3971 — hit@4061 opener=p2_action69 dist_90f_prior=35.7
- **neutral_loss** Game_20260828T165324.slp:4253 — hit@4343 opener=p2_action53 dist_90f_prior=77.8
- **neutral_loss** Game_20260828T165324.slp:4901 — hit@4991 opener=p2_action213 dist_90f_prior=29.8
- **death_sequence** Game_20260828T165324.slp:3706 — death@3707 elapsed=1f opener=p2_action342
- **death_sequence** Game_20260828T165324.slp:5384 — death@5385 elapsed=1f opener=p2_action20

Appended 10 new gap(s) to `scenarios/gaps.json` (1101 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
