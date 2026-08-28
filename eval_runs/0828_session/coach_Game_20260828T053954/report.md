# Coach Report — 20260828_171153

Set: 1 game(s). Bot port: 1.

## Aggregate

| Metric | Value |
|--------|-------|
| Armed approaches/min (mean) | 0.87 |
| Approaches (total) | 10 |
| Conversions (total) | 0 |
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
| Game_20260828T053954.slp | 0.87 | 10 | 0 | 5 | 1 | 3 | 0 |

## Top drill candidates (appended to gap ledger)

- **dropped_punish** Game_20260828T053954.slp:1586 — opening@1586 start_pct=51.0 window=120f
- **neutral_loss** Game_20260828T053954.slp:524 — hit@614 opener=p2_action213 dist_90f_prior=3.2
- **neutral_loss** Game_20260828T053954.slp:792 — hit@882 opener=p2_action215 dist_90f_prior=11.7
- **neutral_loss** Game_20260828T053954.slp:1851 — hit@1941 opener=p2_action56 dist_90f_prior=23.3
- **neutral_loss** Game_20260828T053954.slp:3372 — hit@3462 opener=p2_action65 dist_90f_prior=43.8
- **neutral_loss** Game_20260828T053954.slp:3676 — hit@3766 opener=p2_action213 dist_90f_prior=8.3
- **death_sequence** Game_20260828T053954.slp:1068 — death@1069 elapsed=1f opener=p2_action27
- **death_sequence** Game_20260828T053954.slp:3160 — death@3161 elapsed=1f opener=p2_action24
- **death_sequence** Game_20260828T053954.slp:4155 — death@4156 elapsed=1f opener=p2_action20

Appended 9 new gap(s) to `scenarios/gaps.json` (1199 total).

## Drills that already exist for these gaps

Existing scenario-manifest types: opponent_behind, tech_chase, edgeguard, getup, idle_deadlock.
Gap types this set surfaced: neutral_losses, dropped_punishes, deaths.
